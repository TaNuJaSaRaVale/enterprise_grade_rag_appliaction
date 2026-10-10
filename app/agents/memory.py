import logfire
from langgraph.checkpoint.memory import MemorySaver
from app.config import settings

# Neon's free tier allows few connections and suspends the database when idle
POOL_MAX_SIZE = 4
POOL_TIMEOUT_S = 15


def build_checkpointer():
    """
    Conversation memory for the graph, keyed by thread_id.

    DATABASE_URL set   -> Postgres (Neon): memory survives restarts and is shared by instances.
    DATABASE_URL unset -> MemorySaver: in-process, lost on restart (local runs, evals).

    If Postgres cannot be reached at startup the app falls back to MemorySaver instead of
    failing to start — a demo without long-term memory is better than no demo.
    """
    if not settings.DATABASE_URL:
        logfire.info("Memory: in-process (MemorySaver) — DATABASE_URL not set")
        return MemorySaver()

    try:
        # Imported here so local runs and evals without Postgres don't need psycopg installed
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        pool = ConnectionPool(
            conninfo=settings.DATABASE_URL,
            min_size=1,
            max_size=POOL_MAX_SIZE,
            timeout=POOL_TIMEOUT_S,
            # check: Neon closes idle connections when it suspends; test one before handing it out
            check=ConnectionPool.check_connection,
            # autocommit + dict_row are required by PostgresSaver;
            # prepare_threshold=None because Neon's pooled endpoint (PgBouncer) rejects prepared statements
            kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
            open=False,
        )
        pool.open(wait=True, timeout=POOL_TIMEOUT_S)

        saver = PostgresSaver(pool)
        saver.setup()  # creates the checkpoint tables on first run; no-op afterwards
        logfire.info("Memory: Postgres checkpointer ready")
        return saver
    except Exception as e:
        logfire.error(f"Memory: Postgres unavailable, using in-process memory instead: {e}")
        return MemorySaver()
