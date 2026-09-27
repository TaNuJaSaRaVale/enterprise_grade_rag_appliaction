"""
Evaluation pipeline — runs every golden through the real RAG system and records
what the judge (evals/metrics.py) and a human debugger need.

This module only *runs* the system; it does not score anything.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVALS_DIR = Path(__file__).parent
DEFAULT_GOLDEN_PATH = EVALS_DIR / "golden_dataset.json"
RESULTS_DIR = EVALS_DIR / "results"
RESULTS_FILE = "pipeline_results.jsonl"

REQUIRED_FIELDS = ("id", "question", "ground_truth")

# Must match the format produced in app/agents/nodes/retriever.py:
#   "[1] Source: pods_autoscale.html\n<chunk text>"
_DOC_PATTERN = re.compile(r"^\[\d+\] Source: (?P<source>[^\n]*)\n(?P<text>.*)$", re.DOTALL)

# Must match the plan marker added in app/agents/nodes/responder.py on a Portkey cache hit
CACHE_HIT_MARKER = "Cache: Hit ⚡"

# Heavy RAG stack (graph, Qdrant, NeMo) — loaded lazily and only once per process
_runtime = None


def load_golden_dataset(path: Path = DEFAULT_GOLDEN_PATH) -> list[dict]:
    """
    Load and validate the golden dataset. Fails fast — before any LLM quota is
    spent — and reports every problem at once.
    """
    with open(path, encoding="utf-8") as f:
        goldens = json.load(f)

    if not isinstance(goldens, list):
        raise ValueError(f"{path} must contain a JSON list of goldens, got {type(goldens).__name__}")

    problems, seen_ids = [], set()
    for index, golden in enumerate(goldens):
        label = golden.get("id", f"#{index}") if isinstance(golden, dict) else f"#{index}"
        if not isinstance(golden, dict):
            problems.append(f"golden {label} is not a JSON object")
            continue

        for field in REQUIRED_FIELDS:
            value = golden.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"golden '{label}' is missing '{field}' (or it is empty)")

        # Resume logic skips ids already done — a duplicate id would be silently skipped
        if golden.get("id") in seen_ids:
            problems.append(f"duplicate id '{golden['id']}'")
        seen_ids.add(golden.get("id"))

    if problems:
        raise ValueError(f"Invalid golden dataset {path}:\n  - " + "\n  - ".join(problems))

    return goldens


def _split_document(doc: str) -> tuple[str, str]:
    """
    Split a retriever document into (source, text).
    The judge must see only the chunk text, not our "[n] Source: ..." label.
    Raises if the format changed in retriever.py, rather than silently
    sending labels to the judge.
    """
    match = _DOC_PATTERN.match(doc)
    if not match:
        raise ValueError(
            "Retrieved document is not in the expected '[n] Source: <name>\\n<text>' format "
            f"(did app/agents/nodes/retriever.py change?). Starts with: {doc[:80]!r}"
        )
    return match.group("source").strip(), match.group("text").strip()


def _get_runtime():
    """
    Import the RAG stack lazily (keeps `import evals.pipeline` instant) and
    initialise guardrails exactly once. Without initialize_rails(), guard()
    silently passes everything — the eval would then report guardrails as
    perfect when they never ran.
    """
    global _runtime
    if _runtime is None:
        from app.agents.graph import rag_agent
        from app.guardrails import initialize_rails, guard

        initialize_rails()
        _runtime = (rag_agent, guard)
    return _runtime


def run_single(golden: dict, run_id: str) -> dict:
    """
    Run ONE golden through the same path as POST /query (guard → LangGraph)
    and return a record with everything the judge and a debugger need.
    Never raises for a per-golden failure: the error is recorded instead.
    """
    rag_agent, guard = _get_runtime()
    question = golden["question"]
    expected_source = golden.get("expected_source")
    # Own memory thread per golden (and per run) so history never leaks between questions
    thread_id = f"eval-{run_id}-{golden['id']}"

    record = {
        "id": golden["id"],
        "question": question,
        "ground_truth": golden["ground_truth"],
        "expected_source": expected_source,
        "type": golden.get("type"),
        "tags": golden.get("tags", []),
        "thread_id": thread_id,
        "guard_fired": False,
        "intent": None,
        "search_query": None,
        "answer": None,
        "retrieved_contexts": [],
        "retrieved_sources": [],
        "expected_source_retrieved": None,
        "cache_hit": False,
        "status": None,
        "latency_s": None,
        "error": None,
    }

    start = time.perf_counter()
    try:
        # Gate 1 — same as app/main.py
        fired, guard_response = guard(question)
        if fired:
            record.update(guard_fired=True, answer=guard_response, status="Blocked by guardrails.")
            return record

        # Gate 2 — initial state must mirror app/main.py query() so we evaluate the real system
        initial_state = {
            "messages": [{"role": "user", "content": question}],
            "current_query": question,
            "documents": [],
            "plan": ["Start"],
            "status": "Initializing Graph...",
        }
        final = rag_agent.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})

        split_docs = [_split_document(doc) for doc in final.get("documents", [])]
        sources = [source for source, _ in split_docs]
        conversational = final.get("current_query") == "CONVERSATIONAL"

        record.update(
            intent="conversational" if conversational else "technical",
            search_query=None if conversational else final.get("current_query"),
            answer=final.get("final_answer"),
            retrieved_contexts=[text for _, text in split_docs],
            retrieved_sources=sources,
            expected_source_retrieved=(expected_source in sources) if expected_source else None,
            cache_hit=CACHE_HIT_MARKER in final.get("plan", []),
            status=final.get("status"),
        )
    except Exception as e:  # not BaseException — Ctrl+C must still stop the run
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        record["latency_s"] = round(time.perf_counter() - start, 2)

    return record


# ── Run bookkeeping ────────────────────────────────────────────────────────────

def _git_info() -> dict:
    """Read-only git lookup: which code produced these results, and was it committed?"""
    def git(*args):
        # rstrip only: porcelain lines start with a status column that may be a space (" M file")
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=EVALS_DIR).stdout.rstrip("\n")

    try:
        dirty_files = [line[3:] for line in git("status", "--porcelain").splitlines() if line]
        return {"git_commit": git("rev-parse", "--short", "HEAD").strip() or None,
                "git_dirty": bool(dirty_files), "dirty_files": dirty_files}
    except FileNotFoundError:  # git not installed
        return {"git_commit": None, "git_dirty": None, "dirty_files": []}


def load_results(results_path: Path) -> dict[str, dict]:
    """Read a results JSONL file. The LAST line per id wins (a resumed retry supersedes its old error)."""
    results = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    results[record["id"]] = record
    return results


def _warm_up():
    """Pay one-time startup costs (NeMo, FlashRank model, Gemini probe) before timing starts."""
    from app.services.retrieval.embedding import embed_query
    from app.services.retrieval.ranking_service import rerank_documents

    _get_runtime()
    embed_query("warm-up")
    rerank_documents("warm-up", [{"content": "warm-up", "source": "warm-up"}])


def summarize(results: dict[str, dict]) -> dict:
    records = list(results.values())
    ok = [r for r in records if not r["error"]]
    with_source = [r for r in ok if r["expected_source"] and not r["guard_fired"]]
    timed = [r["latency_s"] for r in ok if not r["cache_hit"] and not r["guard_fired"]]
    return {
        "total": len(records),
        "completed": len(ok),
        "errors": len(records) - len(ok),
        "error_ids": [r["id"] for r in records if r["error"]],
        # For RAG goldens a guardrail block is a bug — the question never reached RAG
        "guard_blocked_ids": [r["id"] for r in ok if r["guard_fired"]],
        "source_hit_rate": round(sum(r["expected_source_retrieved"] for r in with_source) / len(with_source), 3)
                           if with_source else None,
        "source_miss_ids": [r["id"] for r in with_source if not r["expected_source_retrieved"]],
        "avg_latency_s": round(sum(timed) / len(timed), 2) if timed else None,
        "cache_hits": sum(r["cache_hit"] for r in ok),
    }


def run_pipeline(goldens: list[dict], run_id: str, delay_s: float = 2.0) -> dict:
    """
    Run every golden, appending each record to JSONL immediately (crash-safe).
    Goldens already completed without error in this run_id are skipped (resume).
    """
    import logfire

    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / RESULTS_FILE

    done = {gid for gid, r in load_results(results_path).items() if not r["error"]}
    pending = [g for g in goldens if g["id"] not in done]
    print(f"Run {run_id}: {len(goldens)} goldens, {len(done)} already done, {len(pending)} to run.")

    if pending:
        print("Warming up (NeMo, reranker, embeddings)...")
        _warm_up()

    for i, golden in enumerate(pending, start=1):
        with logfire.span("🧪 Eval golden {golden_id}", golden_id=golden["id"], run_id=run_id):
            record = run_single(golden, run_id)
        with open(results_path, "a", encoding="utf-8") as f:  # append immediately — crash-safe
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        outcome = f"ERROR {record['error'][:80]}" if record["error"] else (
            "BLOCKED by guardrails" if record["guard_fired"] else
            f"sources={record['retrieved_sources']} hit={record['expected_source_retrieved']}")
        print(f"[{i}/{len(pending)}] {golden['id']:10} {record['latency_s']:6.1f}s  {outcome}")

        if i < len(pending):
            time.sleep(delay_s)  # stay under Groq / Gemini per-minute rate limits

    summary = summarize(load_results(results_path))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run golden questions through the RAG system.")
    parser.add_argument("--limit", type=int, help="only run the first N goldens (test runs)")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between goldens (default 2)")
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDEN_PATH)
    args = parser.parse_args()

    # Validate first — fail fast before any setup or quota is spent
    goldens = load_golden_dataset(args.goldens)
    if args.limit:
        goldens = goldens[: args.limit]

    # Configure Logfire BEFORE the lazy app imports so every span is captured (see app/main.py)
    import logfire
    from dotenv import load_dotenv
    load_dotenv()
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="rag-evals", console=False)

    # A dependency enables root INFO logging; NeMo then logs every internal event and httpx
    # every request, drowning the progress lines. Raise only these to WARNING so real
    # warnings/errors (e.g. 429s) still show. Spans still go to Logfire.
    import logging
    for noisy in ("nemoguardrails", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run_id = args.resume or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / run_id
    if args.resume and not run_dir.exists():
        sys.exit(f"No run to resume at {run_dir}")

    git = _git_info()
    if git["git_dirty"]:
        print(f"⚠️  Uncommitted changes in {len(git['dirty_files'])} file(s) — results won't match "
              f"commit {git['git_commit']} exactly (recorded in run_info.json).")

    if not args.resume:
        from app.config import settings
        from app.services.retrieval.ranking_service import RERANK_MODEL, MIN_RERANK_SCORE
        run_dir.mkdir(parents=True, exist_ok=True)
        run_info = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            **git,
            "golden_file": str(args.goldens.name),
            "golden_count": len(goldens),
            "limit": args.limit,
            "models": {"llm": settings.GROQ_MODEL, "fast_llm": settings.GROQ_FAST_MODEL,
                       "reranker": RERANK_MODEL, "min_rerank_score": MIN_RERANK_SCORE},
        }
        (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    try:
        summary = run_pipeline(goldens, run_id, delay_s=args.delay)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Finished goldens are saved. Resume with:\n"
              f"  python -m evals.pipeline --resume {run_id}")
        sys.exit(130)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {RESULTS_DIR / run_id}")


if __name__ == "__main__":
    main()
