# `app/` — the backend

Everything that answers a question lives here: the FastAPI service, the LangGraph agent, the guardrails, the LLM gateway, retrieval, and the offline ingestion job.

## Folder map

| Path | What it holds | Details |
|---|---|---|
| `main.py` | FastAPI service: `/query`, `/health`, `/graph`, rate limit, confidence badge | this file |
| `config.py` | One `Settings` object that reads every key and model name from `.env` | this file |
| `agents/` | LangGraph agent: planner → retriever → responder | [agents/README.md](agents/README.md) |
| `guardrails/` | NeMo Guardrails gate that runs before the agent | [guardrails/README.md](guardrails/README.md) |
| `gateway/` | Portkey client (fallback, retry, cache) used for every answer LLM call | [gateway/README.md](gateway/README.md) |
| `services/retrieval/` | Embeddings, Qdrant search, FlashRank reranking | [services/retrieval/README.md](services/retrieval/README.md) |
| `ingestion/` | Offline job: parse files → chunk → embed → index in Qdrant | [ingestion/README.md](ingestion/README.md) |

## How one request flows

```
POST /query {q, thread_id}
  1. input limits        q is 1–1000 chars, thread_id ≤ 100 chars (pydantic)
  2. rate limit          20 requests per minute, global
  3. guardrails.guard()  layer 1: off-topic / jailbreak / greeting → fixed reply, stop
  4. rag_agent.invoke()  planner → (retriever) → responder
  5. response            answer, thought_process, status, sources, source_files, confidence
```

## `main.py`

**Endpoints**

| Endpoint | Purpose |
|---|---|
| `GET /` | "API is live" message |
| `GET /health` | Liveness check for a container platform. Makes no LLM or database call on purpose. |
| `GET /graph` | PNG drawing of the agent graph |
| `POST /query` | The RAG flow |

**Rate limit.** `_rate_limited()` keeps the timestamps of the last minute in a `deque` and rejects the request with HTTP 429 once 20 are in the window. It is global, not per-IP: in the Docker image every request reaches the API from the Streamlit UI on `127.0.0.1`, so a per-IP limit would see one client. A `threading.Lock` protects the deque because FastAPI runs sync endpoints in a thread pool.

**Confidence badge.** `_confidence()` turns the reranker scores into a level. It never asks the LLM how sure it is.

| Best cross-encoder score | Level |
|---|---|
| ≥ 0.7 | High |
| ≥ 0.3 | Medium |
| below 0.3 | Low |
| no chunk kept | Not in docs |
| conversational or off-topic turn | n/a |

The thresholds were calibrated on this corpus: chunks that answered golden questions scored 0.98–0.99, unrelated chunks about 0.0.

**Other details**

- `_source_files()` reads the unique file names out of the `[n] Source: <file>` label that the retriever puts on each chunk.
- `thread_id` has no shared default. A request without one gets a new UUID, so two anonymous users never share memory.
- `retrieval_scores` is reset to `[]` in the initial state of every turn. Without that, the checkpointer would carry the previous turn's scores into a turn that retrieved nothing.
- `logfire.configure()` runs before any app import, so spans from every module are captured.
- Errors return a JSON body with HTTP 500 and a polite message; the real error goes to Logfire.

## `config.py`

| Setting | Value / source | Note |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | planner + answer synthesis |
| `GROQ_FAST_MODEL` | `openai/gpt-oss-20b` | guardrails LLM object + Portkey fallback |
| `GROQ_SLUG`, `GROQ_SLUG_2` | `rag`, `brag` | Portkey provider slugs for the primary and fallback Groq keys |
| `QDRANT_COLLECTION` | env `QDRANT_COLLECTION`, default `enterprise_rag` | see history below |
| `DATABASE_URL` | env, optional | Postgres connection string for conversation memory; unset means in-process memory (see [agents/README.md](agents/README.md#memory)) |
| `LANGSMITH_*` | env | copied into the `LANGCHAIN_*` variables LangChain reads |

Model names are constants in code, keys come from `.env`. A model change must be made here **and** in the Portkey dashboard config (see [gateway/README.md](gateway/README.md)).

## What changed while building this, and why

| Change | Why |
|---|---|
| Llama models → `openai/gpt-oss-120b` / `gpt-oss-20b` | Groq retired the Llama models the project started with. |
| Added input length limits on `q` and `thread_id` | A public demo on free tiers: one huge prompt would burn the daily LLM quota. |
| Added the global rate limit | Same reason — protect free-tier quotas from a burst of requests. |
| Removed the shared default `thread_id` | With a shared default, every caller without an id read and wrote the same conversation memory. |
| Added `/health` with no downstream calls | A platform health check that calls an LLM would spend quota every few seconds. |
| Added `confidence` and `source_files` to the response | The LLM's own confidence is not trustworthy; the reranker score is a measured signal of whether the docs support the answer. This is the project's "it knows when it doesn't know" feature. |
| `QDRANT_COLLECTION` became env-overridable | The re-ingest with the new chunker was built in a second collection (`enterprise_rag_v2`) while the live demo kept reading `enterprise_rag`. Switching the demo is one environment variable; rolling back is deleting it. |
| Sync `/query` endpoint calling `rag_agent.invoke` | Running the graph synchronously keeps Logfire's context variables, so the trace stays in one piece. |
| `DATABASE_URL` setting and `durability="exit"` on the graph call | Conversation memory moved to Postgres so it survives restarts; saving once per turn keeps it to one database write per question. |
