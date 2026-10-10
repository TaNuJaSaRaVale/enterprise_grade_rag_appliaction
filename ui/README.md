# `ui/` — the Streamlit front end

The chat interface ("KubeRAG"). It holds no RAG logic: it sends the question to the backend and renders what comes back.

## Files

| File | Role |
|---|---|
| `app.py` | The UI in use — live demo, Docker image and local runs |
| `requirements.txt` | Runtime dependencies that Streamlit Community Cloud installs |
| `eval_scorecard.json` | Small summary of an eval run, written by `python -m evals.export_scorecard` |
| `st_cloud_ui.py` | Earlier, simpler UI. It only calls `http://localhost:8000` and has no confidence badge. Kept for reference; nothing runs it. |

## Two ways to reach the backend

`app.py` decides from one environment variable:

| `BACKEND_URL` | Mode | Where it is used |
|---|---|---|
| set | HTTP `POST {BACKEND_URL}/query` | Docker image (`start.sh` sets `http://127.0.0.1:8000`), local two-terminal setup |
| not set | Imports `app.main` and calls `query()` in the same process | Streamlit Community Cloud, which runs a single process |

In-process details:

- `_inprocess_backend()` is wrapped in `st.cache_resource`, so the backend is imported and the guardrails initialised once per server, not once per user or per rerun.
- The repo root is put first on `sys.path`. This file is `ui/app.py`; without that, `import app` would import this UI file instead of the backend package `app/`.
- `query()` returns a dict, or a `JSONResponse` for errors and rate limits. `call_backend()` handles both.

## What the page shows

| Element | Source |
|---|---|
| "Thinking" steps | `thought_process` (the agent's `plan`) |
| Retrieved context, one expander per chunk | `sources` |
| Answer, typed out character by character | `answer` (the typing effect is cosmetic; the backend does not stream) |
| Confidence badge | `confidence.level`, `top_score`, `reason` — High green, Medium blue, Low orange, Not in docs red |
| Source file names | `source_files` |
| Sidebar: knowledge-base topics and example questions | Constants in `app.py` |
| "New conversation" button | Clears the messages and creates a new `session_id` |

**Session and memory.** Each browser session gets a UUID, sent as `thread_id`. The backend's conversation memory is keyed by it, so users do not share history.

**Theme.** The Kubernetes-blue styling is CSS inside `app.py`, not `.streamlit/config.toml`: the Docker image copies only `app/` and `ui/`, so a root config file would not reach it. Colours use transparency so they work in light and dark mode.

## Running it

```bash
# two processes
uvicorn app.main:app --port 8000
BACKEND_URL=http://localhost:8000 streamlit run ui/app.py

# one process (same as the live demo)
streamlit run ui/app.py        # with BACKEND_URL unset
```

`.env.example` sets `BACKEND_URL=http://localhost:8000`, so a copied `.env` gives HTTP mode. Remove that line to try in-process mode.

## Deployment notes

- **Live demo:** https://agentic-rag-d.streamlit.app (Streamlit Community Cloud, in-process mode). It sleeps when idle; the first question after a wake-up can take about a minute while models load.
- **Secrets** are set in the Streamlit Cloud dashboard, not in the repo. The backend reads them as environment variables.
- **`requirements.txt` here must stay in sync with `../requirements-app.txt`.** Streamlit Cloud reads the file next to the app before the root one. It leaves out the eval libraries, the ingestion parsers and torch/sentence-transformers.
- **The demo reads the re-ingested collection** since 2026-10-10, through one secret: `QDRANT_COLLECTION = "enterprise_rag_v2"`. Deleting the secret rolls back to `enterprise_rag`, which is kept for now as the rollback.

## What changed while building this, and why

| Change | Why |
|---|---|
| Added in-process backend mode | Streamlit Community Cloud runs one process and cannot start a separate FastAPI server. Calling the same `query()` function keeps one code path for guardrails, rate limit and confidence. |
| `ui/requirements.txt` added | The full `requirements.txt` (evals, parsers, torch) is too heavy for the free host. |
| `sentence-transformers` left out of the runtime | Its 768-dim fallback cannot query the 3072-dim collection anyway. |
| Confidence badge and source files under each answer | Shows the user how well the docs support the answer, from retrieval scores rather than the model's opinion. |
| Rebranded to "KubeRAG" with a themed layout, topic chips and example questions | A visitor (or recruiter) sees what the knowledge base covers and can try it in one click, including a question that shows the honest "not in the docs" path. |
| Per-session `thread_id` | The earlier shared default mixed every visitor's conversation. |
| `logfire.instrument_requests()` disabled in `app.py` | It hit an OpenTelemetry error (`MeterProvider.get_meter() got multiple values for argument 'version'`). |
| Request timeout 60 s → 120 s | A cold backend can take longer than a minute on the first request. |

`eval_scorecard.json` was re-exported on 2026-10-10 from run `20261009-143328` (all 17 in-scope goldens scored). The full comparison of the three runs is in [`../evals/README.md`](../evals/README.md).
