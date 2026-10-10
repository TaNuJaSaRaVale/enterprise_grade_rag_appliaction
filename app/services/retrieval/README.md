# `app/services/retrieval/` — embeddings, vector search, reranking

Three small services. The retriever node (`app/agents/nodes/retriever.py`) and the ingestion job (`app/ingestion/processor.py`) are their only callers.

## Files

| File | Public functions | Used by |
|---|---|---|
| `embedding.py` | `embed_query()`, `embed_texts()`, `get_embedding_dim()` | Qdrant search (queries), ingestion (documents) |
| `qdrant_service.py` | `search_enterprise_knowledge(query, limit)` | Retriever node |
| `ranking_service.py` | `rerank_documents(query, documents, top_n, min_score)` | Retriever node |

## Two-stage retrieval

```
query ─► Gemini embedding ─► Qdrant: 15 nearest chunks ─► FlashRank cross-encoder ─► top 5 with score ≥ 0.1
              (fast, approximate)                              (slower, precise)
```

Vector search compares one vector per chunk with one vector for the query. It is fast but coarse. A cross-encoder reads the query and the chunk **together** and scores how well the chunk answers the query. It is too slow to run over the whole collection, so it only re-scores the 15 candidates.

## `embedding.py`

| Item | Value |
|---|---|
| Primary model | `models/gemini-embedding-2-preview`, 3072 dimensions |
| Fallback model | `all-mpnet-base-v2` (sentence-transformers, local), 768 dimensions |
| Batch size | 50 texts per request |
| Retry waits | 10, 20, 40, 60, 60 seconds — 6 attempts over about 3 minutes |

- The model is chosen once per process, lazily, by `_init()`: it sends one probe embedding to Gemini and loads the local fallback if that fails.
- **The fallback cannot query a Gemini collection.** Its vectors are 768-dim and the collection is 3072-dim. The ingestion job checks the dimension and stops (`_check_dimension()`); the deployed image does not install sentence-transformers at all.
- Retries apply to `embed_texts()` (ingestion) only. `embed_query()` makes one call, so a user's question never waits minutes.
- Both 429 (our quota) and 503 (Google overloaded) are retried. Both clear without any action.

## `qdrant_service.py`

`search_enterprise_knowledge()` embeds the query, calls `query_points` on `settings.QDRANT_COLLECTION`, and returns a list of `{content, source, score}`.

On any error it logs and returns `[]`. The agent then answers "not in the docs" rather than crashing — which also means a Qdrant outage looks the same to the user as a question the docs don't cover. The error is visible in Logfire.

## `ranking_service.py`

| Item | Value |
|---|---|
| Library | FlashRank — a quantised ONNX cross-encoder that runs locally on CPU |
| Model | `ms-marco-MiniLM-L-12-v2` |
| `MIN_RERANK_SCORE` | 0.1 (scores are 0–1) |
| `top_n` | 5 |

- The ranker is created on first use (`_get_ranker()`), so `logfire.configure()` has already run. The model cache is `/tmp/flashrank`; the Docker image downloads the model at build time.
- Results come back sorted by score. The loop stops at the first score below `min_score` or once `top_n` chunks are kept. Each kept chunk gets a `rerank_score`.
- It can return an empty list. That is the signal for the "not in the docs" path.
- If reranking itself fails, it returns the first `top_n` chunks in Qdrant order with no `rerank_score`. The retriever records those scores as 0.0, so the confidence badge reads Low rather than claiming support it did not measure.

The `rerank_score` of the best chunk is what the API turns into the confidence badge (see [`../../README.md`](../../README.md)).

## What changed while building this, and why

| Change | Why |
|---|---|
| Reranker TinyBERT (FlashRank default) → `ms-marco-MiniLM-L-12-v2` | It ranks clearly better on this corpus (0.99 against 0.56 on the Intel paging document) for about 1–2 s of CPU per 15 passages. |
| Added `MIN_RERANK_SCORE = 0.1` | Before, the top 5 were always passed on, even when all were unrelated. Unrelated context invites hallucinated answers. |
| Added `rerank_score` to each kept chunk | Needed for the confidence badge. |
| Retry waits 1+2+4 s → 10/20/40/60/60 s | Gemini's free-tier limit resets per minute. The old backoff spent every retry inside the same minute and whole files were lost. |
| 503 added to the retried errors | The free-tier preview model returns 503 often; it is transient like 429. |
| Lazy model initialisation | Importing the module no longer makes a network call or loads a model. |
| Collection name read from `settings.QDRANT_COLLECTION` (env-overridable) | Lets the demo be switched between `enterprise_rag` and `enterprise_rag_v2` without a code change. |

**Known limit this folder exposed.** With 1,200-character chunks, a question that asks two things can score below 0.1 against every chunk, because each chunk answers only half of it (golden `pwq-002`: top scores 0.018–0.044). The fix is to search and rerank once per sub-question — see `sub_queries` in [`../../agents/README.md`](../../agents/README.md).

**The reranker is sensitive to wording.** Measured on the chunk that answers the second half of `pwq-002`:

| Query given to the reranker | Score on the correct chunk |
|---|---|
| "What is **suggested instead** for a continuous stream of background work?" (the user's words) | 0.611 |
| "What is the **recommended approach** for a continuous stream of background work **in Redis**?" (an LLM rewording) | 0.011 |

Both mean the same to a person, but only the first clears the 0.1 cutoff. This is why the planner is told to copy the user's words into sub-queries instead of rephrasing them. It also means a fixed cutoff can drop a correct chunk when the question is worded unusually; the system then says "not in the docs" rather than guessing.
