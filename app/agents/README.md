# `app/agents/` — the LangGraph agent

The agent is a small, explicit graph of three nodes. Each node is a plain function that reads the shared state and returns the keys it wants to change.

## Files

| File | Role |
|---|---|
| `state.py` | `AgentState` — the dictionary every node reads and writes |
| `graph.py` | Wires the nodes, the routing rule and the memory checkpointer; exports `rag_agent` |
| `memory.py` | `build_checkpointer()` — Postgres memory when `DATABASE_URL` is set, in-process memory otherwise |
| `nodes/planner.py` | One LLM call: classify intent, rewrite the question into a search query, split multi-part questions |
| `nodes/retriever.py` | Vector search in Qdrant, de-duplicate, rerank, label chunks |
| `nodes/responder.py` | Write the final answer from the chunks (or say the docs don't cover it) |

## The graph

```
           ┌─ off_topic ──────────────► END   (fixed reply, no retrieval, no second LLM call)
planner ───┼─ conversational ─────────► responder ─► END
           └─ technical ─► retriever ─► responder ─► END
```

`route_planner()` in `graph.py` decides the branch from `state["current_query"]`: the planner writes the sentinel `"OFF_TOPIC"` or `"CONVERSATIONAL"` there, or the real search query.

## `state.py`

| Key | Type | Written by | Meaning |
|---|---|---|---|
| `messages` | list of `{role, content}`, **appended** (`operator.add`) | API, planner (off-topic), responder | Conversation history for this thread |
| `current_query` | str | planner | Search query, or `OFF_TOPIC` / `CONVERSATIONAL` |
| `sub_queries` | list of str | planner | One query per part of a multi-part question; empty otherwise |
| `documents` | list of str | retriever | Chunks formatted as `[n] Source: <file>\n<text>` |
| `retrieval_scores` | list of float | retriever | Cross-encoder score of each kept chunk; drives the confidence badge |
| `plan` | list of str | every node | Human-readable steps shown in the UI ("thought process") |
| `status` | str | every node | One-line status |
| `final_answer` | str | responder / planner (off-topic) | The reply |

Only `messages` accumulates. Every other key is overwritten by the latest node that returns it.

## `nodes/planner.py`

One call to `gpt-oss-120b` through Portkey, with JSON-schema structured output (`PlannerDecision`):

| Field | Meaning |
|---|---|
| `intent` | `conversational` (greetings, thanks, questions about the conversation itself), `technical` (anything that needs the docs, **including follow-ups** on the previous topic) or `off_topic` |
| `search_query` | For technical questions: a standalone, keyword-rich query of 5–15 words, with pronouns resolved from the history |
| `sub_queries` | For questions that ask about two or three distinct things: one question per part, **copied from the user's own words** (maximum `MAX_SUB_QUERIES = 3`). Empty for single-topic questions. |

- **Sub-queries copy, they do not rephrase.** The prompt tells the planner to cut the user's sentence into its parts and only add the shared subject each part needs ("In the Redis work-queue Job, ..."). The reason is the reranker: it is very sensitive to wording (see the change table below).
- **Same-topic questions are not split.** "What options does X have and what does each do?" or "When should I use X vs Y?" stay as one query. Every sub-query costs one more embedding call, so splitting is kept for questions that really need two different parts of the docs.
- `search_query` is always filled, even when `sub_queries` is filled. The retriever searches with both.
- The planner sees only the last 6 messages, each cut to 500 characters. Older turns add tokens and noise without helping intent or rewriting.
- `decide()` raises on failure. `planner_node()` catches that and falls back to `intent="technical"` with the raw user message, so a planner hiccup never fails the request. The guardrails eval calls `decide()` directly because it needs errors to surface.
- The prompt says that technical questions the knowledge base may not cover (Istio, Argo CD, Juniper) are still `technical`. They must reach retrieval so the user gets an honest "not in the docs" answer instead of an off-topic refusal.
- Off-topic turns return the same fixed sentence as the NeMo rail (`OFF_TOPIC_REPLY`), so both guardrail layers look identical to the user.

## `nodes/retriever.py`

1. Build the query list: `current_query` plus any `sub_queries` (duplicates removed).
2. For **each** query (`_search_and_rerank`):
   - `search_enterprise_knowledge(query, limit=15)` — 15 candidates from Qdrant.
   - Drop exact-duplicate chunk texts (the same text can be ingested from overlapping sources).
   - `rerank_documents(query, unique, top_n=5)` — keep at most 5 chunks whose cross-encoder score against that query is at least 0.1.
3. Merge (`_merge`): a chunk found by more than one query keeps its highest score. Each query's best chunk is taken first, then the remaining slots are filled by score, up to 5 in total. This stops one part of a multi-part question from filling every slot.
4. Sort the final chunks by score and label each `[n] Source: <file>` so the responder can cite `[1]`, `[2]` and the API can list source files.

It may return zero documents. That is intended: the responder then takes the "not in the docs" path.

For a single-topic question the list holds one query and the result is the same as before decomposition was added. A multi-part question costs one extra embedding call and one extra rerank per sub-query (at most three). In the latest eval run 7 of the 17 in-scope goldens were split.

## `nodes/responder.py`

Picks one of three system prompts:

| Situation | Prompt | Behaviour |
|---|---|---|
| `current_query == "CONVERSATIONAL"` | `CONVERSATIONAL_SYSTEM_PROMPT` | Short reply from the history; no invented technical facts |
| Documents were retrieved | `TECHNICAL_SYSTEM_PROMPT` | Answer only from the numbered excerpts, with `[n]` citations. Commands, code and YAML must be copied from the excerpts. General guidance is allowed **only** when the excerpts do not fully answer, labelled "General guidance (not from the docs):", at most three short bullets and no code. |
| Technical, nothing retrieved | `NO_CONTEXT_SYSTEM_PROMPT` | Must begin "I couldn't find this in the internal documentation." |

- `trim_documents()` is the context-size rule: at most 10,000 characters per chunk and 25,000 in total. Oversized chunks are **trimmed**, not dropped. It is public because `evals/metrics.py` calls it so the judge sees exactly the context the LLM saw.
- The call uses the native Portkey client, not LangChain, so the `x-portkey-cache-status` header can be read and "Cache: Hit ⚡" added to the plan.
- Temperature is 0.1.

## Memory

The graph is compiled with a **checkpointer**: after each turn LangGraph saves the state under the request's `thread_id`, and loads it again on the next turn of the same thread. That is how "what schedule format does **it** use?" can be understood as a question about the CronJob from the previous turn.

`memory.py` picks the checkpointer once, at startup:

| `DATABASE_URL` | Checkpointer | Memory lives | Used for |
|---|---|---|---|
| set | `PostgresSaver` (Neon Postgres) | in the database: survives restarts, shared by all instances | the deployed demo |
| not set | `MemorySaver` | in the process: lost on restart | local runs, evals |

Details that matter:

- **Fallback instead of crash.** If Postgres cannot be reached at startup (wrong URL, Neon down), the app logs an error and starts with `MemorySaver`. The connection attempt waits up to 15 seconds. A failure *during* a request is not caught here: that request returns the API's normal error reply.
- **Connection pool** of at most 4 connections. Each connection is checked before use, because Neon's free tier suspends the database after a few idle minutes and drops its connections. The first request after a suspend is about a second slower. The Neon project is in a US region, close to the Streamlit Cloud servers; from a laptop in India each database round trip takes about 0.3 s, so local runs with `DATABASE_URL` set are 2–3 seconds slower per question.
- **`prepare_threshold=None`**, because Neon's pooled endpoint (PgBouncer) does not support prepared statements.
- **Tables are created automatically** by `saver.setup()` on first start (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`).
- **One write per question.** `main.py` calls the graph with `durability="exit"`, so the state is saved once when the turn ends, not after every node.
- **Evals never write to Postgres.** `evals/pipeline.py` blanks `DATABASE_URL` before the app is imported; each golden has its own throwaway thread and would only fill the free database.

Not done yet: old threads are never deleted, so the tables grow slowly with use. Every saved turn also contains that turn's retrieved chunks.

Known gap found while testing memory: the embedding guardrail (layer 1) answers "What did I just ask you?" with its own capabilities reply, so that message never reaches the graph. Follow-up questions that use the history ("how do I check **its** logs?") work.

## What changed while building this, and why

| Change | Why |
|---|---|
| Planner output: free text → JSON-schema structured output | `gpt-oss` occasionally duplicated free-text output ("query...query..."). A schema makes the output parseable every time. |
| Planner got a third intent, `off_topic` | The embedding guardrail (layer 1) only catches messages near its example phrases; on unseen off-topic topics it blocked 14%. The planner already runs on every message, so a third intent adds a second guardrail layer at no extra LLM call. It took the test split to 100% blocked with 0 false blocks. |
| Off-topic branch goes straight to `END` | No retrieval and no responder call are needed for a fixed refusal — it saves quota and latency. |
| Retriever: added a minimum rerank score (0.1) | Low-scoring chunks are unrelated to the question; sending them to the LLM invites answers stretched from irrelevant text. Returning nothing lets the responder say "not in the docs". |
| Retriever: de-duplicate before reranking | Duplicate chunks wasted the five context slots. |
| Retriever: numbered `[n] Source:` labels | Enables inline citations and the source-file list in the API response. |
| Added `retrieval_scores` to the state | The confidence badge needs the reranker scores outside the retriever. |
| Responder: trim oversized chunks instead of skipping them | Before the chunking fix some chunks were whole documents (60k+ characters). Skipping a chunk larger than the budget left the LLM with an empty context. |
| Responder: separate `NO_CONTEXT_SYSTEM_PROMPT` | Gives a fixed, checkable refusal sentence. `evals/refusal.py` scores it with rules; the out-of-scope goldens pass 3/3. |
| Added query decomposition: `sub_queries` in the planner and state, per-query search and rerank in the retriever | Golden `pwq-002` asks two things whose answers sit in different chunks. After the chunk size dropped to 1,200 characters, the single combined query scored below 0.1 against each chunk (top 0.018–0.044), so nothing was retrieved. Tested by hand, the two split questions scored 0.15–0.67 and 0.85. The split comes from the planner call that already runs, so it adds no LLM call. |
| Sub-query prompt, version 2: "short self-contained questions", always fill `search_query` | Version 1 returned keyword-style sub-queries and an empty `search_query`. For `pwq-002` the keyword sub-query scored 0.008 on the chunk that holds the answer, so only half of the question was fixed (run `20261008-165107`). |
| Sub-query prompt, version 3: **copy the user's words**, with one worked example; do not split same-topic questions | Version 2 still reworded the user: "suggested instead" became "recommended approach ... in Redis". The cross-encoder scored that 0.011 on the right chunk; the user's own wording scores 0.61. `pwq-002` got worse (1 chunk, run `20261009-141503`), and 10 of 17 goldens were split, several of them single-topic. With version 3 `pwq-002` retrieves the right chunk and answers both parts, and 7 of 17 goldens are split (run `20261009-143328`). The example in the prompt is deliberately about a different subject than the golden, so the prompt does not contain the eval's answer. |

**Honest limit of this fix.** It was tuned while looking at one golden (`pwq-002`). The other 16 in-scope goldens did not get worse, but one question is thin evidence that decomposition works for every multi-part question. The planner is an LLM, so it can still reword a sub-query now and then. More multi-part goldens are the next check.
| Memory: `MemorySaver` → Postgres checkpointer when `DATABASE_URL` is set (`memory.py`) | In-process memory was lost on every restart, and Streamlit Cloud restarts the demo whenever it wakes from sleep or redeploys. Tested against a local Postgres and then against Neon (state written by one process was read back by a new one). With the local Postgres and the full app: a follow-up question asked from a **new process** ("What schedule format does it use?") was rewritten to "Kubernetes CronJob schedule format cron syntax". |
| `durability="exit"` on the graph call | The default saves after every node (four writes per question). Only the end-of-turn state is needed for conversation memory, and each write is a network round trip to Neon. Measured: 2 turns → 2 saved checkpoints. |
| Responder prompt tightened: every statement from the excerpts, code only copied from them, no general guidance when the docs fully answer | The scored run `20261009-143328` showed answers adding things the docs do not say: `pwq-002` got faithfulness 0.25 for invented Redis command details and a YAML example. After the change (run `20261010-175028`) the same question is answered in 55 words instead of 190, with no invented example. Across the 17 in-scope goldens, answers with a "General guidance" block went from 7 to 0, and the median answer became about half as long. Source hit rate (17/17) and honest refusal (3/3) are unchanged. The judge scores for this run were still being produced when this was written; see [`../../evals/README.md`](../../evals/README.md). |
| Planner: a follow-up that asks for new technical information is `technical` | Found on the live demo right after memory went live: after "What is a CronJob?", the follow-up "What schedule format does it use?" was labelled `conversational`, so it was answered from the chat history and the model's own knowledge — no retrieval, no citations, no confidence badge. The old wording ("answerable purely from the conversation history") invited that. Same history, same question: old prompt 4 of 4 `conversational`, new prompt 3 of 3 `technical` with the query "Kubernetes CronJob schedule format ...". "What did I just ask you?", "thanks" and "summarise your last answer" stay `conversational`. The guardrails eval is unchanged (system block rate 1.0, false-block rate 0.0, 57 cases). |
| Responder: `normalize_citations()` rewrites `【4】` as `[4]` | The model often writes citations in its own bracket style (`【4】`, `【4†L1-L4】`). The UI and `evals/refusal.py` look for `[n]`. A prompt instruction to use plain brackets was tried first and ignored (2 of 2 test answers still used `【n】`), so it is fixed in code, which works every time. |
