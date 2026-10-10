# `app/agents/` — the LangGraph agent

The agent is a small, explicit graph of three nodes. Each node is a plain function that reads the shared state and returns the keys it wants to change.

## Files

| File | Role |
|---|---|
| `state.py` | `AgentState` — the dictionary every node reads and writes |
| `graph.py` | Wires the nodes, the routing rule and the memory checkpointer; exports `rag_agent` |
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
| `intent` | `conversational`, `technical` or `off_topic` |
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
| Documents were retrieved | `TECHNICAL_SYSTEM_PROMPT` | Answer grounded in the numbered excerpts with `[n]` citations; anything extra must be labelled "General guidance (not from the docs):" |
| Technical, nothing retrieved | `NO_CONTEXT_SYSTEM_PROMPT` | Must begin "I couldn't find this in the internal documentation." |

- `trim_documents()` is the context-size rule: at most 10,000 characters per chunk and 25,000 in total. Oversized chunks are **trimmed**, not dropped. It is public because `evals/metrics.py` calls it so the judge sees exactly the context the LLM saw.
- The call uses the native Portkey client, not LangChain, so the `x-portkey-cache-status` header can be read and "Cache: Hit ⚡" added to the plan.
- Temperature is 0.1.

## Memory

`graph.py` compiles the graph with `MemorySaver`, keyed by `thread_id`. Memory lives in the process: it is lost on restart and not shared between instances. Moving to a Postgres checkpointer is on the roadmap.

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
