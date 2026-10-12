# Enterprise Agentic RAG

An agentic Retrieval-Augmented Generation assistant for enterprise IT documentation (Kubernetes, Intel hardware, networking) — with guardrails, an LLM gateway, **a confidence signal that tells users when the docs don't support an answer**, and **an offline evaluation pipeline that measures quality instead of guessing it**.

**Live demo:** https://agentic-rag-d.streamlit.app

**Try:** "In the nginx HPA example, what replica range is used?" · "What roles do etcd and kube-apiserver play?" · "In the Redis work-queue Job, how does each worker know when to stop, and what is suggested instead for a continuous stream of background work?" (multi-part) · "How do I set up Istio mTLS?" (not in docs) · "tell me a joke" (guardrail)

> The demo sleeps when idle — the first question after a wake-up can take about a minute while models load.

Built entirely on free tiers (Groq, Gemini, Qdrant Cloud, Portkey, Logfire, Streamlit Community Cloud).

---

## What makes it different

**1. It knows when it doesn't know.** Every answer carries a confidence badge computed from the cross-encoder relevance of the retrieved passages — not from the LLM's own opinion:

| Badge | Meaning |
|---|---|
| 🟢 High | A retrieved passage closely matches the question |
| 🔵 Medium | Related passages, may not fully answer it |
| 🟠 Low | Only weakly related passages — verify before relying on it |
| ⚪ Not in docs | Nothing relevant retrieved; any guidance is labelled as general knowledge |

Thresholds are calibrated on this corpus: passages that answered golden questions scored 0.98–0.99; unrelated ones ~0.0.

**2. Quality is measured.** A golden dataset + LLM-as-judge pipeline (Ragas) scores retrieval and generation separately — see [Evaluation](#evaluation) for the numbers and what they revealed.

**3. It splits multi-part questions.** "How does X stop, and what is suggested instead for Y?" is two questions whose answers sit in different parts of a document. The planner cuts such a question into its parts, and each part is searched and reranked on its own — in the same LLM call the planner already makes, so it costs no extra LLM call. See [Query decomposition](#query-decomposition-multi-part-questions).

---

## Architecture

```mermaid
flowchart LR
    U[User] --> UI[Streamlit UI :7860]
    UI --> API[FastAPI :8000 — private]
    API --> RL{Rate limit + input limits}
    RL --> G[NeMo Guardrails<br/>embedding-based intents]
    G -- off-topic / jailbreak / greeting --> UI
    G -- technical --> P[Planner<br/>structured output, query rewrite,<br/>multi-part split]
    P -- off-topic --> UI
    P -- conversational --> R[Responder]
    P -- technical --> RET[Retriever<br/>per sub-query: Qdrant search → FlashRank rerank,<br/>then merge]
    RET --> R
    R --> PK[Portkey gateway<br/>fallback + retry + cache]
    PK --> GROQ[Groq: gpt-oss-120b → gpt-oss-20b]
```

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph (planner → retriever → responder) | Explicit, inspectable agent flow |
| Memory | LangGraph checkpointer per `thread_id`: Neon Postgres when `DATABASE_URL` is set, in-process otherwise | Follow-up questions keep working after a restart; local runs and evals need no database |
| Guardrails | Layer 1: NeMo Guardrails, `embeddings_only` intent matching · Layer 2: planner LLM `off_topic` intent | Layer 1 is free and instant for common cases; layer 2 generalises to any topic at no extra LLM call (the planner already runs) |
| Ingestion | Size-bounded chunker (1,200 characters, 200 overlap), per-file report, post-ingest count check in Qdrant | The reranker only reads about 512 tokens, so a chunk must be small enough to be judged as a whole |
| Retrieval | Gemini embeddings → Qdrant Cloud → FlashRank `ms-marco-MiniLM-L-12-v2`, min-score cutoff; multi-part questions are searched once per part | Cross-encoder reranking removes irrelevant chunks before they reach the LLM |
| Generation | Grounded prompt with `[n]` citations; explicit "not in the docs" path | Reduces hallucination |
| Gateway | Portkey: primary/fallback models, retries, caching | Resilience on free-tier rate limits |
| Observability | Pydantic Logfire spans across UI → API → agent | End-to-end traces |

---

## Evaluation

`evals/` contains a reproducible pipeline:

1. **`golden_dataset.json`** — 20 hand-verified questions across 6 source documents (fact, multi-part, reasoning, comparison, out-of-scope). Every ground-truth fact was checked against its source.
2. **`pipeline.py`** — runs each question through the real system (guardrails → graph), records answer, retrieved contexts, sources, latency; crash-safe JSONL with resume; records git commit + models per run.
3. **`metrics.py`** — Ragas Context Recall / Context Precision / Faithfulness / Answer Relevancy with an LLM judge; applicability rules (N/A is never averaged as 0); the judge sees exactly the context the LLM saw.
4. **`judge.py`** — judge adapters for Ragas and DeepEval with a sliding-window token budget that keeps runs inside Groq's free-tier limits.
5. **`refusal.py`** — honest-refusal check for out-of-scope questions (rule-based, no LLM).
6. **`guardrails_eval.py`** — runs `guardrails_dataset.json` (should-block vs should-pass messages) through both guardrail layers and reports each layer and the combined system.

### What each metric means

| Metric | Question it answers | Measures |
|---|---|---|
| Source hit rate | Did retrieval return the document the answer lives in? | Retrieval |
| Context Recall | Do the retrieved passages contain the facts needed for the correct answer? | Retrieval |
| Context Precision | Are the relevant passages ranked above the irrelevant ones? | Retrieval / reranking |
| Faithfulness | Is every claim in the answer supported by the retrieved passages (no hallucination)? | Generation |
| Answer Relevancy | Does the answer address the question that was asked? | Generation |

All scores are 0–1, higher is better. Retrieval and generation are scored separately so a bad answer can be traced to its cause: wrong passages, or a wrong use of the right passages.

### Baseline results (before retrieval fixes)

| Metric | Score | n |
|---|---|---|
| Source hit rate | 0.71 | 17 |
| Context Recall | 0.69 | 17 |
| Context Precision | 0.74 | 17 |
| Faithfulness | 0.78 | 15 |
| Answer Relevancy | 0.73 | 17 |

Run `20260927-112048` · judge `openai/gpt-oss-20b` · all 17 in-scope goldens scored, 0 judge errors. The 3 out-of-scope goldens have no correct passage to find, so these metrics don't apply to them.

**Why `n` differs:** a metric that does not apply is marked N/A and left out of the average — never counted as 0. Faithfulness has n=15 because two questions retrieved nothing, and an answer cannot be faithful or unfaithful to no context (their Context Recall/Precision *are* counted, as a measured 0).

**Rate limits:** scoring takes ~15k judge tokens per golden, and Groq's free tier allows 200k per day. The run hit the daily limit after 13 goldens; because results are written per golden and the scorer resumes, the remaining ones were scored the next day against the same answers, the same judge and the same code — so all 17 are comparable.

### After the retrieval fixes

Three fixes followed from the baseline: a size-bounded chunker with a full re-ingest into a second collection (`enterprise_rag_v2`), query decomposition for multi-part questions, and a stricter answer prompt.

| Metric | Baseline `20260927-112048` | New chunker `20261003-211038` | + query decomposition `20261009-143328` | + stricter answer prompt `20261010-175028` |
|---|---|---|---|---|
| Source hit rate | 0.71 (12/17) | 0.94 (16/17) | **1.00 (17/17)** | **1.00 (17/17)** |
| Context Recall | 0.69 | 0.94 | **1.00** | **1.00** |
| Context Precision | 0.74 | 0.83 | 0.80 | **0.84** |
| Faithfulness | 0.78 (n=15) | 0.92 (n=16) | 0.89 (n=17) | **0.99** (n=17) |
| Answer Relevancy | 0.73 | 0.79 | 0.76 | 0.79 |
| Honest refusal (out-of-scope) | 3/3 | 3/3 | 3/3 | 3/3 |

Same 20 goldens, same corpus, same judge (`openai/gpt-oss-20b`) in every column. All 17 in-scope questions are scored in every column, with 0 judge errors.

What query decomposition (third column) did and did not do:

- **It fixed retrieval.** Context Recall is 1.00: for every question, the retrieved passages now contain the facts needed for the correct answer.
- **It is not a gain on every metric.** Context Precision and Answer Relevancy are slightly lower (0.83 → 0.80 and 0.79 → 0.76). Sub-queries bring in extra passages, and some rank above more useful ones (`mon-001` precision 1.0 → 0.64).
- **Faithfulness is unchanged where it can be compared** (0.92 → 0.93 on the 16 questions scored in both runs). The headline number drops to 0.89 only because the fixed question `pwq-002` is now counted: it had no context before (N/A), and now scores 0.25, because the answer added Redis command details and a YAML example that are not in the retrieved text. Retrieval for it is right; the answer writer over-explains.
- **Some movement is noise.** `hpa-003` precision (0.75 → 0.25) and `hpa-001` relevancy (0.63 → 0.29) fell although neither question was split, and `arch-002` faithfulness rose 0.5 → 1.0 with no related change. With 17 questions and LLM-written queries and answers, single-question swings like these happen between runs.

What the stricter answer prompt (fourth column) did:

- **It fixed the answer-writing problem.** The prompt now says: state only what the excerpts say, copy code instead of writing it, and add general guidance only when the docs do not fully answer. Faithfulness rose from 0.89 to 0.99. `pwq-002` went from 0.25 to 1.0, and 16 of 17 answers are now fully supported by the retrieved text (the exception is `pwq-001` at 0.83).
- **Answers are shorter and stay in the docs.** In-scope answers with a "General guidance" block fell from 7 to 0, and the median answer length from 150 to 78 words.
- **Retrieval did not change, so its movement is noise.** The retrieval code and the collection are the same in the last two columns, and Context Recall is 1.00 in both. Context Precision still moved 0.80 → 0.84, almost all of it from `hpa-003` (0.25 → 0.92) and `dbx-001` (0.70 → 1.0). Read the Answer Relevancy change (0.76 → 0.79) the same way: within run-to-run variation.

**Which data the live demo uses:** the numbers above were measured locally against `enterprise_rag_v2`. The public demo was switched to that collection on 2026-10-10 with one setting (`QDRANT_COLLECTION`), with no downtime: the old collection kept serving while the new one was built and evaluated. Rolling back is deleting the setting.

### Query decomposition (multi-part questions)

After the re-ingest, one question got worse. `pwq-002` asks two things about the Redis work-queue example: how a worker knows when to stop, and what to use instead for a continuous stream of work. With small chunks the two answers sit in two different chunks. The combined question scored below the 0.1 cutoff against **each** of them (top score 0.04), so nothing was retrieved.

The fix has three small parts:

1. **Planner** — the same LLM call now also returns `sub_queries`: one question per part, for questions that ask about two or three distinct things.
2. **Retriever** — every query (the main one plus each sub-query) is searched and reranked on its own. The results are merged so that each part's best chunk gets a slot.
3. **Eval** — the pipeline records the planner's `search_query` and `sub_queries` for every golden, so a miss can be traced to the query that caused it.

It took three versions of the planner prompt to get right, and the eval showed why each earlier one failed:

| Prompt version | What the planner wrote for the second part | Reranker score on the correct chunk | `pwq-002` result |
|---|---|---|---|
| 1. "one query per part" | keywords: "continuous background work pattern alternative to Redis work queue" | 0.008 | half answered |
| 2. "short self-contained questions" | "What is the recommended approach for a continuous stream of background work in Redis?" | 0.011 | worse — 1 chunk |
| 3. "copy the user's words" | "What is suggested instead for a continuous stream of background work?" | 0.61 | both parts answered, 3 chunks |

The lesson: the cross-encoder is very sensitive to wording. "Suggested instead" and "recommended approach" mean the same to a person, but score 0.61 and 0.011. So the planner must **cut** the user's sentence, not **rewrite** it. Version 3 also stopped needless splitting: 7 of 17 in-scope questions are split, down from 10.

Honest caveats: this was tuned while looking at one golden, and the planner is an LLM, so it can still reword a sub-query now and then. The other 16 in-scope goldens kept their source hit. More multi-part goldens are the next check.

### Out-of-scope questions: honest refusal

The 3 out-of-scope goldens ask about things the docs don't cover (Juniper BGP, Argo CD, Istio mTLS). Retrieval metrics don't apply to them, so they are scored by rules instead of an LLM — free and identical on every run. An answer passes only if it (1) says up front that the docs don't cover it, (2) labels any extra advice "General guidance (not from the docs)", and (3) cites no document excerpts.

| Metric | Score | n |
|---|---|---|
| Honest refusal rate | 1.00 | 3 |

One of the three retrieved unrelated chunks and still refused correctly rather than stretching them into an answer.

### Guardrails

`guardrails_dataset.json` has 37 messages in three splits — messages that should be blocked (off-topic, jailbreak) and technical questions that must pass, including traps like *"How do I **kill** a stuck pod?"* or *"How do I **override** the default resource limits?"*:

- **dev** (10) — the original set, used to diagnose the problem
- **tune** (14) — written before any change, then used while tuning
- **test** (13) — written *after* the tuning data was seen and only scored once at the end, so it is the honest measure of generalisation

The gate matches intents by embedding similarity (no LLM call, ~0.02s), so the eval is free and deterministic. NeMo's threshold is **not a cosine**: it scores `1 - sqrt(2 - 2*cos)/2`, so the original 0.6 required a cosine of about 0.68 — only near-copies of an example phrase were caught.

| Config | Block rate — dev | tune | **test** | False-block rate (all splits) |
|---|---|---|---|---|
| Before: threshold 0.6, original examples | 0.40 | 0.00 | **0.00** | 0.00 (n=17) |
| After: threshold 0.45, examples by category, 16 technical examples | 1.00 | 1.00 | **0.14** | 0.00 (n=17) |

What changed: off-topic and jailbreak examples were added by category (sports, food, creative writing, "no rules" and persona jailbreaks, prompt extraction), and the technical intent got 16 examples so that a real question's closest match is technical — that is what made lowering the threshold safe (0 false blocks).

**What the test split showed:** dev and tune reached 100%, but unseen off-topic topics (party planning, stocks, smartphones, translation) were still missed. An example list cannot enumerate every off-topic subject, so embedding matching can only be a fast first layer for common cases.

**Layer 2 — the planner.** The planner LLM already classifies every message that passes layer 1 (conversational vs technical), so it got a third intent, `off_topic`, at no extra LLM call. Off-topic messages get the same fixed reply as the rail, skipping retrieval and the responder. Its prompt states that technical questions the docs don't cover (Istio, Argo CD, Juniper) are still *technical* — so they keep getting the honest "not in docs" answer. The eval adds the 20 RAG goldens as must-pass questions for this layer:

| Split | Layer 1 block | Layer 2 block | **System block** | System false-block |
|---|---|---|---|---|
| dev | 1.00 (5) | 1.00 (5) | **1.00** | 0.00 (5) |
| tune | 1.00 (8) | 1.00 (8) | **1.00** | 0.00 (6) |
| test | 0.14 (7) | 1.00 (7) | **1.00** | 0.00 (6) |
| goldens (incl. 3 out-of-scope) | — | — | — | **0.00 (20)** |

Honest caveats: the planner's off-topic categories (shopping, finance, translation…) were written after the test misses had been seen, so for layer 2 the test split is not fully unseen; the samples are small (7 test block cases); and the planner is an LLM, so repeated runs can differ. A fresh set of unseen messages is the next check.

### What the evals found

- **A document that was never indexed.** All three CronJob questions missed their source: ingestion had silently dropped `cronjobs.docx` (errors were only logged). Found by the first eval run, not by manual testing.
- **Whole documents stored as single chunks.** VPA questions returned "not in docs" although the answer exists: the article was one 17.7k-character chunk and the cross-encoder only reads the first ~512 tokens, where only HPA content is. The same file answered HPA questions correctly.
- **Judge choice changes scores.** The same answer scored Faithfulness 0.68 with a Qwen judge and 1.0 with gpt-oss-20b — so runs are only compared under the same judge, and verdicts are spot-checked.
- **A fix can break something else.** Smaller chunks fixed the VPA questions but broke a two-part question, because each small chunk now answered only half of it. Only the re-run of the full golden set showed this.
- **An LLM rewording a query can hide the right passage.** The planner's paraphrase of a sub-question scored 0.011 on the correct chunk; the user's own words scored 0.61. See [Query decomposition](#query-decomposition-multi-part-questions).
- **The judge can be lenient.** A spot-check of `hpa-003` (HPA vs VPA) found Faithfulness 1.0 although the answer included details ("stateless", "batch jobs") that were not in the trimmed context the LLM saw. Its Context Recall of 0.0 is accurate — it is the oversized-chunk problem again: the VPA part of the 17.7k-character chunk is trimmed away before the LLM sees it.

---

## Engineering notes

Problems solved while building this, each diagnosed by measuring rather than guessing:

- **Model retirement:** the original Llama models were retired by the provider; migrated to gpt-oss and re-validated.
- **Reasoning-model quirks:** gpt-oss ignored NeMo's completion-style intent prompt and occasionally duplicated free-text output → switched guardrails to embedding-based intents and the planner to JSON-schema structured output.
- **Dependency conflict:** Ragas 0.4.3 required a module removed in `langchain-community` 0.4 → found (via `pip --dry-run` and wheel metadata) that 0.3.31 is the last compatible release, and pinned it.
- **Library telemetry timeouts:** each Ragas score took ~40s because its analytics calls timed out on the network → disabled telemetry (40× faster).
- **Free-tier limits beyond the headers:** reproduced Groq's hidden output-tokens-per-minute limit with concurrent requests, switched judge models, and built a token-budget limiter that settles on actual usage.
- **Safe public demo:** input length limits, global rate limiting, private API behind the UI, per-session memory threads, no secrets in the image.

---

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
uvicorn app.main:app --port 8000          # terminal 1
streamlit run ui/app.py                   # terminal 2
```

Evaluation:
```bash
python -m evals.pipeline                  # run the golden set through the system
python -m evals.metrics --run <RUN_ID>    # score it (re-run the same command to resume after a rate limit)
python -m evals.refusal --run <RUN_ID>    # honest refusal on out-of-scope goldens (no LLM)
python -m evals.guardrails_eval --planner # guardrail layers (omit --planner for the free layer-1-only run)
```

Docker:
```bash
docker build -t enterprise-rag .
docker run -p 7860:7860 --env-file .env enterprise-rag
```

Required environment variables: `GROQ_API_KEY`, `GROQ_FALLBACK_API_KEY`, `PORTKEY_API_KEY`, `GEMINI_API_KEY`, `QDRANT_CLUSTER_ENDPOINT`, `QDRANT_API_KEY`; optional: `DATABASE_URL` (Neon Postgres connection string for persistent memory), `QDRANT_COLLECTION`, `LOGFIRE_TOKEN`, `LANGSMITH_API_KEY` (set `LANGSMITH_TRACING=false` without it), `JUDGE_GROQ` (evals).

---

## Known limitations & roadmap

- **Done — chunking:** documents without paragraph breaks used to become single oversized chunks. A size-bounded splitter with overlap and a full re-ingest fixed this (measured above).
- **Done — ingestion reliability:** failures used to be only logged. The job now prints a per-file report and checks the chunk counts in Qdrant afterwards.
- **The reranker is wording-sensitive.** A fixed 0.1 cutoff can drop a correct chunk when a question is phrased unusually. The system then says "not in the docs" instead of guessing, so the error is on the safe side → try a stronger reranker or a relative cutoff, and measure both.
- **Query decomposition is tested on few questions** → add more multi-part goldens whose answers sit in different chunks.
- **Done — answers that added details not in the retrieved text** (unsupported specifics on `pwq-002`, extra "General guidance" blocks): the responder prompt was tightened and Faithfulness rose from 0.89 to 0.99 (run `20261010-175028`).
- **Scores vary between runs.** Each configuration was run once on 17 questions → repeat runs to report a range, and add more goldens.
- **Done — persistent memory:** conversation memory moved from in-process to a Postgres checkpointer (Neon free tier), with a fallback to in-process memory if the database is unreachable.
- **Memory housekeeping:** old conversation threads are never deleted → add a scheduled clean-up of threads older than a few days.
- **Follow-up questions are tested on a handful of examples only** → add multi-turn goldens (a question plus a follow-up that needs the history) to the eval set.
- **Done — questions about the conversation:** "What did I just ask you?" used to get the layer-1 guardrail's capabilities reply instead of an answer from memory. The rails now have an "ask about conversation" intent that passes such messages to the agent; the guardrails eval is unchanged (block rate 1.0, false-block rate 0.0).
- **Judge bias:** the judge shares a model family with the answer model → spot-check verdicts; move to a cross-family judge when limits allow.
- **Guardrails:** two layers now (embeddings + planner LLM), but the planner is evaluated on a small set that influenced its prompt → evaluate on a fresh, larger unseen set; add a dedicated prompt-injection classifier for injections hidden inside long technical questions.
