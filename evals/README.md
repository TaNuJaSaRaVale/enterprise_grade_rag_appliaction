# `evals/` — the evaluation pipeline

This folder measures the system instead of guessing. Running and scoring are separate steps, so an expensive scoring run can be repeated or resumed against the same saved answers.

## Files

| File | Role | Uses an LLM? |
|---|---|---|
| `golden_dataset.json` | 20 hand-verified questions with ground truth and expected source file | — |
| `guardrails_dataset.json` | 37 messages that should be blocked or must pass | — |
| `pipeline.py` | Runs every golden through the real system and saves what happened | The app's own LLM calls |
| `metrics.py` | Scores a saved run with Ragas | Judge LLM |
| `judge.py` | The judge connection and its rate limiter | Judge LLM |
| `refusal.py` | Rule-based honest-refusal check for out-of-scope goldens | No |
| `guardrails_eval.py` | Measures both guardrail layers | Only with `--planner` |
| `export_scorecard.py` | Writes a small summary to `ui/eval_scorecard.json` | No |
| `results/<run_id>/` | Output of each run | — |

## Commands

```bash
python -m evals.pipeline                    # 1. run the goldens  → prints the RUN_ID
python -m evals.metrics --run <RUN_ID>      # 2. score with the judge (re-run the same command to resume)
python -m evals.refusal --run <RUN_ID>      # 3. honest refusal, no LLM
python -m evals.guardrails_eval             # guardrails layer 1 only (free, deterministic)
python -m evals.guardrails_eval --planner   # + layer 2 (one planner call per case)
python -m evals.export_scorecard --run <RUN_ID>
```

Useful flags: `pipeline --limit N`, `pipeline --resume <RUN_ID>`, `metrics --ids a,b`, `metrics --limit N`.

**Run one eval at a time.** The judge's rate limiter is per process. Two scoring runs on the same key collide and use up the daily quota.

## The golden dataset

| Field | Meaning |
|---|---|
| `id`, `question`, `ground_truth` | Required. The dataset is validated before any quota is spent; all problems are reported at once. |
| `expected_source` | File the answer lives in. `null` marks an out-of-scope question. |
| `type` | `simple_fact` (6), `multi_part` (8), `reasoning` (2), `comparison` (1), `out_of_scope` (3) |

Seventeen goldens are in scope, across six source files. Three are out of scope (Juniper BGP, Argo CD, Istio mTLS): technical questions the docs do not answer.

## `pipeline.py` — run, don't score

For each golden it follows the same path as `POST /query` (guardrail, then the graph) and records: answer, retrieved chunk texts and sources, whether the expected source was retrieved, guardrail result, cache hit, latency, error.

| Feature | Why it is there |
|---|---|
| One JSONL line appended per golden | A crash or Ctrl+C loses nothing |
| `--resume` skips goldens already done without error | Rate limits interrupt runs often |
| Own `thread_id` per golden and per run | History never leaks between questions |
| Warm-up before timing | One-time costs (NeMo, reranker model, Gemini probe) don't distort latency |
| `run_info.json` | Git commit, uncommitted files, Qdrant collection, models, reranker and threshold — what produced these numbers |
| `_split_document()` raises on an unexpected format | If the retriever's `[n] Source:` label changes, the eval fails loudly instead of sending labels to the judge |

`summary.json` holds the retrieval-only numbers: `source_hit_rate`, `source_miss_ids`, `avg_latency_s`, `guard_blocked_ids`.

`run_info.json` records the Qdrant collection (`qdrant_collection`) from run `20261009-143328` onwards. Older runs do not have the field: the baseline used `enterprise_rag`, and every run from `20261003` on used `enterprise_rag_v2`.

Each line of `pipeline_results.jsonl` also holds the planner's `search_query` and `sub_queries`, so a retrieval miss can be traced to the query that caused it.

## `metrics.py` — the four Ragas metrics

| Metric | Question it answers | Measures |
|---|---|---|
| Context recall | Do the retrieved chunks contain the facts of the ground truth? | Retrieval |
| Context precision | Are the relevant chunks ranked above the irrelevant ones? | Reranking |
| Faithfulness | Is every claim in the answer supported by the retrieved chunks? | Generation |
| Answer relevancy | Does the answer address the question asked? | Generation |

All are 0–1, higher is better.

**Applicability rules (`plan_metrics`).** A metric that does not apply is recorded as N/A with a reason and left out of the average. It is never counted as 0.

| Situation | What is recorded |
|---|---|
| Out-of-scope golden | All four N/A (scored by `refusal.py` instead) |
| In scope, nothing retrieved | Recall and precision are a **measured 0**; faithfulness is N/A; relevancy is scored |
| Judge error on a metric | `null` plus the error — a failure is never a 0. Re-running scores only these. |

**The judge sees what the LLM saw.** `judge_contexts()` rebuilds the labelled chunks and applies the responder's own `trim_documents()` before stripping the labels.

## `judge.py`

| Item | Value |
|---|---|
| Judge model | `openai/gpt-oss-20b` on Groq, called directly (not through Portkey), `reasoning_effort="low"`, temperature 0 |
| Key | `JUDGE_GROQ`, falling back to `GROQ_FALLBACK_API_KEY`, then `GROQ_API_KEY` |
| Per-minute budget | 7,000 tokens (the limit is 8,000) |
| Daily limit | 200,000 tokens, rolling 24 hours; scoring costs about 15k per golden |
| Answer-relevancy embeddings | `all-mpnet-base-v2`, local |

`TokenBudget` is a sliding-window limiter. Before each call it reserves an estimate (prompt characters ÷ 4 plus 600 for output); after the call `settle()` replaces the estimate with real usage. It re-checks at least every 2 seconds because settling often frees budget early.

## `refusal.py`

An out-of-scope answer passes only if all three hold:

1. It says in the first 300 characters that the docs don't cover it.
2. Any extra advice is labelled "General guidance (not from the docs)".
3. It contains no `[n]` citations.

Rules, not an LLM: free, and identical on every run.

## `guardrails_eval.py`

| Split | Should block | Should pass | Purpose |
|---|---|---|---|
| dev | 5 | 5 | Used to diagnose the problem |
| tune | 8 | 6 | Written before any change, then used while tuning |
| test | 7 | 6 | Written after tuning and scored once — the honest generalisation number |
| goldens (with `--planner`) | 0 | 20 | The RAG goldens must never be blocked |

It reports `block_rate` and `false_block_rate` for layer 1, layer 2 and the combined system. It calls the rails directly rather than `guard()`, because `guard()` fails open and would turn an error into a fake "passed".

## Results so far

| Metric | Baseline `20260927-112048` | v2 `20261003-211038` | v2 + decomposition `20261009-143328` |
|---|---|---|---|
| Source hit rate | 0.706 (12/17) | 0.941 (16/17) | 1.0 (17/17) |
| Context recall | 0.691 | 0.941 | 1.0 |
| Context precision | 0.736 | 0.828 | 0.799 |
| Faithfulness | 0.780 (n=15) | 0.924 (n=16) | 0.887 (n=17) |
| Answer relevancy | 0.734 | 0.789 | 0.761 |
| Honest refusal | 3/3 | 3/3 | 3/3 |

Same goldens and same judge in every column. All three runs have all 17 in-scope goldens scored with 0 judge errors. (The third was scored over two days, 12 + 5, because of the daily judge limit.)

Reading the last two columns honestly:

| Metric | Change | What is behind it |
|---|---|---|
| Context recall | 0.941 → 1.0 | Only `pwq-002` moved (0 → 1.0). This is the fix. |
| Context precision | 0.828 → 0.799 | `pwq-002` 0 → 0.5 (better), `mon-001` 1.0 → 0.64 (split; an extra chunk ranks high), `hpa-003` 0.75 → 0.25 (not split — run-to-run variation), `arch-003` 0.5 → 0.42. |
| Faithfulness | 0.924 (n=16) → 0.887 (n=17) | On the 16 goldens scored in both runs it is 0.924 → 0.927. The drop comes from `pwq-002` entering the average at 0.25: it was N/A before (no context), and now the answer adds Redis command details and a YAML example that are not in the chunks. |
| Answer relevancy | 0.789 → 0.761 | Mostly `hpa-001` 0.63 → 0.29 (not split) and `pwq-002` 0.88 → 0.71. |

So decomposition did what it was built for (recall), cost a little precision, and exposed an answer-writing problem on the question it fixed. Each configuration was run once; single-golden swings of 0.3–0.5 appear on goldens that no change touched, so differences of 0.03 between the averages are within noise.

- Baseline → v2: the new chunker and the re-ingest into `enterprise_rag_v2`.
- v2 → v2 + decomposition: the planner's `sub_queries` and the per-query search in the retriever. Same collection.

### The `pwq-002` story, run by run

| Run | Planner prompt | Chunks for `pwq-002` | Result |
|---|---|---|---|
| `20261003-211038` | no decomposition | 0 | "not in the docs" |
| `20261008-165107` | v1: one query per part | 3 | first part only; keyword sub-query scored 0.008 on the right chunk |
| `20261009-141503` | v2: short self-contained questions | 1 | worse; the reworded sub-query scored 0.011 |
| `20261009-143328` | v3: copy the user's words | 3 | both parts answered from the docs |

The number of in-scope goldens that were split fell from 10 of 17 (v2 prompt) to 7 of 17 (v3 prompt). The v3 prompt tells the planner not to split questions whose parts are about the same thing.

### Run `20261010-175028` — tightened answer prompt (judge scoring in progress)

Same collection and retrieval code as `20261009-143328`; the only change is the responder's prompt (answers must come from the excerpts, no general guidance when the docs fully answer).

| Check (no judge needed) | `20261009-143328` | `20261010-175028` |
|---|---|---|
| Source hit rate | 17/17 | 17/17 |
| Honest refusal | 3/3 | 3/3 |
| In-scope answers with a "General guidance" block | 7 | 0 |
| Median in-scope answer length | 150 words | 78 words |

The four judge metrics for this run are being scored over two days (daily judge limit) and will be added as a fourth column above. Until then, the effect on faithfulness is expected but **not yet measured**.

Things to know before quoting these:

- `pwq-002` is the one retrieval failure in run `20261003-211038` (0 chunks retrieved). Query decomposition fixed it in run `20261009-143328`. The fix was tuned on this one golden, so treat it as "works here", not "proven in general".
- The worker-loop code chunk of that document is still not retrieved for `pwq-002` (it scores under 0.1 for every wording tried). The first part is answered from the overview chunk instead.
- `arch-002` faithfulness fell from 1.0 to 0.5 in run `20261003-211038`: retrieval was fine; the answer added a "General guidance (not from the docs)" block that the judge counts as unsupported. In run `20261009-143328` it is back at 1.0 with no related code change — an example of run-to-run variation.
- `20261003-210709` and `20261008-164819` are disposable smoke runs. `20261008-165107` and `20261009-141503` are the two failed prompt attempts in the table above; they are kept as evidence and were not judge-scored.
- `ui/eval_scorecard.json` was re-exported from run `20261009-143328`. Re-export it after any new scored run.

## What the evals found

| Finding | How it showed up |
|---|---|
| `cronjobs.docx` was never indexed | All three CronJob goldens missed their source in the first run |
| Whole documents stored as single chunks | VPA questions returned "not in the docs" although the answer exists |
| The judge model changes the score | The same answer scored faithfulness 0.68 with a Qwen judge and 1.0 with `gpt-oss-20b` |
| The judge can be lenient | `hpa-003` got faithfulness 1.0 although the answer held details missing from the trimmed context |
| Layer-1 guardrails do not generalise | 100% on dev and tune, 14% on the unseen test split |

## What changed while building this, and why

| Change | Why |
|---|---|
| Split into run (`pipeline.py`) and score (`metrics.py`) | Scoring is the expensive, rate-limited part. Saved answers can be re-scored or resumed without re-running the system. |
| JSONL append + resume in both steps | The daily judge limit ended the baseline scoring after 13 goldens; the rest were scored the next day against the same answers. |
| Judge: Qwen → `gpt-oss-20b` with low reasoning effort | Groq's free tier caps the Qwen model at 1,000 **output** tokens per minute and reserves the full `max_tokens` per request: about one judge call a minute, and long answers could not be judged. The cost is a known bias risk — the judge shares a model family with the answer model. |
| Judge calls Groq directly | Portkey's locked default config would reroute the judge to the 120b model, and its cache could return old verdicts. |
| Added the `TokenBudget` limiter | To stay inside the per-minute limit instead of failing and retrying. |
| Limiter re-checks every 2 s | Sleeping until the oldest entry expired cost about 60 s per golden. |
| Telemetry disabled for Ragas and DeepEval | Ragas' analytics calls timed out on this network and added about 40 s to every score (40.7 s → 1.0 s). |
| Judge `max_tokens` 1024 → 4096 | Long answers produced verdict JSON that was cut off (seen on `arch-003` faithfulness). |
| `langchain-community` pinned to 0.3.31 | Ragas 0.4.3 needs a module that 0.4 removed. |
| N/A and errors kept out of averages | Counting them as 0 would make the score depend on the mix of goldens and on outages. |
| Added `refusal.py` | Retrieval metrics do not apply to out-of-scope questions, but refusing honestly is the behaviour that matters there. |
| Added the guardrails dataset and eval | Guardrails had only been tried by hand. The first measurement was a 0.40 block rate. |
| Added a held-out `test` split | Dev and tune numbers were tuned against; only an unseen split shows generalisation. |
| Added the `goldens` split to the planner eval | A guardrail that blocks real questions is worse than one that misses a joke. |
| Quieter loggers during runs | NeMo and httpx INFO logs drowned the progress lines. |
| Uncommitted files recorded in `run_info.json` | A run on uncommitted code cannot be reproduced from the commit hash alone. |
| `sub_queries` and `search_query` saved per golden | Needed to see *why* `pwq-002` failed: the chunks were fine, the planner's rewording was not. |
| Qdrant collection saved in `run_info.json` | Two collections exist (`enterprise_rag`, `enterprise_rag_v2`). Without the name, a run cannot be tied to the data it was measured on. |
