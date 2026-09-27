"""
Scores a pipeline run with Ragas, using the Qwen judge (evals/judge.py).

Reads   evals/results/<run_id>/pipeline_results.jsonl   (from evals/pipeline.py)
Writes  evals/results/<run_id>/metrics_results.jsonl    (one line per golden, append + resume)
        evals/results/<run_id>/metrics_summary.json
        evals/results/<run_id>/metrics_info.json         (judge + library versions → comparability)

Averages count only REAL scores. N/A and judge errors are reported separately —
counting them as 0 would make scores depend on the golden mix and on outages.
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from statistics import mean

from evals.pipeline import RESULTS_DIR, RESULTS_FILE, _split_document, load_results

METRICS_FILE = "metrics_results.jsonl"
METRIC_NAMES = ("context_recall", "context_precision", "faithfulness", "answer_relevancy")

# Reasons a metric is not scored — recorded next to the null so nothing is silently hidden
NA_OUT_OF_SCOPE = "not applicable: out-of-scope golden (judged by the honest-refusal metric instead)"
NA_NO_CONTEXT = "not applicable: nothing retrieved, so there is no context to be faithful to"
NA_PIPELINE_ERROR = "not scored: pipeline error for this golden"
NA_GUARD_FIRED = "not scored: guardrail blocked this RAG golden"
MEASURED_ZERO_NO_CONTEXT = "measured 0: in-scope question but no chunks were retrieved"


def judge_contexts(record: dict) -> list[str]:
    """
    The exact context text the responder showed the LLM: rebuild the retriever's
    labelled documents, apply the responder's own trimming rule, strip the labels.
    """
    from app.agents.nodes.responder import trim_documents

    labelled = [f"[{i}] Source: {source}\n{text}"
                for i, (source, text) in enumerate(zip(record["retrieved_sources"], record["retrieved_contexts"]), start=1)]
    return [_split_document(doc)[1] for doc in trim_documents(labelled)]


def plan_metrics(record: dict) -> tuple[list[str], dict, dict]:
    """
    Applicability rules. Returns (metrics to run with the judge, fixed scores, notes).
    """
    if record["error"]:
        return [], {}, {m: NA_PIPELINE_ERROR for m in METRIC_NAMES}
    if record["guard_fired"]:
        return [], {}, {m: NA_GUARD_FIRED for m in METRIC_NAMES}
    if not record["expected_source"]:
        return [], {}, {m: NA_OUT_OF_SCOPE for m in METRIC_NAMES}
    if not record["retrieved_contexts"]:
        # A true measurement, not a failure: 0% of the reference can be found in zero chunks
        fixed = {"context_recall": 0.0, "context_precision": 0.0}
        notes = {"context_recall": MEASURED_ZERO_NO_CONTEXT, "context_precision": MEASURED_ZERO_NO_CONTEXT,
                 "faithfulness": NA_NO_CONTEXT}
        return ["answer_relevancy"], fixed, notes
    return list(METRIC_NAMES), {}, {}


class RagasScorer:
    """Builds the Ragas metric objects once (AnswerRelevancy loads a ~400MB embedding model)."""

    def __init__(self):
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
        from evals.judge import get_ragas_embeddings, get_ragas_llm

        llm = get_ragas_llm()
        self.metrics = {
            "context_recall": ContextRecall(llm=llm),
            "context_precision": ContextPrecision(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=get_ragas_embeddings()),
        }

    def _call(self, name: str, record: dict, contexts: list[str]):
        metric = self.metrics[name]
        q, answer, reference = record["question"], record["answer"] or "", record["ground_truth"]
        if name == "context_recall":
            return metric.ascore(user_input=q, retrieved_contexts=contexts, reference=reference)
        if name == "context_precision":
            return metric.ascore(user_input=q, reference=reference, retrieved_contexts=contexts)
        if name == "faithfulness":
            return metric.ascore(user_input=q, response=answer, retrieved_contexts=contexts)
        return metric.ascore(user_input=q, response=answer)  # answer_relevancy

    async def score_golden(self, record: dict) -> dict:
        to_run, scores, notes = plan_metrics(record)
        errors = {}
        contexts = judge_contexts(record) if to_run else []

        start = time.perf_counter()
        # return_exceptions: one failing metric must not throw away the other three
        results = await asyncio.gather(*(self._call(m, record, contexts) for m in to_run), return_exceptions=True)
        for name, result in zip(to_run, results):
            if isinstance(result, Exception):
                scores[name] = None  # a failure is NEVER recorded as 0
                errors[name] = f"{type(result).__name__}: {str(result)[:300]}"
            else:
                scores[name] = round(float(result.value), 4) if result.value is not None else None

        return {
            "id": record["id"],
            "type": record.get("type"),
            "tags": record.get("tags", []),
            "scores": {m: scores.get(m) for m in METRIC_NAMES},
            "notes": notes,
            "errors": errors,
            "judge_context_chars": sum(map(len, contexts)),
            "scoring_s": round(time.perf_counter() - start, 1),
        }


def _stats(values: list) -> dict:
    real = [v for v in values if v is not None]
    return {"mean": round(mean(real), 3) if real else None, "n": len(real)}


def summarize(results: dict[str, dict]) -> dict:
    rows = list(results.values())
    per_metric = {}
    for m in METRIC_NAMES:
        stats = _stats([r["scores"][m] for r in rows])
        stats["n_not_applicable"] = sum(1 for r in rows if r["scores"][m] is None and m in r["notes"])
        stats["n_errors"] = sum(1 for r in rows if m in r["errors"])
        scored = sorted((r["scores"][m], r["id"]) for r in rows if r["scores"][m] is not None)
        stats["lowest"] = [{"id": gid, "score": s} for s, gid in scored[:3]]  # where to look first
        per_metric[m] = stats

    by_type = {}
    for t in sorted({r["type"] for r in rows if r["type"]}):
        typed = [r for r in rows if r["type"] == t]
        by_type[t] = {m: _stats([r["scores"][m] for r in typed])["mean"] for m in METRIC_NAMES}

    return {"goldens": len(rows), "metrics": per_metric, "by_type": by_type,
            "goldens_with_errors": [r["id"] for r in rows if r["errors"]]}


async def score_run(run_id: str, limit: int | None = None, ids: set | None = None) -> dict:
    run_dir = RESULTS_DIR / run_id
    pipeline = load_results(run_dir / RESULTS_FILE)
    if not pipeline:
        sys.exit(f"No pipeline results at {run_dir / RESULTS_FILE} — run evals.pipeline first.")

    records = [r for gid, r in pipeline.items() if not ids or gid in ids]
    records = records[:limit] if limit else records
    out_path = run_dir / METRICS_FILE
    done = {gid for gid, r in load_results(out_path).items() if not r["errors"]}
    pending = [r for r in records if r["id"] not in done]
    print(f"Scoring run {run_id}: {len(records)} goldens, {len(done & {r['id'] for r in records})} already scored, "
          f"{len(pending)} to score.")

    if pending:
        print("Loading judge + metrics (first time loads the local embedding model)...")
        scorer = RagasScorer()

    for i, record in enumerate(pending, start=1):
        row = await scorer.score_golden(record)
        with open(out_path, "a", encoding="utf-8") as f:  # append immediately — crash-safe
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        shown = "  ".join(f"{m[:12]}={row['scores'][m] if row['scores'][m] is not None else '—'}" for m in METRIC_NAMES)
        flag = f"  ERRORS: {list(row['errors'])}" if row["errors"] else ""
        print(f"[{i}/{len(pending)}] {row['id']:9} {row['scoring_s']:5.1f}s  {shown}{flag}")

    scored = {gid: r for gid, r in load_results(out_path).items() if gid in {r["id"] for r in records}}
    summary = summarize(scored)
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Score a pipeline run with Ragas (Qwen judge).")
    parser.add_argument("--run", required=True, metavar="RUN_ID", help="folder name under evals/results/")
    parser.add_argument("--limit", type=int, help="only score the first N goldens")
    parser.add_argument("--ids", help="comma-separated golden ids to score (testing)")
    args = parser.parse_args()

    # judge_contexts() imports the responder, whose logfire.warning calls would print
    # "LogfireNotConfiguredWarning" — scoring doesn't need Logfire, so keep the terminal clean
    import os
    os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

    import ragas, deepeval
    from evals import judge
    run_dir = RESULTS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics_info.json").write_text(json.dumps({
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge.JUDGE_MODEL, "embedding_model": judge.EMBEDDING_MODEL,
        "ragas_version": ragas.__version__, "deepeval_version": deepeval.__version__,
        "tokens_per_minute_budget": judge.TOKENS_PER_MINUTE,
    }, indent=2))

    ids = set(args.ids.split(",")) if args.ids else None
    try:
        summary = asyncio.run(score_run(args.run, limit=args.limit, ids=ids))
    except KeyboardInterrupt:
        print(f"\nInterrupted. Scored goldens are saved. Re-run the same command to resume.")
        sys.exit(130)

    print("\n=== Metrics summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
