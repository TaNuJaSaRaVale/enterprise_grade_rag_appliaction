"""
Export a small, UI-friendly scorecard from an eval run → ui/eval_scorecard.json

    python -m evals.export_scorecard --run 20260927-112048

The deployed UI shows these measured numbers; the raw results folder is not shipped.
"""
import argparse
import json
from pathlib import Path

from evals.metrics import METRICS_FILE, summarize
from evals.pipeline import RESULTS_DIR, load_results

OUT_PATH = Path(__file__).parent.parent / "ui" / "eval_scorecard.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, metavar="RUN_ID")
    args = parser.parse_args()
    run_dir = RESULTS_DIR / args.run

    pipeline_summary = json.loads((run_dir / "summary.json").read_text())
    info = json.loads((run_dir / "metrics_info.json").read_text())
    scored = {gid: r for gid, r in load_results(run_dir / METRICS_FILE).items() if not r["errors"]}
    metrics = summarize(scored)["metrics"]

    scorecard = {
        "run_id": args.run,
        "judge_model": info["judge_model"],
        "goldens_total": pipeline_summary["total"],
        "goldens_scored": len(scored),
        "source_hit_rate": pipeline_summary["source_hit_rate"],
        "avg_latency_s": pipeline_summary["avg_latency_s"],
        "metrics": {name: {"mean": m["mean"], "n": m["n"]} for name, m in metrics.items()},
    }
    OUT_PATH.write_text(json.dumps(scorecard, indent=2))
    print(json.dumps(scorecard, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
