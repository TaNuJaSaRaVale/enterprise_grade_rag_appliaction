"""
Guardrails eval: does the system block what it should, and ONLY that?

    python -m evals.guardrails_eval              # layer 1 only (free, deterministic)
    python -m evals.guardrails_eval --planner    # + layer 2 (one planner LLM call per case)

Two layers are measured separately and combined:
    layer 1  NeMo rails, embedding similarity to example phrases (no LLM call, ~0.02s)
    layer 2  the planner LLM's "off_topic" intent (app/agents/nodes/planner.py)
    system   blocked if EITHER layer blocks — what a user actually experiences

Metrics per dataset split:
    block_rate        of the should-block cases (off-topic, jailbreak), how many were blocked
    false_block_rate  of the should-pass technical questions, how many were wrongly blocked

Splits in guardrails_dataset.json: dev (diagnosis), tune (used while tuning), test (scored
once, after tuning — the honest generalisation number). With --planner, the 20 RAG goldens
are added as a "goldens" split of must-pass questions — including the out-of-scope ones
(Istio, Argo CD, BGP), which are technical and must reach RAG to get an honest "not in docs".

Cost of --planner: ~1k tokens per case on the app's Groq key, paced by --delay.

Writes evals/results/guardrails-<timestamp>/guardrails_results.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from evals.pipeline import EVALS_DIR, RESULTS_DIR, load_golden_dataset

DATASET = EVALS_DIR / "guardrails_dataset.json"
LAYERS = ("layer1", "layer2", "system")


def rails_fired(rails, indicators, message: str) -> bool:
    # Call the rails directly instead of app.guardrails.guard(): guard() fails OPEN on errors
    # (right for production), which here would turn an error into a fake "passed".
    result = rails.generate(messages=[{"role": "user", "content": message}])
    content = result.get("content", "") if isinstance(result, dict) else str(result)
    return any(indicator in content for indicator in indicators)


def _rates(rows: list[dict], layer: str) -> dict:
    rows = [r for r in rows if r.get(layer) is not None]
    should_block = [r for r in rows if r["should_fire"]]
    should_pass = [r for r in rows if not r["should_fire"]]
    rate = lambda hits, total: round(hits / len(total), 3) if total else None
    return {
        "block_rate": rate(sum(r[layer] for r in should_block), should_block),
        "false_block_rate": rate(sum(r[layer] for r in should_pass), should_pass),
        "n_should_block": len(should_block),
        "n_should_pass": len(should_pass),
    }


def _golden_cases() -> list[dict]:
    return [{"id": g["id"], "message": g["question"], "should_fire": False,
             "category": g.get("type", "technical"), "split": "goldens"} for g in load_golden_dataset()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the guardrail layers.")
    parser.add_argument("--planner", action="store_true", help="also measure the planner's off_topic intent")
    parser.add_argument("--delay", type=float, default=2.5, help="seconds between planner calls")
    args = parser.parse_args()

    cases = json.loads(DATASET.read_text())
    if args.planner:
        cases += _golden_cases()

    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
    import logging
    for noisy in ("nemoguardrails", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from app.guardrails import rails as rails_module
    from app.guardrails.colang_rules import RAIL_INDICATORS
    rails_module.initialize_rails()
    if args.planner:
        from app.agents.nodes.planner import decide

    rows = []
    for i, case in enumerate(cases, start=1):
        row = {**case, "layer1": None, "layer2": None, "system": None, "errors": {}}
        try:
            row["layer1"] = rails_fired(rails_module._rails, RAIL_INDICATORS, case["message"])
        except Exception as e:
            row["errors"]["layer1"] = f"{type(e).__name__}: {str(e)[:200]}"
        if args.planner:
            try:
                decision = decide(case["message"])
                row["layer2"] = decision.intent == "off_topic"
                row["planner_intent"] = decision.intent
            except Exception as e:
                row["errors"]["layer2"] = f"{type(e).__name__}: {str(e)[:200]}"
            if i < len(cases):
                time.sleep(args.delay)
        measured = [row[layer] for layer in ("layer1", "layer2") if row[layer] is not None]
        if measured and not row["errors"]:
            row["system"] = any(measured)
        rows.append(row)

        final = row["system"] if row["system"] is not None else row["layer1"]
        verdict = "ERROR" if row["errors"] else ("ok  " if final == case["should_fire"] else "MISS")
        shown = "  ".join(f"{layer}={'block' if row[layer] else 'pass '}" for layer in ("layer1", "layer2")
                          if row[layer] is not None)
        print(f"[{i}/{len(cases)}] {verdict} expected={'block' if case['should_fire'] else 'pass '}  "
              f"{shown}  {case['id']}: {case['message'][:55]}", flush=True)

    splits = {"all": rows}
    for name in sorted({r["split"] for r in rows}):
        splits[name] = [r for r in rows if r["split"] == name]
    layers = LAYERS if args.planner else ("layer1",)

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "layers": {"layer1": "NeMo embeddings_only (no LLM)",
                   **({"layer2": "planner LLM off_topic intent"} if args.planner else {})},
        "cases": len(rows),
        "errors": [r["id"] for r in rows if r["errors"]],
        "by_split": {name: {layer: _rates(subset, layer) for layer in layers} for name, subset in splits.items()},
        "results": rows,
    }

    run_dir = RESULTS_DIR / f"guardrails-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "guardrails_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'split':8} {'layer':7} {'block rate':>14} {'false-block rate':>18}")
    for name, by_layer in summary["by_split"].items():
        for layer, s in by_layer.items():
            print(f"{name:8} {layer:7} {str(s['block_rate']):>6} (n={s['n_should_block']:2})"
                  f" {str(s['false_block_rate']):>9} (n={s['n_should_pass']:2})")
    print(f"errors: {len(summary['errors'])}")
    print(f"Wrote {run_dir / 'guardrails_results.json'}")
    if summary["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
