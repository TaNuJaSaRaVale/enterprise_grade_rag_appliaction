"""
Honest-refusal metric for out-of-scope goldens (the questions the docs do NOT answer).

Ragas metrics don't apply to them (there is no correct passage to find), so they are
scored here with deterministic rules — no LLM judge, no quota, same result every run:

    states_not_in_docs      the answer says up front that the docs don't cover it
    labels_general_advice   any extra advice is labelled "General guidance (not from the docs)"
    no_doc_citations        no [n] citations — citing excerpts would pass general knowledge off as docs

A golden passes only if all three hold.

    python -m evals.refusal --run 20260927-112048

Reads   evals/results/<run_id>/pipeline_results.jsonl
Writes  evals/results/<run_id>/refusal_results.json
"""
import argparse
import json
import re
import sys

from evals.pipeline import RESULTS_DIR, RESULTS_FILE, load_results

# The refusal must come first, not be buried after a confident-sounding answer
OPENING_CHARS = 300
NOT_IN_DOCS = re.compile(
    r"(couldn'?t|could not|can'?t|cannot|unable to) find|"
    r"(does not|doesn'?t|do not|don'?t) (cover|contain|mention|include)|"
    r"not (covered|mentioned|found) in|no relevant (documentation|information)",
    re.IGNORECASE,
)
GENERAL_LABEL = re.compile(r"general guidance \(not from the docs\)", re.IGNORECASE)
CITATION = re.compile(r"\[\d+\]")


def check_refusal(answer: str) -> dict:
    answer = (answer or "").strip()
    opening = answer[:OPENING_CHARS]
    # "Extra advice" = anything after the first paragraph (the refusal sentence itself)
    has_extra = len(answer.split("\n\n", 1)) > 1 and answer.split("\n\n", 1)[1].strip() != ""
    checks = {
        "states_not_in_docs": bool(NOT_IN_DOCS.search(opening)),
        "labels_general_advice": (not has_extra) or bool(GENERAL_LABEL.search(answer)),
        "no_doc_citations": not CITATION.search(answer),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="Score out-of-scope goldens for honest refusal.")
    parser.add_argument("--run", required=True, metavar="RUN_ID")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run
    records = load_results(run_dir / RESULTS_FILE)
    if not records:
        sys.exit(f"No pipeline results at {run_dir / RESULTS_FILE}")

    out_of_scope = [r for r in records.values() if not r["expected_source"]]
    rows = []
    for r in out_of_scope:
        if r["error"] or r["guard_fired"]:
            # A guardrail block is not an honest refusal of a technical question — report it apart
            rows.append({"id": r["id"], "passed": None,
                         "note": "pipeline error" if r["error"] else "blocked by guardrails"})
            continue
        rows.append({"id": r["id"], **check_refusal(r["answer"]),
                     "answer_opening": (r["answer"] or "")[:120]})

    scored = [row for row in rows if row["passed"] is not None]
    summary = {
        "run_id": args.run,
        "goldens": len(rows),
        "honest_refusal_rate": round(sum(row["passed"] for row in scored) / len(scored), 3) if scored else None,
        "n": len(scored),
        "results": rows,
    }
    (run_dir / "refusal_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    for row in rows:
        mark = "—" if row["passed"] is None else ("PASS" if row["passed"] else "FAIL")
        detail = row.get("note") or "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in row["checks"].items())
        print(f"{mark:4}  {row['id']:8}  {detail}")
    print(f"\nHonest refusal rate: {summary['honest_refusal_rate']} (n={summary['n']})")
    print(f"Wrote {run_dir / 'refusal_results.json'}")


if __name__ == "__main__":
    main()
