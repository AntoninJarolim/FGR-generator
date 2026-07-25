"""Compute per-sample span budgets for the trivial retrieval-score baselines.

The random / lead_k / lexical baselines in eval/retrieval_score.py must select
about the SAME amount of text as the LLM extractors, otherwise the comparison
just measures amount-removed (claude_md/evaluation-metrics-assessment.md §1.4).
This script streams the constrained extraction files and records, per
(subset, qid, doc_id):

    budget  -- mean total deduplicated span chars over the systems that
               produced a non-empty extraction (0 when all are empty, so the
               baselines are no-ops exactly where every LLM was), and
    n_spans -- mean deduplicated span count (drives the random-window count).

Output: data/eval/span_budgets.json  {"<subset>|<qid>|<doc_id>": {budget, n_spans}}

Usage:
    python eval/build_baselines.py [files ...]   # default: all constrained files
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.retrieval_score import dedupe_spans, DEFAULT_BUDGETS, DEFAULT_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="*",
                        help="extraction JSONLs (default: constrained full files)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", default=DEFAULT_BUDGETS)
    args = parser.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(
        args.data_dir, "long-embed-xml-constrained", "*_from0-to12612.jsonl")))
    if not files:
        sys.exit(f"no constrained extraction files under {args.data_dir}")

    totals = {}
    for path in files:
        print(f"streaming {os.path.basename(path)} ...", file=sys.stderr)
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = f"{rec['subset']}|{rec['qid']}|{rec['doc_id']}"
                spans = dedupe_spans(rec.get("selected_spans"))
                entry = totals.setdefault(key, {"chars": [], "counts": []})
                if spans:
                    entry["chars"].append(sum(len(s) for s in spans))
                    entry["counts"].append(len(spans))

    budgets = {}
    for key, entry in totals.items():
        if entry["chars"]:
            budgets[key] = {
                "budget": round(sum(entry["chars"]) / len(entry["chars"])),
                "n_spans": sum(entry["counts"]) / len(entry["counts"]),
            }
        else:
            budgets[key] = {"budget": 0, "n_spans": 1}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(budgets, f)
    os.replace(tmp, args.out)
    n_zero = sum(1 for b in budgets.values() if b["budget"] == 0)
    print(f"wrote {len(budgets)} budgets to {args.out} "
          f"({n_zero} zero-budget samples where every system was empty)")


if __name__ == "__main__":
    main()
