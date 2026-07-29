"""Comprehensive per-dataset results tables across all evaluation metrics.

Prints 5 tables (one per LongEmbed subset + a macro average over subsets):
rows = systems (LLM extractors and trivial baselines, auto-discovered from the
artifacts, so newly generated models appear as soon as their metrics ran);
columns =

    gold-ans   answer-bearing rate (data/eval/answer_containment.json;
               narrativeqa/2wikimqa only — gold answers exist only there)
    judge      pairwise-preference Bradley-Terry win rate
               (data/eval/judge_preference.json; "—" until the judge runs)
    plaus      retrieval plausibility: NDCG@10 of the spans-only pseudo-doc
               against the untouched corpus (data/eval/retrieval_score.json)
    compr      retrieval comprehensiveness: mean relative MaxSim score drop
               after deleting spans±context from the document

Rerun any metric script, then rerun this — it only reads the JSON artifacts.
The visualizer's "span-extraction performance" page renders the same tables
live via build_summary() (see visualize/server.py), so no markdown file needs
to be kept in sync by hand; --md remains for ad-hoc exports.

    python eval/summary_table.py
    python eval/summary_table.py --md /tmp/results-summary.md
    python eval/summary_table.py --plaus-key plausibility@0 --compr-key comprehensiveness@2048
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SUBSETS = ["narrativeqa", "qmsum", "summ_screen_fd", "2wikimqa"]
GOLD_SUBSETS = {"narrativeqa", "2wikimqa"}

AC_PATH = os.path.join(REPO_ROOT, "data/eval/answer_containment.json")
JUDGE_PATH = os.path.join(REPO_ROOT, "data/eval/judge_preference.json")
RETR_PATH = os.path.join(REPO_ROOT, "data/eval/retrieval_score.json")

COLUMNS = ["gold-ans", "judge", "plaus", "compr"]


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def gather(plaus_key, compr_key):
    """-> {system: {subset: {column: value|None}}} over every system any artifact knows."""
    ac, judge, retr = load_json(AC_PATH), load_json(JUDGE_PATH), load_json(RETR_PATH)
    cells = {}

    def cell(system, subset):
        return cells.setdefault(system, {}).setdefault(
            subset, {c: None for c in COLUMNS})

    for system, entry in (ac or {}).get("runs", {}).items():
        for row in entry.get("rows", []):
            if row["group"] in GOLD_SUBSETS and row.get("bearing") is not None:
                cell(system, row["group"])["gold-ans"] = row["bearing"]

    for system, per_subset in (judge or {}).get("runs", {}).items():
        for subset, val in per_subset.items():
            if subset in SUBSETS:
                cell(system, subset)["judge"] = val.get("bt_winrate", val) \
                    if isinstance(val, dict) else val

    for system, entry in (retr or {}).get("runs", {}).items():
        plaus = entry.get(plaus_key, {})
        compr = entry.get(compr_key, {})
        for subset, m in plaus.items():
            if subset in SUBSETS:
                cell(system, subset)["plaus"] = m["ndcg10"]
        for subset, m in compr.items():
            if subset in SUBSETS:
                cell(system, subset)["compr"] = m["mean_rel_drop"]
    return cells


def note_text(plaus_key, compr_key):
    return (f"gold-ans = answer-bearing % (gold answers exist for narrativeqa/2wikimqa only). "
            f"judge = pairwise BT win rate % (— until the judge runs). "
            f"plaus = NDCG@10 % of spans-only pseudo-doc ({plaus_key}). "
            f"compr = mean relative score drop % after span removal ({compr_key}); "
            f"higher is better in every column; compare against the baseline rows, "
            f"not against zero.")


def build_summary(plaus_key="plausibility@0", compr_key="comprehensiveness@2048"):
    """Serializable view of the results tables, consumed by the visualizer.

    Returns a dict with the column list, one table per LongEmbed subset plus a
    macro average, and the caption. Values are raw fractions (None if missing);
    the client multiplies by 100 for display, matching the markdown renderer.
    """
    cells = gather(plaus_key, compr_key)
    systems = order_systems(cells)
    macro = macro_cells(cells)
    groups = [(s, s) for s in SUBSETS] + [("macro average (over subsets with data)", None)]
    tables = []
    for title, subset in groups:
        rows = []
        for system in systems:
            row = macro[system] if subset is None else cells[system].get(subset, {})
            rows.append({"system": short_name(system),
                         "cells": {c: row.get(c) for c in COLUMNS}})
        tables.append({"title": title, "subset": subset, "rows": rows})
    return {
        "columns": COLUMNS,
        "subsets": SUBSETS,
        "tables": tables,
        "plaus_key": plaus_key,
        "compr_key": compr_key,
        "note": note_text(plaus_key, compr_key),
    }


def order_systems(systems):
    def key(s):
        if s.startswith("long-embed-xml-constrained/"):
            group = 0
        elif s.startswith("long-embed-xml/"):
            group = 1
        elif s.startswith("long-embed-json/"):
            group = 2
        elif s.startswith("baseline:"):
            group = 3
        else:
            group = 4
        return (group, s)
    return sorted(systems, key=key)


def short_name(system):
    return (system
            .replace("long-embed-xml-constrained/", "xml-con ")
            .replace("long-embed-xml/", "xml-unc ")
            .replace("long-embed-json/", "json-unc ")
            .replace("google~", "").replace("mistralai~", "")
            .replace("-Instruct-2512", ""))


def fmt(value, column):
    if value is None:
        return "—"
    return f"{value * 100:.1f}"


def macro_cells(cells):
    """Unweighted mean over the subsets where each column has a value."""
    out = {}
    for system, per_subset in cells.items():
        agg = {c: [] for c in COLUMNS}
        for subset in SUBSETS:
            for c in COLUMNS:
                v = per_subset.get(subset, {}).get(c)
                if v is not None:
                    agg[c].append(v)
        out[system] = {c: (sum(v) / len(v) if v else None) for c, v in agg.items()}
    return out


def render(summary):
    lines = []
    columns = summary["columns"]
    for table in summary["tables"]:
        lines.append(f"\n### {table['title']}")
        lines.append("")
        lines.append("| system | " + " | ".join(columns) + " |")
        lines.append("|---" * (len(columns) + 1) + "|")
        for row in table["rows"]:
            vals = [fmt(row["cells"].get(c), c) for c in columns]
            lines.append(f"| {row['system']} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append(summary["note"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plaus-key", default="plausibility@0",
                        help="retrieval_score run key for the plausibility column")
    parser.add_argument("--compr-key", default="comprehensiveness@2048",
                        help="retrieval_score run key for the comprehensiveness column")
    parser.add_argument("--md", default=None,
                        help="also write the tables as markdown to this path")
    args = parser.parse_args()

    summary = build_summary(args.plaus_key, args.compr_key)
    if not summary["tables"] or not summary["tables"][0]["rows"]:
        sys.exit("no artifacts found — run eval/answer_containment.py and/or "
                 "eval/retrieval_score.py first")
    text = render(summary)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("# Evaluation results summary\n\nGenerated by `eval/summary_table.py` "
                    "— rerun it after any metric run; do not edit by hand.\n" + text + "\n")
        print(f"\nwrote {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
