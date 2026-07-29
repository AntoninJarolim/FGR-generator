"""Constrained vs unconstrained answer-bearing, decomposed into failure vs quality.

The headline answer-bearing rate (eval/answer_containment.py) counts an empty
extraction as a miss. So a raw con-vs-unc gap conflates two very different things:

  * FAILURE rate  -- how often the mode produced no usable span at all, and
  * span QUALITY  -- given that BOTH modes produced spans for the same sample,
                     which mode's spans actually contain the gold answer.

This script pairs the two modes of one model on (subset, qid) and reports:

  1. raw bearing per mode (empties = miss) -- reproduces the table;
  2. empty/non-empty 2x2 between the modes;
  3. bearing restricted to samples where BOTH modes produced spans -- the clean
     quality comparison, with a McNemar split (con-only wins vs unc-only wins);
  4. so the raw gap is attributed to failure-rate difference vs quality.

Answers "why is a model better unconstrained?": if the restricted-both bearing
is ~equal but raw differs, the gap is purely who-failed-more (denominator); if
restricted-both still favors one mode, that mode's spans are genuinely more
answer-bearing (constrained decoding forces verbatim document substrings, while
free-form spans can restate the answer wording -- which the containment test
rewards).

    python eval/paired_success_analysis.py                 # every model with both modes
    python eval/paired_success_analysis.py --model google~gemma-4-12B-it
    python eval/paired_success_analysis.py --md claude_md/constrained-vs-unconstrained.md
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.answer_containment import (
    QA_SUBSETS, DEFAULT_DATA_DIR, DEFAULT_GOLD, default_files, load_gold, norm,
    parse_spans_field, fuzzy_contains, parse_model_and_mode)


def sample_bearing(spans, answers):
    """(non_empty, bearing) for one sample given its gold answers."""
    nspans = [ns for ns in (norm(s) for s in spans if isinstance(s, str)) if ns]
    if not nspans:
        return False, False
    bearing = (any(f" {a} " in f" {ns} " for ns in nspans for a in answers)
               or any(fuzzy_contains(ns, a) for ns in nspans for a in answers))
    return True, bearing


def load_mode(path, gold):
    """(subset, qid) -> (non_empty, bearing) for QA subsets with gold answers."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            subset = rec.get("subset")
            if subset not in QA_SUBSETS:
                continue
            answers = gold.get((subset, rec.get("qid")))
            if answers is None:
                continue
            spans = parse_spans_field(rec.get("selected_spans"))
            out[(subset, rec["qid"])] = sample_bearing(spans, answers)
    return out


def analyze(model, con_path, unc_path, gold):
    con, unc = load_mode(con_path, gold), load_mode(unc_path, gold)
    keys = sorted(con.keys() & unc.keys())
    agg = {}
    for subset in list(QA_SUBSETS) + ["macro"]:
        agg[subset] = {"n": 0, "con_ne": 0, "unc_ne": 0, "con_bear": 0,
                       "unc_bear": 0, "both_ne": 0, "both_bear_con": 0,
                       "both_bear_unc": 0, "con_only": 0, "unc_only": 0}
    for key in keys:
        subset = key[0]
        con_ne, con_b = con[key]
        unc_ne, unc_b = unc[key]
        for grp in (subset, "macro"):
            a = agg[grp]
            a["n"] += 1
            a["con_ne"] += con_ne
            a["unc_ne"] += unc_ne
            a["con_bear"] += con_b
            a["unc_bear"] += unc_b
            if con_ne and unc_ne:
                a["both_ne"] += 1
                a["both_bear_con"] += con_b
                a["both_bear_unc"] += unc_b
                a["con_only"] += con_b and not unc_b
                a["unc_only"] += unc_b and not con_b
    return agg


def _pct(num, den):
    return None if not den else num / den


def rows_for(model, agg):
    rows = []
    for subset in list(QA_SUBSETS) + ["macro"]:
        a = agg[subset]
        # macro = unweighted mean over the two subsets, matching answer_containment
        rows.append({
            "group": subset,
            "n": a["n"],
            "con_fail": _pct(a["n"] - a["con_ne"], a["n"]),
            "unc_fail": _pct(a["n"] - a["unc_ne"], a["n"]),
            "con_bear_raw": _pct(a["con_bear"], a["n"]),
            "unc_bear_raw": _pct(a["unc_bear"], a["n"]),
            "both_ne": a["both_ne"],
            "con_bear_both": _pct(a["both_bear_con"], a["both_ne"]),
            "unc_bear_both": _pct(a["both_bear_unc"], a["both_ne"]),
            "con_only": a["con_only"],
            "unc_only": a["unc_only"],
        })
    # replace macro subset-aggregate with unweighted mean of the QA subsets
    qa = [r for r in rows if r["group"] in QA_SUBSETS]
    macro = next(r for r in rows if r["group"] == "macro")
    for col in ("con_fail", "unc_fail", "con_bear_raw", "unc_bear_raw",
                "con_bear_both", "unc_bear_both"):
        vals = [r[col] for r in qa if r[col] is not None]
        macro[col] = sum(vals) / len(vals) if vals else None
    return rows


def render(model, rows):
    pc = lambda v: "   —" if v is None else f"{v*100:5.1f}"
    lines = [f"\n### {model}", "",
             "| subset | n | con fail% | unc fail% | con bear% (raw) | unc bear% (raw) "
             "| both-nonempty n | con bear% (both) | unc bear% (both) | con-only wins | unc-only wins |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['n']} | {pc(r['con_fail'])} | {pc(r['unc_fail'])} "
            f"| {pc(r['con_bear_raw'])} | {pc(r['unc_bear_raw'])} | {r['both_ne']} "
            f"| {pc(r['con_bear_both'])} | {pc(r['unc_bear_both'])} "
            f"| {r['con_only']} | {r['unc_only']} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--model", default=None,
                        help="restrict to one model id (e.g. google~gemma-4-12B-it)")
    parser.add_argument("--md", default=None, help="also write markdown here")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    by_model = {}
    for path in default_files(args.data_dir):
        model, mode = parse_model_and_mode(path)
        by_model.setdefault(model, {})[mode] = path

    chunks = []
    for model in sorted(by_model):
        if args.model and model != args.model:
            continue
        modes = by_model[model]
        # Pair a model's constrained arm with its matching unconstrained arm of
        # the same output format (XML span-grammar preferred, else JSON-guided).
        con = modes.get("long-embed-xml-constrained")
        unc = modes.get("long-embed-xml") or modes.get("long-embed-json")
        if not con or not unc:
            print(f"skip {model}: need constrained+unconstrained, have {list(modes)}", file=sys.stderr)
            continue
        agg = analyze(model, con, unc, gold)
        rows = rows_for(model, agg)
        chunk = render(model, rows)
        print(chunk)
        chunks.append(chunk)

    if args.md:
        header = ("# Constrained vs unconstrained: failure rate vs span quality\n\n"
                  "Generated by `eval/paired_success_analysis.py`. Columns: fail% = "
                  "share of samples with no usable span; bear% (raw) = answer-bearing "
                  "counting empties as misses (matches the summary table); both-nonempty "
                  "n = samples where BOTH modes produced spans; bear% (both) = "
                  "answer-bearing on that paired subset (the clean quality comparison); "
                  "con-/unc-only wins = of the both-nonempty samples, how many one mode "
                  "gets right and the other misses (McNemar split).\n")
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(chunks) + "\n")
        print(f"\nwrote {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
