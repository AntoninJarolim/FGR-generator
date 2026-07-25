"""Answer-bearing rate over extracted relevance spans (narrativeqa, 2wikimqa).

For every (query, document) sample of the two QA-derived LongEmbed subsets:
does at least one extracted span contain a gold answer? Deterministic and
judge-free (claude_md/downstream-evaluation.md Proposal 5). Requires the gold
answers file produced by eval/fetch_gold_answers.py.

Matching semantics:

  * both span and answer are SQuAD-normalized (lowercase, punctuation and
    articles stripped, whitespace collapsed) after straightening curly
    quotes/dashes;
  * a hit is TOKEN-BOUNDARY containment of the answer in a SINGLE span
    (" answer " in " span ") -- "no" does not match inside "nothing", and
    spans are never concatenated (that would invent cross-span matches);
  * bearing = any span x any gold answer; a sample without spans counts as
    not-bearing (reported as #empty); a sample whose query has no gold answer
    (join miss) is excluded from the denominator (reported as #joinmiss);
  * fuzzy is an exclusive near-miss band on the non-bearing samples: the
    answer aligns inside a span within a ~5% edit budget (answers shorter
    than 8 chars are never fuzzy-matched); +fuzzy is cumulative;
  * #short counts evaluated samples whose gold answers are ALL <= 3 chars
    after normalization (yes/no etc.) -- accidental-containment risk.

Rows: each QA subset, their micro average, and the unweighted macro average
(narrativeqa outweighs 2wikimqa ~35:1 in the micro view).

Usage:
    python eval/answer_containment.py             # all constrained+unconstrained files
    python eval/answer_containment.py FILE [...]  # specific file(s)
    python eval/answer_containment.py --limit 500 --no-cache  # quick sanity pass
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.eval_squad import normalize_answer
from visualize.stats import APPROX_RATE, NORM_TABLE, semiglobal_min_err
from custom_utils.artifact_hash import artifact_hash

# Bump when matching/aggregation semantics change; invalidates sidecar caches.
AC_VERSION = 1

QA_SUBSETS = ("narrativeqa", "2wikimqa")
SHORT_ANSWER_CHARS = 3
MODES = ("long-embed-json", "long-embed-json-constrained",
         "long-embed-xml", "long-embed-xml-constrained")

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(EVAL_DIR, ".cache")
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data/extracted_relevancy")
DEFAULT_GOLD = os.path.join(REPO_ROOT, "data/eval/gold_answers.jsonl")
DEFAULT_JSON_OUT = os.path.join(REPO_ROOT, "data/eval/answer_containment.json")


def norm(text):
    return normalize_answer(text.translate(NORM_TABLE))


def parse_spans_field(raw):
    """``selected_spans`` is a native JSON list in most runs, but the newer
    Karolina-cluster runs serialize it as a Python-repr STRING, e.g.
    "['a', 'b']" (single quotes -- not valid JSON). Iterating that string
    directly would silently yield one-character "spans". Normalize both
    shapes to an actual list of strings."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def fuzzy_contains(norm_span, norm_answer):
    """The answer aligns somewhere inside the span within a ~5% edit budget.
    Spans are short enough to run the semi-global DP whole, without the
    piece-anchoring heuristic of stats.approx_matches (whose 16-char window
    grid can truncate patterns as short as answers are)."""
    if len(norm_answer) < 8:
        return False
    max_err = max(2, round(len(norm_answer) * APPROX_RATE))
    return semiglobal_min_err(norm_span, norm_answer) <= max_err


def load_gold(path):
    """(subset, qid) -> list of normalized answers, or None for join misses /
    queries whose answers all normalize away."""
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            answers = [norm(a) for a in rec["answers"]]
            answers = [a for a in answers if a]
            gold[(rec["subset"], rec["qid"])] = answers if rec["join"] != "miss" and answers else None
    return gold


def _new_acc():
    return {"n": 0, "bearing": 0, "fuzzy": 0, "n_empty": 0,
            "n_join_miss": 0, "n_short": 0}


def compute_rows(path, gold, limit=None):
    """One streaming pass -> rows [narrativeqa, 2wikimqa, All (micro), macro]."""
    accs = {subset: _new_acc() for subset in QA_SUBSETS}
    n_other = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            subset = rec.get("subset")
            if subset not in accs:
                n_other += 1
                continue
            acc = accs[subset]
            answers = gold.get((subset, rec.get("qid")))
            if answers is None:
                acc["n_join_miss"] += 1
                continue
            acc["n"] += 1
            if all(len(a) <= SHORT_ANSWER_CHARS for a in answers):
                acc["n_short"] += 1
            spans = [s for s in parse_spans_field(rec.get("selected_spans")) if isinstance(s, str)]
            nspans = [ns for ns in (norm(s) for s in spans) if ns]
            if not nspans:
                acc["n_empty"] += 1
                continue
            if any(f" {a} " in f" {ns} " for ns in nspans for a in answers):
                acc["bearing"] += 1
            elif any(fuzzy_contains(ns, a) for ns in nspans for a in answers):
                acc["fuzzy"] += 1

    def row(group, acc):
        n = acc["n"]
        return {
            "group": group,
            "bearing": acc["bearing"] / n if n else None,
            "fuzzy": acc["fuzzy"] / n if n else None,
            "n": n,
            "n_empty": acc["n_empty"],
            "n_join_miss": acc["n_join_miss"],
            "n_short": acc["n_short"],
        }

    rows = [row(subset, accs[subset]) for subset in QA_SUBSETS]
    micro = _new_acc()
    for acc in accs.values():
        for key in micro:
            micro[key] += acc[key]
    rows.append(row("All (micro)", micro))
    rated = [r for r in rows[:len(QA_SUBSETS)] if r["bearing"] is not None]
    rows.append({
        "group": "macro",
        "bearing": sum(r["bearing"] for r in rated) / len(rated) if rated else None,
        "fuzzy": sum(r["fuzzy"] for r in rated) / len(rated) if rated else None,
        "n": None, "n_empty": None, "n_join_miss": None, "n_short": None,
    })
    rows.append({"group": "(other subsets skipped)", "bearing": None, "fuzzy": None,
                 "n": n_other, "n_empty": None, "n_join_miss": None, "n_short": None})
    return rows


# --------------------------- CLI ---------------------------

def _cache_path(path, data_dir):
    rel = os.path.relpath(path, data_dir)
    if rel.startswith(".."):
        rel = os.path.basename(path)
    return os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9._-]", "_", rel) + ".ac.json")


def cached_rows(path, gold_path, gold, data_dir, use_cache=True):
    st, gst = os.stat(path), os.stat(gold_path)
    sig = [st.st_mtime, st.st_size, gst.st_mtime, gst.st_size, AC_VERSION]
    cpath = _cache_path(path, data_dir)
    if use_cache:
        try:
            with open(cpath, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("sig") == sig:
                return data["rows"], True
        except (OSError, json.JSONDecodeError):
            pass
    t0 = time.time()
    print(f"computing {os.path.basename(path)} ...", file=sys.stderr)
    rows = compute_rows(path, gold)
    print(f"computed in {time.time() - t0:.1f}s", file=sys.stderr)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = cpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "rows": rows}, f)
        os.replace(tmp, cpath)
    except OSError as e:
        print(f"warning: could not write cache {cpath}: {e}", file=sys.stderr)
    return rows, False


def print_table(rows):
    pc = lambda v: "     —" if v is None else f"{v * 100:5.1f}%"
    ct = lambda v: "        —" if v is None else f"{v:9,}"
    header = (f"{'group':24s} {'bearing':>7s} {'fuzzy':>6s} {'+fuzzy':>6s} "
              f"{'#eval':>9s} {'#empty':>9s} {'#joinmiss':>9s} {'#short':>9s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        cum = None if r["bearing"] is None else r["bearing"] + r["fuzzy"]
        print(f"{r['group']:24s} {pc(r['bearing'])} {pc(r['fuzzy'])} {pc(cum)} "
              f"{ct(r['n'])} {ct(r['n_empty'])} {ct(r['n_join_miss'])} {ct(r['n_short'])}")


def parse_model_and_mode(path):
    """Model id and mode from an extraction path. The mode is the parent
    directory name -- one of the four canonical dirs (long-embed-json,
    long-embed-json-constrained, long-embed-xml, long-embed-xml-constrained);
    the file is <model>_from<a>-to<b>.jsonl (the mode is never a filename
    suffix)."""
    name = os.path.basename(path)
    stem = name[:-len(".jsonl")] if name.endswith(".jsonl") else name
    return stem.split("_from")[0], os.path.basename(os.path.dirname(path))


def run_key(path):
    model, mode = parse_model_and_mode(path)
    return f"{mode}/{model}"


def save_json_artifact(json_out, results):
    old = None
    try:
        with open(json_out, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    base = old if isinstance(old, dict) and "runs" in old else {"version": AC_VERSION, "runs": {}}
    # Copy, don't alias: mutating `old` in place would make the change-detection
    # hash compare the merged dict to itself and wrongly report "unchanged".
    data = {**base, "runs": dict(base.get("runs", {}))}
    for key, entry in results.items():
        data["runs"][key] = entry
    if old is not None and artifact_hash(data) == artifact_hash(old):
        print(f"json artifact unchanged: {json_out}")
        return
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    tmp = json_out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, json_out)
    print(f"wrote json artifact: {json_out}")


def default_files(data_dir):
    """One file per (mode, model): the widest generation range, skipping
    partial/broken leftovers (e.g. *_from0-to10.jsonl, *.broken-empty.jsonl).
    Scans the four canonical mode dirs (MODES); long-embed-json-superseded/
    (the small-model JSON-prompt runs replaced by their XML span-grammar runs)
    is deliberately not listed, so it stays excluded from every metric."""
    patterns = [os.path.join(data_dir, mode, "*.jsonl") for mode in MODES]
    best = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)
            m = re.search(r"_from(\d+)-to(\d+)\.jsonl$", name)
            if m is None or "broken" in name or "partial" in name:
                continue
            key, width = run_key(path), int(m.group(2)) - int(m.group(1))
            if key not in best or width > best[key][0]:
                best[key] = (width, path)
    return [path for _width, path in best.values()]


def main():
    parser = argparse.ArgumentParser(
        description="Answer-bearing rate: %% of narrativeqa/2wikimqa samples where "
                    "some extracted span contains a gold answer.")
    parser.add_argument("files", nargs="*",
                        help="Extraction JSONL file(s); default: constrained + "
                             "unconstrained files under --data-dir")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT,
                        help="Combined JSON artifact ('' disables)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N lines of each file (sanity "
                             "runs; disables cache and json artifact)")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        sys.exit(f"gold answers file not found: {args.gold}\n"
                 f"run: python eval/fetch_gold_answers.py")
    gold = load_gold(args.gold)

    files = args.files or default_files(args.data_dir)
    if not files:
        sys.exit(f"no .jsonl files found under {args.data_dir}")

    results = {}
    for i, path in enumerate(files):
        if args.limit is not None:
            rows, from_cache = compute_rows(path, gold, limit=args.limit), False
        else:
            rows, from_cache = cached_rows(path, args.gold, gold, args.data_dir,
                                           use_cache=not args.no_cache)
        results[run_key(path)] = {"file": os.path.relpath(path, REPO_ROOT), "rows": rows}
        if i:
            print()
        print(f"== {run_key(path)}{' (cached)' if from_cache else ''} ==")
        print_table(rows)

    if args.json_out and args.limit is None:
        save_json_artifact(args.json_out, results)


if __name__ == "__main__":
    main()
