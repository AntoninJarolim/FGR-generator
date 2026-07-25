"""Fetch gold QA answers for the LongEmbed narrativeqa / 2wikimqa subsets.

LongEmbed's qrels carry no answers (their ``text`` field is empty), so the
answer-containment metric (eval/answer_containment.py) needs answers joined in
from the original QA datasets:

  * narrativeqa -- the official ``qaps.csv`` of the DeepMind repo (test split;
    LongEmbed narrativeqa queries are verbatim test-set questions);
  * 2wikimqa    -- LongBench's length-uniform sample ``2wikimqa_e`` (the split
    LongEmbed sampled from), shipped inside ``data.zip`` of the
    THUDM/LongBench dataset repo; plain ``2wikimqa`` is unioned in as a
    fallback when coverage stays below 100%.

Joining is by query TEXT (LongEmbed qid/doc_id are internal ids that mean
nothing to the source datasets), tried at three normalization tiers:
raw strip -> lowercase + collapsed whitespace -> SQuAD-normalized. Some
narrativeqa question texts repeat across different books; those join
ambiguously and receive the UNION of the answers (flagged ``ambiguous``).

Writes one line per (subset, qid) -- including misses -- so the metric script
never needs the gold sources:

    {"subset", "qid", "query", "answers", "join": "raw|loose|full|miss",
     "ambiguous", "n_gold_docs"}

Usage:
    python eval/fetch_gold_answers.py            # -> data/eval/gold_answers.jsonl
    python eval/fetch_gold_answers.py --queries-from <extraction.jsonl>  # offline
"""
import argparse
import io
import json
import os
import sys
import zipfile
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.eval_squad import normalize_answer
from visualize.stats import NORM_TABLE
from custom_utils.longembed import LONGEMBED_DATASET_ID

QAPS_URL = "https://github.com/google-deepmind/narrativeqa/raw/master/qaps.csv"
SUBSETS = ["narrativeqa", "2wikimqa"]
TIERS = ["raw", "loose", "full"]


def tier_key(query, tier):
    if tier == "raw":
        return query.strip()
    if tier == "loose":
        return " ".join(query.split()).lower()
    return normalize_answer(query.translate(NORM_TABLE))


def _add_row(maps, question, answers, doc_id):
    answers = [a.strip() for a in answers if isinstance(a, str) and a.strip()]
    for tier in TIERS:
        key = tier_key(question, tier)
        if not key:
            continue
        entry = maps[tier].setdefault(key, {"answers": [], "doc_ids": set()})
        for a in answers:
            if a not in entry["answers"]:
                entry["answers"].append(a)
        entry["doc_ids"].add(doc_id)


def build_narrativeqa_maps(qaps_url):
    import pandas as pd

    # keep_default_na=False: answers like "None" or "NA" must stay strings.
    df = pd.read_csv(qaps_url, keep_default_na=False)
    df = df[df["set"] == "test"]
    maps = {tier: {} for tier in TIERS}
    for row in df.itertuples(index=False):
        _add_row(maps, row.question, [row.answer1, row.answer2], row.document_id)
    print(f"narrativeqa gold: {len(df)} test-split rows from {qaps_url}")
    return maps


def load_longbench_rows(config):
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip",
                               repo_type="dataset")
    with zipfile.ZipFile(zip_path) as z:
        member = next((n for n in z.namelist()
                       if n.endswith(f"{config}.jsonl") and "._" not in n), None)
        if member is None:
            sys.exit(f"{config}.jsonl not found inside {zip_path}")
        with z.open(member) as f:
            return [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8")]


def build_2wikimqa_maps(configs):
    maps = {tier: {} for tier in TIERS}
    for config in configs:
        rows = load_longbench_rows(config)
        for row in rows:
            _add_row(maps, row["input"], row.get("answers") or [], row.get("_id", ""))
        print(f"2wikimqa gold: +{len(rows)} rows from LongBench/{config}")
    return maps


def load_longembed_queries(subset):
    from datasets import load_dataset

    rows = load_dataset(LONGEMBED_DATASET_ID, subset, split="queries")
    return [(row["qid"], row["text"]) for row in rows]


def harvest_queries_from_file(path):
    """Offline fallback: collect (subset, qid, query) by streaming one
    extraction JSONL (any model's file works -- all carry the same pairs)."""
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("subset") in SUBSETS:
                seen.setdefault((rec["subset"], rec["qid"]), rec["query"])
    out = {subset: [] for subset in SUBSETS}
    for (subset, qid), query in seen.items():
        out[subset].append((qid, query))
    return out


def join_subset(subset, queries, maps):
    results, tier_hits, misses, n_ambiguous = [], Counter(), [], 0
    for qid, query in queries:
        entry, hit_tier = None, "miss"
        for tier in TIERS:
            entry = maps[tier].get(tier_key(query, tier))
            if entry is not None:
                hit_tier = tier
                break
        tier_hits[hit_tier] += 1
        ambiguous = bool(entry) and len(entry["doc_ids"]) > 1
        n_ambiguous += ambiguous
        if entry is None:
            misses.append(query)
        results.append({
            "subset": subset,
            "qid": qid,
            "query": query,
            "answers": entry["answers"] if entry else [],
            "join": hit_tier,
            "ambiguous": ambiguous,
            "n_gold_docs": len(entry["doc_ids"]) if entry else 0,
        })
    return results, tier_hits, misses, n_ambiguous


def main():
    parser = argparse.ArgumentParser(
        description="Join gold QA answers onto LongEmbed narrativeqa/2wikimqa "
                    "queries (by query text) for the answer-containment metric.")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "data/eval/gold_answers.jsonl"))
    parser.add_argument("--qaps-url", default=QAPS_URL,
                        help="URL or local path of the narrativeqa qaps.csv")
    parser.add_argument("--queries-from", default=None,
                        help="Extraction JSONL to harvest (subset,qid,query) from "
                             "instead of loading the LongEmbed queries via HF datasets.")
    args = parser.parse_args()

    if args.queries_from:
        queries = harvest_queries_from_file(args.queries_from)
    else:
        queries = {subset: load_longembed_queries(subset) for subset in SUBSETS}
    for subset in SUBSETS:
        print(f"LongEmbed[{subset}]: {len(queries[subset])} queries")

    gold_maps = {"narrativeqa": build_narrativeqa_maps(args.qaps_url)}
    maps_2wiki = build_2wikimqa_maps(["2wikimqa_e"])
    covered = sum(any(tier_key(q, t) in maps_2wiki[t] for t in TIERS)
                  for _, q in queries["2wikimqa"])
    if covered < len(queries["2wikimqa"]):
        print(f"2wikimqa: only {covered}/{len(queries['2wikimqa'])} covered by "
              f"2wikimqa_e -- unioning in the plain 2wikimqa config")
        for tier, m in build_2wikimqa_maps(["2wikimqa"]).items():
            for key, entry in m.items():
                tgt = maps_2wiki[tier].setdefault(key, {"answers": [], "doc_ids": set()})
                for a in entry["answers"]:
                    if a not in tgt["answers"]:
                        tgt["answers"].append(a)
                tgt["doc_ids"] |= entry["doc_ids"]
    gold_maps["2wikimqa"] = maps_2wiki

    all_results = []
    for subset in SUBSETS:
        results, tier_hits, misses, n_ambiguous = join_subset(
            subset, queries[subset], gold_maps[subset])
        all_results.extend(results)
        total = len(results)
        joined = total - tier_hits["miss"]
        n_answers = Counter(len(r["answers"]) for r in results if r["join"] != "miss")
        print(f"\n== {subset}: {joined}/{total} joined ==")
        print(f"  per tier: " + ", ".join(f"{t}={tier_hits[t]}" for t in TIERS + ["miss"]))
        print(f"  ambiguous (question text on >1 gold doc): {n_ambiguous}")
        print(f"  answers-per-query histogram: {dict(sorted(n_answers.items()))}")
        for q in misses[:10]:
            print(f"  MISS: {q[:120]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    print(f"\nwrote {len(all_results)} records to {args.out}")


if __name__ == "__main__":
    main()
