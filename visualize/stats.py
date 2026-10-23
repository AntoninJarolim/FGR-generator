"""Span match-rate statistics over an extracted-relevancy JSONL file.

Classifies every selected span with the SAME three matching tiers the viewer
frontend uses (keep in sync with static/index.html):

  * em     -- the span occurs verbatim in the passage;
  * norm   -- it occurs after conservative normalization only (lowercase,
              straight quotes/dashes, collapsed whitespace), contiguously;
  * approx -- it aligns somewhere with at most APPROX_RATE (~5%) of its length
              in character edits (min 2), found via piece anchoring + a
              semi-global Levenshtein alignment (numpy);
  * nf     -- not found (the remainder; tiers are mutually exclusive).

Aggregation is MACRO, per the analysis requirement: first average the 0/1 tier
indicators over the spans of one sample, then average those per-sample values
over the samples of a group -- so every sample weighs the same no matter how
many spans it has. Samples with zero spans cannot have a span-level rate and
are excluded from the averages (reported separately as ``n_nospan``).

Groups: "All datasets" plus each subset, in file order.

Also runnable directly to print the tables in the terminal:

    python visualize/stats.py                       # every file under data/extracted_relevancy
    python visualize/stats.py path/to/output.jsonl  # specific file(s)
    python visualize/stats.py --no-cache ...        # force recomputation

It shares the server's sidecar cache (visualize/.cache), so anything the
server already computed prints instantly, and vice versa.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

# Run directly (``python visualize/stats.py``) sys.path[0] is visualize/, not
# the repo root, so make the repo importable before pulling in custom_utils.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Span matching -- tiers, normalization, approximate alignment -- lives in
# custom_utils/span_match.py so eval/heuristic_spans.py, which needs the match
# OFFSETS and not just the tier, shares this exact implementation. One
# implementation is what keeps the located spans and these tier rates in
# agreement. Re-exported here because callers already import them from stats.
from custom_utils.span_match import (  # noqa: F401  (re-exported)
    APPROX_RATE, NORM_TABLE, approx_matches, locate_span, normalize,
    normalize_with_map, semiglobal_min_err,
)

# Bump when classification/aggregation semantics change: cached stats sidecars
# with a different version are recomputed instead of served.
STATS_VERSION = 4


class _NormCache:
    """Tiny FIFO cache of normalized passages. Output rows are grouped by
    document, so a handful of entries removes nearly all recomputation."""

    def __init__(self, cap=8):
        self.cap = cap
        self.entries = {}

    def get(self, key, passage):
        if key not in self.entries:
            if len(self.entries) >= self.cap:
                self.entries.pop(next(iter(self.entries)))
            self.entries[key] = normalize(passage)
        return self.entries[key]


def classify_spans(passage, spans, norm_passage_fn):
    """Tier per span: 'em' | 'norm' | 'approx' | 'nf' (first tier that hits).
    Non-string span items (malformed model output in some old files) are 'nf'."""
    tiers = []
    for s in spans:
        if not isinstance(s, str):
            tiers.append("nf")
            continue
        if s and s in passage:
            tiers.append("em")
            continue
        ns = normalize(s).strip() if s else ""
        if not ns:
            tiers.append("nf")
        elif ns in norm_passage_fn():
            tiers.append("norm")
        elif approx_matches(norm_passage_fn(), ns):
            tiers.append("approx")
        else:
            tiers.append("nf")
    return tiers


ALL_GROUP = "All datasets"


def compute_stats(path):
    """One pass over the file -> ordered rows of macro-aggregated tier rates:
    [{group, em, norm, approx, n, n_nospan}, ...], ALL_GROUP first, then each
    subset in order of first appearance. Rates are fractions (or None when a
    group has no sample with spans)."""
    groups = {ALL_GROUP: _new_acc()}
    cache = _NormCache()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            spans = rec.get("selected_spans") or []
            accs = [groups[ALL_GROUP],
                    groups.setdefault(rec.get("subset") or "?", _new_acc())]
            if not spans:
                for a in accs:
                    a["n"] += 1
                    a["n_nospan"] += 1
                continue
            # "clean_text"/"document_id" are the older FDM output schema.
            passage = rec.get("passage") or rec.get("clean_text") or ""
            key = rec.get("doc_id") or rec.get("document_id"), rec.get("subset")
            tiers = classify_spans(passage, spans, lambda: cache.get(key, passage))
            # Sample-level outcome, alongside the span-level rates. What survives
            # locating is what the judge and the retrieval metrics actually see
            # (eval/heuristic_spans.py drops 'nf'), so the question "is this
            # sample used as generated, repaired, or gutted?" is answered here and
            # not by any span-level average:
            #   all_em    -- every located span IS the generated span
            #   located   -- all spans survive, but >=1 needed normalization/approx
            #   partial   -- some spans located, >=1 dropped as not-found
            #   all_nf    -- nothing located; the sample ends up with no spans
            n_nf = tiers.count("nf")
            bucket = ("all_nf" if n_nf == len(tiers) else
                      "partial" if n_nf else
                      "all_em" if tiers.count("em") == len(tiers) else "located")
            for a in accs:
                a["n"] += 1
                a[bucket] += 1
                for tier in ("em", "norm", "approx"):
                    a[tier] += tiers.count(tier) / len(tiers)
    rows = []
    for name, a in groups.items():
        n_spanned = a["n"] - a["n_nospan"]
        rows.append({
            "group": name,
            "em": a["em"] / n_spanned if n_spanned else None,
            "norm": a["norm"] / n_spanned if n_spanned else None,
            "approx": a["approx"] / n_spanned if n_spanned else None,
            "n": a["n"],
            "n_nospan": a["n_nospan"],
            # Sample counts, and their share of the samples that had spans. The
            # four are mutually exclusive and sum to n_spanned.
            "samples": {k: a[k] for k in SAMPLE_BUCKETS},
            "sample_rates": ({k: a[k] / n_spanned for k in SAMPLE_BUCKETS}
                             if n_spanned else None),
        })
    return rows


#: Sample-level outcome buckets, mutually exclusive; see compute_stats.
SAMPLE_BUCKETS = ("all_em", "located", "partial", "all_nf")


def _new_acc():
    return dict({"n": 0, "n_nospan": 0, "em": 0.0, "norm": 0.0, "approx": 0.0},
                **{k: 0 for k in SAMPLE_BUCKETS})


# --------------------------- CLI ---------------------------

VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VIZ_DIR)
CACHE_DIR = os.path.join(VIZ_DIR, ".cache")
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data/extracted_relevancy")


def _cache_path(path, data_dir):
    """Same sidecar naming as the viewer server, so both share one cache."""
    rel = os.path.relpath(path, data_dir)
    if rel.startswith(".."):
        rel = os.path.basename(path)
    return os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9._-]", "_", rel) + ".stats.json")


def cached_stats(path, data_dir, use_cache=True):
    """Stats rows for ``path``: from the sidecar cache when it matches the
    file's (mtime, size), else computed (and cached)."""
    st = os.stat(path)
    sig = [st.st_mtime, st.st_size]
    cpath = _cache_path(path, data_dir)
    if use_cache:
        try:
            with open(cpath, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("sig") == sig and data.get("v") == STATS_VERSION:
                return data["stats"], True
        except (OSError, json.JSONDecodeError):
            pass
    t0 = time.time()
    print(f"computing {os.path.basename(path)} ...", file=sys.stderr)
    rows = compute_stats(path)
    print(f"computed in {time.time() - t0:.1f}s", file=sys.stderr)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = cpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "v": STATS_VERSION, "stats": rows}, f)
        os.replace(tmp, cpath)
    except OSError as e:
        print(f"warning: could not write cache {cpath}: {e}", file=sys.stderr)
    return rows, False


def print_table(rows):
    pc = lambda v: "     —" if v is None else f"{v * 100:5.1f}%"
    header = (f"{'group':16s} {'EM':>6s} {'norm':>6s} {'approx':>6s} "
              f"{'+norm':>6s} {'+approx':>7s} {'#samples':>9s} {'#no-spans':>10s}")
    print(header)
    print("-" * len(header))
    for g in rows:
        em, norm, approx = g["em"], g["norm"], g["approx"]
        cum_norm = None if em is None else em + norm
        cum_approx = None if cum_norm is None else cum_norm + approx
        print(f"{g['group']:16s} {pc(em)} {pc(norm)} {pc(approx)} "
              f"{pc(cum_norm)} {pc(cum_approx):>7s} {g['n']:9,} {g['n_nospan']:10,}")

    # Sample-level outcome: what a downstream metric actually gets to score,
    # since locating drops the not-found spans.
    if not any(g.get("sample_rates") for g in rows):
        return
    print()
    header = (f"{'group':16s} {'as generated':>13s} {'located':>13s} "
              f"{'partly lost':>13s} {'nothing found':>14s} {'no spans':>10s}")
    print(header)
    print("-" * len(header))
    for g in rows:
        r, c = g.get("sample_rates"), g.get("samples") or {}
        if not r:
            continue
        cell = lambda k: f"{r[k] * 100:5.1f}% {c[k]:6,}"
        print(f"{g['group']:16s} {cell('all_em'):>13s} {cell('located'):>13s} "
              f"{cell('partial'):>13s} {cell('all_nf'):>14s} {g['n_nospan']:10,}")


def main():
    parser = argparse.ArgumentParser(
        description="Span match-rate statistics (EM / normalized / approximate, "
                    "exclusive and cumulative), macro-aggregated per sample.")
    parser.add_argument("files", nargs="*",
                        help="JSONL output file(s); default: every *.jsonl under --data-dir")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Scanned when no files are given; also anchors cache names.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute even when a fresh cache entry exists.")
    args = parser.parse_args()

    files = args.files
    if not files:
        files = sorted(
            os.path.join(root, fn)
            for root, _dirs, fns in os.walk(args.data_dir)
            for fn in fns if fn.endswith(".jsonl")
        )
    if not files:
        sys.exit(f"no .jsonl files found under {args.data_dir}")

    for i, path in enumerate(files):
        rows, from_cache = cached_stats(path, args.data_dir, use_cache=not args.no_cache)
        if i:
            print()
        print(f"== {os.path.basename(path)}{' (cached)' if from_cache else ''} ==")
        print_table(rows)


if __name__ == "__main__":
    main()
