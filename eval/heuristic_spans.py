"""Materialize heuristically-located spans for an extraction run.

Unconstrained decoding produces spans that are often NOT verbatim substrings of
the document -- paraphrased, re-cased, re-quoted, or outright invented. Feeding
those straight into the retrieval metrics is unsafe: eval/retrieval_score.py
builds a spans-only pseudo-document, so any non-verbatim text would be embedded
as if it were part of the document, and a model that paraphrases fluently scores
well on spans that do not exist. (Constrained runs are unaffected: their spans
are verbatim by construction.)

This pass resolves each selected span against its passage with the SAME three
tiers the viewer and visualize/stats.py report -- em / norm / approx -- via the
shared custom_utils/span_match.locate_span, and writes the passage substring
each match actually lands on. Those substrings are verbatim by construction, so
downstream metrics only ever see real document text.

Spans that resolve to ``nf`` are DROPPED, not repaired: they are hallucinations
with respect to the document. A sample whose every span is ``nf`` ends up with an
empty span list, which the retrieval metrics score as zero -- the correct
penalty, and the reason this pass must run before scoring an unconstrained arm.

Output: one sidecar JSONL per run, keyed on (subset, qid, doc_id) -- the join key
the rest of eval already uses. A sidecar rather than a new column in the
extraction file because passages dominate those files (~3.6 GB each), and the
spans alone are a few MB.

    {"subset", "qid", "doc_id",
     "heuristic_spans": [str, ...],   # verbatim passage substrings, in order
     "spans_out": [[start, end], ...],# their char offsets in the passage
     "tiers": ["em"|"norm"|"approx"|"nf", ...],  # per INPUT span, aligned
     "n_spans", "n_found"}

The per-tier rates printed at the end use the same macro aggregation as
visualize/stats.py (mean over spans within a sample, then mean over samples,
zero-span samples excluded) so they are directly comparable to that table.

Run in the pylate env (any env with numpy works):

    source eval/eval_env.sh
    $PYTHON eval/heuristic_spans.py --all-unconstrained
    $PYTHON eval/heuristic_spans.py data/extracted_relevancy/long-embed-xml/*.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from custom_utils.span_match import locate_span, normalize_with_map  # noqa: E402
from eval.answer_containment import parse_spans_field, run_key  # noqa: E402

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data/extracted_relevancy")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data/eval/heuristic_spans")

#: Only these arms need the pass; a constrained arm's spans are already verbatim.
UNCONSTRAINED_MODES = ("long-embed-xml", "long-embed-json")

TIERS = ("em", "norm", "approx", "nf")


def sidecar_path(system, out_dir=DEFAULT_OUT_DIR):
    """``<mode>/<model>`` -> sidecar file (mirrors retrieval_score's encoding)."""
    return os.path.join(out_dir, system.replace("/", "__") + ".jsonl")


def locate_record(rec):
    """-> (spans, offsets, tiers) for one extraction record.

    ``spans`` are passage substrings for every input span that resolved; the
    passage is normalized at most once per record, and not at all when every
    span is already verbatim (the common case for a constrained run).
    """
    passage = rec.get("passage") or ""
    spans_in = parse_spans_field(rec.get("selected_spans"))
    norm = None                      # lazily built: it costs O(len(passage))

    spans, offsets, tiers = [], [], []
    for span in spans_in:
        if isinstance(span, str) and span and span in passage:
            tier, start = "em", passage.find(span)
            end = start + len(span)
        else:
            if norm is None:
                norm = normalize_with_map(passage)
            tier, start, end = locate_span(span, passage, norm[0], norm[1])
        tiers.append(tier)
        if tier != "nf" and end > start:
            located = passage[start:end]
            assert located in passage           # verbatim by construction
            spans.append(located)
            offsets.append([start, end])
    return spans, offsets, tiers


def process(path, out_dir, force=False, limit=None):
    system = run_key(path)
    out = sidecar_path(system, out_dir)
    if os.path.exists(out) and not force:
        print(f"{system}: sidecar present, skipping (use --force)")
        return out, None

    os.makedirs(out_dir, exist_ok=True)
    tmp = out + ".tmp"
    # Macro accumulators, matching visualize/stats.py: per-sample tier rates
    # first, then the mean over samples; samples with no spans are excluded.
    per_sample = {t: 0.0 for t in TIERS}
    n_samples = n_nospan = n_spans_tot = n_found_tot = 0
    t0 = time.time()

    with open(path, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if limit and i >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            spans, offsets, tiers = locate_record(rec)
            fout.write(json.dumps({
                "subset": rec.get("subset"), "qid": rec.get("qid"),
                "doc_id": rec.get("doc_id"),
                "heuristic_spans": spans, "spans_out": offsets, "tiers": tiers,
                "n_spans": len(tiers), "n_found": len(spans),
            }, ensure_ascii=False) + "\n")

            n_spans_tot += len(tiers)
            n_found_tot += len(spans)
            if tiers:
                n_samples += 1
                for t in TIERS:
                    per_sample[t] += tiers.count(t) / len(tiers)
            else:
                n_nospan += 1
            if (i + 1) % 2000 == 0:
                print(f"  {system}: {i + 1} records, {time.time() - t0:.0f}s",
                      file=sys.stderr, flush=True)

    os.replace(tmp, out)
    rates = {t: (per_sample[t] / n_samples if n_samples else 0.0) for t in TIERS}
    stats = {"system": system, "n_samples": n_samples, "n_nospan": n_nospan,
             "n_spans": n_spans_tot, "n_found": n_found_tot, "rates": rates,
             "seconds": round(time.time() - t0, 1)}
    print(f"{system}: {n_spans_tot} spans, {n_found_tot} located "
          f"({100.0 * n_found_tot / max(1, n_spans_tot):.1f}%) -> {out}")
    return out, stats


def print_table(all_stats):
    print("\nMacro per-sample tier rates (same aggregation as visualize/stats.py;"
          "\nzero-span samples excluded and counted in nospan)\n")
    head = f"{'system':52s} " + " ".join(f"{t:>7s}" for t in TIERS) + \
           f" {'found%':>7s} {'n':>6s} {'nospan':>7s}"
    print(head)
    print("-" * len(head))
    for s in all_stats:
        if s is None:
            continue
        r = s["rates"]
        found = 100.0 * s["n_found"] / max(1, s["n_spans"])
        print(f"{s['system']:52s} " +
              " ".join(f"{r[t] * 100:6.1f}%" for t in TIERS) +
              f" {found:6.1f}% {s['n_samples']:6d} {s['n_nospan']:7d}")


def default_files(data_dir):
    """Widest generation range per (unconstrained mode, model), skipping the
    partial/broken leftovers default_files in answer_containment also skips."""
    best = {}
    for mode in UNCONSTRAINED_MODES:
        for path in sorted(glob.glob(os.path.join(data_dir, mode, "*.jsonl"))):
            name = os.path.basename(path)
            if "broken" in name or "partial" in name:
                continue
            import re
            m = re.search(r"_from(\d+)-to(\d+)\.jsonl$", name)
            if not m:
                continue
            key, width = run_key(path), int(m.group(2)) - int(m.group(1))
            if key not in best or width > best[key][0]:
                best[key] = (width, path)
    return [p for _w, p in best.values()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="extraction JSONL(s)")
    ap.add_argument("--all-unconstrained", action="store_true",
                    help="process every unconstrained run under --data-dir")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N records (sanity runs)")
    ap.add_argument("--force", action="store_true",
                    help="recompute sidecars that already exist")
    args = ap.parse_args()

    files = args.files or (default_files(args.data_dir)
                           if args.all_unconstrained else [])
    if not files:
        sys.exit("nothing to do: pass files or --all-unconstrained")

    all_stats = []
    for path in files:
        _out, stats = process(path, args.out_dir, force=args.force,
                              limit=args.limit)
        all_stats.append(stats)
    if any(s for s in all_stats):
        print_table(all_stats)


if __name__ == "__main__":
    main()
