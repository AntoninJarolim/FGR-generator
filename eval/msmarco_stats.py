"""Dataset-paper statistics for the MS MARCO span supervision (gemma-4-31B-it,
XML, reasoning), for the constrained and unconstrained decoding arms.

Per arm and split: split sizes, query/passage/span lengths, spans per pair,
span coverage of the passage, relative span start position and empty rate.
Where both arms exist for a split, also an overlaid comparison figure with
per-bin deltas and paired agreement statistics on the shared pairs.

Writes figs/msmarco_supervision/{arm}_{split}_span_stats.png,
{split}_length_stats.png and comparison_{split}_span_stats.png.

Run from the repo root:  python eval/msmarco_stats.py
"""
import collections
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ARMS = {
    "constrained": "data/extracted_spans/ms-marco-xml-constrained-reasoning/google~gemma-4-31B-it_",
    "unconstrained": "data/extracted_spans/ms-marco-xml-reasoning/google~gemma-4-31B-it_",
}
SPLITS = ("train", "validation")
OUT = "figs/msmarco_supervision"
COLORS = {"constrained": "#4C72B0", "unconstrained": "#DD8452"}
W = lambda s: len(s.split())  # noqa: E731


def rel_pos(i, p, s):
    """Relative start: offset / (passage_len - span_len); 0 if the span is the passage."""
    return i / (len(p) - len(s)) if len(p) > len(s) else 0.0


def load(path):
    """One pass over an extraction file. Returns summary dict + per-pair records."""
    qlen = []; plen = []; nsp = []; splen = []; cov = []; pos = []
    empty = 0; n = 0; notsub = 0
    qids = set(); pids = set(); qdup = collections.Counter()
    pairs = []  # (key, passage_len, [(start,end)|None ...]) for paired comparison
    for line in open(path):
        d = json.loads(line); n += 1
        q = d["query"]; p = d["passage"]; sp = d.get("selected_spans") or []
        qids.add(d["qid"]); pids.add(d["doc_id"]); qdup[d["qid"]] += 1
        qlen.append(W(q)); plen.append(W(p)); nsp.append(len(sp))
        locs = []
        if not sp:
            empty += 1
        else:
            tot = 0
            for s in sp:
                splen.append(W(s)); i = p.find(s); tot += len(s)
                if i < 0:
                    notsub += 1; locs.append(None)
                else:
                    pos.append(rel_pos(i, p, s)); locs.append((i, i + len(s)))
            cov.append(min(1.0, tot / len(p)))
        pairs.append(((d["qid"], d["doc_id"]), len(p), locs))
    pct = lambda a: (np.mean(a), np.median(a), np.percentile(a, 90), np.max(a))  # noqa: E731
    summary = dict(
        n=n, uq=len(qids), up=len(pids), multi=sum(1 for v in qdup.values() if v > 1),
        empty=empty, notsub=notsub, total_spans=len(splen),
        qlen=pct(qlen), plen=pct(plen), splen=pct(splen), cov=pct(cov),
        nsp=collections.Counter(nsp), nsp_nonempty_mean=np.mean([x for x in nsp if x]),
        single_frac=np.mean([x == 1 for x in nsp if x]),
    )
    arrays = dict(cov=cov, pos=pos, splen=splen, nsp=nsp, qlen=qlen, plen=plen)
    return summary, arrays, pairs


def print_summary(label, r):
    print(f"\n== {label}: pairs={r['n']:,} unique queries={r['uq']:,} unique passages={r['up']:,} "
          f"queries with >1 passage={r['multi']:,}")
    print(f"empty={r['empty']:,} ({100 * r['empty'] / r['n']:.1f}%)  spans not verbatim substrings="
          f"{r['notsub']:,} ({100 * r['notsub'] / max(1, r['total_spans']):.2f}%)  total spans={r['total_spans']:,}")
    for k in ("qlen", "plen", "splen", "cov"):
        m, md, p90, mx = r[k]
        f = (lambda x: f"{x:.3f}") if k == "cov" else (lambda x: f"{x:.1f}")
        print(f"{k:6s} mean={f(m)} median={f(md)} p90={f(p90)} max={f(mx)}")
    c = r["nsp"]; tot = sum(c.values())
    print("spans/pair dist:", {k: f"{100 * v / tot:.1f}%" for k, v in sorted(c.items())[:7]},
          f"| mean (non-empty)={r['nsp_nonempty_mean']:.2f} single-span={100 * r['single_frac']:.1f}%")


def add_pct_axis(a, total):
    """Right-hand axis in percent of `total`, locked to the left (count) axis."""
    r = a.twinx(); lo, hi = a.get_ylim(); r.set_ylim(lo / total * 100, hi / total * 100)
    r.yaxis.set_major_formatter(PercentFormatter(decimals=0)); r.set_ylabel("% of " + a.get_ylabel())
    return r


# Panel definitions shared by the per-arm and comparison figures:
# (key, title, xlabel, bins, range, population label, transform)
PANELS = [
    ("cov", "Span coverage of passage (chars)", "fraction of passage covered", 25, (0, 1), "pairs", lambda a: a),
    ("splen", "Span length (words)", "words (clipped at 60)", 30, (0, 60), "spans", lambda a: np.clip(a, 0, 60)),
    ("nsp", "Spans per pair", "#spans (0 = empty)", 7, (-0.5, 6.5), "pairs", lambda a: np.clip(a, 0, 6)),
    ("pos", "Relative span start position", "0 = passage start, 1 = end", 20, (0, 1), "spans", lambda a: a),
]


def arm_figure(arm, split, n, arrays):
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for a, (key, title, xlabel, bins, rng, pop, tf) in zip(ax, PANELS):
        vals = tf(np.asarray(arrays[key]))
        a.hist(vals, bins=bins, range=rng, color=COLORS[arm])
        a.set_title(title); a.set_xlabel(xlabel); a.set_ylabel(pop)
        if key == "nsp":
            a.set_xticks(range(7)); a.set_xticklabels([str(i) for i in range(6)] + ["6+"])
        add_pct_axis(a, len(vals))
    fig.suptitle(f"MS MARCO {split}, gemma-4-31B-it (XML, {arm}, reasoning), n={n:,} pairs", y=1.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/{arm}_{split}_span_stats.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def length_figure(split, n, arrays):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(arrays["qlen"], bins=range(0, 30), color="#4C72B0"); ax[0].set_title("Query length (words)")
    ax[0].set_xlabel("words")
    ax[1].hist(np.clip(arrays["plen"], 0, 200), bins=40, color="#55A868"); ax[1].set_title("Passage length (words)")
    ax[1].set_xlabel("words (clipped at 200)")
    for a in ax:
        a.set_ylabel("pairs"); add_pct_axis(a, n)
    fig.suptitle(f"MS MARCO {split}, n={n:,} pairs", y=1.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/{split}_length_stats.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def comparison_figure(split, arrays_by_arm):
    """Top row: both arms as percent histograms. Bottom row: unconstrained minus constrained, pp."""
    arms = list(arrays_by_arm)
    fig, ax = plt.subplots(2, 4, figsize=(18, 6.6), gridspec_kw={"height_ratios": [2, 1]})
    for col, (key, title, xlabel, bins, rng, pop, tf) in enumerate(PANELS):
        top, bot = ax[0, col], ax[1, col]
        hists = {}
        for arm in arms:
            vals = tf(np.asarray(arrays_by_arm[arm][key]))
            h, edges = np.histogram(vals, bins=bins, range=rng)
            hists[arm] = 100 * h / len(vals)
            top.stairs(hists[arm], edges, fill=True, alpha=0.45, color=COLORS[arm], label=f"{arm} (n={len(vals):,})")
        top.set_title(title); top.set_ylabel(f"% of {pop}"); top.legend(fontsize=8)
        delta = hists[arms[1]] - hists[arms[0]]
        centers = (edges[:-1] + edges[1:]) / 2
        bot.bar(centers, delta, width=np.diff(edges) * 0.9, color=np.where(delta >= 0, "#DD8452", "#4C72B0"))
        bot.axhline(0, color="k", lw=0.8)
        bot.set_xlabel(xlabel); bot.set_ylabel(f"{arms[1]} − {arms[0]} (pp)")
        bot.set_title(f"max |Δ| = {np.abs(delta).max():.2f} pp", fontsize=9)
        if key == "nsp":
            for a in (top, bot):
                a.set_xticks(range(7)); a.set_xticklabels([str(i) for i in range(6)] + ["6+"])
    fig.suptitle(f"MS MARCO {split}: constrained vs unconstrained decoding, gemma-4-31B-it (XML, reasoning)", y=1.01)
    fig.tight_layout(); fig.savefig(f"{OUT}/comparison_{split}_span_stats.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def paired_stats(split, pairs_by_arm):
    """Agreement on the shared (qid, doc_id) pairs: empties, exact span-set match, char-level Jaccard."""
    a_name, b_name = list(pairs_by_arm)
    A = {k: (L, locs) for k, L, locs in pairs_by_arm[a_name]}
    B = {k: (L, locs) for k, L, locs in pairs_by_arm[b_name]}
    shared = A.keys() & B.keys()
    both_empty = only_a = only_b = both = exact = 0
    jac = []; cov_delta = []; nsp_delta = []
    for k in shared:
        La, la = A[k]; Lb, lb = B[k]
        ea, eb = (len(la) == 0), (len(lb) == 0)
        nsp_delta.append(len(lb) - len(la))
        if ea and eb: both_empty += 1; continue
        if ea: only_a += 1
        elif eb: only_b += 1
        else: both += 1
        ma = np.zeros(La, bool); mb = np.zeros(Lb, bool)
        for loc in la:
            if loc: ma[loc[0]:loc[1]] = True
        for loc in lb:
            if loc: mb[loc[0]:loc[1]] = True
        if len(ma) != len(mb):  # passages identical by construction; guard anyway
            continue
        inter = (ma & mb).sum(); union = (ma | mb).sum()
        if union: jac.append(inter / union)
        if ma.sum() == mb.sum() == inter: exact += 1
        cov_delta.append(mb.mean() - ma.mean())
    n = len(shared); jac = np.asarray(jac)
    print(f"\n== paired {split} ({a_name} vs {b_name}) on {n:,} shared pairs")
    print(f"both empty={100 * both_empty / n:.2f}%  only {a_name} empty={100 * only_a / n:.2f}%  "
          f"only {b_name} empty={100 * only_b / n:.2f}%  both non-empty={100 * both / n:.2f}%")
    print(f"among both non-empty: identical char coverage={100 * exact / max(1, both):.1f}%  "
          f"char Jaccard mean={jac.mean():.3f} median={np.median(jac):.3f}  "
          f"Jaccard>=0.5: {100 * (jac >= 0.5).mean():.1f}%  Jaccard==0: {100 * (jac == 0).mean():.1f}%")
    print(f"mean per-pair delta ({b_name} − {a_name}): coverage={100 * np.mean(cov_delta):+.2f} pp  "
          f"#spans={np.mean(nsp_delta):+.3f}")
    return jac


def main():
    os.makedirs(OUT, exist_ok=True)
    for split in SPLITS:
        arrays_by_arm = {}; pairs_by_arm = {}
        for arm, prefix in ARMS.items():
            path = prefix + split + ".jsonl"
            if not os.path.exists(path):
                print(f"\n(skip: {path} missing)"); continue
            summary, arrays, pairs = load(path)
            print_summary(f"{arm} {split}", summary)
            arm_figure(arm, split, summary["n"], arrays)
            arrays_by_arm[arm] = arrays; pairs_by_arm[arm] = pairs
            if arm == next(iter(ARMS)):
                length_figure(split, summary["n"], arrays)
        if len(arrays_by_arm) == 2:
            comparison_figure(split, arrays_by_arm)
            jac = paired_stats(split, pairs_by_arm)
            fig, a = plt.subplots(figsize=(5, 3.2))
            a.hist(jac, bins=20, range=(0, 1), color="#8172B2"); a.set_xlabel("char-level Jaccard of covered spans")
            a.set_ylabel("pairs (both non-empty)"); add_pct_axis(a, len(jac))
            a.set_title(f"MS MARCO {split}: per-pair agreement, constrained vs unconstrained", fontsize=9)
            fig.tight_layout(); fig.savefig(f"{OUT}/comparison_{split}_jaccard.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
    print("\nfigs written to", OUT)


if __name__ == "__main__":
    main()
