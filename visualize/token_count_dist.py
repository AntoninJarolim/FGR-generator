"""Token-count distribution figures for the LongEmbed and MS MARCO document
pools, styled after Figure 5 ("LoCoV1 Document Token Count Distributions") of
Zhu et al. 2024 (arXiv:2402.07440): horizontal violins with interior quartile
lines, one panel per subset, cycling warm/cool fills.

Tokens are counted with a plain whitespace tokenizer (``text.split()``) so the
numbers are model-agnostic and reproducible.

Run inside the fgr-generator conda env:
    python visualize/token_count_dist.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from custom_utils.longembed import NON_SYNTHETIC_SUBSETS, load_longembed

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")

# Figure-5 palette: red / green / amber, cycled across panels.
PANEL_COLORS = ["#D64541", "#2E8B3E", "#E8A317"]

# Pretty panel titles matching the paper's CamelCase style.
LONGEMBED_TITLES = {
    "narrativeqa": "NarrativeQA",
    "summ_screen_fd": "SummScreenFD",
    "qmsum": "QMSum",
    "2wikimqa": "2WikiMQA",
}

MSMARCO_DATASET_ID = "bclavie/msmarco-2m-triplets"
MSMARCO_SAMPLE = 200_000  # cap rows tokenized; the pool is 2M positives


def whitespace_len(text):
    """Whitespace token count."""
    return len(text.split())


def style_violin(ax, tokens, color, title):
    """Draw one horizontal violin + quartile lines in the paper's style."""
    tokens = np.asarray(tokens, dtype=float)
    parts = ax.violinplot(
        tokens,
        vert=False,
        showextrema=False,
        widths=0.9,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(1.0)

    # Interior quartile lines (Q1/median/Q3), as in Figure 5.
    q1, med, q3 = np.percentile(tokens, [25, 50, 75])
    for x in (q1, med, q3):
        ax.vlines(x, 0.62, 1.38, color="black", lw=0.8, alpha=0.55)

    ax.set_title(title, fontsize=10, pad=4)
    ax.set_yticks([])
    ax.set_ylim(0.5, 1.5)
    ax.set_xlim(left=max(0, tokens.min() - np.ptp(tokens) * 0.02))
    ax.tick_params(axis="x", labelsize=7.5, length=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#666666")


def grid_figure(panels, ncols, suptitle, out_path,
                xlabel="Document token count (whitespace)"):
    """panels: list of (title, tokens, color). Lay out in a grid of violins."""
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.4 * ncols, 1.55 * nrows),
        squeeze=False,
    )
    for idx, (title, tokens, color) in enumerate(panels):
        r, c = divmod(idx, ncols)
        style_violin(axes[r][c], tokens, color, title)
    # Blank any unused cells.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.supxlabel(xlabel, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def build_longembed_panels():
    records = load_longembed(NON_SYNTHETIC_SUBSETS)
    panels = []
    for i, subset in enumerate(NON_SYNTHETIC_SUBSETS):
        # Dedup documents by doc_id -- a document distribution, not per-pair.
        seen = {}
        for rec in records:
            if rec["subset"] == subset:
                seen[rec["doc_id"]] = rec["passage"]
        tokens = [whitespace_len(t) for t in seen.values()]
        title = LONGEMBED_TITLES.get(subset, subset)
        print(f"  {title}: {len(tokens)} docs, "
              f"median={int(np.median(tokens))} tokens")
        panels.append((title, tokens, PANEL_COLORS[i % len(PANEL_COLORS)]))
    return panels


def build_msmarco_panels():
    from datasets import load_dataset

    ds = load_dataset(MSMARCO_DATASET_ID, split="train")
    n = min(MSMARCO_SAMPLE, len(ds))
    positives = ds.select(range(n))["positive"]
    queries = ds.select(range(n))["query"]
    doc_tokens = [whitespace_len(t) for t in positives]
    q_tokens = [whitespace_len(t) for t in queries]
    print(f"  MS MARCO passages: {n} sampled, "
          f"median={int(np.median(doc_tokens))} tokens")
    print(f"  MS MARCO queries: median={int(np.median(q_tokens))} tokens")
    return [
        ("Passage (positive)", doc_tokens, PANEL_COLORS[0]),
        ("Query", q_tokens, PANEL_COLORS[1]),
    ]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("LongEmbed:")
    le_panels = build_longembed_panels()
    grid_figure(
        le_panels, ncols=2,
        suptitle="LongEmbed Document Token Count Distributions",
        out_path=os.path.join(OUT_DIR, "longembed_token_dist.png"),
    )

    print("MS MARCO:")
    mm_panels = build_msmarco_panels()
    grid_figure(
        mm_panels, ncols=2,
        suptitle="MS MARCO Token Count Distributions",
        out_path=os.path.join(OUT_DIR, "msmarco_token_dist.png"),
        xlabel="Token count (whitespace)",
    )


if __name__ == "__main__":
    main()
