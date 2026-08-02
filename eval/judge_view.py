"""Rendered span views for the pairwise preference judge.

A "view" is what a judge (API or local) actually reads for one system on one
sample: that system's selected spans, each shown inside its surrounding document
text so the excerpt is readable, with the selection itself marked. Both judges
consume the same views, so their verdicts are comparable.

Why per-system views rather than one shared document window: the two systems
being compared chose different parts of the document, and plausibility is
*defined* on reading the selection alone ("a person reading only the span would
be convinced the document is relevant"). Showing each system's own excerpts is
therefore the correct stimulus, not a cost compromise. The cost is that the same
document region can appear in both views when both systems selected near it.

Construction, per system and sample:

  * spans come from the eval/heuristic_spans.py sidecar -- the located, verbatim
    ones the retrieval metrics use, never the raw model output (see that module
    for why: unconstrained arms emit spans that are not substrings of their own
    document, and those must not reach a judge as if they were document text);
  * exact-duplicate offsets are dropped, and spans are ordered by position in the
    document rather than by model output order;
  * each span is padded by CONTEXT chars each side and *overlapping windows are
    merged*, so neighbouring spans share one excerpt instead of repeating the
    text between them. A merged excerpt keeps every span's markers;
  * an excerpt edge that is not a document boundary gets an ellipsis, so the
    judge can see the text is cut;
  * total rendered text is capped (MAX_VIEW_CHARS). Whole excerpts are kept in
    document order and a truncation note states how much was dropped -- the cap
    is set high enough that it fires on a few percent of samples.

The baseline systems have no extraction file; their spans are synthesized on the
fly by eval/retrieval_score.baseline_spans from the same span budgets the
retrieval metric and the viewer use, so all three agree on what the baseline is.

Run directly to build the view cache (one streaming pass per system; the
extraction files are ~3.6 GB each, the cache is a few MB):

    source eval/eval_env.sh
    $PYTHON eval/judge_view.py --systems-file eval/judge_systems.json \
        --items data/eval/judge_items.json
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.answer_containment import parse_spans_field, run_key  # noqa: E402
from eval.heuristic_spans import sidecar_path  # noqa: E402

#: Context shown each side of a span. Matches the viewer's default ctxSize, so a
#: judge and a human annotator looking at the same sample see the same thing.
CONTEXT = 300

#: Cap on rendered characters per system per sample. Measured on the real span
#: distributions: at 12k this truncates a few percent of samples, where at 6k it
#: truncated 10-20% -- systematic truncation would bias the minimality judgement
#: toward whichever system was hiding the most text.
MAX_VIEW_CHARS = 12000

SPAN_OPEN, SPAN_CLOSE = "[[SPAN]]", "[[/SPAN]]"
ELLIPSIS = "…"
EMPTY_VIEW = "(no spans selected)"

DEFAULT_VIEW_DIR = os.path.join(REPO_ROOT, "data/eval/judge_views")
DEFAULT_BUDGETS = os.path.join(REPO_ROOT, "data/eval/span_budgets.json")


def item_key(rec):
    return (rec["subset"], rec["qid"], rec["doc_id"])


# --------------------------- view construction ---------------------------

def merge_windows(spans, passage_len, context=CONTEXT):
    """Span offsets -> excerpt windows, each carrying the spans inside it.

    -> [(win_start, win_end, [(span_start, span_end), ...]), ...] in document
    order. Overlapping windows are merged so adjacent spans share one excerpt.
    """
    uniq = sorted({(int(s), int(e)) for s, e in spans})
    out = []
    for s, e in uniq:
        ws, we = max(0, s - context), min(passage_len, e + context)
        if out and ws <= out[-1][1]:
            out[-1][1] = max(out[-1][1], we)
            out[-1][2].append((s, e))
        else:
            out.append([ws, we, [(s, e)]])
    return [(ws, we, sp) for ws, we, sp in out]


def render_excerpt(passage, win_start, win_end, spans):
    """One excerpt: the window text with every span inside it marked."""
    parts, cur = [], win_start
    for s, e in spans:
        s, e = max(s, win_start), min(e, win_end)
        if s < cur:            # overlapping spans: keep text, drop the marker
            continue
        parts += [passage[cur:s], SPAN_OPEN, passage[s:e], SPAN_CLOSE]
        cur = e
    parts.append(passage[cur:win_end])
    body = "".join(parts)
    if win_start > 0:
        body = ELLIPSIS + body
    if win_end < len(passage):
        body = body + ELLIPSIS
    return body


def build_view(passage, spans, context=CONTEXT, max_chars=MAX_VIEW_CHARS):
    """-> (view_text, stats). ``spans`` are (start, end) offsets into passage.

    Deliberately carries no span/character counts into the text: minimality is
    not just a number, and with the cap set high nearly everything is shown in
    full anyway. When the cap does fire, the note says how much was dropped.
    """
    n_raw = len(spans)
    wins = merge_windows(spans, len(passage), context)
    stats = {"n_spans": n_raw,
             "n_spans_unique": len({(int(s), int(e)) for s, e in spans}),
             "selected_chars": sum(e - s for s, e in {(int(s), int(e))
                                                     for s, e in spans}),
             "n_excerpts": len(wins), "truncated": False}
    if not wins:
        return EMPTY_VIEW, stats

    blocks, used, shown = [], 0, 0
    for ws, we, sp in wins:
        body = render_excerpt(passage, ws, we, sp)
        block = f"Excerpt {shown + 1}:\n{body}"
        if shown and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
        shown += 1
    if shown < len(wins):
        dropped = sum(we - ws for ws, we, _ in wins[shown:])
        blocks.append(f"[{ELLIPSIS} truncated, {dropped} more characters not shown]")
        stats["truncated"] = True
    stats["n_excerpts_shown"] = shown
    return "\n\n".join(blocks), stats


# --------------------------- span sources ---------------------------

def load_sidecar_spans(system, keys, out_dir=None):
    """(subset, qid, doc_id) -> located span offsets, for the sampled keys only."""
    path = sidecar_path(system) if out_dir is None else \
        os.path.join(out_dir, system.replace("/", "__") + ".jsonl")
    if not os.path.exists(path):
        sys.exit(f"no sidecar for {system} at {path}\n"
                 f"run: $PYTHON eval/heuristic_spans.py <extraction file>")
    want, got = set(keys), {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = item_key(r)
            if k in want:
                got[k] = [tuple(o) for o in r.get("spans_out", [])]
    return got


def baseline_offsets(kind, rec, budgets):
    """Synthesized baseline spans, located back to offsets.

    Uses eval/retrieval_score.baseline_spans -- the same function the retrieval
    metric and the viewer's baseline column call, so a baseline means one thing
    everywhere.
    """
    from eval.retrieval_score import baseline_spans
    passage = rec.get("passage") or ""
    key = f"{rec['subset']}|{rec['qid']}|{rec['doc_id']}"
    b = budgets.get(key) or {"budget": 0, "n_spans": 1}
    offs, cursor = [], 0
    for span in baseline_spans(kind, rec.get("query") or "", passage,
                               int(b["budget"]), int(round(b["n_spans"]))):
        if not span:
            continue
        pos = passage.find(span, cursor)          # in-order search keeps
        if pos == -1:                              # repeated sentences distinct
            pos = passage.find(span)
        if pos == -1:
            continue
        offs.append((pos, pos + len(span)))
        cursor = pos + len(span)
    return offs


# --------------------------- cache building ---------------------------

def build_cache(system, source, keys, spans_by_key, out_dir, budgets=None,
                baseline_kind=None, context=CONTEXT, max_chars=MAX_VIEW_CHARS):
    """Stream ``source`` once, rendering views for ``keys``. -> output path."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, system.replace("/", "__").replace(":", "--") + ".jsonl")
    tmp, want, n = out + ".tmp", set(keys), 0
    with open(source, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = item_key(rec)
            if k not in want:
                continue
            passage = rec.get("passage") or ""
            offs = (baseline_offsets(baseline_kind, rec, budgets)
                    if baseline_kind else spans_by_key.get(k, []))
            # Offsets come from this system's own passage; a mismatch means the
            # sidecar and the extraction file have drifted apart.
            offs = [(s, e) for s, e in offs if 0 <= s < e <= len(passage)]
            view, stats = build_view(passage, offs, context, max_chars)
            fout.write(json.dumps({
                "subset": k[0], "qid": k[1], "doc_id": k[2],
                "query": rec.get("query") or "", "view": view, **stats,
            }, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"  {system}: {n}/{len(want)} views", file=sys.stderr, flush=True)
    os.replace(tmp, out)
    print(f"{system}: {n} views -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems-file", required=True,
                    help="JSON: [{label, path} ...]; path is an extraction JSONL "
                         "or 'baseline:<kind>'")
    ap.add_argument("--items", required=True,
                    help="JSON list of [subset, qid, doc_id] to render")
    ap.add_argument("--out-dir", default=DEFAULT_VIEW_DIR)
    ap.add_argument("--budgets", default=DEFAULT_BUDGETS)
    ap.add_argument("--baseline-source",
                    help="extraction JSONL to stream passages from for baselines "
                         "(defaults to the first non-baseline system)")
    ap.add_argument("--context", type=int, default=CONTEXT)
    ap.add_argument("--max-chars", type=int, default=MAX_VIEW_CHARS)
    args = ap.parse_args()

    systems = json.load(open(args.systems_file))
    keys = [tuple(k) for k in json.load(open(args.items))]
    print(f"rendering {len(keys)} items x {len(systems)} systems")

    budgets = {}
    if any(s["path"].startswith("baseline:") for s in systems):
        with open(args.budgets) as f:
            budgets = json.load(f)
    default_src = next((s["path"] for s in systems
                        if not s["path"].startswith("baseline:")), None)

    for s in systems:
        label, path = s["label"], s["path"]
        if path.startswith("baseline:"):
            build_cache(label, args.baseline_source or default_src, keys, {},
                        args.out_dir, budgets=budgets,
                        baseline_kind=path.split(":", 1)[1],
                        context=args.context, max_chars=args.max_chars)
        else:
            spans = load_sidecar_spans(run_key(path), keys)
            build_cache(label, path, keys, spans, args.out_dir,
                        context=args.context, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
