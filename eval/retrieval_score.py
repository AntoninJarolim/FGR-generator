"""Retrieval-model plausibility / comprehensiveness scores for extracted spans.

Two directions, per claude_md/evaluation-metrics-assessment.md §1.4, both scored
with the LongEmbed retrieval setup pylate-fgr uses (GTE-ModernColBERT-v1 MaxSim,
document_length=32768, query_length=32):

  * plausibility (additive)      -- replace the gold document by the concatenated
    spans (± context) and rank it against the untouched distractor corpus. High
    NDCG@10 = the spans alone keep the document retrievable for its query.
  * comprehensiveness (ablative) -- delete the spans (± context) from the gold
    document, re-embed it, and measure the score drop / rank change. Large drop =
    the spans carried the document's retrievability ("if removed, no longer
    relevant"). Expect small absolute drops on narrativeqa (spans cover ~0.1% of
    a 250K-char doc and the embedder window ends far earlier); compare systems
    against the matched random baseline, not against zero.

Because every corpus is tiny (197-355 docs) no ANN index is needed: the original
query x doc MaxSim score matrix is computed exactly once per subset (stage
"corpus") and cached; each system/baseline then only encodes its modified gold
documents and re-ranks them against the cached distractor scores (stage "score").
Stage "aggregate" folds all per-pair rows into data/eval/retrieval_score.json for
eval/summary_table.py.

Systems are extraction JSONLs (same files eval/answer_containment.py reads) or
budget-matched trivial baselines synthesized on the fly:

    baseline:random   -- random same-length windows of the document
    baseline:lead_k   -- the document prefix of the same total length
    baseline:lexical  -- query-term-overlap top sentences, same total length

(budgets per sample come from eval/build_baselines.py -> data/eval/span_budgets.json).

Run in the pylate env (NOT fgr-generator); eval/eval_env.sh activates it and
sets $PYTHON, so no path is hardcoded:

    source eval/eval_env.sh
    $PYTHON eval/retrieval_score.py --stage corpus
    $PYTHON eval/retrieval_score.py --stage score --system data/extracted_relevancy/long-embed-xml-constrained/google~gemma-4-12B-it_from0-to12612.jsonl
    $PYTHON eval/retrieval_score.py --stage score --system baseline:random
    $PYTHON eval/retrieval_score.py --stage aggregate

Every stage is resumable; interrupted "score" runs continue where they stopped.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np

# A single 32K-token document peaks at ~17 GiB during encoding (sdpa materializes
# the sliding-window mask); expandable segments make that fit a 24 GiB 3090.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODEL_ID = "lightonai/GTE-ModernColBERT-v1"
DOCUMENT_LENGTH = 32768   # pylate-fgr benchmarks.py: model card's LongEmbed setup
QUERY_LENGTH = 32
SUBSETS = ["narrativeqa", "qmsum", "summ_screen_fd", "2wikimqa"]
BASELINES = ("baseline:random", "baseline:lead_k", "baseline:lexical")
MODES = ("long-embed-json", "long-embed-xml", "long-embed-xml-constrained")

# Chars ~ 4 x tokens; if a modified doc is byte-identical to the original within
# this prefix, the embedder (32768 tokens) cannot see the change -> skip encode.
CHAR_WINDOW = DOCUMENT_LENGTH * 4
CHAR_WINDOW_SAFE = int(CHAR_WINDOW * 1.5)

CACHE_DIR = os.path.join(REPO_ROOT, "data/eval/retrieval_cache")
ROWS_DIR = os.path.join(REPO_ROOT, "data/eval/retrieval_rows")
DEFAULT_BUDGETS = os.path.join(REPO_ROOT, "data/eval/span_budgets.json")
DEFAULT_JSON_OUT = os.path.join(REPO_ROOT, "data/eval/retrieval_score.json")
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data/extracted_relevancy")

_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"[^.!?\n]*[.!?\n]+|[^.!?\n]+$")


# --------------------------- span / text preparation ---------------------------

def parse_spans_field(raw):
    """``selected_spans`` is a native JSON list in most runs, but newer
    Karolina-cluster runs serialize it as a Python-repr STRING, e.g.
    "['a', 'b']" (single quotes -- not valid JSON). Iterating that string
    directly would silently yield one-character "spans"."""
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


def dedupe_spans(raw):
    seen, out = set(), []
    for s in parse_spans_field(raw):
        if not isinstance(s, str):
            continue
        key = s.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def span_intervals(passage, spans, context):
    """Merged (start, end) intervals of the spans' first occurrences ± context.
    Spans not found verbatim (unconstrained outputs) are returned separately."""
    ivals, missing = [], []
    for s in spans:
        pos = passage.find(s)
        if pos == -1:
            missing.append(s)
            continue
        ivals.append((max(0, pos - context), min(len(passage), pos + len(s) + context)))
    ivals.sort()
    merged = []
    for start, end in ivals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged, missing


def ablate(passage, merged):
    if not merged:
        return passage
    parts, prev = [], 0
    for start, end in merged:
        parts.append(passage[prev:start])
        prev = end
    parts.append(passage[prev:])
    return "".join(parts)


def pseudo_doc(passage, merged):
    """Spans-only document: ONLY text that actually occurs in the passage.

    Spans that could not be located are deliberately dropped rather than
    appended verbatim. Appending them (as this did originally) embeds text that
    is absent from the document, so an unconstrained model that paraphrases
    fluently scores well on spans that do not exist -- and a fully hallucinated
    output still produced a non-empty pseudo-document. With them dropped, an
    extraction whose every span is unlocatable yields an empty pseudo-document,
    which scores zero: the correct penalty.

    Run eval/heuristic_spans.py first for unconstrained arms, so spans that
    differ only cosmetically (case, quotes, whitespace) or within the approximate
    edit budget are recovered as real passage substrings instead of being lost.
    """
    return "\n".join(passage[start:end] for start, end in merged)


# --------------------------- baseline span synthesis ---------------------------

def baseline_spans(kind, query, passage, budget, n_spans):
    """Budget-matched trivial extraction: list of verbatim substrings."""
    if budget <= 0 or not passage:
        return []
    budget = min(budget, len(passage))
    if kind == "lead_k":
        return [passage[:budget]]
    if kind == "random":
        rng = random.Random(hashlib.sha256(
            f"{budget}|{n_spans}|{passage[:64]}".encode()).hexdigest())
        n = max(1, min(n_spans, budget))
        piece = max(1, budget // n)
        out = []
        for _ in range(n):
            start = rng.randrange(0, max(1, len(passage) - piece))
            out.append(passage[start:start + piece])
        return out
    if kind == "lexical":
        q_terms = set(_WORD.findall(query.lower()))
        scored = []
        for m in _SENT.finditer(passage):
            sent = m.group(0)
            terms = _WORD.findall(sent.lower())
            if not terms:
                continue
            overlap = sum(t in q_terms for t in terms)
            if overlap:
                scored.append((overlap / (len(terms) ** 0.5), m.start(), sent))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out, used = [], 0
        for _score, _pos, sent in scored:
            if used >= budget:
                break
            out.append(sent)
            used += len(sent)
        return out
    raise ValueError(kind)


# --------------------------- MaxSim scoring ---------------------------

def load_model(device):
    from pylate import models
    return models.ColBERT(model_name_or_path=MODEL_ID, device=device,
                          document_length=DOCUMENT_LENGTH, query_length=QUERY_LENGTH)


def maxsim(q_emb, d_emb):
    """MaxSim for one (query tokens x dim, doc tokens x dim) pair on the same device."""
    import torch
    with torch.no_grad():
        sim = q_emb @ d_emb.T                      # (q_tok, d_tok)
        return float(sim.max(dim=1).values.sum())


def score_matrix(model, queries, docs, device, batch_queries=256):
    """Exact MaxSim scores (n_queries x n_docs); doc-by-doc, chunked queries."""
    import torch
    q_embs = model.encode(queries, is_query=True, show_progress_bar=False)
    q_pad = torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(np.asarray(e), dtype=torch.float16) for e in q_embs],
        batch_first=True).to(device)               # (nq, max_qtok, dim)
    scores = np.zeros((len(queries), len(docs)), dtype=np.float32)
    for j, d_text in enumerate(docs):
        d = model.encode([d_text], is_query=False, show_progress_bar=False)[0]
        d_t = torch.as_tensor(np.asarray(d), dtype=torch.float16, device=device)
        for i0 in range(0, len(queries), batch_queries):
            q = q_pad[i0:i0 + batch_queries]        # (b, qtok, dim)
            sim = torch.einsum("bqd,td->bqt", q, d_t)
            scores[i0:i0 + batch_queries, j] = (
                sim.max(dim=2).values.sum(dim=1).float().cpu().numpy())
        if (j + 1) % 25 == 0:
            print(f"  corpus scoring: {j + 1}/{len(docs)} docs", file=sys.stderr)
    return scores, q_pad


# --------------------------- stage: corpus ---------------------------

def load_subset_data(subset):
    from custom_utils.longembed import LONGEMBED_DATASET_ID
    from datasets import load_dataset
    corpus = load_dataset(LONGEMBED_DATASET_ID, subset, split="corpus")
    queries = load_dataset(LONGEMBED_DATASET_ID, subset, split="queries")
    qrels = load_dataset(LONGEMBED_DATASET_ID, subset, split="qrels")
    doc_ids = [r["doc_id"] for r in corpus]
    doc_texts = [r["text"] for r in corpus]
    qid2text = {r["qid"]: r["text"] for r in queries}
    gold = {r["qid"]: r["doc_id"] for r in qrels}
    qids = [qid for qid in qid2text if qid in gold]
    return doc_ids, doc_texts, qids, [qid2text[q] for q in qids], gold


def stage_corpus(args):
    import torch
    model = load_model(args.device)
    for subset in args.subsets:
        out_dir = os.path.join(CACHE_DIR, subset)
        done = os.path.join(out_dir, "scores.npy")
        if os.path.exists(done) and not args.force:
            print(f"{subset}: corpus cache present, skipping")
            continue
        t0 = time.time()
        doc_ids, doc_texts, qids, q_texts, gold = load_subset_data(subset)
        print(f"{subset}: scoring {len(qids)} queries x {len(doc_ids)} docs")
        scores, q_pad = score_matrix(model, q_texts, doc_texts, args.device)
        os.makedirs(out_dir, exist_ok=True)
        np.save(done + ".tmp.npy", scores)
        os.replace(done + ".tmp.npy", done)
        torch.save(q_pad.cpu(), os.path.join(out_dir, "q_pad.pt"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"doc_ids": doc_ids, "qids": qids,
                       "gold": {q: gold[q] for q in qids},
                       "model": MODEL_ID, "document_length": DOCUMENT_LENGTH}, f)
        print(f"{subset}: done in {time.time() - t0:.0f}s")


class SubsetCache:
    def __init__(self, subset, device):
        import torch
        d = os.path.join(CACHE_DIR, subset)
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        self.doc_ids = meta["doc_ids"]
        self.qids = meta["qids"]
        self.gold = meta["gold"]
        self.doc_index = {doc: i for i, doc in enumerate(self.doc_ids)}
        self.q_index = {q: i for i, q in enumerate(self.qids)}
        self.scores = np.load(os.path.join(d, "scores.npy"))
        self.q_pad = torch.load(os.path.join(d, "q_pad.pt"), map_location=device)

    def rank_of(self, qi, gold_j, gold_score):
        """1-based rank of the (possibly re-scored) gold doc among distractors."""
        distr = np.delete(self.scores[qi], gold_j)
        return 1 + int((distr > gold_score).sum())


# --------------------------- stage: score ---------------------------

def system_name(system):
    """Model id and mode ("<mode>/<model>") from an extraction path, or the
    baseline token unchanged. The mode is the parent directory name -- one of
    the three canonical dirs (long-embed-json, long-embed-xml,
    long-embed-xml-constrained)."""
    if system.startswith("baseline:"):
        return system
    name = os.path.basename(system)
    stem = name[:-len(".jsonl")] if name.endswith(".jsonl") else name
    mode = os.path.basename(os.path.dirname(system))
    model = stem.split("_from")[0]
    return f"{mode}/{model}"


def _encode_name(name):
    return name.replace("/", "__").replace(":", "--")


def _decode_name(safe):
    return safe.replace("--", ":").replace("__", "/")


def rows_path(system, direction, context):
    return os.path.join(
        ROWS_DIR, f"{_encode_name(system_name(system))}.{direction}.ctx{context}.jsonl")


def sampled_in(rec, frac):
    """Deterministic per-pair subsample, identical for every system (hash of the
    pair key, not of the system), so cross-system comparisons stay paired."""
    if frac >= 1.0:
        return True
    key = f"retr|{rec['subset']}|{rec['qid']}|{rec['doc_id']}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def iter_pairs(source_file, subsets, narrativeqa_frac=1.0):
    with open(source_file, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("subset") not in subsets:
                continue
            if rec["subset"] == "narrativeqa" and not sampled_in(rec, narrativeqa_frac):
                continue
            yield rec


def build_modified(rec, system, direction, context, budgets, heuristic=None):
    """-> (modified_text or None-if-no-op, stats dict).

    ``heuristic`` maps (subset, qid, doc_id) -> heuristically located spans (see
    eval/heuristic_spans.py). When supplied it REPLACES the raw selected_spans,
    so both directions score the same, verbatim span set.
    """
    passage = rec["passage"]
    if system.startswith("baseline:"):
        key = f"{rec['subset']}|{rec['qid']}|{rec['doc_id']}"
        b = budgets.get(key) or {"budget": 0, "n_spans": 1}
        spans = baseline_spans(system.split(":", 1)[1], rec["query"], passage,
                               int(b["budget"]), int(round(b["n_spans"])))
    elif heuristic is not None:
        spans = dedupe_spans(heuristic.get(
            (rec["subset"], rec["qid"], rec["doc_id"]), []))
    else:
        spans = dedupe_spans(rec.get("selected_spans"))
    merged, missing = span_intervals(passage, spans, context)
    removed = sum(end - start for start, end in merged)
    stats = {"n_spans": len(spans), "n_missing": len(missing), "removed_chars": removed,
             "removed_in_window": sum(min(end, CHAR_WINDOW) - min(start, CHAR_WINDOW)
                                      for start, end in merged)}
    if direction == "plausibility":
        text = pseudo_doc(passage, merged)
        return (text if text.strip() else None), stats
    if not merged:
        return None, stats
    modified = ablate(passage, merged)
    if modified[:CHAR_WINDOW_SAFE] == passage[:CHAR_WINDOW_SAFE]:
        return None, stats     # change invisible to the embedder window
    return modified, stats


def load_heuristic_spans(system, mode):
    """(subset, qid, doc_id) -> located spans, or None to use selected_spans.

    ``mode``: "auto" uses the sidecar when eval/heuristic_spans.py has produced
    one (the unconstrained arms) and the raw spans otherwise (constrained arms
    are already verbatim, so locating them is a no-op); "never" always uses the
    raw spans; "require" refuses to score without a sidecar, which is the safe
    setting for an unconstrained arm -- scoring one without it embeds
    unlocatable text in the pseudo-document.
    """
    if mode == "never" or system.startswith("baseline:"):
        return None
    from eval.heuristic_spans import sidecar_path
    path = sidecar_path(system_name(system))
    if not os.path.exists(path):
        if mode == "require":
            sys.exit(f"--heuristic-spans require: no sidecar at {path}\n"
                     f"run: $PYTHON eval/heuristic_spans.py {system}")
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["subset"], r["qid"], r["doc_id"])] = r.get("heuristic_spans", [])
    print(f"heuristic spans: {len(out)} records from {os.path.basename(path)}",
          file=sys.stderr)
    return out


def stage_score(args):
    import torch
    budgets = {}
    if args.system.startswith("baseline:"):
        with open(args.budgets) as f:
            budgets = json.load(f)
        source = args.source or _any_extraction_file(args.data_dir)
        print(f"baseline passages streamed from {source}", file=sys.stderr)
    else:
        source = args.system

    heuristic = load_heuristic_spans(args.system, args.heuristic_spans)

    model = load_model(args.device)
    caches = {s: SubsetCache(s, args.device) for s in args.subsets}

    for direction in args.directions:
        for context in args.contexts:
            if direction == "plausibility" and context != args.contexts[0] and args.plausibility_first_context_only:
                continue
            out = rows_path(args.system, direction, context)
            os.makedirs(ROWS_DIR, exist_ok=True)
            done = set()
            if os.path.exists(out):
                with open(out, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            done.add((r["subset"], r["qid"], r["doc_id"]))
                        except json.JSONDecodeError:
                            pass
            print(f"== {system_name(args.system)} {direction} ctx={context} "
                  f"({len(done)} pairs already done) ==", file=sys.stderr)
            n, t0 = 0, time.time()
            with open(out, "a", encoding="utf-8") as sink:
                batch_texts, batch_meta = [], []

                def flush():
                    nonlocal batch_texts, batch_meta
                    if not batch_texts:
                        return
                    embs = model.encode(batch_texts, is_query=False,
                                        batch_size=args.batch_size,
                                        show_progress_bar=False)
                    for emb, meta in zip(embs, batch_meta):
                        cache, qi, gold_j, rec, stats = meta
                        d_t = torch.as_tensor(np.asarray(emb), dtype=torch.float16,
                                              device=args.device)
                        q_emb = cache.q_pad[qi]
                        score_mod = maxsim(q_emb, d_t)
                        _write_row(sink, cache, qi, gold_j, rec, stats, score_mod,
                                   direction, context)
                    batch_texts, batch_meta = [], []

                for rec in iter_pairs(source, args.subsets, args.narrativeqa_frac):
                    key = (rec["subset"], rec["qid"], rec["doc_id"])
                    if key in done:
                        continue
                    if args.limit and n >= args.limit:
                        break
                    cache = caches[rec["subset"]]
                    qi = cache.q_index.get(rec["qid"])
                    gold_j = cache.doc_index.get(rec["doc_id"])
                    if qi is None or gold_j is None:
                        continue
                    n += 1
                    text, stats = build_modified(rec, args.system, direction,
                                                 context, budgets, heuristic)
                    if text is None:
                        # no-op: plausibility -> empty pseudo-doc scores nothing;
                        # comprehensiveness -> unchanged doc keeps its score.
                        score_mod = (float("-inf") if direction == "plausibility"
                                     else float(cache.scores[qi, gold_j]))
                        _write_row(sink, cache, qi, gold_j, rec, stats, score_mod,
                                   direction, context)
                    else:
                        batch_texts.append(text)
                        batch_meta.append((cache, qi, gold_j, rec, stats))
                        if len(batch_texts) >= args.encode_chunk:
                            flush()
                    if n % 500 == 0:
                        flush()
                        sink.flush()
                        print(f"  {n} pairs, {time.time() - t0:.0f}s", file=sys.stderr)
                flush()
            print(f"  finished {n} new pairs in {time.time() - t0:.0f}s", file=sys.stderr)


def _write_row(sink, cache, qi, gold_j, rec, stats, score_mod, direction, context):
    score_orig = float(cache.scores[qi, gold_j])
    finite_mod = score_mod if score_mod != float("-inf") else None
    rank_mod = cache.rank_of(qi, gold_j, score_mod if finite_mod is not None else -1e9)
    rank_orig = cache.rank_of(qi, gold_j, score_orig)
    sink.write(json.dumps({
        "subset": rec["subset"], "qid": rec["qid"], "doc_id": rec["doc_id"],
        "direction": direction, "context": context,
        "score_orig": score_orig, "score_mod": finite_mod,
        "rank_orig": rank_orig, "rank_mod": rank_mod,
        "ndcg10_mod": 1.0 / np.log2(rank_mod + 1) if rank_mod <= 10 else 0.0,
        **stats}, ensure_ascii=False) + "\n")


def _any_extraction_file(data_dir):
    for mode in MODES:
        hits = sorted(glob.glob(os.path.join(data_dir, mode, "*_from0-to12612.jsonl")))
        if hits:
            return hits[0]
    sys.exit("no extraction file found to stream passages from (--source)")


# --------------------------- stage: aggregate ---------------------------

def stage_aggregate(args):
    from custom_utils.artifact_hash import artifact_hash
    runs = {}
    for path in sorted(glob.glob(os.path.join(ROWS_DIR, "*.jsonl"))):
        name = os.path.basename(path)[:-len(".jsonl")]
        safe_system, direction, ctx = name.rsplit(".", 2)
        system = _decode_name(safe_system)
        context = int(ctx[len("ctx"):])
        per_subset = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                a = per_subset.setdefault(r["subset"], {
                    "n": 0, "ndcg10": 0.0, "rank_mod": 0.0, "drop": 0.0,
                    "rel_drop": 0.0, "n_noop": 0, "removed_chars": 0.0,
                    "in_window_frac": 0.0, "n_missing_spans": 0})
                a["n"] += 1
                a["ndcg10"] += r["ndcg10_mod"]
                a["rank_mod"] += r["rank_mod"]
                a["n_missing_spans"] += r.get("n_missing", 0)
                a["removed_chars"] += r.get("removed_chars", 0)
                if r.get("removed_chars"):
                    a["in_window_frac"] += r["removed_in_window"] / r["removed_chars"]
                if r["score_mod"] is None or r["score_mod"] == r["score_orig"]:
                    a["n_noop"] += 1
                if r["score_mod"] is not None and r["score_orig"]:
                    drop = r["score_orig"] - r["score_mod"]
                    a["drop"] += drop
                    a["rel_drop"] += drop / abs(r["score_orig"])
        entry = runs.setdefault(system, {})
        entry[f"{direction}@{context}"] = {
            subset: {
                "n": a["n"],
                "ndcg10": a["ndcg10"] / a["n"],
                "mean_rank": a["rank_mod"] / a["n"],
                "mean_drop": a["drop"] / a["n"],
                "mean_rel_drop": a["rel_drop"] / a["n"],
                "noop_frac": a["n_noop"] / a["n"],
                "mean_removed_chars": a["removed_chars"] / a["n"],
                "removed_in_window_frac": (a["in_window_frac"] / a["n"]) if a["n"] else 0,
                "missing_spans": a["n_missing_spans"],
            } for subset, a in per_subset.items()}
    data = {"model": MODEL_ID, "document_length": DOCUMENT_LENGTH, "runs": runs}
    old = None
    try:
        with open(args.json_out, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    if old is not None and artifact_hash(old) == artifact_hash(data):
        print(f"json artifact unchanged: {args.json_out}")
        return
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    tmp = args.json_out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, args.json_out)
    print(f"wrote {args.json_out} ({len(runs)} systems)")


# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=["corpus", "score", "aggregate"])
    parser.add_argument("--system", help="extraction JSONL path or baseline:{random,lead_k,lexical}")
    parser.add_argument("--source", help="extraction file to stream passages from (baselines)")
    parser.add_argument("--subsets", nargs="*", default=SUBSETS)
    parser.add_argument("--directions", nargs="*",
                        default=["plausibility", "comprehensiveness"])
    parser.add_argument("--contexts", nargs="*", type=int, default=[0, 2048],
                        help="context chars each side of a span (the ladder rungs)")
    parser.add_argument("--plausibility-first-context-only", action="store_true",
                        help="run plausibility only at the first context rung")
    parser.add_argument("--heuristic-spans", choices=["auto", "never", "require"],
                        default="auto",
                        help="use the eval/heuristic_spans.py sidecar (verbatim "
                             "located spans) instead of raw selected_spans: auto "
                             "when a sidecar exists, never, or require it "
                             "(recommended for unconstrained arms).")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="encoder batch size (a single 32K-token doc peaks ~17 GiB)")
    parser.add_argument("--encode-chunk", type=int, default=16)
    parser.add_argument("--narrativeqa-frac", type=float, default=1.0,
                        help="deterministic paired subsample of narrativeqa (its "
                             "10,449 pairs x ~8.5s/32K-token re-encode dominate cost; "
                             "0.2 keeps ~2,090 pairs, plenty for system ranking)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage == "corpus":
        stage_corpus(args)
    elif args.stage == "score":
        if not args.system:
            sys.exit("--stage score requires --system")
        stage_score(args)
    else:
        stage_aggregate(args)


if __name__ == "__main__":
    main()
