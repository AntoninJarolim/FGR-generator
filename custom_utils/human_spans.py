"""Human span annotations for MS MARCO dev-small, in the pipeline's record shape.

The human data lives in the ``explaining-ms-marco`` repo as ONE merged file,
``data/spans/merged/spans.jsonl`` (regenerated there with
``python annotation_stats.py --write``). A copy is kept at
``data/human_spans/msmarco-spans.jsonl``; refresh it by copying, never by editing.

One merged row per (qid, psgid)::

    qid, psgid, query, passage,            # passage is the repaired text
    n_annotators,
    annotators: { "<slot>": { round, raw_spans, n_spans,
                              spans: [ {text, start, end, match, ambiguous} ] } }

``start``/``end`` are character offsets into ``passage``; ``match`` is one of
``exact | ascii | fuzzy | unmatched``. Slots are ``r<round>_a<id>``; round 1
(Apr 2025) is 3 annotators x 60 items with 20 items shared by all, round 2
(Oct 2025) is 3 x 120 with 60-item pairwise overlaps. No item is in both rounds.

This module flattens that into the ``{"query", "passage", "split", "qid",
"doc_id", "selected_spans"}`` records the extraction outputs use (see
``custom_utils.msmarco``), one record per (pair, annotator), so a human slot can
be scored with exactly the code that scores a model. ``selected_spans`` are the
LOCATED substrings ``passage[start:end]`` (verbatim by construction) rather
than the raw highlighted strings; spans that could not be located are dropped.

All 319 human pairs are in the BEIR ``validation`` split (MS MARCO dev-small),
so every record carries ``split="validation"`` and string ids like the adapter.

CAVEAT when joining against a model's extraction file: the human ``passage`` is
the REPAIRED text, while BEIR passages keep their mojibake (``â\\x80\\x99`` for
a curly apostrophe, stray ``Â`` before nbsp, ...). 79 of the 319 passages differ
that way and 45 of the 677 human spans are therefore not verbatim substrings
of the BEIR passage. Compare on the human ``passage``: locate the model's
spans in it with ``custom_utils.span_match.locate_span`` (the ``norm`` /
``approx`` tiers absorb the mojibake) rather than the other way round.
"""
import collections
import json
import os

DEFAULT_PATH = os.path.join("data", "human", "msmarco-spans.jsonl")
SPLIT = "validation"


def read_merged(path=DEFAULT_PATH):
    """The merged rows as written by explaining-ms-marco, untouched."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def located_spans(passage, annotation, keep_unmatched=False):
    """Verbatim span texts for one annotator's entry on one row."""
    spans = []
    for span in annotation["spans"]:
        if span["match"] == "unmatched":
            if keep_unmatched:
                spans.append(span["text"])
            continue
        spans.append(passage[span["start"]:span["end"]])
    return spans


def flatten(rows, min_annotators=1, slots=None):
    """One pipeline-shaped record per (pair, annotator).

    :param min_annotators: keep only pairs annotated by at least this many
        slots (2 gives the 201 pairs usable for agreement).
    :param slots: optional iterable of slot names to keep (e.g. round 2 only).
    """
    slots = set(slots) if slots is not None else None
    records = []
    for row in rows:
        if row["n_annotators"] < min_annotators:
            continue
        for slot, annotation in sorted(row["annotators"].items()):
            if slots is not None and slot not in slots:
                continue
            records.append({
                "query": row["query"],
                "passage": row["passage"],
                "split": SPLIT,
                "qid": str(row["qid"]),
                "doc_id": str(row["psgid"]),
                "annotator": slot,
                "round": annotation["round"],
                "selected_spans": located_spans(row["passage"], annotation),
            })
    return records


def load_human_spans(path=DEFAULT_PATH, min_annotators=1, slots=None):
    return flatten(read_merged(path), min_annotators, slots)


def by_pair(records):
    """``{(qid, doc_id): {slot: [spans]}}`` for joining against a model's
    extraction file keyed the same way."""
    out = collections.defaultdict(dict)
    for rec in records:
        out[(rec["qid"], rec["doc_id"])][rec["annotator"]] = rec["selected_spans"]
    return dict(out)


def _self_test():
    passage = "Ethanol is an alcohol-based fuel produced by fermenting crops."
    fc = passage.find("fermenting crops")
    rows = [{
        "qid": 8798, "psgid": 7346304, "query": "what is ethanol",
        "passage": passage, "n_annotators": 2,
        "annotators": {
            "r2_a1": {"round": "r2", "n_spans": 1, "raw_spans": ["Ethanol "],
                      "spans": [{"text": "Ethanol", "start": 0, "end": 7,
                                 "match": "exact", "ambiguous": False}]},
            "r2_a0": {"round": "r2", "n_spans": 2, "raw_spans": ["fermenting crops", "zzz"],
                      "spans": [{"text": "fermenting crops", "start": fc, "end": fc + 16,
                                 "match": "exact", "ambiguous": False},
                                {"text": "zzz", "start": -1, "end": -1,
                                 "match": "unmatched", "ambiguous": False}]},
        },
    }, {
        "qid": 1, "psgid": 2, "query": "q", "passage": "single annotator passage",
        "n_annotators": 1,
        "annotators": {"r1_a4": {"round": "r1", "n_spans": 0, "raw_spans": [], "spans": []}},
    }]

    recs = flatten(rows)
    assert [r["annotator"] for r in recs] == ["r2_a0", "r2_a1", "r1_a4"], recs
    assert recs[0]["selected_spans"] == ["fermenting crops"], recs[0]  # unmatched dropped
    assert recs[1]["selected_spans"] == ["Ethanol"] and recs[1]["qid"] == "8798"
    assert recs[1]["doc_id"] == "7346304" and recs[1]["split"] == SPLIT
    assert recs[2]["selected_spans"] == []

    assert len(flatten(rows, min_annotators=2)) == 2
    assert [r["annotator"] for r in flatten(rows, slots=["r1_a4"])] == ["r1_a4"]
    assert by_pair(recs)[("8798", "7346304")] == {"r2_a0": ["fermenting crops"],
                                                  "r2_a1": ["Ethanol"]}

    if os.path.exists(DEFAULT_PATH):
        real = load_human_spans()
        pairs = by_pair(real)
        n_multi = sum(1 for v in pairs.values() if len(v) >= 2)
        print(f"{DEFAULT_PATH}: {len(real)} (pair, annotator) records, "
              f"{len(pairs)} pairs, {n_multi} with >=2 annotators")
    print("human_spans self-test OK")


if __name__ == "__main__":
    _self_test()
