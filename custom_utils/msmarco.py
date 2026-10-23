"""MS MARCO (BEIR packaging) input adapter.

The generation pipeline expects flat records that carry BOTH a query and its
relevant passage on the same row (the span-extraction templates render
``{{query}}`` and ``{{passage}}``). BEIR ships MS MARCO as three pieces --

    * ``BeIR/msmarco``       config ``corpus``  : ``_id`` + ``title`` + ``text``
    * ``BeIR/msmarco``       config ``queries`` : ``_id`` + ``title`` + ``text``
    * ``BeIR/msmarco-qrels`` splits train/validation/test :
                                                 ``query-id`` + ``corpus-id`` + ``score``

so a query and its passage live in different rows, exactly like LongEmbed. This
module joins them through qrels into
``{"query", "passage", "split", "qid", "doc_id"}`` records.

Two differences from ``custom_utils.longembed`` that will silently corrupt the
join if ignored:

* **ID types.** qrels ids are INTEGERS (``query-id: 1185869``) while corpus and
  queries ids are STRINGS (``_id: "1185869"``). Every lookup casts with ``str``.
* **Graded scores.** LongEmbed qrels are uniformly positive so its adapter
  ignores the score. Here only the train and validation splits are uniformly
  ``score=1``; ``test`` is TREC-DL 2019 with graded 0-3 judgments of which 5158
  of 9260 are ``score=0``, i.e. explicitly NOT relevant. Feeding those to the
  extractor would ask which spans make an irrelevant passage relevant, so
  ``build_pairs`` filters on ``min_score``.

``title`` is ignored: it is empty for every MS MARCO passage.
"""

MSMARCO_DATASET_ID = "BeIR/msmarco"
MSMARCO_QRELS_ID = "BeIR/msmarco-qrels"

# train (532,751 pairs) is the supervision set; validation (7,437) is the
# dev-small subset, kept as a held-out set. test is TREC-DL 2019 and is excluded
# on purpose -- see the module docstring on graded scores.
DEFAULT_SPLITS = ["train", "validation"]


def is_msmarco(input_name):
    """True when ``input_name`` refers to the MS MARCO dataset, spelled any of
    the usual ways (``BeIR/msmarco``, ``ms_marco``, ``ms-marco``)."""
    normalized = input_name.lower().replace("_", "").replace("-", "")
    return "msmarco" in normalized


def id_text_map(rows):
    """``{id_string: text}`` from corpus or queries rows.

    Reads whole columns when given a HuggingFace ``Dataset`` (one Arrow-to-Python
    conversion instead of 8.8M per-row dict constructions for the corpus) and
    falls back to per-row access for the plain lists of dicts used in tests.
    """
    try:
        ids, texts = rows["_id"], rows["text"]
    except TypeError:
        ids = [row["_id"] for row in rows]
        texts = [row["text"] for row in rows]
    return {str(doc_id): text for doc_id, text in zip(ids, texts)}


def build_pairs(doc_text, query_text, qrels_rows, split, min_score=1):
    """Join qrels against ``{id: text}`` lookups into flat (query, passage) records.

    Kept free of any ``datasets`` dependency so the join logic is testable with
    plain dicts and lists. ``qrels_rows`` drives the pairing: each row with
    ``score >= min_score`` becomes one supervision sample.

    Rows are emitted in qrels order, which makes the output deterministic and
    therefore safe to address by index from sharded generation jobs.

    No deduplication: ~30k train queries carry two positives, and each
    (query, passage) pair is a distinct sample.
    """
    records = []
    for rel in qrels_rows:
        if rel["score"] < min_score:
            continue
        qid, doc_id = str(rel["query-id"]), str(rel["corpus-id"])
        if qid not in query_text or doc_id not in doc_text:
            continue
        records.append({
            "query": query_text[qid],
            "passage": doc_text[doc_id],
            "split": split,
            "qid": qid,
            "doc_id": doc_id,
        })
    return records


def load_msmarco(splits=None, min_score=1):
    """Download the BEIR MS MARCO pieces and return the joined records.

    The corpus is 8,841,823 passages, so the ``{doc_id: text}`` lookup costs
    several GB of RAM and minutes to build. Call this ONCE on a machine with
    internet and materialize the result (see ``prefetch_hf_assets.py``) rather
    than re-joining inside every generation job.

    :param splits: qrels splits to load (defaults to train + validation).
    :param min_score: minimum qrels score to treat as relevant.
    """
    from datasets import load_dataset

    if splits is None:
        splits = DEFAULT_SPLITS

    print(f"MS MARCO: loading corpus and queries from {MSMARCO_DATASET_ID}")
    doc_text = id_text_map(load_dataset(MSMARCO_DATASET_ID, "corpus", split="corpus"))
    query_text = id_text_map(load_dataset(MSMARCO_DATASET_ID, "queries", split="queries"))
    print(f"MS MARCO: {len(doc_text)} passages, {len(query_text)} queries")

    records = []
    for split in splits:
        qrels_rows = load_dataset(MSMARCO_QRELS_ID, split=split)
        split_records = build_pairs(doc_text, query_text, qrels_rows, split, min_score)
        print(f"MS MARCO[{split}]: {len(split_records)} query-passage pairs "
              f"from {len(qrels_rows)} qrels (min_score={min_score})")
        records.extend(split_records)

    print(f"MS MARCO: {len(records)} total pairs from {len(splits)} splits")
    return records


def _self_test():
    """Join assertions that need no network: id-type coercion and score filtering."""
    corpus = [{"_id": "0", "title": "", "text": "passage zero"},
              {"_id": "49", "title": "", "text": "passage forty-nine"}]
    queries = [{"_id": "1185869", "title": "", "text": "query one"},
               {"_id": "597651", "title": "", "text": "query two"}]
    # Integer ids, as the real qrels ship them, against the string _ids above.
    qrels = [
        {"query-id": 1185869, "corpus-id": 0, "score": 1},
        {"query-id": 597651, "corpus-id": 49, "score": 0},    # judged NOT relevant
        {"query-id": 597651, "corpus-id": 999, "score": 1},   # doc not in corpus
        {"query-id": 111, "corpus-id": 0, "score": 1},        # query not in queries
    ]

    doc_text, query_text = id_text_map(corpus), id_text_map(queries)
    assert doc_text == {"0": "passage zero", "49": "passage forty-nine"}, doc_text

    records = build_pairs(doc_text, query_text, qrels, "train")
    assert len(records) == 1, f"expected 1 record, got {len(records)}: {records}"
    assert records[0] == {
        "query": "query one",
        "passage": "passage zero",
        "split": "train",
        "qid": "1185869",
        "doc_id": "0",
    }, records[0]

    # min_score=0 keeps the graded-zero row (still dropping the unjoinable two).
    graded = build_pairs(doc_text, query_text, qrels, "test", min_score=0)
    assert len(graded) == 2, f"expected 2 records, got {len(graded)}"

    assert is_msmarco("BeIR/msmarco") and is_msmarco("ms_marco") and is_msmarco("ms-marco")
    assert not is_msmarco("dwzhu/LongEmbed")

    print("msmarco self-test OK")


if __name__ == "__main__":
    _self_test()
