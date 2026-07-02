"""LongEmbed (dwzhu/LongEmbed) input adapter.

The generation pipeline expects flat records that carry BOTH a query and its
relevant passage on the same row (the ``long-embed`` template renders
``{{query}}`` and ``{{passage}}``). LongEmbed does not ship that shape: every
subset is split into three parts that all share the columns ``doc_id`` /
``text`` / ``qid`` --

    * ``corpus``  : ``doc_id`` + ``text`` (the passage), ``qid`` empty
    * ``queries`` : ``qid`` + ``text`` (the query),   ``doc_id`` empty
    * ``qrels``   : ``qid`` + ``doc_id`` relevance pair, ``text`` empty

so a query and its passage live in different rows. This module downloads those
splits for the non-synthetic subsets and joins them through ``qrels`` into
``{"query", "passage", "subset", "qid", "doc_id"}`` records.
"""

LONGEMBED_DATASET_ID = "dwzhu/LongEmbed"

# passkey / needle are synthetic recall probes -- excluded on purpose.
NON_SYNTHETIC_SUBSETS = ["narrativeqa", "summ_screen_fd", "qmsum", "2wikimqa"]


def is_longembed(input_name):
    """True when ``input_name`` refers to the LongEmbed dataset."""
    return "longembed" in input_name.lower()


def build_pairs(corpus_rows, queries_rows, qrels_rows, subset):
    """Join already-loaded split rows into flat (query, passage) records.

    Kept free of any ``datasets`` dependency so the join logic is unit-testable
    with plain lists of dicts. ``qrels_rows`` drives the pairing: each row is a
    positive (qid, doc_id) relevance pair.
    """
    doc_text = {row["doc_id"]: row["text"] for row in corpus_rows}
    query_text = {row["qid"]: row["text"] for row in queries_rows}

    records = []
    for rel in qrels_rows:
        qid, doc_id = rel["qid"], rel["doc_id"]
        if qid not in query_text or doc_id not in doc_text:
            continue
        records.append({
            "query": query_text[qid],
            "passage": doc_text[doc_id],
            "subset": subset,
            "qid": qid,
            "doc_id": doc_id,
        })
    return records


def load_longembed(subsets=None):
    """Download the LongEmbed splits and return the joined records.

    :param subsets: subset names to load (defaults to the non-synthetic ones).
    """
    from datasets import load_dataset

    if subsets is None:
        subsets = NON_SYNTHETIC_SUBSETS

    records = []
    for subset in subsets:
        corpus_rows = load_dataset(LONGEMBED_DATASET_ID, subset, split="corpus")
        queries_rows = load_dataset(LONGEMBED_DATASET_ID, subset, split="queries")
        qrels_rows = load_dataset(LONGEMBED_DATASET_ID, subset, split="qrels")

        subset_records = build_pairs(corpus_rows, queries_rows, qrels_rows, subset)
        print(f"LongEmbed[{subset}]: {len(subset_records)} query-passage pairs")
        records.extend(subset_records)

    print(f"LongEmbed: {len(records)} total pairs from {len(subsets)} subsets")
    return records
