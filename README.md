# Fine-Grained Relevancy generator


## Human span annotations (MS MARCO dev-small)

`data/human/msmarco-spans.jsonl` is a copy of the merged human dataset from the
`explaining-ms-marco` repo (`data/spans/merged/spans.jsonl`): 319 query-passage
pairs, one row per pair, every annotator's highlighted spans as character
offsets. Refresh it by copying that file; do not edit it here.

`custom_utils/human_spans.py` flattens it into the same
`{query, passage, split, qid, doc_id, selected_spans}` records the extraction
outputs use, one record per (pair, annotator), so a human slot can be scored
like a model (`python custom_utils/human_spans.py` runs its self-test). All 319
pairs are in the BEIR `validation` split and are covered by
`data/extracted_relevancy/ms-marco-*/google~gemma-4-31B-it_validation.jsonl`.
