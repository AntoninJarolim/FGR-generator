# llm2colbert notebooks

Analysis notebooks from the archived llm2colbert-distillation project (Feb–May 2025):

- `human-annotations.ipynb`: agreement between the three MS MARCO span annotators and
  Gemma-2 27B (pairwise P/R/F1, Fleiss' kappa), plus the span-count histogram
- `accuracy_annotations.ipynb`: accuracy-dataset judgements and annotator time
- `explanations.ipynb`: the 35-sample model comparison (Gemma-2, Llama-3.1, GPT-4o, TF-IDF)

`explainable_dataset.py` and `utils/text_utils.py` are the old llm2colbert versions.
The notebooks rely on their API (`q_key`, dict spans), which differs from the
`explainable_dataset.py` at the FGR-generator root.

Run them from this directory. They read from:

| variable | path | contents |
|---|---|---|
| `LEGACY` | `../../data/extracted_spans/msmarco-legacy-gemma2` | extractions, 35-sample and accuracy-dataset files |
| `RELEVANCE` | `../../../explaining-ms-marco/data/relevance` | accuracy-dataset assignments and judgements |
| `COLBERT_DATA` | `../../../ColBERT/data` | 800k train / dev-small extraction TSVs (span-count histogram) |

Besides FGR-generator's requirements, you need `statsmodels`, `jsonlines`, `scikit-learn`,
`seaborn` and a Jupyter kernel. `explanations.ipynb` loads the gated
`meta-llama/Llama-3.1-70B-Instruct` tokenizer, so it needs a Hugging Face token.
Running the notebooks overwrites `fig/*.pdf`, and `explanations.ipynb` rewrites
`$LEGACY/experiments/other/f1_stats.json`.
