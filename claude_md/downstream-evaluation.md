# Downstream evaluation of extracted relevance spans

**Context.** We generate fine-grained relevance supervision: for a (query, document) pair known to be
relevant (LongEmbed qrels), a model extracts verbatim spans that make the document relevant. Since
span-constrained decoding, EM = 100% — every span is guaranteed in-document — so *locating* spans is
solved and evaluation shifts entirely to *span quality*: do the spans actually carry the relevance?

The two questions from the project definition map exactly onto the two standard rationale-quality
criteria from the explainability literature (ERASER; DeYoung et al., ACL 2020):

| Project question | ERASER concept | Test shape |
|---|---|---|
| "Reading only the spans, are you convinced the document is relevant?" | **Sufficiency** (plausibility) | judge(query, spans-only) |
| "Is this part of the document relevant to the query — yes/no?" | per-extract **relevance precision** | judge(query, one span + local context) |
| (complement, from our own prompt's definition) | **Comprehensiveness** | judge(query, document − spans) |

---

## Proposal 1 — Span-sufficiency judgment (spans-only relevance test)

Show a judge (LLM at scale, human on a subset) the query and the concatenated spans **only** —
no document. Ask for a relevance decision. Score = fraction of gold-positive pairs judged relevant
from spans alone ("sufficiency rate"), per model.

**Design details**
- **Binary vs graded:** binary matches the qrels semantics and maximizes judge agreement; a graded
  scale (0–3, TREC-style; Sormunen 2002) gives more statistical resolution between our three models.
  Recommendation: 4-level graded, collapsed to binary for headline numbers.
- **Negative controls are mandatory.** A judge that says "relevant" to anything inflates all models
  equally. Mix in (a) spans paired with a *wrong* query from the same subset, (b) random same-length
  document snippets. Report sufficiency *above* the control rate (or as AUC over positives/controls).
- **No-span samples** count as sufficiency = 0 (the model failed to justify a known-relevant pair).
  This penalizes empty-output collapse, which raw EM does not.

**Pros:** directly answers the project question; cheap (spans are short — pennies per judgment with an
LLM); works without any gold rationale annotations; sensitive to both missing spans and vacuous spans.
**Cons:** LLM-judge biases (verbosity, self-preference — see the companion document); a span can be
"convincing" yet factually mislocated in context (less of an issue since spans are verbatim);
narrativeqa queries sometimes require multi-hop context a short span cannot carry — expect a
subset-dependent ceiling.

## Proposal 2 — Comprehensiveness via span ablation

Delete the extracted spans from the document and ask whether the remainder is still relevant
(judge-based), or measure the drop in a retriever/reranker relevance score
(score(q, doc) − score(q, doc−spans); cf. ERASER comprehensiveness, AOPC).

**Pros:** tests the "if the span were removed, the document would no longer be relevant" clause of our
own annotation instructions; automatic when using a scoring model (e.g., the repo's cross-encoder
`naver/trecdl22-crossencoder-debertav3` or a ColBERT MaxSim score).
**Cons — and this is serious for LongEmbed:** documents are up to 135K tokens; removing a 20-token
span from a 100K-token narrative rarely moves any score, and the document usually stays "relevant"
via redundant evidence. Comprehensiveness discriminates poorly on long documents (known issue with
ERASER-style metrics on long inputs). Mitigation: evaluate on a *window* around the span (± 2K chars)
rather than the whole document, or report comprehensiveness only for summ_screen_fd/qmsum-style tasks
where evidence is more localized. Treat as secondary metric.

## Proposal 3 — Per-span relevance precision

Judge each span independently: "does this text fragment (shown with ±N chars of context) support the
query — yes/no". Precision = judged-relevant spans / all spans; complements sufficiency (which is
per-sample) by punishing shotgun extraction (many spans, few relevant — our constrained models emit
more spans than baseline, so this matters).

**Pros:** localizes errors; produces span-level labels reusable as silver training data; natural to
annotate in the existing visualizer.
**Cons:** ignores complementarity (two individually weak spans may jointly justify relevance);
more judgments per sample (cost scales with span count, p90 = 5).

## Proposal 4 — Extrinsic: train a retriever on the supervision

The actual purpose of the dataset. Fine-tune a late-interaction model (ColBERT-style; the repo
already touches `GTE-ModernColBERT-v1`) or a long-context bi-encoder, using span-level signal
(e.g., MaxSim alignment targets on span tokens, or span-cropped positives) vs. a baseline trained on
document-level labels only. Evaluate nDCG@10 / recall@k on held-out LongEmbed splits and zero-shot
transfer (BEIR subset).

**Pros:** the only evaluation that measures what the dataset is *for*; publishable headline result
("span supervision improves long-document retrieval by X").
**Cons:** expensive (training runs, hyperparameter fairness between conditions); confounded — a null
result may indict the training recipe, not the spans; slow iteration. Sequence it *after* the cheap
intrinsic metrics agree on a model ranking.

## Proposal 5 — Answer-bearing rate (narrativeqa / 2wikimqa only)

These subsets originate from QA datasets with gold answers. Recover the answers from the source
datasets (they are not in our qrels join) and compute the fraction of samples where some extracted
span contains (or fuzzy-contains) the gold answer string.

**Pros:** fully automatic, zero judge cost, objective; strong external validity for two subsets
covering 85% of samples.
**Cons:** answers are short entities — a span can be relevant without containing the literal answer
(and vice versa for accidental containment); does not apply to summ_screen_fd/qmsum (summary-style
queries). Use as a sanity metric, not the headline.

---

## Recommended next steps (ordered)

1. **Answer-bearing rate** (Proposal 5): one script, no judges — immediate model ranking signal.
2. **LLM-judge sufficiency** (Proposal 1) over all 3 × 12,612 outputs with negative controls,
   two judge models from different families than the generators (to dodge self-preference).
3. **Per-span precision** (Proposal 3) on a stratified sample (e.g., 300 samples × 3 models),
   double-annotated by humans in the visualizer → validates the LLM judge (report Cohen's κ).
4. **Comprehensiveness** (Proposal 2) windowed variant, secondary metric, cross-encoder based.
5. **Retriever fine-tuning** (Proposal 4) once 1–3 agree on which generator's data to invest in.

Report per subset (narrativeqa dominates 83% of samples — always show macro-over-subsets alongside),
with bootstrap CIs over samples, and compare the three constrained models *and* the unconstrained
baseline outputs (post-hoc filtered to valid spans) to quantify what the constraint bought beyond EM.
