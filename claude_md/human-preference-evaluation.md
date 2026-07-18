# Human (and LLM) preference evaluation between span-extraction models

**Context.** Three constrained models (gemma-4-12B, gemma-4-E4B, Ministral-3-14B) produce verbatim
spans for the same 12,612 (query, document) pairs. EM is saturated at 100% by construction, so model
comparison must come from *judged quality*. This document designs the pairwise-preference study: the
interface, the annotation target, aggregation, and whether an LLM can stand in for humans.

---

## 1. Interface: side-by-side diff view

Extend the existing visualizer (it already renders per-model columns over the same document with
highlighted spans) into an **annotation mode**:

- Two models' extractions for the same (query, document), highlighted in the *same* document view —
  either two synchronized columns (current layout) or a single document with two highlight colors
  (true "diff" view). Single-document dual-color is preferable: differences pop visually and
  scrolling is halved; columns remain useful when span sets overlap heavily.
- **Blinding and randomization:** model identities hidden; left/right (or color) assignment randomized
  per item — position bias is well documented in both humans and LLM judges (Zheng et al., NeurIPS
  2023). Log the permutation.
- Judgment buttons + keyboard (`a` left better / `b` right better / `t` tie / `x` both useless),
  a skip-with-reason option, and persistence via a small POST endpoint into a JSONL next to the data.
- Sample selection: stratified by subset and by disagreement (pairs where models differ most —
  measured by span-set Jaccard — are the informative ones; agreeing pairs waste annotation budget).

**Pros:** reuses ~80% of existing code; annotators see spans in context (crucial — a span's quality is
context-dependent); fast (target < 30 s/item with keyboard flow).
**Cons:** long documents make full reading impossible — annotators judge from the highlighted regions
plus local context, which slightly favors "plausible-looking" over "actually comprehensive" spans;
mitigate with the proximity view (already built) defaulting to union of both models' span windows.

## 2. What exactly should be annotated?

Three candidate protocols, orderable by cognitive load:

**(a) Single overall pairwise preference (+ tie).**
"Which extraction better justifies why this document is relevant to this query?"
*Pros:* the RLHF literature converged on pairwise comparison precisely because it is the most
reliable elicitation (Ouyang et al. 2022; Bai et al. 2022); highest throughput; directly feeds
Bradley–Terry/Elo. *Cons:* conflates dimensions — a judge may prefer fewer-but-cleaner spans while
another prefers coverage; criteria drift across annotators unless anchored.

**(b) Per-dimension pairwise judgments.**
Same pairwise mechanics, but asked twice per item with our own prompt's definitions (reuse them
verbatim as the annotation guideline — they are already crisp):
- **Plausibility:** "reading only the spans, which set is more convincing that the document is relevant?"
- **Comprehensiveness:** "which set better covers *all* the material that makes the document relevant?"
Optionally a third: **minimality** (least irrelevant text dragged along) — this is where the
constrained models' duplicate/shotgun spans would be penalized.
*Pros:* diagnostic (a model can win plausibility and lose minimality — that is a finding, not noise);
dimensions map 1:1 to the dataset's stated construction criteria, which reviewers will like.
*Cons:* ~2–3× annotation cost; dimensions correlate strongly in practice (halo effect), so the extra
signal may be modest.

**(c) Per-span accept/reject (checkbox on each highlight).**
*Pros:* yields reusable span-level gold; supports precision/recall-style metrics rather than only
preferences. *Cons:* highest cost; does not directly answer "which model is better"; recall side is
unmeasurable without exhaustive gold spans (annotators would have to find missed evidence in a 100K-token
document — unrealistic).

**Recommendation:** protocol (b) with the two project-native dimensions, plus an overall preference as
the final question of each item (three keypresses per item). Run (c) only on the ~300-sample subset
used for LLM-judge validation.

## 3. Aggregation and study size

- **Model ranking:** Bradley–Terry (or Elo with fixed K) over pairwise outcomes; 3 models → 3 pairings.
  With ~a 55/45 true preference and 95% power, plan for **≈ 800 comparisons per model pair**; if
  budget-limited, 300/pair still resolves clear gaps. Stratify across the 4 subsets.
- **Agreement:** double-annotate ≥ 20% of items; report Krippendorff's α (target > 0.6 for pairwise);
  adjudicate systematic disagreements — they usually reveal guideline gaps, iterate the guideline once.
- **Guideline:** one page, the two definitions from the system prompt + 4 worked examples (one per
  subset), including a "both useless" example.

## 4. Can a model do this evaluation? (LLM-as-judge)

Yes, with controls — this is now standard practice in IR: LLM relevance assessors reach or exceed
crowdworker agreement with TREC gold (Thomas et al., SIGIR 2024; UMBRELA, Upadhyay et al. 2024;
TREC RAG 2024 used LLM judges officially). For pairwise quality judging, MT-Bench-style protocols
(Zheng et al. 2023) report ~80% human agreement — comparable to human–human agreement.

**Required bias controls:**
- **Position bias:** judge every pair in both orders; keep only order-consistent verdicts (or average).
- **Self-preference / family bias:** LLMs prefer their own outputs (Panickssery et al. 2024) — do not
  judge Gemma outputs with a Gemma judge; use ≥ 2 judge models from families disjoint from all three
  generators, report per-judge and consensus results.
- **Length/verbosity bias:** judges favor longer outputs; report span-count/length per model alongside,
  and instruct the judge that extra irrelevant spans should *lower* the verdict (anchors minimality).
- **Context handling:** the judge cannot read 100K-token documents cheaply; give it the same
  proximity view a human gets (union of both models' span windows ± context). This is a deliberate,
  documented restriction — identical information for human and LLM judges makes agreement meaningful.

**Validation protocol:** LLM judges run on the full set; humans annotate the ~300-item stratified
subset with protocol (b); report human–LLM agreement (accuracy against human majority + Cohen's κ)
per dimension. If κ ≥ ~0.5 and the *model ranking* matches, the LLM-judged full-set result is
reportable with the human study as its anchor — the standard pattern in current literature.

**Pros of LLM judging:** scales to all 37,836 outputs; cheap; repeatable across dataset iterations.
**Cons:** biases above; circularity risk (LLM judging LLM extractions rewards LLM-typical style);
reviewers will expect the human anchor regardless — budget it from the start.

---

## Next steps (ordered)

1. Freeze the annotation guideline (reuse system-prompt definitions; add examples).
2. Add annotation mode to the visualizer: dual-color diff view, blinding, randomized order,
   keypress judgments (plausibility, comprehensiveness, overall), JSONL persistence endpoint.
3. Pilot: 30 items, 2 annotators → measure time/item and α; fix guideline; then main study
   (300–800 per model pair, stratified, ≥ 20% double-annotated).
4. In parallel: LLM-judge pipeline with both-orders + cross-family judges on the full set.
5. Report: BT ranking (human), BT ranking (LLM), human–LLM κ, per-dimension wins, per-subset splits.
