# Evaluation-metric assessment for the long-embed span data

**Purpose of this document.** Decide which of the proposed evaluation metrics are sound, which are weak or trash, and how the keepers complement each other — so the best generator model can be picked and the pick defended in one short paper subsection. Synthesized from four independent analyses: (1) a literature review (2020–2026), (2) a mapping of the metric outline against the two existing design docs (`downstream-evaluation.md`, `human-preference-evaluation.md`), (3) an adversarial methodology critique, and (4) an audit of the actually-generated data.

**Fixed decisions this assessment assumes:** audience = internal model selection, briefly defensible in a paper; hallucination measured on the unconstrained baseline only; downstream = doc-replacement retrieval (fixed off-the-shelf embedder, NDCG@10); resources = commercial judge APIs + 8×H100 + realistic human annotation of a few hundred items; scope = 3 constrained systems + unconstrained baseline, full 12,612 pairs.

---

## 0. Verdict table (TL;DR)

| Proposed metric | Verdict | One-line reason | Feasible on our data? |
|---|---|---|---|
| Hallucination (baseline only) | **KEEP** (as design justification, not selection metric) | Can't discriminate the 3 constrained systems (all 0% by construction) | ✅ now — already measured: 37.7% (unc-12B) / 50.2% (unc-E4B) non-verbatim spans. ⚠️ unc-Ministral is 100% empty output → 2-of-3 comparison only |
| Absolute judge score | **DEMOTE** | Poor calibration/consistency (G-Eval score clustering, ICC 0.58–0.77); differences between near-clone systems sit inside judge noise | ✅ now |
| Relative judge score (pairwise + human agreement) | **KEEP — primary metric** | Literature's preferred protocol for model selection (~80–85% human agreement, Zheng 2023); needs both-orders, cross-family judges, length stratification | ✅ now, after span dedup + empty-output policy |
| Downstream retrieval Δ — span-ablation (doc − spans, the intended design) | **KEEP-WITH-FIXES** | Right semantics (tests "if removed, no longer relevant") and keeps docs comparable — but it is retrieval-ERASER-comprehensiveness: spans cover 0.02–0.09% of narrativeqa docs, so NDCG@10 will be flat; measure continuous score/rank-margin drop vs a matched random-removal control | ✅ now — corpora/qrels local, pipeline in `pylate-fgr`; needs per-(query,doc) re-embed (docs serve ~29 queries) |
| — spans / +context / +more-context ladder | **KEEP — it is the necessary amplifier in the ablation design** | bare-span removal moves nothing on 250K-char docs; wider windows create measurable drops — but the random-removal control must match the removal size at each rung | trivial ~20-line context extractor (offsets recoverable via `passage.find`) |
| Cross-encoder score | **KEEP span-vs-span only; TRASH spans-vs-doc** | 512-token truncation makes the long side meaningless; κ<0.3 vs humans out-of-domain | needs work — CE cached locally, reusable code in `multidoc2dial-reranking` |
| Judge: sufficiency | **KEEP** | Judge reads only query+spans — short inputs, no long-context degradation; ERASER sufficiency without the classifier machinery | ✅ now |
| Judge: comprehensiveness | **TRASH** (full-doc judge form) | Judge must verify absence of evidence in 135K tokens = measures judge attention, not coverage; redundant narrative evidence breaks it (Carton 2020; docs already demoted it) | partial rescue: gold supporting facts on 2wikimqa only |
| Judge: downstream preference | **MERGE into pairwise** | A second preference pass doubles cost and invites "why do your two preference metrics disagree?" | — |
| **Missing: answer-containment** | **ADD — cheapest strong signal** | Deterministic, judge-free; RECOMP/EXIT precedent; covers narrativeqa+2wikimqa (85% of samples) | ⚠️ gold answers NOT local — qrels `text` field is empty; requires downloading source QA datasets + join on query text |
| **Missing: trivial floors** (lead-k, BM25-top-sentences, random same-length spans) | **ADD — mandatory** | Without a floor, "best of 4 LLMs" is uninterpretable; if BM25 sentences match the winner, there is no result | ✅ trivial to generate |
| **Missing: span-statistics table** | **ADD — free** | Catches pathologies every other metric averages away; feeds the length-bias controls | ✅ computed already (see §4) |

---

## 1. Metric-by-metric assessment

### 1.1 Hallucination rate (unconstrained baseline only) — KEEP, reframed

It proves that constrained decoding was necessary, not which constrained model is best. Two fixes:

- **Fuzzy/normalized matching** (whitespace, unicode, casing, quotes) with a separately-reported near-miss band — otherwise byte-level artifacts are counted as hallucinations.
- **Pair it with a format-tax check**: constrained vs unconstrained outputs of the *same* model on the same judge metric. Tam et al. (EMNLP-Industry 2024) show constrained decoding can degrade quality; reporting "removed X% non-verbatim spans at no measured quality cost" (or the honest opposite) pre-empts the obvious reviewer attack. Data supports this for gemma-12B and E4B; Jaccard(con-12B, unc-12B) = 0.63 with 54% identical span sets, so the 12B format tax is likely small and measurable. **unc-Ministral produced 100% empty outputs** — either re-run it or scope the claim to the two gemma models.

### 1.2 Absolute judge score — DEMOTE to sanity gate

Pointwise LLM scores cluster on few scale values, drift in calibration, and correlate only moderately with humans (G-Eval SummEval Spearman ~0.51; RAGAS-style relevance ~0.55). With four systems that all output "a span from a known-relevant document," mean-score differences will be noise. **But absolute scoring does the one thing pairwise cannot: detect that all systems are bad.** Pairwise forced choice manufactures confident winners among indistinguishable outputs.

Keep it as: 3–4-level rubric (not 1–10), on a 1–2K stratified sample, ≥2 samples per item, reported as a **distribution per system per subset** ("≥80% of spans are rubric-level relevant"), never as the ranking. **Negative controls are mandatory** (the design docs already say so): spans paired with a wrong query from the same subset + random same-length snippets; report scores above the control rate, otherwise a judge that says "relevant" to everything inflates all systems equally.

### 1.3 Pairwise judge preference + human validation — KEEP as the PRIMARY metric

The literature's canonical model-selection protocol: GPT-4-class pairwise agreement with humans ≈80–85%, on par with human–human ≈81% (Zheng et al., NeurIPS 2023); LLM relevance assessors are accepted for *relative system comparison* (Thomas et al. SIGIR 2024, κ 0.20–0.64; UMBRELA system-ranking τ ≈ 0.8–0.9 despite modest per-item κ ≈ 0.4). The Soboroff/Clarke–Dietz circularity critique is why the human anchor is non-negotiable.

Mandatory controls (all cheap, all standard, all already in `human-preference-evaluation.md`):

- **Both presentation orders**; count a win only if order-consistent, else tie. Position bias flips ~⅓ of verdicts and is worst exactly when gaps are small — ours are.
- **≥2 judge models from families disjoint from all generators.** Two generators are Gemma → no Gemini/Gemma judge; one is Ministral → no Mistral judge (self-preference is causal: Panickssery et al., NeurIPS 2024).
- **Length-stratified win rates.** Judges favor longer outputs (~+17% audited bias) and our systems differ hugely in span statistics (median span 34 vs 68 vs 43 chars; median count 1 vs 2 vs 5). One length-binned table defuses the verbosity-bias reviewer.
- **4-way outcome** (A / B / tie / both-useless), not 3-way. With empty outputs at 3.6–15.4% per system, "both useless" is a real state; folding it into "tie" corrupts Bradley–Terry.
- **Bradley–Terry over all pairings + bootstrap CIs**, cluster by document (narrativeqa has ~29 queries per doc — observations are not independent).
- **Task-anchored rubric** ("which span set better enables answering the query / justifies the document's relevance") — this absorbs the outline's separate "downstream judge preference" item.
- Sample ~2–3K items per system pair (stratified by subset and by span-set Jaccard disagreement — cross-model Jaccard is 0.075–0.095, so outputs genuinely differ and preference judging is informative), not all 12,612: the binding constraint is judge validity, not n.

**Human validation:** ~250 stratified pairwise items, 2 annotators + adjudication, in the existing visualizer's annotation mode (blinded, randomized, proximity view = union of both models' span windows — the *same* view the LLM judge gets). Report instance-level κ (expect ±0.10 CI at n≈250) and raw % agreement side-by-side with human–human agreement (the Thomas/Zheng bar is judge≈human, not judge=perfect), **plus** sign agreement on the 6 system pairs with bootstrap CIs. "Same ranking" alone is too weak a bar (1/24 by chance); instance-level agreement is the real evidence.

### 1.4 Downstream retrieval delta — the intended design is span-ABLATION (doc − spans), not spans-only

**Clarified semantics.** The metric replaces each document with the document *minus* the extracted spans (then minus spans+context, minus spans+more context), re-embeds, retrieves again, and measures the *drop*. This is deliberately not the spans-union pseudo-document design — a 100K-token doc and a 200-char span set are not comparable index units. The two designs differ on almost every property:

| Property | (A) spans-only pseudo-doc (additive) | (B) doc − spans (ablative — the intended design) |
|---|---|---|
| What it measures | sufficiency: spans alone keep the doc retrievable | comprehensiveness: nothing retrievable was left behind — directly tests the dataset's own "if the span were removed, the document would no longer be relevant" instruction |
| Literature lineage | doc2query, Dense X, RECOMP | ERASER comprehensiveness / AOPC, moved from a classifier to a retriever; `downstream-evaluation.md` Proposal 2 is the cross-encoder version of exactly this |
| Length comparability | broken: a short query-focused pseudo-doc vs 100K-token distractors → documented short/literal/early-content retriever biases inflate everything | preserved: the doc shrinks by 0.02–0.6% (median span coverage), corpus stays symmetric — the "hard to compare" problem is solved ✓ |
| Query leakage | inflates: retrieving your own prompt; a degenerate query-echoing extractor scores brilliantly | still query-conditioned, but deflating: removing the most query-*similar* text maximizes the drop whether or not it is the true evidence → a similarity-removal floor is mandatory (below) |
| Expected signal size | large, possibly ceiling'd | **tiny on narrativeqa**: median coverage is 0.02–0.09% of a 253K-char doc; deleting that from one pooled embedding moves essentially nothing. NDCG@10 will be flat; any signal lives in continuous score/rank-margin drops |
| Redundancy | immune | penalized: narratives restate evidence, so a *correct* extraction can produce zero drop — the exact failure Carton et al. 2020 documented for ERASER comprehensiveness and Proposal 2 already predicted for this data |
| Context ladder (spans → +context → +more) | pure length confound | the *necessary amplifier*: ±2K windows make removals large enough to move embeddings — but the random-removal control must be matched to the same removal size at every rung, or the ladder trend just measures amount-removed |
| Empty outputs | needs a policy decision | auto-penalized: nothing removed → zero drop → worst score, which is the correct penalty for a model that failed to extract ✓ |
| Embedder truncation | pseudo-doc always fits the window | spans beyond the embedder's context window have zero effect when removed — silently ignored. ~49–53% of docs were truncated at *generation* time (spans live in the prefix), so pick an embedder window ≥ the generation window, or restrict/report separately the docs that don't fit |
| Mechanics | one corpus swap | removal is query-conditioned but docs serve ~29 queries each (narrativeqa) → per-(query,doc) re-embed the single modified doc and swap it in at scoring time (≈12,612 extra doc embeddings per system per rung — cheap) |

**Verdict: KEEP-WITH-FIXES.** The ablative design has the right semantics (it operationalizes the annotation instruction itself) and fixes the additive variant's fatal comparability problem — but it is the retrieval twin of ERASER comprehensiveness and inherits its weak discrimination on long redundant documents. Concretely: expect near-zero variance on narrativeqa (83% of items) at the bare-spans rung; discrimination should appear at the +context rungs and on qmsum/summ_screen_fd (3.7–5.4% coverage).

**Required fixes:**

- **Measure continuous drops, not NDCG@10 alone**: Δ similarity score(q, d), rank-margin over the best distractor, and gold-doc rank change. With 197–355-doc corpora of mutually unrelated narratives, the gold doc will usually stay rank 1 even after removal — NDCG@10 is a step function that won't move. Report it, but expect it flat and say so.
- **Matched-size random-removal control**: for each (query, doc, rung), remove random windows of the same total length from the same doc; report each system as Δ over its control. This isolates "removed the *right* text" from "removed text".
- **Similarity-removal floor**: greedily remove the top query-similar sentences (BM25 or embedding similarity) at the same budget. If the LLM extractors don't clearly beat this floor, the metric is measuring similarity removal, not relevance extraction.
- **Truncation accounting**: report the fraction of removed characters that actually fall inside the embedder window, per system; a system whose spans sit past the window gets a free pass otherwise.
- **Bootstrap over queries, cluster by document** (removals on the same doc across its ~29 queries are correlated).

**Also run the additive direction (A) at a matched token budget** — the infrastructure is identical, and (A)+(B) together are the retrieval analog of ERASER sufficiency + comprehensiveness: (A) is where the discrimination will actually come from, (B) validates the "if removed, no longer relevant" claim. If (A) is run, its own controls apply: random same-length spans, lead-k, BM25-top-sentences at the same budget, symmetric replacement of *all* corpus docs, common token budget across systems, Recall@10/MRR alongside NDCG@10, and an explicit statement of the query-leakage caveat (Dense X / doc2query index query-*agnostic* units; ours are query-conditioned by design).

**Feasibility:** everything local — LongEmbed corpora/queries/qrels in `~/.hfcache`, BEIR-style NDCG@10 pipeline in `/mnt/data/ijarolim/pylate-fgr` (`longembed_data.py`, `retrieve.py`, `benchmarks.py`). The ablation variant additionally needs the per-(query,doc) embedding swap and a ±N-chars window extractor (~20 lines; offsets recoverable via `passage.find`).

### 1.5 Cross-encoder score — span-vs-span only

MS MARCO cross-encoders (incl. the cached `naver/trecdl22-crossencoder-debertav3`) truncate at 512 tokens: "scoring" a 135K-token document scores its first ~448 tokens — a spans-vs-doc comparison is structurally meaningless (and narrative lead content is scene-setting). Reranker-judges also agree poorly with humans out-of-domain (κ<0.3 on TREC-DL for MonoT5-class models; narrativeqa/qmsum are far from MS MARCO). Verdict: score **(query, span-set)** identically formatted across systems as a cheap secondary signal correlated with the judge — never as an independent downstream result, never with the full doc on one side. If it disagrees with the pairwise judge, trust the judge.

### 1.6 Sufficiency judge — KEEP (best of the "downstream judge" trio)

Judge reads only query + concatenated spans — short input, no long-context degradation, cheap at full scale. Directly answers the project question ("reading only the spans, are you convinced?") and is ERASER sufficiency without the untrainable 135K-token classifier. Apply the same negative controls as §1.2, count no-span samples as sufficiency=0 (penalizes empty-output collapse — 15.4% for con-12B — which no other automatic metric catches), expect a subset-dependent ceiling on narrativeqa multi-hop queries.

### 1.7 Comprehensiveness judge — TRASH in the proposed form

Both prior analysis and the literature agree:

- A judge must verify that **no other relevant evidence exists** in a 100–135K-token document — exactly the regime of "lost in the middle" judge degradation. You'd measure judge attention.
- Narratives/meetings restate evidence; deleting a correct span barely moves anything (even *human gold* rationales fail ERASER comprehensiveness — Carton et al., EMNLP 2020).
- The design doc already demoted it ("discriminates poorly on long documents … treat as secondary").

**Partial rescues:** (a) 2wikimqa ships gold supporting facts → deterministic supporting-fact recall on that subset (300 samples); (b) ROUGE-recall of gold summary content against the span set for qmsum/summ_screen_fd (weak proxy); (c) the §1.4 span-ablation retrieval metric *is* comprehensiveness — measured by a retriever instead of a judge, so it avoids the long-context judge problem, but it shares the redundancy caveat: on narrativeqa expect it near-flat, so do not claim comprehensiveness on narrativeqa from any of these.

### 1.8 Downstream preference judge — merge into §1.3

One pairwise protocol with a task-anchored rubric, one human-validation study. Two preference metrics = double cost + a reviewer question you don't want.

---

## 2. What the outline is missing (all cheap, several mandatory)

1. **Answer-containment / answer-bearing rate** — the design docs' recommended step 1 and the critique's "big free win": does a span (fuzzy-)contain the gold answer? Deterministic, judge-free, RECOMP/EXIT precedent, covers narrativeqa+2wikimqa = 85% of samples. **Blocker:** gold answers are not in the local data (LongEmbed qrels `text` is empty) — download source datasets and join on query text. Sanity metric, not headline (short-entity answers ≠ relevance), but it is the only fully objective quality signal available.
2. **Trivial floors** — lead-k sentences, BM25-top-sentences, random same-length spans, in *every* results table (sufficiency, NDCG, cross-encoder). Lead-N is the canonical embarrassingly-strong baseline; without floors, "best of 4 LLMs" is uninterpretable.
3. **Span-statistics table** — empty rate, span count/length distributions, duplicate rate, grammar-cap-hit rate, position-in-document distribution, query-term-copy rate. Free, catches pathologies averages hide, and directly feeds the length-bias controls. (Most of it is already computed — §4.)
4. **Per-span relevance precision** (design docs' Proposal 3) — the only metric that punishes shotgun extraction; matters because con-Ministral emits a median of 5 spans with 60% duplicate instances and hits the 32-span cap on 35% of samples. Run on the ~300-item human subset (per-span checkboxes in the visualizer) to validate the LLM judge at span level.
5. **Per-subset breakdown + macro-average, everywhere.** narrativeqa is 83% of items — the micro-average *is* narrativeqa. If the winner flips on 2wikimqa (multi-hop), that's a finding.
6. **Statistical machinery**: paired bootstrap/permutation over queries, doc-clustered; randomized Tukey HSD or Holm–Bonferroni across the 6 system pairs; effect sizes + CIs, not p-values (at n≈12.6K everything is "significant").

---

## 3. How the keepers complement each other

Each leg covers the others' known failure modes — this is the defensible structure for the paper:

| Leg | What it measures | Its blind spot | Covered by |
|---|---|---|---|
| Answer-containment + supporting-fact recall (deterministic) | objective task grounding | relevance without literal answer; 2 subsets only | sufficiency judge, pairwise judge |
| Pairwise LLM judge + human anchor (primary) | holistic span-set quality, human-validated | can't detect "all bad"; judge biases | absolute-rubric floor; both-orders/cross-family/length controls |
| Absolute rubric (demoted, with negative controls) | absolute floor, "both bad" detection | noisy ranking | pairwise does the ranking |
| Span-ablation retrieval Δ over matched random-removal control (+ optional spans-only direction) | extrinsic comprehensiveness: the spans carried the doc's retrievability | redundancy dilution → near-flat on narrativeqa; blind past embedder window; gamed by removing query-similar text | similarity-removal floor + matched-size control make Δ interpretable; sufficiency judge and the additive direction cover the sufficiency side |
| Hallucination rate + format-tax check (baseline only) | justifies constrained decoding | says nothing about which constrained model | everything else |
| Span-statistics table | pathologies (emptiness, duplication, cap-hits, position bias) | not a quality measure | feeds bias controls of all legs |

Selection rule: rank by pairwise Bradley–Terry (human-anchored); require the winner to also lead or tie on answer-containment and the ablation retrieval Δ and to clear the trivial floors; use the absolute rubric to certify the winner is good in absolute terms, not merely least-bad.

---

## 4. Data problems to fix BEFORE running any metric

From the full-data audit (all numbers verified on `data/extracted_relevancy/long-embed-*`):

1. **unc-Ministral: 100% empty raw generations** (12,612/12,612) — re-run or drop that arm; scope hallucination/format-tax claims to gemma-12B and E4B.
2. **Empty span lists**: con-12B 15.4%, con-Ministral 9.3%, con-E4B 3.6%; 18.6% of pairs lack at least one constrained system's output. Decide the policy once (sufficiency=0; "both useless" allowed in pairwise; in the ablation retrieval metric empties auto-score zero drop, which is the correct penalty) and apply it uniformly.
3. **Duplicates & cap artifacts**: duplicate span instances 45.5% (con-12B) / 60.4% (con-Ministral); 32-span grammar cap hit on 35.1% of con-Ministral samples; con-12B contains pre-cap records (max 908 spans) — the cap was applied inconsistently within that run. **Dedupe spans before every metric** and report cap-hit rate as a system property.
4. **Generation-time document truncation**: ~49–53% of samples saw only a document prefix — spans can never come from the tail. This is a shared handicap but interacts with position-in-document analysis; the `was_truncated` meta is missing from con-12B (0/12,612) and partial in con-E4B (6,980/12,612) — recompute or join from the unconstrained files.
5. **Joins must key on `(subset, qid, doc_id)`** — qid/doc_id values collide across subsets.
6. **No char offsets stored** — recover via `passage.find(span)` (verified 0 mismatches for constrained outputs; `custom_utils/text_utils.py:find_span` exists); first occurrence is ambiguous for repeated text — acceptable for context windows, document it.
7. **Old unconstrained `long-embed-json` prompt runs are superseded** (28.7–50.4% non-verbatim; glm file has non-string spans) — exclude from all metrics. (Note: the JSON-*constrained* large-model runs in `long-embed-json-constrained` are a legacy arm — produced by the now-removed `--vllm_guided_json` option, which constrained the JSON schema but not verbatimness; kept for now, not regenerable.)

---

## 5. Recommended execution order

1. **Data hygiene** (§4): dedupe, empty-output policy, (subset,qid,doc_id) keys, recompute truncation meta. Optionally re-run unc-Ministral.
2. **Free deterministic tier**: span-statistics table; hallucination + near-miss band (gemma arms); download gold answers → answer-containment (narrativeqa, 2wikimqa) + supporting-fact recall (2wikimqa) + ROUGE-coverage (qmsum, summ_screen_fd); generate lead-k / BM25 / random-span floors. → first model ranking signal, zero judge cost.
3. **Span-ablation retrieval Δ** with all §1.4 fixes (continuous score/rank-margin drop, matched-size random-removal control, similarity-removal floor, truncation accounting); optionally the additive spans-only direction at matched budget, since the infrastructure is identical (pipeline exists in `pylate-fgr`).
4. **Pairwise LLM judge** (§1.3 controls) on ~2–3K stratified items per pair; absolute rubric with negative controls on a 1–2K sample as the floor/both-bad gate; cross-encoder span-vs-span as a correlation check.
5. **Human anchor**: visualizer annotation mode → ~250 pairwise items (2 annotators, adjudicated, ≥20% double), per-span checkboxes on the same items; report κ, % agreement vs human–human, pair-sign agreement with CIs.
6. **Paper paragraph**: BT ranking (human-anchored) + answer-containment + ablation retrieval Δ over the floors + hallucination/format-tax sentence. Everything per-subset + macro, bootstrap CIs, doc-clustered.

---

## 6. Key citations

Judging: Zheng et al. NeurIPS 2023 (MT-Bench; position/verbosity/self-bias, ~80–85% human agreement) · Thomas et al. SIGIR 2024 (Bing LLM assessor, κ 0.20–0.64) · Upadhyay et al. 2024/2025 (UMBRELA; per-item κ≈0.4 but system-ranking τ≈0.8–0.9) · Soboroff arXiv:2409.15133 + Clarke & Dietz arXiv:2412.17156 (circularity critique) · Panickssery et al. NeurIPS 2024 (self-preference is causal) · JudgeBench ICLR 2025 (judges near chance on hard pairs) · Liu et al. G-Eval EMNLP 2023 (pointwise calibration).

Rationales/attribution: DeYoung et al. ACL 2020 (ERASER) · Carton et al. EMNLP 2020 (human rationales fail suff./compr.; length sensitivity) · Gao et al. ALCE EMNLP 2023 · LongCite 2024 / L-CiteEval ACL 2025 (long-context citation eval precedent).

Extraction/compression: Xu et al. RECOMP ICLR 2024 · EXIT Findings-ACL 2025 · Chen et al. Dense X EMNLP 2024 · doc2query 2019 · Gospodinov et al. ECIR 2023 (Doc2Query--).

Constrained decoding: Tam et al. EMNLP-Industry 2024 (format tax). Retriever biases: Coelho et al. SIGIR 2024 (dwell-in-the-beginning) · "Collapse of Dense Retrievers" 2025 (short/literal bias) · Zhu et al. LongEmbed EMNLP 2024.

Faithfulness: Grusky et al. NAACL 2018 (coverage/density) · Ladhak et al. ACL 2022 (extractiveness confound) · Maynez et al. ACL 2020.

Statistics: Smucker et al. CIKM 2007 · Carterette TOIS 2012 (randomized Tukey HSD) · Deutsch et al. TACL 2021 / NAACL 2022 (system-level correlation pitfalls) · Bujang & Baharum 2017 (κ sample size: n≈250 → CI ±0.10).
