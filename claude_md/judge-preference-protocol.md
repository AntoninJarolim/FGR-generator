# LLM-judge pairwise preference protocol (operational plan)

**Purpose.** Turn the "pairwise judge = primary metric" decision from `evaluation-metrics-assessment.md` §1.3 into a runnable protocol: exactly what the judge sees, the exact prompt, token budgets, sampling, aggregation, and how your own annotations calibrate and validate it. Companion to `human-preference-evaluation.md` (which designed the human side); this doc makes the two sides *identical by construction*.

---

## 1. What is actually being evaluated?

The spans exist to be **fine-grained relevance supervision for training retrievers**. A "better" extraction is the one that would make better training signal. That decomposes into four testable properties, in priority order:

1. **Support (precision).** Every highlighted span genuinely helps explain why this document matches this query. A span that is off-topic, vacuous ("Mark"), or merely mentions a query word without carrying evidence is a defect.
2. **Local coverage.** Within the text shown, the span set captures the distinct pieces of evidence. Missing an evidence sentence that the other model caught is a defect. (This is deliberately *local* — no judge or human can verify global comprehensiveness over a 250K-char document; the retrieval-ablation metric owns that claim.)
3. **Minimality.** No duplicate spans, no near-duplicate rephrasings of the same highlight, no dragging along whole paragraphs where a sentence carries the evidence, no fragment confetti (10 two-word shards where one sentence would do).
4. **Standalone readability.** A person reading only the span (with the query) should be convinced without needing invisible context. Cut-off fragments that depend on unseen text score worse.

The single overall question the judge answers: **"Which extraction would better teach a retrieval model *which text* makes this document relevant to this query?"** — support violations are disqualifying, coverage beats minimality, minimality beats readability.

What is deliberately NOT evaluated: global comprehensiveness (unverifiable here), factual correctness of the document itself, style/grammar of spans (they are verbatim quotes), and span count per se (more spans is neither better nor worse — only what they contain).

## 2. What the judge sees: the paired diff view

One judged item = one (query, document, model-A spans, model-B spans) tuple, rendered as **merged evidence windows** — the union of both models' span regions with shared context:

```
QUERY: Why is Bobolink eventually eager to help Martin?

DOCUMENT EXCERPTS (2 of 3 regions shown; 1 more region with only B-highlights was cut for length):

[region 1 | document position ~12%]
...preceding context... [[A: span text highlighted only by A ]] ...shared
context... [[AB: text both models highlighted ]] ...trailing context...

[region 2 | document position ~57%]
...context... [[B: text only B highlighted ]] ...context...

MODEL A: 4 spans total (1 in cut regions)   MODEL B: 2 spans total (1 in cut regions)
```

Construction rules:

- Recover offsets by `passage.find(span)` (constrained spans are verbatim; for unconstrained systems, non-locatable spans are listed under the excerpts as `A (not found verbatim in document): "..."`).
- **Dedupe spans first** (exact-duplicate instances are 45–60% in two systems — they must not be shown, only counted; the minimality dimension is judged from a shown `A: n spans / B: m spans` count plus the visible near-duplicates).
- Merge overlapping/adjacent (±context) regions of BOTH models into shared windows so identical evidence appears once with an `[[AB: ...]]` marking — this *is* the diff view: differences pop as `[[A:` or `[[B:` segments. Overlapping-but-not-identical spans are split into `[[A:`, `[[AB:`, `[[B:` segments.
- An empty extraction renders as `MODEL B: (no spans extracted)` and the item is still judged (an empty set loses to a useful set, but beats a garbage set only via `both_useless`).
- Document position % shown per region (cheap, lets the judge sense spread without seeing the whole doc).

## 3. Token budget

Target **≤ 3,000 tokens of rendered content** per item (≈ what a careful human reads in the visualizer's proximity view; also keeps 2 judges × 2 orders × ~6K items affordable).

- Context per region: **±120 words (~160 tokens)** around the merged span group.
- Cap at **8 regions**, keeping the regions with the largest highlighted mass; disclose cuts explicitly ("N more regions with only A/B highlights were cut", plus the per-model total span counts) — silent truncation would corrupt the coverage dimension.
- Overflow handling in order: shrink context to ±60 words → drop lowest-mass regions (with disclosure) → hard-truncate individual spans over 600 tokens (rare; whole-paragraph spans get shown head+tail with `[... span continues ...]`).
- Never feed the full document. This is a *deliberate, documented restriction*: the LLM judge and the human annotator see byte-identical renderings, which is what makes their agreement meaningful (`human-preference-evaluation.md` §4).

## 4. The judge prompt

System prompt (frozen after the pilot — see §7):

```
You are evaluating two competing "relevance span" extractions, A and B. Both
were asked to highlight the verbatim text that makes a document relevant to a
query. The extractions will be used to teach retrieval models WHICH text in a
document carries the relevance — that purpose defines quality.

You see the query and merged document excerpts in which highlighted text is
marked [[A: ...]], [[B: ...]], or [[AB: ...]] (both). Regions may have been cut
for length; cuts and total span counts are disclosed. Judge ONLY from what is
shown.

Evaluate, in this priority order:
1. support — does each highlighted span genuinely evidence the document's
   relevance to the query? Irrelevant, vacuous, or keyword-only highlights are
   defects. This outweighs everything else.
2. coverage — within the shown excerpts, which extraction captures more of the
   distinct evidence? Missing evidence the other model caught is a defect.
3. minimality — duplicates, near-duplicate rephrasings, fragment confetti, or
   whole paragraphs where a sentence suffices are defects. Do NOT reward an
   extraction merely for highlighting more text; extra irrelevant text is a
   defect, not coverage.
4. readability — spans a person could read standalone (with the query) and be
   convinced beat cut-off fragments needing unseen context.

Output STRICT JSON, nothing else:
{"support": "A"|"B"|"tie", "coverage": "A"|"B"|"tie",
 "minimality": "A"|"B"|"tie",
 "overall": "A"|"B"|"tie"|"both_useless",
 "reason": "<one sentence>"}
"both_useless": neither extraction contains a single span that genuinely
supports the document's relevance (an empty extraction vs a garbage one is
both_useless, not a win).
```

User message: the rendered item from §2. Temperature 0. The `reason` field costs a few tokens and is kept — it is the raw material for §7's calibration and for error analysis.

Notes on why this shape:

- **Per-dimension verdicts + overall** (protocol (b) of `human-preference-evaluation.md` §2): diagnostic at ~zero extra cost in a single call; a model that wins support but loses minimality is a finding.
- **Anti-verbosity instruction is explicit** ("extra irrelevant text is a defect") — judges' documented length bias must be pushed against in-prompt AND checked post-hoc (§6).
- **No absolute scores** — the assessment demoted pointwise scoring; the only absolute escape hatch is `both_useless`.

## 5. Sampling: which pairs, which items

- **System pairs:** all pairs of constrained systems (3 systems → 3 pairs; each newly generated model adds its pairs against all incumbents). Optionally +3 constrained-vs-unconstrained same-model pairs (the format-tax check, §1.1 of the assessment) — cheap because they agree heavily (Jaccard 0.63 for 12B → most items skippable as identical).
- **Skip identical items:** if A and B span sets are equal after dedup/normalization, record `tie` without a judge call (4–6% of pairs between constrained systems, 54% for con-vs-unc 12B).
- **Both-empty items:** record `both_useless` without a judge call.
- **Per-pair sample (~2,000):** 2wikimqa 300 (all) + summ_screen_fd 336 (all) + qmsum 500 + narrativeqa 900. Within qmsum/narrativeqa: 80% sampled by *disagreement* (lowest span-set Jaccard first — that's where the information is), 20% uniform random (so the reported win-rate is not conditioned on disagreement only; report both strata separately).
- **Both presentation orders** for every judged item (A/B swapped). Overall verdict counts only if order-consistent after unswapping; else `tie`. Report the position-flip rate per judge (it is itself a judge-quality diagnostic).
- **Cost envelope:** 3 pairs × ~2,000 items × 2 orders × ~3.4K tokens ≈ 41M input tokens per judge; with two mid-tier API judges this is tens of dollars, not hundreds. The 8×H100 alternative (a strong open judge, e.g. a large Qwen/DeepSeek — families disjoint from Gemma/Mistral) is the fallback if API budget is a problem.

## 6. Judges and aggregation

- **≥2 judges from families disjoint from ALL generators.** Generators are Gemma (12B, E4B, likely the new 31B) and Ministral → **no Google and no Mistral judges**. Use one OpenAI + one Anthropic mid-tier model (exact snapshot ids recorded in the artifact); optionally a third open-weights judge (Qwen-family) on the H100s for tie-breaks.
- **Consensus rule:** a system-pair item counts as a win only when the order-consistent verdicts of the judges agree; judge-disagreement → tie. Report per-judge results too (they should rank systems identically; if not, that's a red flag to investigate before trusting either).
- **Aggregation:** Bradley–Terry over all pairings (ties split; `both_useless` excluded from BT but reported as its own rate — it is an absolute-quality signal). **Bootstrap CIs clustered by document** (narrativeqa has ~29 queries per doc). Report per-subset + macro, and **win rates within span-length-ratio bins** (A much longer / similar / B much longer) — if the winner only wins when longer, that's the verbosity bias, not quality.

## 7. Syncing with YOUR preference (the human anchor)

The LLM-judge result is only reportable anchored to human judgment (`evaluation-metrics-assessment.md` §1.3). Three phases:

**Phase 1 — pilot, ~30 items (you, ~30 min).** I generate 30 stratified items rendered exactly as §2 and give them to you (visualizer annotation mode once it exists; a plain markdown file with one item per section works immediately). For each item you give a verdict and — this is the important part — **a one-line reason in your own words**. I then (a) compare your verdicts to the draft judge's, (b) distill your reasons into concrete rule adjustments in the §4 prompt (e.g., how you weigh few-clean-spans vs broad-coverage — that trade-off is the main calibration knob and I cannot guess it for you), and (c) freeze the prompt.

**Phase 2 — validation, ~250 items (you + ideally 1 colleague).** Stratified like §5, ≥20% double-annotated between the two humans. Deliverable numbers: instance-level Cohen's κ and raw % agreement of judge-vs-human-majority, side by side with human–human agreement on the doubly annotated part (the acceptance bar is judge≈human-human, per Thomas/Zheng), plus sign agreement on the system pairs. κ ≥ ~0.5 with matching pair signs → the full-set LLM result is reportable with this study as its anchor.

**Phase 3 — full run.** Judges run the §5 sample; results reported per §6 with the Phase-2 anchor.

### What you should tell me (annotation guideline + feedback format)

When you annotate, apply the §1 priority order and record, per item: `item_id`, `verdict` (A / B / tie / both_useless), and optionally a reason. The reasons that help calibration most are the ones naming a **trade-off**, e.g.:

- "A because B's extra spans are just query-word mentions" (support > coverage),
- "B despite duplicates — A missed the actual answer sentence" (coverage > minimality),
- "tie — both found it, A drags a whole paragraph" (how much does minimality matter to you?),
- "both_useless — spans are from the right scene but nothing explains the WHY of the query".

Concrete signals I need from you at least once during the pilot: (1) fewer-but-cleaner vs more-but-noisier — which do you want the dataset to prefer? (2) do duplicates/fragments actually bother you or only extreme cases? (3) does a span need to be convincing standalone, or is "correct region of the document" enough? (4) any subset-specific rule (e.g., for qmsum summary-queries, is scattered partial coverage acceptable?). Your answers become literal prompt lines; disagreements you flag after the pilot become regression items I re-test every prompt revision against.

Feedback format: whatever is easiest — the visualizer's annotation JSONL (`{item_id, verdict, reason}` per line) or a plain list in chat ("item 7: A — ..."). I take either.

## 8. Implementation plan (files, when built)

1. `eval/judge_items.py` — builds the rendered items (JSONL: `item_id`, `pair`, `subset`, `qid`, `doc_id`, `order`, `rendered_text`, span stats, Jaccard) from two extraction files + sampling config. The renderer is the single source of truth reused for judge calls, the pilot markdown, and the visualizer annotation mode.
2. `eval/judge_run.py` — sends items to the configured judge APIs (both orders), validates the JSON verdicts, retries malformed outputs once, writes `data/eval/judge_verdicts.jsonl`; resumable.
3. `eval/judge_aggregate.py` — consistency filtering, consensus, BT fit, doc-clustered bootstrap, length-bin table → `data/eval/judge_preference.json` (feeds `eval/summary_table.py`, which already reserves the column).
4. Visualizer annotation mode (from `human-preference-evaluation.md` §1) reusing the §2 renderer for the human phases.

Open inputs needed from you before Phase 3: which two judge APIs/keys to use, and the per-pair sample size budget (default 2,000) — everything else is decided above.
