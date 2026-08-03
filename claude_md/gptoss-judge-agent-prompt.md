# Agent brief: run the gpt-oss second judge on the cluster

Hand this to an agent with cluster access. Everything it needs is in the repo on
branch `format-vs-constraint-dirs`.

---

## What you are doing and why

We are comparing six span-extraction systems with a blind pairwise judge. An API
judge (`claude-opus-5`) has already judged **4,357 comparisons**. Your job is to
run a **second, independent judge** — `gpt-oss` — over a **superset** of the same
comparisons, so we can compute how much the two judges agree.

The second judge is not a cost-saving substitute for the first. Its whole purpose
is that it is a *different model family*, which is what makes an agreement rate
meaningful rather than circular.

**Do not substitute the judge model.** Three of the six systems under comparison
are `gemma-4-31B-it` arms and one is `Qwen3.6-27B`. A gemma or Qwen judge would be
scoring its own family's output and would carry self-preference bias, which is
exactly the confound this run exists to avoid. Use gpt-oss. If gpt-oss cannot be
made to run, stop and report back rather than swapping in another model.

---

## What to copy to the cluster

Only **27 MB**. Do **not** copy `data/extracted_relevancy/` — those are six 3.6 GB
files and they are not needed; the spans have already been rendered into views.

```
eval/judge_vllm.py
eval/judge_preference.py          # imports: prompt, schema, pairing, skip rules
eval/judge_view.py                # imports: EMPTY_VIEW, item_key
eval/answer_containment.py        # imported transitively
eval/heuristic_spans.py           # imported transitively
custom_utils/span_match.py        # imported transitively
eval/judge_systems.json
data/eval/judge_items_full.json   # 1200 items
data/eval/judge_views/*.jsonl     # 6 files, ~27 MB total, 1200 views each
```

## Environment

vLLM plus the repo's light deps (numpy, jsonlines). `karolina_setup_env.sh` in the
repo root builds the project conda env on IT4I; adapt for your cluster. No GPU is
needed for anything except the generation itself.

---

## Run it

### 1. Smoke test first — small model, tiny slice

Prove the plumbing before booking a large allocation. Any small instruct model
works here; this step is about the runner, prompt rendering and JSON parsing, not
about the verdicts, so the family-disjointness rule does not apply to it.

```bash
python eval/judge_vllm.py \
    --model mistralai/Ministral-3-14B-Instruct-2512 \
    --items data/eval/judge_items_full.json \
    --limit 20 --tag smoke
```

Then check `data/eval/judge_work/smoke.verdicts.jsonl`:

- 20 lines, **zero** carrying an `"error"` key;
- each has `plausible_a`, `plausible_b`, `minimality`, `overall` with values from
  the allowed sets (`yes`/`no`, and `a`/`b`/`tie`);
- `reasoning` is prose, with no stray XML or channel markers in it.

### 2. Full run

```bash
python eval/judge_vllm.py \
    --model openai/gpt-oss-120b \
    --items data/eval/judge_items_full.json \
    --tag gptoss \
    --tensor-parallel-size <#GPUs> \
    --reasoning-parser <parser>      # see the gpt-oss note below
```

**Workload:** 1200 items × 15 pairs = 18,000 comparisons, of which 1,754 resolve
without inference (both systems empty, or byte-identical selections), leaving
**16,246 to generate** — roughly **51 M input tokens** and ~125 output tokens each.
Size the allocation accordingly; it is an overnight job on a modest node.

---

## Invariants — breaking any of these silently invalidates the comparison

1. **`--seed 0`** (the default). The seed determines the A/B assignment for every
   comparison. A different seed produces a different assignment, and the two
   judges would no longer be answering the same questions.
2. **Do not rebuild the views.** Ship the `judge_views/*.jsonl` as they are. They
   encode exactly what the API judge saw; regenerating them risks a different
   rendering and makes the agreement rate meaningless.
3. **Do not edit `SYSTEM_PROMPT` or `SCHEMA`** in `judge_preference.py`. Both
   judges must be asked the identical question in the identical format.
4. **Do not change the item list.** `judge_items_full.json` (1200) is a strict
   superset of the API judge's 320, which is what makes the overlap computable —
   verified: all 4,357 API comparisons are contained in your pool.

---

## gpt-oss specifics — the one real technical risk

gpt-oss emits a separate reasoning channel. Guided JSON decoding constrains the
*answer*, and if the reasoning channel is not separated out you will get either
unparseable output or reasoning text leaking into the `reasoning` field.

Handle it in this order:

1. Pass the appropriate vLLM `--reasoning-parser` for gpt-oss (check your vLLM
   version's supported parser names — do not guess; `vllm serve --help` or the
   docs list them).
2. If parsing still fails, the smoke test will show it as `"error": "parse: ..."`
   records. Report what the raw output looks like rather than hand-patching the
   schema.
3. `temperature=0` is set by the runner. Unlike the Claude judge (whose model
   generation rejects sampling parameters outright), this one is genuinely
   deterministic — a re-run should reproduce byte-identical verdicts, which is a
   useful check if something looks off.

---

## What to report back

Bring back **`data/eval/judge_work/gptoss.verdicts.jsonl`** and
**`gptoss.resolved.jsonl`**, plus:

- counts: comparisons generated, parsed OK, unparseable;
- the **tie rate** on `overall` (the API judge's was **1.8%** — a wildly different
  number means the model is not following the "use tie sparingly" instruction and
  the verdicts need scrutiny before use);
- the **absolute plausibility rate per system** (a quick group-by on the
  `plausible` field) — for reference the API judge's pilot gave lexical ≈ 24% and
  the model arms 78–100%, so a judge that rates lexical near the models is not
  discriminating and that is worth flagging;
- wall-clock and any OOM/truncation warnings.

Do **not** run the aggregation — that happens back on the main machine, where
both judges' verdicts are combined into the win matrix, the agreement rate and
the Bradley–Terry fit.
