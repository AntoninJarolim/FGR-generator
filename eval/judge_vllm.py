"""Local/cluster judge: the same pairwise protocol, run through vLLM.

A second judge from a different model family is what turns the API judge's
verdicts from one model's opinion into a measurable agreement rate. gpt-oss is
family-disjoint from every system under comparison (GLM, Qwen, gemma), so it
carries no self-preference bias -- do NOT substitute a gemma or Qwen model here,
since three of the six systems are gemma-4-31B arms and one is Qwen.

Everything except the inference call is shared with eval/judge_preference.py --
the same views, system prompt, JSON schema, pairing, A/B randomisation and skip
rules -- so verdicts from the two judges are directly comparable and land in the
same aggregation. Two differences from the API path:

  * temperature=0 IS settable here (the Claude models reject sampling params),
    so decoding is actually deterministic;
  * there is no marginal cost per comparison, so the item pool is a superset of
    the API judge's. The overlap is what the agreement rate is computed on.

Output is written in the same record shape eval/judge_preference.py --stage
aggregate already reads, tagged by judge so the two can be separated.

    python eval/judge_vllm.py --model openai/gpt-oss-120b \\
        --items data/eval/judge_items_full.json --tag gptoss
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.judge_preference import (  # noqa: E402
    DEFAULT_SYSTEMS, DEFAULT_VIEW_DIR, DEFAULT_WORK, SCHEMA, SYSTEM_PROMPT,
    build_comparisons, load_views, user_prompt,
)


def build_llm(args):
    from vllm import LLM
    kwargs = {}
    if args.tensor_parallel_size:
        kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.reasoning_parser:
        # gpt-oss and other reasoning models interleave a thinking channel; the
        # parser keeps it out of the JSON that guided decoding is constraining.
        kwargs["reasoning_parser"] = args.reasoning_parser
    return LLM(model=args.model, dtype=args.dtype,
               gpu_memory_utilization=args.gpu_memory_utilization, **kwargs)


def sampling_params(args):
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        so = {"structured_outputs": StructuredOutputsParams(json=SCHEMA)}
    except ImportError:                      # older vLLM
        from vllm.sampling_params import GuidedDecodingParams
        so = {"guided_decoding": GuidedDecodingParams(json=SCHEMA)}
    return SamplingParams(temperature=0.0, max_tokens=args.max_tokens, **so)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True,
                    help="HF id of the judge; must NOT be from a family under "
                         "comparison (no gemma, no Qwen)")
    ap.add_argument("--items", default=os.path.join(REPO_ROOT, "data/eval/judge_items_full.json"))
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS)
    ap.add_argument("--view-dir", default=DEFAULT_VIEW_DIR)
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--tag", default="gptoss")
    ap.add_argument("--limit", type=int, default=0, help="cap comparisons (smoke test)")
    ap.add_argument("--seed", type=int, default=0,
                    help="MUST match the API run's seed so the A/B assignment "
                         "and therefore the comparison set are identical")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--reasoning-parser", default=None,
                    help="vLLM reasoning parser name, for models that emit a "
                         "thinking channel (e.g. gpt-oss)")
    args = ap.parse_args()

    systems = json.load(open(args.systems))
    items = [tuple(k) for k in json.load(open(args.items))]
    views = load_views(systems, args.view_dir)
    to_judge, resolved = build_comparisons(systems, items, views, seed=args.seed)
    print(f"{len(items)} items, {len(to_judge)} comparisons to judge, "
          f"{len(resolved)} resolved without inference")
    if args.limit:
        to_judge = to_judge[:args.limit]
        print(f"--limit -> {len(to_judge)}")

    llm, sp = build_llm(args), sampling_params(args)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt(c["query"], c["view_a"], c["view_b"])}],
            tokenize=False, add_generation_prompt=True)
        for c in to_judge
    ]

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"generated {len(outputs)} in {time.time() - t0:.0f}s")

    os.makedirs(args.work, exist_ok=True)
    out = os.path.join(args.work, f"{args.tag}.verdicts.jsonl")
    n_ok = n_bad = 0
    with open(out, "w", encoding="utf-8") as f:
        for c, o in zip(to_judge, outputs):
            meta = {k: c[k] for k in ("pair", "item", "system_a", "system_b", "swapped")}
            meta["judge"] = args.model
            try:
                v = json.loads(o.outputs[0].text)
                missing = set(SCHEMA["required"]) - set(v)
                if missing:
                    raise ValueError(f"missing fields {sorted(missing)}")
            except (json.JSONDecodeError, ValueError) as e:
                n_bad += 1
                f.write(json.dumps({**meta, "error": f"parse: {e}"}) + "\n")
                continue
            n_ok += 1

            def sysname(lbl):
                return meta["system_a"] if lbl == "a" else meta["system_b"]
            f.write(json.dumps({
                **meta, "raw": v,
                "plausible": {meta["system_a"]: v["plausible_a"],
                              meta["system_b"]: v["plausible_b"]},
                "minimality_winner": "tie" if v["minimality"] == "tie" else sysname(v["minimality"]),
                "overall_winner": "tie" if v["overall"] == "tie" else sysname(v["overall"]),
            }, ensure_ascii=False) + "\n")
    # Resolved-without-inference records are written by the API run for its own
    # item set; write this judge's own so its rates are computed over its pool.
    with open(os.path.join(args.work, f"{args.tag}.resolved.jsonl"), "w",
              encoding="utf-8") as f:
        for r in resolved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{n_ok} ok, {n_bad} unparseable -> {out}")


if __name__ == "__main__":
    main()
