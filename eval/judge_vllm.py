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
        --items data/eval/judge_items_full.json --tag gptoss \\
        --reasoning-parser openai_gptoss --reasoning-effort low

The reasoning parser is what keeps the thinking channel out of the JSON that
guided decoding constrains; `answer_text` then strips the channel framing off
the completion. See karolina_run_judge_vllm.sh for the cluster wrapper.
"""
import argparse
import json
import os
import re
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
    # skip_special_tokens=False: on a reasoning model the JSON sits in the last
    # harmony channel, and the channel markers ARE special tokens. Stripping them
    # would glue the thinking text onto the JSON with nothing left to split on.
    return SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                          skip_special_tokens=False, **so)


#: gpt-oss answers in harmony channels: an `analysis` (thinking) channel, then
#: `final` carrying the answer. The reasoning parser makes guided decoding wait
#: for the final header, so everything after its LAST occurrence is the JSON.
#:
#: The header is NOT a fixed string -- the model optionally announces the
#: response format, giving `<|channel|>final <|constrain|>json<|message|>` as
#: well as the bare `<|channel|>final<|message|>`. Matching only the bare form
#: silently discards otherwise-valid verdicts, so accept anything between the
#: channel name and the message marker.
_FINAL_RE = re.compile(r"<\|channel\|>final\b.*?<\|message\|>", re.DOTALL)
_END_MARKERS = ("<|return|>", "<|end|>", "<|call|>")


def answer_text(text):
    """The JSON payload, with any reasoning channel and end markers removed.

    A no-op for plain models, which emit the JSON and nothing else.
    """
    starts = [m.end() for m in _FINAL_RE.finditer(text)]
    if starts:
        text = text[starts[-1]:]
    for m in _END_MARKERS:
        j = text.find(m)
        if j != -1:
            text = text[:j]
    return text.strip()


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
    # 600 was enough for the API judge, whose thinking was disabled. A reasoning
    # model spends most of its budget in the analysis channel before the JSON
    # starts, and a budget that runs out mid-thought yields no JSON at all.
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--reasoning-effort", default=None,
                    help="chat-template reasoning_effort (gpt-oss: low|medium|"
                         "high). 'low' is the closest analogue to the API "
                         "judge, which ran with thinking disabled")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--reasoning-parser", default=None,
                    help="vLLM reasoning parser name, for models that emit a "
                         "thinking channel (e.g. gpt-oss)")
    ap.add_argument("--swap", action="store_true",
                    help="invert every A/B assignment. Same comparisons, each "
                         "presented the other way round, so averaging a swapped "
                         "arm with an unswapped one cancels position bias")
    ap.add_argument("--repeats", type=int, default=1,
                    help="draw N independent samples in one process, avoiding a "
                         "weight reload per sample. Writes <tag>.rep<i>.verdicts.jsonl")
    args = ap.parse_args()

    systems = json.load(open(args.systems))
    items = [tuple(k) for k in json.load(open(args.items))]
    views = load_views(systems, args.view_dir)
    to_judge, resolved = build_comparisons(systems, items, views, seed=args.seed,
                                           swap=args.swap)
    print(f"{len(items)} items, {len(to_judge)} comparisons to judge, "
          f"{len(resolved)} resolved without inference"
          f"{' [SWAPPED A/B]' if args.swap else ''}")
    if args.limit:
        to_judge = to_judge[:args.limit]
        print(f"--limit -> {len(to_judge)}")

    llm, sp = build_llm(args), sampling_params(args)
    tok = llm.get_tokenizer()
    tmpl_kwargs = {}
    if args.reasoning_effort:
        tmpl_kwargs["reasoning_effort"] = args.reasoning_effort
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt(c["query"], c["view_a"], c["view_b"])}],
            tokenize=False, add_generation_prompt=True, **tmpl_kwargs)
        for c in to_judge
    ]

    os.makedirs(args.work, exist_ok=True)
    for rep in range(args.repeats):
        # One tag means one file, so a single-sample run keeps its old name and
        # anything already reading <tag>.verdicts.jsonl is unaffected.
        suffix = "" if args.repeats == 1 else f".rep{rep}"
        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        print(f"[rep {rep}] generated {len(outputs)} in {time.time() - t0:.0f}s")
        write_verdicts(args, to_judge, outputs,
                       os.path.join(args.work, f"{args.tag}{suffix}.verdicts.jsonl"))

    # Resolved-without-inference records are written by the API run for its own
    # item set; write this judge's own so its rates are computed over its pool.
    with open(os.path.join(args.work, f"{args.tag}.resolved.jsonl"), "w",
              encoding="utf-8") as f:
        for r in resolved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_verdicts(args, to_judge, outputs, out):
    n_ok = n_bad = n_trunc = 0
    with open(out, "w", encoding="utf-8") as f:
        for c, o in zip(to_judge, outputs):
            meta = {k: c[k] for k in ("pair", "item", "system_a", "system_b", "swapped")}
            meta["judge"] = args.model
            comp = o.outputs[0]
            try:
                v = json.loads(answer_text(comp.text))
                missing = set(SCHEMA["required"]) - set(v)
                if missing:
                    raise ValueError(f"missing fields {sorted(missing)}")
            except (json.JSONDecodeError, ValueError) as e:
                n_bad += 1
                # A hit --max-tokens is a different failure from malformed JSON:
                # it means the budget, not the parser, needs raising. Keep the
                # raw text so the two can be told apart without a re-run.
                truncated = comp.finish_reason == "length"
                n_trunc += truncated
                f.write(json.dumps({
                    **meta, "error": f"parse: {e}", "truncated": truncated,
                    "finish_reason": comp.finish_reason,
                    "n_output_tokens": len(comp.token_ids),
                    "raw_text": comp.text[-2000:],
                }, ensure_ascii=False) + "\n")
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
    print(f"{n_ok} ok, {n_bad} unparseable ({n_trunc} of them truncated at "
          f"--max-tokens {args.max_tokens}) -> {out}")


if __name__ == "__main__":
    main()
