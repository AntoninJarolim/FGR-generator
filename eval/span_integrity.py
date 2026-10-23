"""Check the integrity and span quality of a constrained+reasoning run.

Two questions, both of which a run can fail silently:

1. **Did the grammar actually hold?** The whole promise of the constrained arm
   is that every span is a verbatim contiguous substring of its own passage, so
   span locating is solved by construction and evaluation reduces to span
   quality. Anything below 100% EM means decoding was not really constrained --
   a reasoning parser that does not match the model leaves the bitmask unfilled
   and decoding degrades to free with nothing in the log. Reported per span and
   per row, with offenders printed rather than just a percentage.

   EM is measured AFTER stripping the reasoning block, using llm_extraction's
   own helpers, so the boundary matches what the pipeline parses. Measuring it
   on the raw output instead harvests spans the chain of thought was merely
   rehearsing and reports a spurious failure.

2. **Is the model selecting only PART of the passage?** The prompt asks for
   minimal phrase- or clause-level spans, not whole sentences. On short passages
   that instruction is under real pressure, so the coverage ratio
   (span chars / passage chars) is reported as a distribution with the
   whole-passage and empty rates called out. A distribution piled up near 1.0
   means the model is returning the passage rather than extracting from it.

Chain-of-thought lengths and the share of samples that hit
--thinking_token_budget come along for free from the same pass; a binding budget
is worth knowing about, though it is harmless as long as those samples still
produce spans.

Reads the RAW batch files (which retain the verbatim output including the
reasoning block) and the extracted jsonl.

Usage:
  MODE=ms-marco-xml-constrained-reasoning
  RUN=google~gemma-4-31B-it_train
  python eval/span_integrity.py \
      --generated-dir data/generated/$MODE/$RUN \
      --extracted data/extracted_relevancy/$MODE/$RUN.jsonl \
      --tokenizer google/gemma-4-31B-it \
      --thinking-budget 3072
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_extraction import (REASONING_DELIMITERS, get_sorted_generation_files,
                            process_output, strip_reasoning)
from jsonlines import jsonlines

# Grammar-level cap in VLLMGenerator; sample counts at the cap mean the run is
# hitting it rather than choosing to stop.
MAX_SPANS = 32


def split_reasoning(text):
    """``(chain_of_thought, answer)`` for one raw output.

    The answer comes from ``strip_reasoning`` so it is byte-identical to what the
    pipeline parses spans from; the CoT is the text the same delimiters enclose.
    An output with no reasoning block yields an empty CoT.
    """
    for open_tag, close_tag in REASONING_DELIMITERS:
        if close_tag not in text:
            continue
        close_at = text.rindex(close_tag)
        start = text.index(open_tag) + len(open_tag) if open_tag in text else 0
        return text[start:close_at], strip_reasoning(text)
    return "", text


def describe(name, values, unit="", precision=1, low_percentiles=False):
    """One line of percentiles, or a clear note when there is nothing to report.

    ``precision`` matters for the coverage ratio: at 1 decimal place LongEmbed's
    genuinely tiny ratios all render as "0.0", which reads as a broken metric
    rather than a small one. ``low_percentiles`` reports p10/p25 instead of
    p95/p99, since for coverage the interesting tail is the LOW one (spans that
    cover almost nothing) as well as the high one.
    """
    if not len(values):
        print(f"  {name:<26} (no data)")
        return
    arr = np.asarray(values, dtype=float)
    w = 8
    if low_percentiles:
        p10, p50, p90, p99 = np.percentile(arr, [10, 50, 90, 99])
        print(f"  {name:<26} p10={p10:>{w}.{precision}f}  p50={p50:>{w}.{precision}f}  "
              f"p90={p90:>{w}.{precision}f}  p99={p99:>{w}.{precision}f}  "
              f"mean={arr.mean():>{w}.{precision}f} {unit}")
        return
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    print(f"  {name:<26} p50={p50:>{w}.{precision}f}  p90={p90:>{w}.{precision}f}  "
          f"p95={p95:>{w}.{precision}f}  p99={p99:>{w}.{precision}f}  "
          f"max={arr.max():>{w}.{precision}f} {unit}")


def token_counter(tokenizer_name):
    """Token-length function, or a whitespace fallback if the tokenizer is absent.

    CoT budgets are expressed in model tokens, so the real tokenizer matters;
    falling back is still better than refusing to report anything.
    """
    if not tokenizer_name:
        return (lambda t: len(t.split())), "whitespace words"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        return (lambda t: len(tok(t, add_special_tokens=False)["input_ids"])), \
               f"{tokenizer_name} tokens"
    except Exception as e:
        print(f"WARNING: could not load tokenizer {tokenizer_name} ({e}); "
              f"falling back to whitespace word counts.", file=sys.stderr)
        return (lambda t: len(t.split())), "whitespace words (tokenizer unavailable)"


def read_raw_outputs(generated_dir):
    """``{row_id: raw_output_text}`` across every batch file in the run dir."""
    responses = {}
    for path in get_sorted_generation_files(generated_dir):
        responses.update(process_output(path))
    return responses


def analyze_generation(generated_dir, count_tokens, thinking_budget):
    responses = read_raw_outputs(generated_dir)
    if not responses:
        raise SystemExit(f"No *_output.jsonl batch files in {generated_dir}")

    cot_tokens, answer_tokens = [], []
    n_with_reasoning = n_no_spans_tag = 0
    for text in responses.values():
        cot, answer = split_reasoning(text)
        if cot:
            n_with_reasoning += 1
        cot_tokens.append(count_tokens(cot))
        answer_tokens.append(count_tokens(answer))
        if not answer.lstrip().startswith("<spans>"):
            n_no_spans_tag += 1

    n = len(responses)
    print(f"\n=== Generation ({n} samples from {generated_dir}) ===")
    print(f"  reasoning block present   {n_with_reasoning}/{n} "
          f"({100.0 * n_with_reasoning / n:.1f}%)")
    # Under --constrained the pipeline aborts on any such sample, so a non-zero
    # count here means this pilot was NOT a constrained run.
    print(f"  answer without <spans>    {n_no_spans_tag}/{n}")
    describe("chain-of-thought length", cot_tokens)
    describe("answer length", answer_tokens)

    if thinking_budget:
        # Within 1% of the budget: vLLM force-closed the CoT rather than the
        # model choosing to stop, so the budget is binding for these samples.
        at_budget = sum(1 for t in cot_tokens if t >= 0.99 * thinking_budget)
        print(f"  CoT at the {thinking_budget}-token budget: {at_budget}/{n} "
              f"({100.0 * at_budget / n:.1f}%) -- these were cut off, not finished")


def analyze_spans(extracted_path, psg_key, count_tokens):
    ratios, span_counts, passage_tokens = [], [], []
    n_empty = n_whole = n_error = 0
    # EM: the whole promise of the constrained arm is that every span is a
    # verbatim contiguous substring of its OWN passage, so span locating is
    # solved by construction and evaluation reduces to span quality. Anything
    # below 100% means the grammar did not actually gate the output -- report the
    # offenders rather than a bare percentage.
    n_spans = n_verbatim = n_bad_rows = 0
    offenders = []

    with jsonlines.open(extracted_path) as reader:
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No rows in {extracted_path}")

    for row in rows:
        if row.get('extraction_error'):
            n_error += 1
        passage = row.get(psg_key) or ""
        spans = row.get('selected_spans') or []
        passage_tokens.append(count_tokens(passage))
        span_counts.append(len(spans))
        row_ok = True
        for span in spans:
            n_spans += 1
            if span in passage:
                n_verbatim += 1
            else:
                row_ok = False
                if len(offenders) < 5:
                    offenders.append((row.get('qid'), span, passage))
        if not row_ok:
            n_bad_rows += 1
        if not passage:
            continue
        ratio = sum(len(s) for s in spans) / len(passage)
        ratios.append(ratio)
        if not spans:
            n_empty += 1
        elif ratio > 0.95:
            n_whole += 1

    n = len(rows)
    print(f"\n=== Spans ({n} rows from {extracted_path}) ===")
    # Two denominators, because they are easy to confuse: a row contributes as
    # many spans as the model emitted, so the span total is the larger number and
    # is NOT a count of query-passage pairs.
    em = 100.0 * n_verbatim / n_spans if n_spans else float('nan')
    row_em = 100.0 * (n - n_bad_rows) / n if n else float('nan')
    verdict = "OK" if n_verbatim == n_spans else "*** GRAMMAR DID NOT HOLD ***"
    print(f"  EM per SPAN : {n_verbatim}/{n_spans} spans = {em:.2f}%   {verdict}")
    print(f"  EM per ROW  : {n - n_bad_rows}/{n} rows  = {row_em:.2f}%   "
          f"(row passes iff all its spans are verbatim)")
    print(f"  spans per row (mean)  : {n_spans / n:.2f}" if n else "")
    for qid, span, passage in offenders:
        print(f"    qid={qid} span={span!r}")
        print(f"      passage={passage[:160]!r}")
    describe("passage length", passage_tokens)
    describe("spans per sample", span_counts)
    describe("coverage ratio", ratios, "(span chars / passage chars)",
             precision=4, low_percentiles=True)
    print(f"  whole passage (>0.95)     {n_whole}/{n} ({100.0 * n_whole / n:.1f}%)"
          f"   <- the prompt asks for PART of the passage")
    print(f"  no spans at all           {n_empty}/{n} ({100.0 * n_empty / n:.1f}%)")
    print(f"  at the {MAX_SPANS}-span grammar cap  "
          f"{sum(1 for c in span_counts if c >= MAX_SPANS)}/{n}")
    if n_error:
        print(f"  marked extraction_error   {n_error}/{n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generated-dir', required=True,
                    help="Raw batch-file dir for the pilot run (under data/generated/).")
    ap.add_argument('--extracted', required=True,
                    help="Extracted jsonl for the same run.")
    ap.add_argument('--tokenizer', default=None,
                    help="HF tokenizer for token counts (use the generating model). "
                         "Omitted: whitespace word counts, which are NOT comparable "
                         "to the token budgets.")
    ap.add_argument('--thinking-budget', type=int, default=None,
                    help="The --thinking_token_budget the pilot ran with, to report "
                         "how often it was binding.")
    ap.add_argument('--psg-key', default='passage',
                    help="Passage field in the extracted rows (default: passage).")
    args = ap.parse_args()

    count_tokens, unit = token_counter(args.tokenizer)
    print(f"Token counts in: {unit}")

    analyze_generation(args.generated_dir, count_tokens, args.thinking_budget)
    analyze_spans(args.extracted, args.psg_key, count_tokens)


if __name__ == '__main__':
    main()
