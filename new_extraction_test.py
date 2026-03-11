import copy
import functools
import argparse
import os.path
import time
from symtable import Class

import torch
import json
import tqdm
from jinja2 import Template
from sympy import true
from torch import newaxis, Tensor
from find_BLO import TokenByteFinder
from llm_runner import LLMRunner, get_model_config
from utils.text_utils import find_span, extract_span, remove_special_token, read_template, get_token_span, \
    find_first_token, find_last_token
from utils.squad_dataset import load_or_create_gt_data
from utils.other import TimedList


def create_prompt(template, **kwargs):
    start_span_tokens = kwargs.pop("start_span_tokens")
    end_span_tokens = kwargs.pop("end_span_tokens")

    if type(template) == str:
        text_to_check = template
    elif type(template) == Template:
        text_to_check = template.template_text
    else:
        raise TypeError("Template must be str or Template")

    # Remove all special tokens
    remove_special_token(text_to_check, *start_span_tokens, *end_span_tokens, error_on_detection=True)

    # Add all examples to the kwargs template
    for i, (start_t, end_t) in enumerate(zip(start_span_tokens, end_span_tokens)):
        suffix = "" if i == 0 else f"_{i}"
        kwargs[f"start_span_token{suffix}"] = start_t
        kwargs[f"end_span_token{suffix}"] = end_t
    return template.render(**kwargs)


def remove_special_tokens(data, start_span_tokens, end_span_tokens):
    data_removed_special = []
    all_tokens = [*start_span_tokens, *end_span_tokens]
    for record in tqdm.tqdm(data, desc="Preprocessing data"):
        data_removed_special.append(
            {
                "question": remove_special_token(record["question"], *all_tokens),
                "answer": [
                    remove_special_token(a, *all_tokens)
                    for a in record["answer"]
                ],
                "context": remove_special_token(record["context"], *all_tokens),
                "id": record["id"]
            }
        )
    return data_removed_special


def find_max_pos(logits, insert_token_ids):
    # logits is Batch, Time, Vocab

    # Max over tokens that could generate the token we care about
    selected_logits = torch.max(logits[:, :, insert_token_ids], dim=-1).values

    # get the position
    pos = torch.argmax(selected_logits, dim=-1).item()
    return pos


def lowest_index_greater_zero(logits, insert_token_ids,
                              plot_label=None, gt_range=None, start_index=0, return_id=False, mask=None):
    # logits is Batch, Time, Vocab

    if mask is not None:
        # No constrains (masking) of last logit, as it is predicting just after sequence - where everything is allowed
        mask = mask.to(logits.device)
        logits[:, :-1, insert_token_ids] = logits[:, :-1, insert_token_ids].masked_fill(~mask, float('-inf'))

    # Batch, Time, Vocab
    greedy_next = torch.argmax(logits, dim=-1, keepdim=False).to("cpu")[..., newaxis]
    insert_tag_ids = torch.tensor(insert_token_ids)[newaxis, newaxis, :]
    in_generating = torch.eq(greedy_next, insert_tag_ids).any(dim=-1)
    if not torch.any(in_generating):
        return -1
    # argmax selects FIRST true
    first_greater = torch.argmax(in_generating.type(torch.int), dim=-1).item()

    show_it = False
    if show_it:
        mask = torch.zeros(logits.size(-1)).type(torch.bool)
        mask.scatter_(dim=-1, index=torch.tensor(insert_token_ids), value=1)

        max_tag = torch.amax(logits[..., mask], dim=-1)
        max_other = torch.amax(logits[..., ~mask], dim=-1)

        logits_ = max_tag - max_other

        plot_logits_at_positions(plot_label, first_greater, logits_, gt_range, start_at=start_index)

    if return_id:
        return first_greater, greedy_next[:, first_greater]

    return first_greater


def plot_logits_at_positions(plot_label, pos, selected_logits, gt_range, start_at):
    values = selected_logits[0].cpu()
    import matplotlib.pyplot as plt

    x = list(range(start_at, start_at + len(values)))

    plt.figure(figsize=(8, 4))
    plt.plot(x, values, marker='o', label="Values")

    plt.plot(pos + start_at, values[pos], 'ro', markersize=6, label="First >0")

    plt.axvspan(gt_range[0], gt_range[1] - 1, color='green', alpha=0.3, label="GT span")

    # correct limits
    plt.xlim(0, start_at + len(values) - 1)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Values vs. Index for {plot_label}")
    plt.grid(True)
    plt.legend()
    plt.show()


def append_tokens_batched(x, append_toks):
    if isinstance(append_toks, list):
        append_toks = torch.tensor(append_toks)
    return torch.cat(
        [
            x,
            append_toks.to(x.device).repeat(x.size(0), 1)
        ],
        dim=1
    )


# -----------------------------
# Loop functions
# -----------------------------


def generate_standard_answers(model, data, start_span_tokens, end_span_tokens, template):
    assert len(start_span_tokens) == 1 and len(end_span_tokens) == 1, (
        "generate_standard_answers supports at most one tag pair"
    )
    start_span_token = start_span_tokens[0]
    end_span_token = end_span_tokens[0]

    results = TimedList()

    tokens_finder = TokenByteFinder(model.tokenizer)
    start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_token)
    end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_token)

    for record in tqdm.tqdm(data, desc="Standard generation"):
        prompt = create_prompt(
            template,
            start_span_tokens=[start_span_token],
            end_span_tokens=[end_span_token],
            question=record["question"],
            context=record["context"]
        )

        logits, generated, generated_ids = model.tokenize_run(prompt)
        answer, start_ch, end_ch = extract_span(
            start_span_token,
            end_span_token,
            generated,
            return_start_end=True
        )

        if start_ch == -1 or end_ch == -1:
            start_tok = None
            end_tok = None
        else:
            generated_before = generated[:start_ch]
            generated_after = generated[end_ch:]

            start_tok = find_first_token(generated_before, generated_ids, model.tokenizer)
            end_tok = find_last_token(generated_after, generated_ids, model.tokenizer)

            if generated_ids[start_tok] not in start_span_tokens_id:
                print(f"Warning generated_ids[start_tok] {generated_ids[start_tok]}")
                print(f"\tcontext: generated_ids[start_tok-2:start_tok+2] {generated_ids[start_tok - 2:start_tok + 2]}")

            if generated_ids[end_tok] not in end_span_tokens_id:
                print(f"Warning generated_ids[end_tok] {generated_ids[end_tok]}")
                print(f"\tcontext: generated_ids[end_tok-2:end_tok+2] {generated_ids[end_tok - 2:end_tok + 2]}")

        # print_topk_logits_decoded(
        #     tokens_finder,
        #     model.tokenizer,
        #     logits,
        #     start_span_tokens_id=start_span_tokens_id
        # )

        results.append(
            {
                **record,
                "prediction": answer,
                "raw_output": generated,
                "ctx_enc": generated_ids,
                "start": start_tok,
                "end": end_tok,
            }
        )
    return results


def generate_parallel_answers(model, data, template, start_span_tokens, end_span_tokens, one_char=True):
    assert len(start_span_tokens) == 1 and len(end_span_tokens) == 1, (
        "generate_parallel_answers supports at most one tag pair"
    )
    end_span_token = end_span_tokens[0]
    start_span_token = start_span_tokens[0]

    start_span_str = start_span_token[0]
    end_span_str = end_span_token[0]

    results = TimedList()

    if one_char:
        start_span_tokens_id = model.tokenize_char(start_span_str)
        end_span_tokens_id = model.tokenize_char(end_span_str)
    else:
        tokens_finder = TokenByteFinder(model.tokenizer)
        start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_str)
        end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_str)

    data = remove_special_tokens(data, start_span_token, end_span_token)

    for record in tqdm.tqdm(data, desc="Parallel generation"):
        context = record['context']
        question = record['question']

        prompt = create_prompt(
            template,
            start_span_tokens=start_span_token,
            end_span_tokens=end_span_token,
            question=question,
            context=context
        )

        logits, ctx_enc = model.encode_ctx(prompt, context)
        start = find_max_pos(logits, start_span_tokens_id)
        start_context, end_context = model.split_context_by_token_pos(context, start)
        start_context = start_context + start_span_str

        if end_context == "":
            annotated = start_context + start_span_str + end_span_str
            end = None
            pass
        else:
            prompt_ctx = prompt + start_context
            logits, _ = model.encode_ctx(prompt_ctx, end_context)
            end = find_max_pos(logits, end_span_tokens_id)
            end_start, end_end = model.split_context_by_token_pos(end_context, end)
            annotated = start_context + end_start + end_span_str + end_end

        answer = extract_span(start_span_str, end_span_str, annotated)
        results.append(
            {
                **record,
                "prediction": answer,
                "raw_output": annotated,
                "ctx_enc": ctx_enc,
                "start": start,
                "end": end
            }
        )
    return results


def generate_parallel_answers_diff(model, data, template, start_span_tokens, end_span_tokens):
    assert len(start_span_tokens) == 1 and len(end_span_tokens) == 1, (
        "generate_parallel_answers_diff supports at most one tag pair"
    )
    end_span_token = end_span_tokens[0]
    start_span_token = start_span_tokens[0]

    start_span_str = start_span_token[0]
    end_span_str = end_span_token[0]

    tokens_finder = TokenByteFinder(model.tokenizer)
    start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_str)
    end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_str)

    data = remove_special_tokens(data, start_span_token, end_span_token)

    results = TimedList()
    for record in tqdm.tqdm(data, desc="Parallel diff generation"):
        context = record['context']
        question = record['question']
        gt_span_pos = get_token_span(model.tokenizer, context, find_span(context, record['answer'][0]))

        prompt = create_prompt(
            template,
            start_span_tokens=start_span_token,
            end_span_tokens=end_span_token,
            question=question,
            context=context
        )

        logits, ctx_enc = model.encode_ctx(prompt, context)
        start = lowest_index_greater_zero(logits, start_span_tokens_id, plot_label="Start", gt_range=gt_span_pos)
        start_context, end_context = model.split_context_by_token_pos(context, start)

        # print_topk_logits_decoded(
        #     tokens_finder,
        #     model.tokenizer,
        #     logits,
        #     start_span_tokens_id=start_span_tokens_id,
        # )

        prompt_ctx = prompt + start_context + " " + start_span_str
        if end_context == "":
            annotated = start_context + start_span_str + end_span_str
            end = None
            pass
        else:
            logits, _ = model.encode_ctx(prompt_ctx, end_context)
            end = lowest_index_greater_zero(logits, end_span_tokens_id, plot_label="end", gt_range=gt_span_pos,
                                            start_index=start)
            end_start, end_end = model.split_context_by_token_pos(end_context, end)
            annotated = start_context + " " + start_span_str + end_start + end_span_str + end_end

        answer = extract_span(start_span_str, end_span_str, annotated)

        results.append(
            {
                **record,
                "prediction": answer,
                "raw_output": annotated,
                "ctx_enc": ctx_enc,
                "start": start,
                "end": end,
            }
        )
    return results


def argmax_logit_id(logits, next_valid):
    mask = torch.full_like(logits, float("-inf"))
    mask[:, :, next_valid] = logits[:, :, next_valid]
    next_span_token_id = mask.argmax(dim=-1)
    return next_span_token_id


def generate_parallel_answers_tokens_only(model, data, template, start_span_tokens, end_span_tokens):
    assert len(start_span_tokens) == 1 and len(end_span_tokens) == 1, (
        "generate_parallel_answers_tokens_only supports at most one tag pair"
    )
    end_span_token = end_span_tokens[0]
    start_span_token = start_span_tokens[0]

    start_span_str = start_span_token[0]
    end_span_str = end_span_token[0]

    tokens_finder = TokenByteFinder(model.tokenizer)
    start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_str)
    end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_str)

    start_span_token_id = tokens_finder.get_single_token(start_span_str)
    end_span_token_id = tokens_finder.get_single_token(end_span_str)

    start_id_next_valid_f = tokens_finder.get_valid_next_func(start_span_str)
    end_id_next_valid_f = tokens_finder.get_valid_next_func(end_span_str)

    data = remove_special_tokens(data, start_span_token, end_span_token)

    results = TimedList()
    for record in tqdm.tqdm(data, desc="Parallel diff generation"):
        context = record['context']
        question = record['question']
        gt_span_pos = get_token_span(model.tokenizer, context, find_span(context, record['answer'][0]))

        prompt = create_prompt(
            template,
            start_span_tokens=start_span_token,
            end_span_tokens=end_span_token,
            question=question,
            context=context
        )

        logits, ctx_enc = model.encode_ctx(prompt, context)
        start = lowest_index_greater_zero(
            logits,
            start_span_tokens_id,
            plot_label="Start",
            gt_range=gt_span_pos
        )

        # Last position selected or nothing selected
        selected_end = (start == ctx_enc.size(1)) or (start == -1)

        if selected_end:
            answer = ""
            annotated = context + start_span_str + end_span_str
            annotated_tokens = ctx_enc[0].tolist()
            end = start + 1
        else:
            start_span_token_id = logits[:, start].argmax()
            before = append_tokens_batched(ctx_enc[:, :start], start_span_token_id)
            while True:
                next_valid = start_id_next_valid_f(before[0])
                if not next_valid:
                    # next_valid returns empty set if the target string is generated
                    break
                if len(next_valid) == 1:
                    # No need to perform forward, if only one token allowed
                    before = append_tokens_batched(before, next_valid)
                else:
                    # Required to make forward to distinguish what to generate
                    logits, _ = model.encode_ctx(prompt, context=None, before=before)
                    next_span_token_id = argmax_logit_id(logits, next_valid)
                    before = append_tokens_batched(before, next_span_token_id)

            assert "�" not in model.tokenizer.decode(ctx_enc[:, start:][0])
            assert "�" not in model.tokenizer.decode(ctx_enc[:, :start][0])

            # <start> tag is generated at this point; now catch back with middle
            first_middle = ctx_enc[:, start].item()
            next_valid = tokens_finder.get_generating_tokens(first_middle)

            logits, _ = model.encode_ctx(prompt, context=None, before=before)
            catch_tok = argmax_logit_id(logits, next_valid)
            before = append_tokens_batched(before, catch_tok)

            tmp_after = model.tokenizer.decode(ctx_enc[:, start:][0]).lstrip()
            tmp_after_catch = tmp_after.removeprefix(model.tokenizer.decode(catch_tok[0]))

            logits, ctx_enc = model.encode_ctx(prompt, context=tmp_after_catch, before=before)
            end = lowest_index_greater_zero(
                logits,
                end_span_tokens_id,
                plot_label="End", gt_range=gt_span_pos, start_index=start
            )

            # Note that middle is missing catch_tok at the start, but it's okay cuz last 'before' contains it
            middle = ctx_enc[:, :end]
            after = ctx_enc[:, end:]

            annotated_tokens = torch.cat([
                before,
                append_tokens_batched(middle, end_span_token_id),
                after
            ],
                dim=1
            )[0].tolist()

            annotated = model.tokenizer.decode(annotated_tokens)
            answer = extract_span(start_span_str, end_span_str, annotated)

        # print_topk_logits_decoded(
        #     tokens_finder,
        #     model.tokenizer,
        #     logits,
        #     start_span_tokens_id=start_span_tokens_id,
        # )

        results.append(
            {
                **record,
                "prediction": answer,
                "raw_output": annotated,
                "ctx_enc": annotated_tokens,
                "start": start,
                "end": end,
            }
        )
    return results


def verify_tokens_cmp(verify_logits, verify_tokens):
    """
    Verifies how many tokens are verified by comparing greedy next and tokens to be verified.
    Returns number of verified tokens.
    """
    if verify_tokens.size(1) == 0:
        # Nothing to verify
        return 0
    greedy_next = torch.argmax(verify_logits, dim=-1, keepdim=False)
    not_eq = ~greedy_next[0].eq(verify_tokens.to(greedy_next.device))
    if not not_eq.any():
        return verify_tokens.size(1)
    first_negative = torch.argmax(not_eq.to(torch.int)).item()
    return first_negative


def build_mask(completing_bytes, sgts_stripped) -> Tensor:
    mask = torch.empty((len(completing_bytes), len(sgts_stripped)), dtype=torch.bool)
    for i, c in enumerate(completing_bytes):
        for j, sgt_stripped in enumerate(sgts_stripped):
            if sgt_stripped.startswith(c):
                x = sgt_stripped[len(c):]
                # Only allowed if nothing remains or if only space remains
                is_space_or_empty = (x == b"" or x == b" ")
                mask[i, j] = is_space_or_empty
            else:
                mask[i, j] = False
    return mask


def create_allowed_mask(tokens_finder, force_tokens, next_tags):
    """
    Filters allowed_tokens for each time position by eliminating tokens, that do not share prefix.
    Returns only a mask (time, allowed_tokens) that should be interpreted as mask for allowed_tokens at each position.
    """

    sgts = [
        [x for x in tokens_finder.get_generating_tokens(tag_string)]
        for tag_string in next_tags
    ][0]

    sgts_stripped = [
        tokens_finder.get_stripped_sgts(next_tag)
        for next_tag in next_tags
    ][0]

    completing_bytes = tokens_finder.return_token_completing_bytes(force_tokens[0].tolist())

    if any(len(x) > 0 for x in completing_bytes):
        # There is some continuation in the seq - todo: check good behaviour
        pass

    # Create another dimension for batch
    mask = build_mask(completing_bytes, sgts_stripped)[newaxis]
    return sgts, mask, completing_bytes


def forward_verify_predict(model, tokens_finder, static_prefix, verify_tokens, force_tokens, next_tags):
    """
    :param static_prefix: tokens that were already generated, only for context
    :param verify_tokens: rest of tokens to generate valid tag
    :param force_tokens: teacher forced tokens - proposed by tokenizer
    :next_tags: finds position to insert one of these tags (only if all tokens are verified)
    returns list of verified + forced tokens + one next

    Never inserts tag to a position that would generate invalid string.
    """
    # Todo-batch: this needs to verified per-batch somehow
    nr_tokens_verify = verify_tokens.size(1)

    # One forward pass
    logits = model.encode_ctx_verify(static_prefix, force_tokens, verify_tokens=verify_tokens)

    #  Verify guessed tokens
    verify_logits, logits = logits[:, :nr_tokens_verify], logits[:, nr_tokens_verify:]
    nr_verified = verify_tokens_cmp(verify_logits, verify_tokens)

    # Not all tokens are verified
    if nr_verified < nr_tokens_verify:
        greedy_next_id = verify_logits[nr_verified].argmax(dim=-1).item()
        verify_tokens = verify_tokens[:nr_verified]
        forced = torch.empty(1, 0).to(torch.int)
        forced_bytes = b""
    else:
        sgts_tag, mask_allow_at, completing_bytes = create_allowed_mask(tokens_finder, force_tokens, next_tags)
        tag_position, greedy_next_id = lowest_index_greater_zero(
            logits,
            sgts_tag,
            mask=mask_allow_at,
            return_id=True
        )

        forced = force_tokens[:, :tag_position]
        forced_bytes = tokens_finder.return_token_bytes(forced[0], as_bytes=True) + completing_bytes[tag_position]

    return {
        'verified': verify_tokens,
        'forced': forced,
        'forced_bytes': forced_bytes,
        'greedy': greedy_next_id,
    }


class Tag:
    def __init__(self, str_seq: str, tokens_finder: TokenByteFinder):
        self.str = str_seq
        self.all_paths = tokens_finder.find_all_paths(str_seq)
        self.all_paths_tokens = [[token_id for token_id, t_bytes in path] for path in self.all_paths]

    def __str__(self):
        return self.str

    def __getitem__(self, item):
        return self.all_paths_tokens[item]

class Tags:
    def __init__(self, tags_string:list[str], tokens_finder):
        self.tags = [Tag(s, tokens_finder) for s in tags_string]

    def to_str(self):
        return [str(tag) for tag in self.tags]

def guess_from_last(tags: Tags, generated_tokens):
    if not generated_tokens:
        return None
    return str(tags[0])


def generate_verify_guess_parallel(model, data, template, start_span_tokens, end_span_tokens):

    tokens_finder = TokenByteFinder(model.tokenizer)

    opening_tags = Tags(start_span_tokens, tokens_finder)
    closing_tags = Tags(end_span_tokens, tokens_finder)

    batch_size = 1
    results = TimedList()
    for record in tqdm.tqdm(data, desc="Parallel diff generation"):
        context = record['context']
        question = record['question']

        prefix = create_prompt(
            template,
            start_span_tokens=opening_tags.to_str(),
            end_span_tokens=closing_tags.to_str(),
            question=question,
            context=context
        )
        prefix = model.tokenizer(prefix, return_tensors="pt")["input_ids"]

        generate_bytes = context.encode("utf-8")
        generated_tokens = []

        current_tags = opening_tags  # Opening tags that can be generated
        next_tags = closing_tags  # Closing tags that can be generated

        while generate_bytes:
            # Tags that are valid trough this tag generation, will shrink when first token of tag is generated
            possible_tags = copy.deepcopy(current_tags)

            while True:
                # todo: this must always select correct closing tag
                guessed_tag = guess_from_last(possible_tags, generated_tokens)

                # verify guessed_tag
                verify_tokens = (torch.empty(batch_size, 0).to(torch.int)
                                 if guessed_tag is None else
                                 model.tokenizer.tokenize(guessed_tag))
                force_tokens = model.tokenizer(
                    generate_bytes.decode("utf-8"),
                    add_special_tokens=False, return_tensors="pt"
                )["input_ids"]

                out = forward_verify_predict(
                    model,
                    tokens_finder,
                    prefix,  # todo + generated tokens
                    verify_tokens,
                    force_tokens,
                    next_tags.to_str()
                )

                assert generate_bytes.startswith(out['forced_bytes'])
                generate_bytes = generate_bytes.removeprefix(out['forced_bytes'])

                # always concat verified tokens (can be empty)
                # always concat forced tokens -- is empty when not all tokens are verified

                completed_tag = False  # todo: match end of generated_tokens to detect generated tokens / verified token
                if completed_tag:
                    current_tags.pop(completed_tag)  # todo: remove confirmed tag from list
                    break

            # swap current and next
            current_tags, next_tags = next_tags, current_tags

        raw_output = model.tokenizer.decode(generated_tokens)
        answer = extract_span(opening_tags[0], closing_tags[0], raw_output)  # todo: multi-span extraction

        results.append(
            {
                **record,
                "prediction": answer,
                "raw_output": raw_output,
                "ctx_enc": generated_tokens,
            }
        )

        # todo: adjust output for multi-annotated spans
        # todo: adjust all other algs. to follow this output
        # todo: regenerate to match outputs


def parse_args():
    parser = argparse.ArgumentParser(description="LLM extractive QA processing.")

    parser.add_argument("--min_ans_size", type=int, default=5, help="Minimum answer word count.")
    parser.add_argument("--max_dataset_size", type=int, default=None,
                        help="Maximum dataset size to process.")

    parser.add_argument("--model", type=str, default="google/gemma-7b-it", help="HF model id to use.")
    parser.add_argument("--start_span_tokens", type=str, nargs="*", default=["⟦"],
                        help="List of opening (start) span tokens. Default: ⟦")
    parser.add_argument("--end_span_tokens", type=str, nargs="*", default=["⟧"],
                        help="List of closing (end) span tokens. Default: ⟧")

    parser.add_argument("--extracted_data_path", type=str, required=True,
                        help="Path to the directory to save extracted data.")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template.")
    parser.add_argument("--method", type=str,
                        choices=['standard', 'standard_custom_decode',
                                 'parallel', 'parallel_multiple', 'parallel_multiple_diff', 'parallel_tokens_only',
                                 'verify_guess_parallel',
                                 'all'],
                        default='all',
                        help="Which method to run and save.")

    args = parser.parse_args()

    # Opening and closing tags lists must have same len
    assert len(args.end_span_tokens) == len(args.start_span_tokens), \
        f"Invalid arguments: {args.end_span_tokens} does not match {args.start_span_tokens} length"

    return args


def safe_run_generation(method_name, generation_func, **kwargs):
    results = []
    error_message = None
    try:
        results = generation_func(**kwargs)
    except Exception as e:
        print(f"Error running '{method_name}': {e}")
        error_message = str(e)
        import traceback
        traceback.print_exc()
    return results, error_message


def run_and_save_results(method_name, generation_func, output_dir, **kwargs):
    """Runs a generation function and saves the results to a JSON file."""
    print(f"Running '{method_name}' method...")

    start_time = time.time()
    results, error_message = safe_run_generation(method_name, generation_func, **kwargs)
    elapsed_time = time.time() - start_time

    model = kwargs.get("model")
    tokenizer_info, gen_config = get_model_config(model)

    output_data = {
        "parameters": {
            "start_span_tokens": kwargs.get("start_span_tokens"),
            "end_span_tokens": kwargs.get("end_span_tokens"),
            "tokenizer": tokenizer_info,
            "generation_config": gen_config,
        },
        "results": sorted(results, key=lambda r: r["id"]) if results else [],
        "generation_stats": {
            "generation_time_sec": elapsed_time
        },
        "error_message": error_message
    }

    output_path = os.path.join(output_dir, f"{method_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Saved {method_name} method results to {output_path}")


def main():
    args = parse_args()

    args.extracted_data_path = os.path.join(args.extracted_data_path, args.model.strip("/").split("/")[-1])

    template = read_template(args.template_path)
    model = LLMRunner(args.model)

    data = load_or_create_gt_data(
        args.extracted_data_path,
        args.min_ans_size,
        args.max_dataset_size
    )

    generation_methods = {
        "standard": functools.partial(generate_standard_answers),
        "parallel": functools.partial(generate_parallel_answers, one_char=True),
        "parallel_multiple": functools.partial(generate_parallel_answers, one_char=False),
        "parallel_multiple_diff": functools.partial(generate_parallel_answers_diff),
        "parallel_tokens_only": functools.partial(generate_parallel_answers_tokens_only),
        "verify_guess_parallel": functools.partial(generate_verify_guess_parallel)
    }

    common_params = {
        "model": model,
        "data": data,
        "template": template,
        "start_span_tokens": args.start_span_tokens,
        "end_span_tokens": args.end_span_tokens,
    }

    for method_name in generation_methods:
        if args.method != "all" and args.method != method_name:
            continue

        assert method_name in generation_methods, f"'{method_name}' method is not implemented."

        run_and_save_results(
            method_name,
            generation_methods[method_name],
            args.extracted_data_path,
            **common_params
        )


if __name__ == "__main__":
    main()
