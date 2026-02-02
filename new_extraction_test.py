import functools
import argparse
import os.path
import time
import torch
import json
import tqdm
from jinja2 import Template
from torch import newaxis
from find_BLO import TokenByteFinder
from llm_runner import LLMRunner, get_model_config
from utils.text_utils import find_span, extract_span, remove_special_token, read_template, get_token_span, \
    find_first_token, find_last_token
from utils.squad_dataset import load_or_create_gt_data
from utils.other import TimedList


def create_prompt(template, **kwargs):
    start_span_token = kwargs.get("start_span_token")
    end_span_token = kwargs.get("end_span_token")

    if type(template) == str:
        text_to_check = template
    elif type(template) == Template:
        text_to_check = template.template_text
    else:
        raise TypeError("Template must be str or Template")

    remove_special_token(text_to_check, start_span_token, end_span_token, error_on_detection=True)
    return template.render(**kwargs)


def remove_special_tokens(data, end_span_token, start_span_token):
    data_removed_special = []
    for record in tqdm.tqdm(data, desc="Preprocessing data"):
        data_removed_special.append(
            {
                "question": remove_special_token(record["question"], start_span_token, end_span_token),
                "answer": [
                    remove_special_token(a, start_span_token, end_span_token)
                    for a in record["answer"]
                ],
                "context": remove_special_token(record["context"], start_span_token, end_span_token),
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


def lowest_index_greater_zero(logits, insert_token_ids, plot_label=None, gt_range=None, start_index=0):
    # logits is Batch, Time, Vocab

    # Batch, Time, Vocab
    greedy_next = torch.argmax(logits, dim=-1, keepdim=False).to("cpu")[..., newaxis]
    insert_tag_ids = torch.tensor(insert_token_ids)[newaxis, newaxis, :]
    in_generating = torch.eq(greedy_next, insert_tag_ids).any(dim=-1)
    first_greater = torch.argmax(in_generating.type(torch.int), dim=-1).item()

    show_it = False
    if show_it:
        mask = torch.zeros(logits.size(-1)).type(torch.bool)
        mask.scatter_(dim=-1, index=torch.tensor(insert_token_ids), value=1)

        max_tag = torch.amax(logits[..., mask], dim=-1)
        max_other = torch.amax(logits[..., ~mask], dim=-1)

        logits_ = max_tag - max_other

        plot_logits_at_positions(plot_label, first_greater, logits_, gt_range, start_at=start_index)

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


def append_one_token_batched(x, one_token):
    return torch.cat(
        [
            x,
            one_token.repeat(x.size(0))[:, None]
        ],
        dim=1
    )

# -----------------------------
# Loop functions
# -----------------------------


def generate_standard_answers(model, data, start_span_token, end_span_token, template):
    results = TimedList()

    tokens_finder = TokenByteFinder(model.tokenizer)
    start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_token)
    end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_token)

    for record in tqdm.tqdm(data, desc="Standard generation"):
        prompt = create_prompt(
            template,
            start_span_token=start_span_token,
            end_span_token=end_span_token,
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
                print(f"\tcontext: generated_ids[start_tok-2:start_tok+2] {generated_ids[start_tok-2:start_tok+2]}")

            if generated_ids[end_tok] not in end_span_tokens_id:
                print(f"Warning generated_ids[end_tok] {generated_ids[end_tok]}")
                print(f"\tcontext: generated_ids[end_tok-2:end_tok+2] {generated_ids[end_tok-2:end_tok+2]}")

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


def generate_parallel_answers(model, data, template, start_span_token, end_span_token, one_char=True):
    start_span_str = start_span_token
    end_span_str = end_span_token

    results = TimedList()

    if one_char:
        start_span_tokens_id = model.tokenize_char(start_span_str)
        end_span_tokens_id = model.tokenize_char(end_span_str)
    else:
        tokens_finder = TokenByteFinder(model.tokenizer)
        start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_str)
        end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_str)

    data = remove_special_tokens(data, end_span_str, start_span_str)

    for record in tqdm.tqdm(data, desc="Parallel generation"):
        context = record['context']
        question = record['question']

        prompt = create_prompt(
            template,
            start_span_token=start_span_str,
            end_span_token=end_span_str,
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


def generate_parallel_answers_diff(model, data, template, start_span_token, end_span_token):
    start_span_str = start_span_token
    end_span_str = end_span_token

    tokens_finder = TokenByteFinder(model.tokenizer)
    start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_str)
    end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_str)

    data = remove_special_tokens(data, end_span_str, start_span_str)

    results = TimedList()
    for record in tqdm.tqdm(data, desc="Parallel diff generation"):
        context = record['context']
        question = record['question']
        gt_span_pos = get_token_span(model.tokenizer, context, find_span(context, record['answer'][0]))

        prompt = create_prompt(
            template,
            start_span_token=start_span_str,
            end_span_token=end_span_str,
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


def generate_parallel_answers_tokens_only(model, data, template, start_span_token, end_span_token):
    start_span_str = start_span_token
    end_span_str = end_span_token

    tokens_finder = TokenByteFinder(model.tokenizer)
    start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_str)
    end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_str)

    start_span_token_id = tokens_finder.get_singe_token(start_span_str)
    end_span_token_id = tokens_finder.get_singe_token(end_span_str)

    start_span_token_id = torch.tensor([start_span_token_id], dtype=torch.int)
    end_span_token_id = torch.tensor([end_span_token_id], dtype=torch.int)

    data = remove_special_tokens(data, end_span_str, start_span_str)

    results = TimedList()
    for record in tqdm.tqdm(data, desc="Parallel diff generation"):
        context = record['context']
        question = record['question']
        gt_span_pos = get_token_span(model.tokenizer, context, find_span(context, record['answer'][0]))

        prompt = create_prompt(
            template,
            start_span_token=start_span_str,
            end_span_token=end_span_str,
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

        before = append_one_token_batched(ctx_enc[:, :start], start_span_token_id)
        tmp_after = ctx_enc[:, start:]

        logits, ctx_enc = model.encode_ctx(prompt, context=tmp_after, before=before)
        end = lowest_index_greater_zero(
            logits,
            end_span_tokens_id,
            plot_label="End", gt_range=gt_span_pos, start_index=start
        )

        middle = ctx_enc[:, :end]
        after = ctx_enc[:, end:]

        annotated_tokens = torch.cat([
                before,
                append_one_token_batched(middle, end_span_token_id),
                after
            ],
            dim=1
        )[0].tolist()

        annotated = model.tokenizer.decode(annotated_tokens)
        answer = model.tokenizer.decode(middle[0])

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


def parse_args():
    parser = argparse.ArgumentParser(description="LLM extractive QA processing.")

    parser.add_argument("--min_ans_size", type=int, default=5, help="Minimum answer word count.")
    parser.add_argument("--max_dataset_size", type=int, default=None,
                        help="Maximum dataset size to process.")

    parser.add_argument("--model", type=str, default="google/gemma-7b-it", help="HF model id to use.")
    parser.add_argument("--start_span_token", type=str, default="⟦", help="Start span token.")
    parser.add_argument("--end_span_token", type=str, default="⟧", help="End span token.")

    parser.add_argument("--extracted_data_path", type=str, required=True,
                        help="Path to the directory to save extracted data.")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template.")
    parser.add_argument("--method", type=str,
                        choices=['standard', 'standard_custom_decode',
                                 'parallel', 'parallel_multiple', 'parallel_multiple_diff', 'parallel_tokens_only',
                                 'all'],
                        default='all',
                        help="Which method to run and save.")
    return parser.parse_args()


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
            "start_span_token": kwargs.get("start_span_token"),
            "end_span_token": kwargs.get("end_span_token"),
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
    }
    
    common_params = {
        "model": model,
        "data": data,
        "template": template,
        "start_span_token": args.start_span_token,
        "end_span_token": args.end_span_token,
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
