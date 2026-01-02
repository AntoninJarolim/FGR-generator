import argparse
import os.path

import torch
import json

import tqdm
from jinja2 import Template

from datasets import load_dataset

from find_BLO import TokenByteFinder
from llm_runner import LLMRunner


def long_ans_squad(min_ans_size, max_dataset_size):
    """
    Load SQuAD dataset and filter answers longer than min_ans_size words.
    Return at most max_dataset_size examples.
    """
    dataset = load_dataset("squad", split="validation")

    records = []
    for i, example in enumerate(dataset):
        context = example["context"]
        question = example["question"]
        answers = example["answers"]["text"]  # this is a list

        if any(len(a.split()) < min_ans_size for a in answers):
            continue

        records.append({"question": question, "answer": answers, "context": context})

        # Stop if reached max_dataset_size
        if max_dataset_size and len(records) >= max_dataset_size:
            break

    return records


def remove_special_token(text, *args, error_on_detection=False):
    for special_token in args:
        if special_token in text:
            if error_on_detection:
                raise AssertionError(f"Special token '{special_token}' cannot in text \n{text}")

            print(f"Warning: Removing special token '{special_token}' from text present in: \n{text}")
            text = text.replace(special_token, "")
    return text


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


def extract_span(start_str, end_str, generated_text):
    """Extract text between start and end tokens."""
    start_idx = generated_text.find(start_str)
    end_idx = generated_text.find(end_str, start_idx + 1)
    if start_idx != -1 and end_idx != -1:
        return generated_text[start_idx + len(start_str):end_idx]
    return ""


def encode_with_llm(template, context):
    """Return logits for each position (placeholder)."""
    # For now, random tensor for demonstration
    return torch.rand(len(context))


def find_max_pos(logits, insert_token_ids):
    # logits is Batch, Time, Vocab

    # Max over tokens that could generate the token we care about
    selected_logits = torch.max(logits[:, :, insert_token_ids], dim=-1).values

    # get the position
    pos = torch.argmax(selected_logits, dim=-1).item()
    return pos


def ensure_created_directory(path: str):
    if os.path.isdir(path):
        print(f"WARNING: Directory '{path}' already exists.")
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Loop functions
# -----------------------------


def generate_standard_answers(model, data, start_span_token, end_span_token, template):
    results = []
    for record in tqdm.tqdm(data):
        prompt = create_prompt(
            template,
            start_span_token=start_span_token,
            end_span_token=end_span_token,
            question=record["question"],
            context=record["context"]
        )

        generated = model.tokenize_run(prompt)
        answer = extract_span(start_span_token, end_span_token, generated)

        results.append(
            {
                "question": record["question"],
                "context": record["context"],
                "prediction": answer,
                "raw_output": generated,
            }
        )
    return results


def generate_standard_answers_custom_decode(model, data, start_span_token, end_span_token, template):
    results = []
    for record in tqdm.tqdm(data):
        prompt = create_prompt(
            template,
            start_span_token=start_span_token,
            end_span_token=end_span_token,
            question=record["question"],
            context=record["context"]
        )

        generated = model.tokenize_run_custom(
            prompt,
            targets_text=record['context'],
            special=start_span_token,
        )
        answer = extract_span(start_span_token, end_span_token, generated)

        results.append(
            {
                "question": record["question"],
                "context": record["context"],
                "prediction": answer,
                "raw_output": generated,
            }
        )
    return results


def read_template(path):
    with open(path, 'r') as f:
        template_text = f.read()
    template = Template(template_text)
    template.template_text = template_text
    return template


# -----------------------------
# Main function
# -----------------------------

def generate_parallel_answers(model, data, template, start_span_token, end_span_token, one_char=True):
    start_span_str = start_span_token
    end_span_str = end_span_token

    results = []

    if one_char:
        start_span_tokens_id = model.tokenize_char(start_span_str)
        end_span_tokens_id = model.tokenize_char(end_span_str)
    else:
        tokens_finder = TokenByteFinder(model.tokenizer)
        start_span_tokens_id = tokens_finder.get_generating_tokens(start_span_str)
        end_span_tokens_id = tokens_finder.get_generating_tokens(end_span_str)

    data = remove_special_tokens(data, end_span_str, start_span_str)

    for record in tqdm.tqdm(data):
        context = record['context']
        question = record['question']

        prompt = create_prompt(
            template,
            start_span_token=start_span_str,
            end_span_token=end_span_str,
            question=question,
            context=context
        )

        logits = model.encode_ctx(prompt, context)
        start = find_max_pos(logits, start_span_tokens_id)
        start_context, end_context = model.split_context_by_token_pos(context, start)
        start_context = start_context + start_span_str

        prompt_ctx = prompt + start_context
        logits = model.encode_ctx(prompt_ctx, end_context)
        end = find_max_pos(logits, end_span_tokens_id)
        end_start, end_end = model.split_context_by_token_pos(end_context, end)
        annotated = start_context + end_start + end_span_str + end_end

        answer = extract_span(start_span_str, end_span_str, annotated)
        results.append(
            {
                "question": question,
                "context": context,
                "prediction": answer,
                "raw_output": annotated,
            }
        )
    return results


def remove_special_tokens(data, end_span_token, start_span_token):
    data_removed_special = []
    for record in tqdm.tqdm(data):
        data_removed_special.append(
            {
                "question": remove_special_token(record["question"], start_span_token, end_span_token),
                "answer": [
                    remove_special_token(a, start_span_token, end_span_token)
                    for a in record["answer"]
                ],
                "context": remove_special_token(record["context"], start_span_token, end_span_token)
            }
        )
    return data_removed_special


def parse_args():
    parser = argparse.ArgumentParser(description="LLM extractive QA processing.")

    parser.add_argument("--min_ans_size", type=int, default=5, help="Minimum answer word count.")
    parser.add_argument("--max_dataset_size", type=int, default=300, help="Maximum dataset size to process.")

    parser.add_argument("--model", type=str, default="google/gemma-7b-it", help="HF model id to use.")
    parser.add_argument("--start_span_token", type=str, default="⟦", help="Start span token.")
    parser.add_argument("--end_span_token", type=str, default="⟧", help="End span token.")

    parser.add_argument("--extracted_data_path", type=str, required=True,
                        help="Path to the directory to save extracted data.")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template.")
    parser.add_argument("--method", type=str,
                        choices=['standard', 'parallel', 'standard_custom_decode', 'all'],
                        default='all',
                        help="Which method to run and save.")
    return parser.parse_args()


def run_and_save_results(method_name, generation_func, output_dir, **kwargs):
    """Runs a generation function and saves the results to a JSON file."""
    print(f"Running '{method_name}' method...")
    results = generation_func(**kwargs)
    output_path = os.path.join(output_dir, f"{method_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved {method_name} method results to {output_path}")


def main():
    args = parse_args()

    template = read_template(args.template_path)
    model = LLMRunner(args.model)

    # Load filtered dataset
    data = long_ans_squad(args.min_ans_size, args.max_dataset_size)

    # Create output directory
    ensure_created_directory(args.extracted_data_path)

    # Save ground truth
    gt_data = [{"question": r["question"], "context": r["context"], "answers": r["answer"]} for r in data]
    gt_path = os.path.join(args.extracted_data_path, "gt.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=4)
    print(f"Saved ground truth data to {gt_path}")

    generation_methods = {
        "standard": generate_standard_answers,
        "standard_custom_decode": generate_standard_answers_custom_decode,
        "parallel": generate_parallel_answers,
    }

    common_params = {
        "model": model,
        "data": data,
        "template": template,
        "start_span_token": args.start_span_token,
        "end_span_token": args.end_span_token,
    }

    methods_to_run = generation_methods.keys() if args.method == 'all' else [args.method]

    for method_name in methods_to_run:
        if method_name in generation_methods:
            run_and_save_results(
                method_name,
                generation_methods[method_name],
                args.extracted_data_path,
                **common_params
            )


if __name__ == "__main__":
    main()
