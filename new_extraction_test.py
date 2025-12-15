import argparse
import os.path
import shutil
import sys
from collections import namedtuple
from curses import start_color

import torch
import json

import tqdm
from jinja2 import Template

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

Record = namedtuple("Record", ["question", "answer", "context"])


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

        records.append(Record(question=question, answer=answers, context=context))

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


def extract_span(start_token, end_token, generated_text):
    """Extract text between start and end tokens."""
    start_idx = generated_text.find(start_token)
    end_idx = generated_text.find(end_token, start_idx + 1)
    if start_idx != -1 and end_idx != -1:
        return generated_text[start_idx + len(start_token):end_idx]
    return ""


def encode_with_llm(template, context):
    """Return logits for each position (placeholder)."""
    # For now, random tensor for demonstration
    return torch.rand(len(context))


def max_insert(logits, insert_token_id):
    # In real use, scatter logits and return argmax
    selected_logits = logits[:, :, insert_token_id]
    pos = torch.argmax(logits[:, :, insert_token_id], dim=-1).item()
    return pos


def ensure_created_directory(path: str):
    if os.path.isdir(path):
        print(f"WARNING: Directory '{path}' already exists.")
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Loop functions
# -----------------------------


class LLMRunner:
    def __init__(self, model_name, device=None, max_new_tokens=1024, do_sample=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )

        self.model.eval()
        torch.set_grad_enabled(False)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    def tokenize_char(self, character):
        assert len(character) == 1

        tokenized = self.tokenizer(character, return_tensors="pt", add_special_tokens=False)
        assert len(tokenized["input_ids"][0]) == 1, "This character is not tokenized as one token."

        return tokenized["input_ids"][0][0]

    def tokenize_run(self, prompt: str) -> str:
        """
        Tokenizes the prompt, runs the model autoregressively, and returns decoded text.
        """
        # Encode the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate output
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample
        )

        # Decode to string
        input_size = len(inputs['input_ids'][0])
        generated_text = self.tokenizer.decode(outputs[0][input_size:], skip_special_tokens=True)
        return generated_text

    def encode_ctx(self, template, context) -> str:
        # tokenize context alone (only for length, no tensors needed)
        ctx = self.tokenizer(context, add_special_tokens=False, return_length=True)
        context_len = ctx["length"][0]

        full_text = template + context
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        # Generate output
        outputs = self.model(**inputs)

        return outputs['logits'][:, -context_len:]

    def split_context_by_tokens(self, text: str, start_context: int):
        # Tokenize without special tokens
        ctx = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        )

        # Get token ids (1D tensor)
        token_ids = ctx["input_ids"][0]

        # Split tokens
        left_tokens = token_ids[:start_context]
        right_tokens = token_ids[start_context:]

        # Decode back to text
        left_text = self.tokenizer.decode(left_tokens, skip_special_tokens=True)
        right_text = self.tokenizer.decode(right_tokens, skip_special_tokens=True)

        return left_text, right_text


def generate_standard_answers(model, data, start_span_token, end_span_token, template):
    results = []
    for question, _, context in tqdm.tqdm(data):
        prompt = create_prompt(
            template,
            start_span_token=start_span_token,
            end_span_token=end_span_token,
            question=question,
            context=context
        )

        outputs = model.tokenize_run(prompt)

        answer = extract_span(start_span_token, end_span_token, outputs)
        results.append(((question, context), answer))
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

def generate_parallel_answers(data, template, model, start_span_token, end_span_token):
    results = []
    start_span_token_id = model.tokenize_char(start_span_token)
    end_span_token_id = model.tokenize_char(end_span_token)

    data = remove_special_tokens(data, end_span_token, start_span_token)

    for question, _, context in tqdm.tqdm(data):
        prompt = create_prompt(
            template,
            start_span_token=start_span_token,
            end_span_token=end_span_token,
            question=question,
            context=context
        )

        logits = model.encode_ctx(prompt, context)
        start = max_insert(logits, start_span_token_id)
        start_context, end_context = model.split_context_by_tokens(context, start)
        start_context = start_context + start_span_token

        prompt_ctx = prompt + start_context
        logits = model.encode_ctx(prompt_ctx, end_context)
        end = max_insert(logits, end_span_token_id)
        end_start, end_end = model.split_context_by_tokens(end_context, end)
        annotated = start_context + end_start + end_span_token + end_end

        answer = extract_span(start_span_token, end_span_token, annotated)
        results.append(((question, context), answer))
    return results


def remove_special_tokens(data, end_span_token, start_span_token):
    data_removed_special = []
    for question, answers, context in tqdm.tqdm(data):
        data_removed_special.append(
            (
                remove_special_token(question, start_span_token, end_span_token),
                [
                    remove_special_token(a, start_span_token, end_span_token)
                    for a in answers
                ],
                remove_special_token(context, start_span_token, end_span_token)
            )
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
    parser.add_argument("--method", type=str, choices=['standard', 'parallel', 'all'], default='all',
                        help="Which method to run and save.")
    return parser.parse_args()


def main():
    args = parse_args()

    template = read_template(args.template_path)
    model = LLMRunner(args.model)

    # Load filtered dataset
    data = long_ans_squad(args.min_ans_size, args.max_dataset_size)

    # Create output directory
    ensure_created_directory(args.extracted_data_path)

    # Save ground truth
    gt_data = [(q, c, a) for q, a, c in data]
    gt_path = os.path.join(args.extracted_data_path, "gt.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=4)
    print(f"Saved ground truth data to {gt_path}")

    if args.method == 'standard' or args.method == 'all':
        standard_results = generate_standard_answers(
            template=template,
            model=model,
            data=data,
            start_span_token=args.start_span_token,
            end_span_token=args.end_span_token
        )
        standard_path = os.path.join(args.extracted_data_path, "standard.json")
        with open(standard_path, "w", encoding="utf-8") as f:
            json.dump(standard_results, f, ensure_ascii=False, indent=4)
        print(f"Saved standard method results to {standard_path}")

    if args.method == 'parallel' or args.method == 'all':
        parallel_results = generate_parallel_answers(
            data,
            template,
            model=model,
            start_span_token=args.start_span_token,
            end_span_token=args.end_span_token
        )
        parallel_path = os.path.join(args.extracted_data_path, "parallel.json")
        with open(parallel_path, "w", encoding="utf-8") as f:
            json.dump(parallel_results, f, ensure_ascii=False, indent=4)
        print(f"Saved parallel method results to {parallel_path}")


if __name__ == "__main__":
    main()
