import argparse
import os.path
from collections import namedtuple

import torch
import json
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


def create_prompt(question, context):
    """Create the prompt for the LLM (placeholder)."""
    return f"Question: {question}\nContext: {context}\nLabeled Context:"


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


def max_insert(logits, insert_token):
    """
    Returns the position where the insert_token should be inserted.
    Scatter logic placeholder.
    """
    # In real use, scatter logits and return argmax
    pos = torch.argmax(logits).item()
    return pos


def update_prompt(template, start_context):
    """Update the template with partially labeled context."""
    return template.replace("Context:", f"Context: {start_context}")


# -----------------------------
# Loop functions
# -----------------------------


class LLMRunner:
    def __init__(self, model_name, device=None, max_new_tokens=1024, do_sample=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

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
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def generate_standard_answers(model, data, start_span_token, end_span_token, template):
    results = []
    for question, _, context in data:

        prompt = template.render(
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
        template = Template(f.read())
    return template


# -----------------------------
# Main function
# -----------------------------

def generate_parallel_answers(data, start_span_token, end_span_token):
    results = []
    for _, _, context in data:
        template = create_prompt("", context)

        logits = encode_with_llm(template, context)
        start = max_insert(logits, start_span_token)
        start_context = context[:start] + start_span_token
        end_context = context[start:]

        prompt = update_prompt(template, start_context)
        logits = encode_with_llm(prompt, end_context)
        end = max_insert(logits, end_span_token)

        answer = context[start:end]
        results.append((context, answer))
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="LLM extractive QA processing.")

    parser.add_argument("--min_ans_size", type=int, default=5, help="Minimum answer word count.")
    parser.add_argument("--max_dataset_size", type=int, default=300, help="Maximum dataset size to process.")

    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="HF model id to use.")
    parser.add_argument("--start_span_token", type=str, default="⟦", help="Start span token.")
    parser.add_argument("--end_span_token", type=str, default="⟧", help="End span token.")

    parser.add_argument("--extracted_data_path", type=str, required=True, help="Path to save extracted data.")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template.")
    return parser.parse_args()


def main():
    args = parse_args()

    template = read_template(args.template_path)
    model = LLMRunner(args.model)

    # Load filtered dataset
    data = long_ans_squad(args.min_ans_size, args.max_dataset_size)

    data_out = {"gt": [], "standard": [], "parallel": []}

    # Fill ground truth
    for question, answer, context in data:
        data_out["gt"].append((question, context, answer))

    # Standard LLM generation
    data_out["standard"] = generate_standard_answers(
        template=template,
        model=model,
        data=data,
        start_span_token=args.start_span_token,
        end_span_token=args.end_span_token
    )

    # Parallel span selection
    data_out["parallel"] = generate_parallel_answers(
        data,
        start_span_token=args.start_span_token,
        end_span_token=args.end_span_token
    )


    if not os.path.exists(args.extracted_data_path):
        os.makedirs(args.extracted_data_path)

    with open(args.extracted_data_path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)

    print(f"Saved extracted data to {args.extracted_data_path}")


if __name__ == "__main__":
    main()
