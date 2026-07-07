import os
import json
import uuid
from datasets import load_dataset

def ensure_created_directory(path: str):
    if os.path.isdir(path):
        print(f"WARNING: Directory '{path}' already exists.")
    os.makedirs(path, exist_ok=True)

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

def load_or_create_gt_data(output_dir, min_ans_size, max_dataset_size):
    """
    Loads ground truth data if it exists and has IDs, otherwise creates it.
    Returns the dataset in the format required for generation functions.
    """
    gt_path = os.path.join(output_dir, "gt.json")
    ensure_created_directory(output_dir)

    regenerate = False
    if os.path.exists(gt_path):
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            if not isinstance(gt_data, list) or not gt_data or 'id' not in gt_data[0]:
                print("Found existing ground truth file, but it's missing IDs or is invalid. Regenerating...")
                regenerate = True
        except (json.JSONDecodeError, IndexError):
            print("Found corrupted or empty ground truth file. Regenerating...")
            regenerate = True
    else:
        # File doesn't exist, so we need to generate it
        regenerate = True

    if regenerate:
        print("Creating new ground truth data with unique IDs...")
        squad_data = long_ans_squad(min_ans_size, max_dataset_size)

        data = []
        for record in squad_data:
            record['id'] = uuid.uuid4().hex
            data.append(record)

        gt_data = [{"id": r["id"], "question": r["question"], "context": r["context"], "answers": r["answer"]} for r in
                   data]
        gt_data.sort(key=lambda x: x['id'])  # Sort by ID for consistent order
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=4)
        print(f"Saved new ground truth data ({len(gt_data)} samples) with IDs to {gt_path}")
    else:
        print(f"Ground truth file '{gt_path}' with IDs already exists. Loading it.")
        print("Ignoring dataset creation arguments (--min_ans_size, --max_dataset_size).")
        # gt_data is already loaded from the check above.
        # Re-create the 'data' structure as it's used by generation functions
        data = [{"id": r["id"], "question": r["question"], "context": r["context"], "answer": r["answers"]} for r in
                gt_data]

    return data
