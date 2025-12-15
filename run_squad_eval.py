import json
import argparse
import os
from eval.eval_squad import f1_score, exact_match_score, metric_max_over_ground_truths


def evaluate(predictions, ground_truths_map):
    total_f1 = 0
    total_em = 0

    if not predictions:
        return 0.0, 0.0

    for key_as_list, prediction_text in predictions:
        key = tuple(key_as_list)
        ground_truths = ground_truths_map.get(key)

        if ground_truths:
            total_f1 += metric_max_over_ground_truths(f1_score, prediction_text, ground_truths)
            total_em += metric_max_over_ground_truths(exact_match_score, prediction_text, ground_truths)

    avg_f1 = total_f1 / len(predictions) if predictions else 0
    avg_em = total_em / len(predictions) if predictions else 0

    return avg_f1, avg_em


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQuAD predictions.")
    parser.add_argument("input_dir", help="Path to the directory with prediction files.")
    args = parser.parse_args()

    # Load ground truth
    gt_path = os.path.join(args.input_dir, "gt.json")
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_path}")
        return
        
    ground_truths_map = {tuple(item[0:2]): item[2] for item in gt_data}

    # Evaluate standard method
    standard_path = os.path.join(args.input_dir, "standard.json")
    if os.path.exists(standard_path):
        with open(standard_path, 'r', encoding='utf-8') as f:
            standard_data = json.load(f)
        
        print("Evaluating 'standard' method...")
        std_f1, std_em = evaluate(standard_data, ground_truths_map)
        print(f"  Average F1 Score: {std_f1:.4f}")
        print(f"  Average Exact Match Score: {std_em:.4f}")
    else:
        print("Skipping 'standard' method: file not found.")

    # Evaluate parallel method
    parallel_path = os.path.join(args.input_dir, "parallel.json")
    if os.path.exists(parallel_path):
        with open(parallel_path, 'r', encoding='utf-8') as f:
            parallel_data = json.load(f)

        print("\nEvaluating 'parallel' method...")
        par_f1, par_em = evaluate(parallel_data, ground_truths_map)
        print(f"  Average F1 Score: {par_f1:.4f}")
        print(f"  Average Exact Match Score: {par_em:.4f}")
    else:
        print("\nSkipping 'parallel' method: file not found.")


if __name__ == "__main__":
    main()
