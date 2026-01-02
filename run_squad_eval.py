import json
import argparse
import os
from eval.eval_squad import f1_score, exact_match_score, metric_max_over_ground_truths


def evaluate(predictions, ground_truths_map):
    total_f1 = 0
    total_em = 0

    if not predictions:
        return 0.0, 0.0

    for pred_item in predictions:
        key = (pred_item["question"], pred_item["context"])
        prediction_text = pred_item["prediction"]
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
        
    ground_truths_map = {(item["question"], item["context"]): item["answers"] for item in gt_data}

    # Find and evaluate all prediction files in the input directory
    prediction_files = [f for f in os.listdir(args.input_dir) if f.endswith('.json') and f != 'gt.json']

    if not prediction_files:
        print("\nNo prediction files (ending in .json, excluding gt.json) found to evaluate.")

    for filename in sorted(prediction_files):  # sorted for consistent order
        method_name = os.path.splitext(filename)[0]
        file_path = os.path.join(args.input_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)

            print(f"\nEvaluating '{method_name}' method from '{filename}'...")
            if not predictions:
                print("  File is empty or contains no predictions. Skipping.")
                continue

            f1, em = evaluate(predictions, ground_truths_map)
            print(f"  Average F1 Score: {f1:.4f}")
            print(f"  Average Exact Match Score: {em:.4f}")

        except FileNotFoundError:
            # This case is unlikely due to os.listdir, but good practice
            print(f"\nError: File not found at {file_path}. Skipping.")
        except json.JSONDecodeError:
            print(f"\nError: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"\nAn unexpected error occurred with {file_path}: {e}")


if __name__ == "__main__":
    main()
