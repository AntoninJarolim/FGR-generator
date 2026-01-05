import json
import argparse
import os
from eval.eval_squad import f1_score, exact_match_score, metric_max_over_ground_truths


def evaluate(predictions, ground_truths_map):
    """
    Evaluates predictions and returns per-datapoint F1 and EM scores.
    """
    results = {}
    if not predictions:
        return results

    for pred_item in predictions:
        # Use a tuple of (question, context) as the key to handle potential duplicate questions
        key = (pred_item["question"], pred_item["context"])
        prediction_text = pred_item["prediction"]
        ground_truths = ground_truths_map.get(key)

        f1 = 0.0
        em = 0.0
        if ground_truths:
            f1 = metric_max_over_ground_truths(f1_score, prediction_text, ground_truths)
            em = metric_max_over_ground_truths(exact_match_score, prediction_text, ground_truths)

        results[key] = {
            "f1": f1,
            "em": em,
            "prediction": prediction_text,
        }
    return results


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
    try:
        prediction_files = [f for f in os.listdir(args.input_dir) if f.endswith('.json') and f != 'gt.json']
    except FileNotFoundError:
        print(f"Error: Input directory not found at {args.input_dir}")
        return
        

    if not prediction_files:
        print("\nNo prediction files (ending in .json, excluding gt.json) found to evaluate.")

    results_by_method = {}

    for filename in sorted(prediction_files):
        method_name = os.path.splitext(filename)[0]
        file_path = os.path.join(args.input_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)

            print(f"\nEvaluating '{method_name}' method from '{filename}'...")
            if not predictions:
                print("  File is empty or contains no predictions. Skipping.")
                continue
            
            # Get detailed results
            detailed_results = evaluate(predictions, ground_truths_map)
            
            if method_name in ["standard", "parallel_multiple_diff"]:
                 results_by_method[method_name] = detailed_results

            # Calculate and print averages
            if not detailed_results:
                avg_f1, avg_em = 0.0, 0.0
            else:
                total_f1 = sum(res['f1'] for res in detailed_results.values())
                total_em = sum(res['em'] for res in detailed_results.values())
                count = len(detailed_results)
                avg_f1 = total_f1 / count
                avg_em = total_em / count
            
            print(f"  Average F1 Score: {avg_f1:.4f}")
            print(f"  Average Exact Match Score: {avg_em:.4f}")

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\nAn error occurred while processing {filename}: {e}. Skipping.")
        except Exception as e:
            print(f"\nAn unexpected error occurred with {filename}: {e}")

    # Comparison logic
    if "standard" in results_by_method and "parallel_multiple_diff" in results_by_method:
        standard_results = results_by_method["standard"]
        parallel_results = results_by_method["parallel_multiple_diff"]

        differences = []
        common_keys = set(standard_results.keys()).intersection(set(parallel_results.keys()))

        for key in common_keys:
            standard_f1 = standard_results[key]["f1"]
            parallel_f1 = parallel_results[key]["f1"]
            diff = abs(standard_f1 - parallel_f1)
            
            if diff > 0:
                question, context = key
                ground_truths = ground_truths_map.get(key, [])
                differences.append({
                    "diff": diff,
                    "question": question,
                    "context": context,
                    "ground_truths": ground_truths,
                    "standard_f1": standard_f1,
                    "parallel_f1": parallel_f1,
                    "standard_pred": standard_results[key]["prediction"],
                    "parallel_pred": parallel_results[key]["prediction"],
                })

        differences.sort(key=lambda x: x["diff"], reverse=True)

        print("\n\n--- Top 30 F1 Score Differences (standard vs. parallel_multiple_diff) ---")
        for i, item in enumerate(differences[:30]):
            print(f"\n--- Example {i+1} (Diff: {item['diff']:.4f}) ---")
            print(f"Question: {item['question']}")
            print(f"Context: {item['context']}")
            print(f"Ground Truth: {item['ground_truths']}")
            print(f"  Standard F1: {item['standard_f1']:.4f}, Pred: '{item['standard_pred']}'")
            print(f"  Parallel F1: {item['parallel_f1']:.4f}, Pred: '{item['parallel_pred']}'")
    else:
        print("\nSkipping comparison: 'standard' and/or 'parallel_multiple_diff' results not found.")


if __name__ == "__main__":
    main()