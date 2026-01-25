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
            "ctx_enc": pred_item["ctx_enc"],
        }
    return results


def normalize_text(text):
    return "".join(text.split())


def check_is_valid(record, start_token, end_token):
    raw_output = record.get("raw_output", "")
    context = record.get("context", "")

    # Remove all instances of start/end tokens
    # Note: If the model generates them multiple times or incorrectly, we remove them all to check if the base text is preserved.
    cleaned_output = raw_output.replace(start_token, "").replace(end_token, "")

    return normalize_text(cleaned_output) == normalize_text(context)


def check_score_diff_conditioned_on_tokenization(standard_results, parallel_results):
    diff_scores_set = set()
    for key in standard_results.keys():

        st_obj = standard_results[key]
        pl_obj = parallel_results[key]

        if st_obj["f1"] != pl_obj["f1"]:
            diff_scores_set.add(key)


    for key in diff_scores_set:
        assert standard_results[key]["ctx_enc"] != parallel_results[key]["ctx_enc"]

    print("For all different scores, the tokenization is different.")



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
        return

    # 1. Load all method data
    methods_data = {}
    for filename in sorted(prediction_files):
        method_name = os.path.splitext(filename)[0]
        file_path = os.path.join(args.input_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_loaded = json.load(f)
            methods_data[method_name] = data_loaded
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\nAn error occurred while loading {filename}: {e}. Skipping.")

    if not methods_data:
        print("No valid method data loaded.")
        return

    # 2. Find valid IDs from 'standard' method
    assert "standard" in methods_data
    reference_method = "standard"

    print(f"Using '{reference_method}' as the reference for valid IDs.")

    ref_data = methods_data[reference_method]
    ref_preds = ref_data["results"]
    ref_params = ref_data["parameters"]
    start_span_token = ref_params["start_span_token"]
    end_span_token = ref_params["end_span_token"]

    assert all(
        [
            # start and end span tokens must match for valid comparison
            data['parameters'] == methods_data[reference_method]['parameters']
            for method_name, data in methods_data.items()
         ]
    )

    valid_ids = set()
    all_ids = set()

    # Check for validity in reference method
    cheating_count = 0
    for pred in ref_preds:
        pid = pred.get("id")
        if pid:
            all_ids.add(pid)
            if check_is_valid(pred, start_span_token, end_span_token):
                valid_ids.add(pid)
            else:
                cheating_count += 1

    print(f"Found {len(valid_ids)} valid IDs out of {len(all_ids)} total in '{reference_method}'.")
    if cheating_count > 0:
        print(f"Excluded {cheating_count} cheating examples from comparison set.")

    # Helper function to run evaluation on a specific set of IDs
    def run_evaluation_on_ids(target_ids, label):
        print(f"\n\n{'=' * 20} Running Evaluation: {label} {'=' * 20}")
        print(f"Evaluating on {len(target_ids)} IDs.")

        current_results_by_method = {}

        for method_name, data in methods_data.items():
            print(f"\nEvaluating '{method_name}'...")
            predictions = data["results"]

            # Filter predictions to only those in target_ids
            filtered_preds = [p for p in predictions if p["id"] in target_ids]

            # Get detailed results
            detailed_results = evaluate(filtered_preds, ground_truths_map)
            current_results_by_method[method_name] = detailed_results

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

        return current_results_by_method


    # 3. Run evaluation on ALL IDs (Incorrect Analysis)
    run_evaluation_on_ids(all_ids, "ALL IDs (Including Cheating)")

    # 4. Run evaluation on valid IDs (Primary)
    results_by_method_valid = run_evaluation_on_ids(valid_ids, "Valid (Non-Cheating) IDs from Standard")

    # Comparison logic (using valid IDs results)
    results_by_method = results_by_method_valid
    
    if not "standard" in results_by_method or not "parallel_multiple_diff" in results_by_method:
        print("\nSkipping comparison: 'standard' and/or 'parallel_multiple_diff' results not found.")
        exit(0)

    print("Printing some examples of differences only on non-cheating examples.")
    standard_results = results_by_method["standard"]
    parallel_results = results_by_method["parallel_multiple_diff"]
    
    check_score_diff_conditioned_on_tokenization(standard_results, parallel_results)
    
    differences = []
    common_keys = set(standard_results.keys()).intersection(set(parallel_results.keys()))
    print(f"Length of Common  (standard and. parallel_multiple_diff): {len(common_keys)}")

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
        print(f"\n--- Example {i + 1} (Diff: {item['diff']:.4f}) ---")
        print(f"Question: {item['question']}")
        print(f"Context: {item['context']}")
        print(f"Ground Truth: {item['ground_truths']}")
        print(f"  Standard F1: {item['standard_f1']:.4f}, Pred: '{item['standard_pred']}'")
        print(f"  Parallel F1: {item['parallel_f1']:.4f}, Pred: '{item['parallel_pred']}'")


if __name__ == "__main__":
    main()
