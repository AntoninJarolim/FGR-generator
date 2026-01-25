import json
import argparse
import os
from argparse import Namespace
from typing import Any, Literal

from eval.eval_squad import f1_score, exact_match_score, metric_max_over_ground_truths


def evaluate(predictions, ground_truths_data_map):
    """
    Evaluates predictions and returns per-datapoint F1 and EM scores.
    """
    results = {}
    if not predictions:
        return results

    for pred_item in predictions:
        # Use 'id' as the key
        key = pred_item["id"]
        prediction_text = pred_item["prediction"]
        ground_truths = ground_truths_data_map.get(key)["answers"]

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


def get_valid_ids(ref_preds, reference_method: str, start_span_token, end_span_token) -> set[Any]:
    # Check for validity in reference method
    valid_ids = set()
    for pred_id, pred_data in ref_preds.items():
        if check_is_valid(pred_data, start_span_token, end_span_token):
            valid_ids.add(pred_id)

    cheating_count = len(ref_preds) - len(valid_ids)
    print(f"Found {len(valid_ids)} valid IDs out of {len(ref_preds)} total in '{reference_method}'.")
    if cheating_count > 0:
        print(f"Excluded {cheating_count} cheating examples from comparison set.")
    return valid_ids


def get_diff_tokens_ids(standard_data, parallel_data, valid_ids):
    standard_data = standard_data["results"]
    parallel_data = parallel_data["results"]

    diff_tokens_ids = set()
    for key in valid_ids:
        std_ctx = standard_data[key]['ctx_enc']
        prl_ctx = parallel_data[key]['ctx_enc']

        if std_ctx[-1] == 128009:  # todo: eos_token_id:
            std_ctx = std_ctx[:-1]

        if std_ctx != prl_ctx:
            diff_tokens_ids.add(key)

    return diff_tokens_ids


def get_diff_raw_outputs(standard_data, parallel_data, ids_to_inspect):
    def white_space_fix(text):
        return ' '.join(text.split())

    standard_data = standard_data["results"]
    parallel_data = parallel_data["results"]

    diff_texts = set()
    for k in ids_to_inspect:
        std_ctx = standard_data[k]['raw_output']
        prl_ctx = parallel_data[k]['raw_output']

        if white_space_fix(std_ctx) != white_space_fix(prl_ctx):
            diff_texts.add(k)

    return diff_texts


def get_diff_prediction_span(standard_data, parallel_data, ids_to_inspect):
    standard_data = standard_data["results"]
    parallel_data = parallel_data["results"]

    diff_pred_ids = set()
    for k in ids_to_inspect:
        std_ctx = standard_data[k]['prediction']
        prl_ctx = parallel_data[k]['prediction']

        if not exact_match_score(std_ctx, prl_ctx):
            diff_pred_ids.add(k)

    return diff_pred_ids


# Helper function to run evaluation on a specific set of IDs
def run_evaluation_on_ids(target_ids, label, methods_data, ground_truths_data_map):
    print(f"\n\n{'=' * 20} Running Evaluation: {label} {'=' * 20}")
    print(f"Evaluating on {len(target_ids)} IDs.")

    current_results_by_method = {}

    for method_name, data in methods_data.items():
        print(f"\nEvaluating '{method_name}'...")
        predictions = data["results"]

        # Filter predictions to only those in target_ids
        filtered_preds = [p_data for p_id, p_data in predictions.items()
                          if p_id in target_ids]

        # Get detailed results
        detailed_results = evaluate(filtered_preds, ground_truths_data_map)
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


def main():
    args = parse_args()

    # Load ground truth data with answers
    gt_data = get_gt_data(args)
    ground_truths_data_map = {item["id"]: item for item in gt_data}

    # Find and evaluate all prediction files in the input directory
    prediction_files = [f for f in os.listdir(args.input_dir) if f.endswith('.json') and f != 'gt.json']
    assert prediction_files, "No prediction files found to evaluate."

    # 1. Load all method data
    methods_data = load_all_data(args, prediction_files)

    assert "standard" in methods_data
    reference_method = "standard"
    print(f"Using '{reference_method}' as the reference for valid IDs.")

    ref_data = methods_data[reference_method]
    ref_preds = ref_data["results"]
    ref_params = ref_data["parameters"]
    start_span_token = ref_params["start_span_token"]
    end_span_token = ref_params["end_span_token"]

    assert_coherent_params(methods_data, reference_method)

    # 2. Get the differences in the most prominent method and standard eval baseline
    all_ids = set(ref_preds.keys())

    # Filter examples with invalid generation using even standard method
    valid_ids = get_valid_ids(ref_preds, reference_method, start_span_token, end_span_token)

    # Find differences of context tokenization
    diff_tokens_ids = get_diff_tokens_ids(
        methods_data[reference_method],
        methods_data["parallel_multiple_diff"],
        valid_ids
    )

    diff_texts_ids = get_diff_raw_outputs(
        methods_data[reference_method],
        methods_data["parallel_multiple_diff"],
        diff_tokens_ids
    )

    diff_prediction_ids = get_diff_prediction_span(
        methods_data[reference_method],
        methods_data["parallel_multiple_diff"],
        diff_tokens_ids
    )

    # 3. Run evaluation on ALL IDs (Incorrect Analysis)
    run_evaluation_on_ids(
        all_ids,
        "ALL IDs (Including Cheating)",
        methods_data, ground_truths_data_map
    )

    # 4. Run evaluation on valid IDs (Primary)
    results_by_method_valid = run_evaluation_on_ids(
        valid_ids,
        "Valid (Non-Cheating) IDs from Standard",
        methods_data,
        ground_truths_data_map
    )

    # Comparison logic (using valid IDs results)
    results_by_method = results_by_method_valid

    if not "standard" in results_by_method or not "parallel_multiple_diff" in results_by_method:
        print("\nSkipping comparison: 'standard' and/or 'parallel_multiple_diff' results not found.")
        exit(0)

    print("Printing some examples of differences only on non-cheating examples.")
    standard_results = results_by_method["standard"]
    parallel_results = results_by_method["parallel_multiple_diff"]

    differences = []
    common_keys = set(standard_results.keys()).intersection(set(parallel_results.keys()))
    print(f"Length of Common  (standard and. parallel_multiple_diff): {len(common_keys)}")

    for key in common_keys:
        standard_f1 = standard_results[key]["f1"]
        parallel_f1 = parallel_results[key]["f1"]
        diff = abs(standard_f1 - parallel_f1)

        if diff > 0:
            gt_item = ground_truths_data_map.get(key)
            question = gt_item["question"]
            context = gt_item["context"]
            ground_truths = gt_item["answers"]
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


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SQuAD predictions.")
    parser.add_argument("input_dir", help="Path to the directory with prediction files.")
    args = parser.parse_args()
    return args


def get_gt_data(args: Namespace) -> Any:
    gt_path = os.path.join(args.input_dir, "gt.json")
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    assert gt_data, f"Error: Ground truth file not found at {gt_path}"
    return gt_data


def assert_coherent_params(methods_data: dict[Any, Any], reference_method: str):
    """
    Check that all the methods were generated using the same generation params.
    Invalid comparison would arise otherwise
    """
    assert all(
        [
            # start and end span tokens must match for valid comparison
            data['parameters'] == methods_data[reference_method]['parameters']
            for method_name, data in methods_data.items()
        ]
    )


def load_all_data(args: Namespace, prediction_files: list[str | Literal['gt.json']]) -> dict[Any, Any]:
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
        exit(1)

    for method_name, data in methods_data.items():
        results_list = data["results"]
        # Convert list of results to the dict of results, to allow indexing by key
        data['results'] = {r['id']: r for r in results_list}

    return methods_data


if __name__ == "__main__":
    main()
