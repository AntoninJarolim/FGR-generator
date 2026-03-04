import json
import argparse
import os
from argparse import Namespace
from typing import Any, Literal

from eval.eval_squad import f1_score, exact_match_score, metric_max_over_ground_truths
from utils.print_examples import print_evaluation_stats, get_examples_dir


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


def get_valid_ids(ref_preds, start_span_token, end_span_token) -> set[Any]:
    """Check for validity using standard method."""
    valid_ids = set()
    for pred_id, pred_data in ref_preds.items():
        if check_is_valid(pred_data, start_span_token, end_span_token):
            valid_ids.add(pred_id)

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
def get_diff_toks_bef_start(standard_data, parallel_data, ids_to_inspect):
    standard_data = standard_data["results"]
    parallel_data = parallel_data["results"]


    diff_pred_ids = set()
    for k in ids_to_inspect:
        assert not exact_match_score(standard_data[k]['prediction'], parallel_data[k]['prediction'])

        # Get the index of the token starting <start> tag
        std_start = standard_data[k]['start']
        prl_start = parallel_data[k]['start']

        if not std_start or not prl_start:
            continue

        start_i = min(std_start, prl_start)

        std_ctx_start = standard_data[k]['ctx_enc'][:start_i]
        prl_ctx_start = parallel_data[k]['ctx_enc'][:start_i]

        # finding different starts in the ctx - teacher forcing different tokens
        if std_ctx_start != prl_ctx_start:
            diff_pred_ids.add(k)

    return diff_pred_ids


def run_evaluation_on_ids(target_ids, label, methods_data, ground_truths_data_map):
    current_results_by_method = {}

    for method_name, data in methods_data.items():
        detailed_results = run_evaluation_on_ids_one_method(data, method_name, ground_truths_data_map, target_ids)
        current_results_by_method[method_name] = detailed_results

    return current_results_by_method


def run_evaluation_on_ids_one_method(data, method_name, ground_truths_data_map, target_ids) -> dict[Any, Any]:
    predictions = data["results"]

    # Filter predictions to only those in target_ids
    filtered_preds = [p_data for p_id, p_data in predictions.items()
                      if p_id in target_ids]

    # Get detailed results
    return evaluate(filtered_preds, ground_truths_data_map)


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
    other_methods = sorted(m for m in methods_data if m != "standard")

    ref_data = methods_data["standard"]
    ref_preds = ref_data["results"]
    ref_params = ref_data["parameters"]
    start_span_token = ref_params["start_span_token"]
    end_span_token = ref_params["end_span_token"]

    assert_coherent_params(methods_data)

    # 2. Run evaluation on ALL IDs and valid IDs (needed for per-method stats and examples)
    all_ids = set(ref_preds.keys())
    valid_ids = get_valid_ids(ref_preds, start_span_token, end_span_token)

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

    results_by_method = results_by_method_valid
    standard_results = results_by_method.get("standard") or {}
    examples_dir = get_examples_dir(args.input_dir)
    os.makedirs(examples_dir, exist_ok=True)

    # Per-method: run hierarchy (standard vs this method), build stats and examples, store one file per method
    for method_name in other_methods:
        diff_tokens_ids = get_diff_tokens_ids(
            methods_data["standard"],
            methods_data[method_name],
            valid_ids,
        )
        diff_texts_ids = get_diff_raw_outputs(
            methods_data["standard"],
            methods_data[method_name],
            diff_tokens_ids,
        )
        diff_prediction_ids = get_diff_prediction_span(
            methods_data["standard"],
            methods_data[method_name],
            diff_texts_ids,
        )
        diff_toks_bef_start = get_diff_toks_bef_start(
            methods_data["standard"],
            methods_data[method_name],
            diff_prediction_ids,
        )
        common_keys = set(standard_results.keys()) & set((results_by_method.get(method_name) or {}).keys())
        method_pair = ["standard", method_name]
        results_subset = {k: results_by_method[k] for k in method_pair if k in results_by_method}
        statistics = build_statistics(
            all_ids, valid_ids, diff_tokens_ids, diff_texts_ids,
            diff_prediction_ids, diff_toks_bef_start,
            results_subset, common_keys, method_pair,
        )
        parallel_results = results_by_method.get(method_name) or {}
        differences = []
        for key in common_keys:
            standard_f1 = standard_results[key]["f1"]
            parallel_f1 = parallel_results[key]["f1"]
            diff = abs(standard_f1 - parallel_f1)
            if diff > 0:
                gt_item = ground_truths_data_map.get(key)
                differences.append({
                    "diff": diff,
                    "question": gt_item["question"],
                    "context": gt_item["context"],
                    "ground_truths": gt_item["answers"],
                    "standard_f1": standard_f1,
                    "parallel_f1": parallel_f1,
                    "standard_pred": standard_results[key]["prediction"],
                    "parallel_pred": parallel_results[key]["prediction"],
                })
        differences.sort(key=lambda x: x["diff"], reverse=True)
        artifacts = {"statistics": statistics, "examples": differences}
        out_path = os.path.join(examples_dir, f"eval_artifacts_{method_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2, ensure_ascii=False)
        print_evaluation_stats(statistics)


def build_statistics(
    all_ids, valid_ids, diff_tokens_ids, diff_texts_ids,
    diff_prediction_ids, diff_toks_bef_start,
    results_by_method, common_keys, method_names: list[str],
) -> dict[str, Any]:
    """Build a statistics dict for storage alongside examples."""
    n_all = len(all_ids)
    n_valid = len(valid_ids)
    n_diff_tok = len(diff_tokens_ids)
    n_diff_text = len(diff_texts_ids)
    n_diff_pred = len(diff_prediction_ids)
    n_diff_toks_bef = len(diff_toks_bef_start)
    counts = {
        "all": n_all,
        "valid": n_valid,
        "invalid": n_all - n_valid,
        "same_tokenization": n_valid - n_diff_tok,
        "different_tokenization": n_diff_tok,
        "same_raw_output": n_diff_tok - n_diff_text,
        "different_raw_output": n_diff_text,
        "same_prediction": n_diff_text - n_diff_pred,
        "different_prediction": n_diff_pred,
        "same_ctx_tokens_before_start": n_diff_pred - n_diff_toks_bef,
        "different_ctx_tokens_before_start": n_diff_toks_bef,
        "common_keys_all_methods": len(common_keys),
    }
    averages = {}
    for method_name, detailed in results_by_method.items():
        if not detailed:
            averages[method_name] = {"avg_f1": 0.0, "avg_em": 0.0}
        else:
            n = len(detailed)
            averages[method_name] = {
                "avg_f1": sum(r["f1"] for r in detailed.values()) / n,
                "avg_em": sum(r["em"] for r in detailed.values()) / n,
            }
    return {"counts": counts, "averages": averages, "methods": method_names}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SQuAD predictions.")
    parser.add_argument("input_dir", help="Path to the directory with prediction files (all .json except gt.json are methods).")
    args = parser.parse_args()
    return args


def get_gt_data(args: Namespace) -> Any:
    gt_path = os.path.join(args.input_dir, "gt.json")
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    assert gt_data, f"Error: Ground truth file not found at {gt_path}"
    return gt_data


def assert_coherent_params(methods_data: dict[Any, Any]) -> None:
    """
    Check that all the methods were generated using the same generation params.
    Invalid comparison would arise otherwise
    """
    ref_params = methods_data["standard"]["parameters"]

    for method_name, data in methods_data.items():
        assert data["parameters"] == ref_params, (
            f"Parameter mismatch for method '{method_name}'. "
            f"Expected {ref_params}, got {data['parameters']}."
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
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    if not methods_data:
        exit(1)

    for method_name, data in methods_data.items():
        results_list = data["results"]
        # Convert list of results to the dict of results, to allow indexing by key
        data['results'] = {r['id']: r for r in results_list}

    return methods_data


if __name__ == "__main__":
    main()
