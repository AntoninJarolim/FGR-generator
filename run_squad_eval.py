import json
import argparse
import os
from argparse import Namespace
from typing import Any, Literal

from eval.eval_squad import f1_score, exact_match_score, metric_max_over_ground_truths
from utils.print_examples import print_evaluation_stats, get_examples_dir
from utils.artifact_hash import artifact_hash
from utils import wandb_integration


def save_artifacts_if_new(
    artifacts: dict,
    out_path: str,
) -> bool:
    """If artifact content is new (by hash): save to out_path and register for wandb. Returns True if data was new."""
    new_hash = artifact_hash(artifacts)
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if artifact_hash(existing) == new_hash:
                return False
        except (OSError, json.JSONDecodeError):
            pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2, ensure_ascii=False)
    return True


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


def _get_ids_where_differ(
    standard_data: dict,
    parallel_data: dict,
    ids_to_inspect: set,
    differ_fn: Any,
) -> set:
    """
    Return set of keys in ids_to_inspect where differ_fn(std_results, prl_results, key) is True.
    standard_data/parallel_data are full method dicts (with "results" key).
    """
    std_res = standard_data["results"]
    prl_res = parallel_data["results"]
    return {k for k in ids_to_inspect if differ_fn(std_res, prl_res, k)}


def get_diff_tokens_ids(standard_data, parallel_data, valid_ids):
    EOS_TOKEN_ID = 128009  # todo: eos_token_id

    def differ(std_res, prl_res, k):
        std_ctx = list(std_res[k]["ctx_enc"])
        if std_ctx and std_ctx[-1] == EOS_TOKEN_ID:
            std_ctx = std_ctx[:-1]
        return std_ctx != prl_res[k]["ctx_enc"]

    return _get_ids_where_differ(standard_data, parallel_data, valid_ids, differ)


def get_diff_raw_outputs(standard_data, parallel_data, ids_to_inspect):
    def white_space_fix(text):
        return " ".join(text.split())

    def differ(std_res, prl_res, k):
        return white_space_fix(std_res[k]["raw_output"]) != white_space_fix(prl_res[k]["raw_output"])

    return _get_ids_where_differ(standard_data, parallel_data, ids_to_inspect, differ)


def get_diff_prediction_span(standard_data, parallel_data, ids_to_inspect):
    def differ(std_res, prl_res, k):
        return not exact_match_score(std_res[k]["prediction"], prl_res[k]["prediction"])

    return _get_ids_where_differ(standard_data, parallel_data, ids_to_inspect, differ)


def get_diff_toks_bef_start(standard_data, parallel_data, ids_to_inspect):
    def differ(std_res, prl_res, k):
        assert not exact_match_score(std_res[k]["prediction"], prl_res[k]["prediction"])
        std_start = std_res[k]["start"]
        prl_start = prl_res[k]["start"]
        if not std_start or not prl_start:
            return False
        start_i = min(std_start, prl_start)
        return std_res[k]["ctx_enc"][:start_i] != prl_res[k]["ctx_enc"][:start_i]

    return _get_ids_where_differ(standard_data, parallel_data, ids_to_inspect, differ)


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

    # 2. Run evaluation on ALL IDs and valid IDs (needed for per-method stats)
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
    examples_dir = get_examples_dir(args.input_dir)
    os.makedirs(examples_dir, exist_ok=True)

    # Phase 1: Compute avg_f1 and avg_em for every method (no comparison yet)
    method_averages = {
        method_name: compute_method_averages(results_by_method.get(method_name) or {})
        for method_name in methods_data
    }

    # Phase 2a: Standard — counts only (no comparison), plus its averages
    counts_standard = compute_counts_standard_only(all_ids, valid_ids)
    statistics_standard = {
        "counts": counts_standard,
        "averages": {"standard": method_averages["standard"]},
    }
    artifacts_standard = {"statistics": statistics_standard}
    out_path_standard = os.path.join(examples_dir, "eval_artifacts_standard.json")
    print_evaluation_stats(statistics_standard)

    had_new_data = False
    if save_artifacts_if_new(artifacts_standard, out_path_standard):
        wandb_integration.register_for_log(args.input_dir, "standard", statistics_standard)
        had_new_data = True

    # Phase 2b: Each other method — comparison counts vs standard, plus that method's averages
    standard_results = results_by_method.get("standard") or {}
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
        counts = compute_counts_vs_standard(
            all_ids,
            valid_ids,
            diff_tokens_ids,
            diff_texts_ids,
            diff_prediction_ids,
            diff_toks_bef_start,
            n_common=len(common_keys),
        )
        statistics_for_file = {
            "counts": counts,
            "averages": {method_name: method_averages[method_name]},
        }
        artifacts = {"statistics": statistics_for_file}
        out_path = os.path.join(examples_dir, f"eval_artifacts_{method_name}.json")
        print_evaluation_stats(statistics_for_file)

        wandb_integration.register_for_log(args.input_dir, method_name, statistics_for_file)
        if save_artifacts_if_new(artifacts, out_path):
            had_new_data = True

    if had_new_data:
        wandb_integration.log_run(args.input_dir)
    else:
        print("No changes detected, no wandb logging.")


def compute_method_averages(results: dict) -> dict[str, float]:
    """Compute avg_f1 and avg_em for one method from its id->{f1, em, ...} results."""
    if not results:
        return {"avg_f1": 0.0, "avg_em": 0.0}
    n = len(results)
    return {
        "avg_f1": sum(r["f1"] for r in results.values()) / n,
        "avg_em": sum(r["em"] for r in results.values()) / n,
    }


def compute_counts_standard_only(all_ids: set, valid_ids: set) -> dict[str, Any]:
    """Counts for standard method only (no comparison): valid/invalid and trivial same_* = n_valid, different_* = 0."""
    n_all = len(all_ids)
    n_valid = len(valid_ids)
    return {
        "all": n_all,
        "valid": n_valid,
        "invalid": n_all - n_valid,
        "same_tokenization": None,
        "different_tokenization": None,
        "same_raw_output": None,
        "different_raw_output": None,
        "same_prediction": None,
        "different_prediction": None,
        "same_ctx_tokens_before_start": None,
        "different_ctx_tokens_before_start": None,
        "common_keys_all_methods": None,
    }


def compute_counts_vs_standard(
    all_ids: set,
    valid_ids: set,
    diff_tokens_ids: set,
    diff_texts_ids: set,
    diff_prediction_ids: set,
    diff_toks_bef_start: set,
    n_common: int | None = None,
) -> dict[str, Any]:
    """Counts for one method vs standard (hierarchy of diffs)."""
    n_all = len(all_ids)
    n_valid = len(valid_ids)
    n_diff_tok = len(diff_tokens_ids)
    n_diff_text = len(diff_texts_ids)
    n_diff_pred = len(diff_prediction_ids)
    n_diff_toks_bef = len(diff_toks_bef_start)
    return {
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
        "common_keys_all_methods": n_common if n_common is not None else n_valid,
    }


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
