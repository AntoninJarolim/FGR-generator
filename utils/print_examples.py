"""
Print evaluation statistics from stored eval_artifacts.json.
Does not print any examples, only stats (counts + averages).
"""
import argparse
import json
import os
import sys


def get_examples_dir(input_dir: str) -> str:
    """Same as run_squad_eval: data/eval/<input_dir>/ with leading data/ stripped."""
    normalized = input_dir.removeprefix("data/").lstrip("/")
    return os.path.join("data", "eval", normalized) if normalized else os.path.join("data", "eval")


def print_evaluation_stats(statistics: dict) -> None:
    """Print evaluation statistics (counts hierarchy + averages). No examples."""
    counts = statistics["counts"]
    averages = statistics["averages"]

    n_all = counts["all"]
    n_valid = counts["valid"]
    n_invalid = counts["invalid"]
    n_diff_tok = counts["different_tokenization"]
    n_same_tok = counts["same_tokenization"]
    n_diff_text = counts["different_raw_output"]
    n_same_text = counts["same_raw_output"]
    n_diff_pred = counts["different_prediction"]
    n_same_pred = counts["same_prediction"]
    n_diff_toks_bef = counts["different_ctx_tokens_before_start"]
    n_same_toks_bef = counts["same_ctx_tokens_before_start"]

    def pct(n: int, total: int) -> str:
        if total == 0:
            return "0.0%"
        return f"({n / total * 100:.1f}%)"

    print("--- Evaluation statistics ---")
    print(f"Common keys: {counts.get('common_keys_all_methods', 'N/A')}")
    print()
    print("Filtering hierarchy:")
    print(f"ALL {n_all}")
    print(f"-- invalid {n_invalid} {pct(n_invalid, n_all)}")
    print(f"-- valid {n_valid} {pct(n_valid, n_all)}")
    print(f"  -- same tokenization {n_same_tok} {pct(n_same_tok, n_valid)}")
    print(f"  -- different tokenization {n_diff_tok} {pct(n_diff_tok, n_valid)}")
    print(f"    -- same raw output {n_same_text} {pct(n_same_text, n_diff_tok)}")
    print(f"    -- different raw output {n_diff_text} {pct(n_diff_text, n_diff_tok)}")
    print(f"      -- same prediction {n_same_pred} {pct(n_same_pred, n_diff_text)}")
    print(f"      -- different prediction {n_diff_pred} {pct(n_diff_pred, n_diff_text)}")
    print(f"        -- same ctx tokens before start {n_same_toks_bef} {pct(n_same_toks_bef, n_diff_pred)}")
    print(f"        -- different ctx tokens before start {n_diff_toks_bef} {pct(n_diff_toks_bef, n_diff_pred)}")
    print()
    print("Averages (valid IDs):")
    for method_name, avgs in averages.items():
        print(f"  {method_name}: F1={avgs['avg_f1']:.4f}, EM={avgs['avg_em']:.4f}")
    print("---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print evaluation stats from stored eval artifacts (no examples).")
    parser.add_argument("input_dir", help="Input dir used for run_squad_eval (e.g. data/squad_v1).")
    parser.add_argument("--method", help="Method name for per-method file (eval_artifacts_<method>.json). If omitted, reads eval_artifacts.json.")
    args = parser.parse_args()

    examples_dir = get_examples_dir(args.input_dir)
    if args.method:
        path = os.path.join(examples_dir, f"eval_artifacts_{args.method}.json")
    else:
        path = os.path.join(examples_dir, "eval_artifacts.json")
    if not os.path.isfile(path):
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    statistics = data.get("statistics")
    if not statistics:
        print(f"Error: no 'statistics' in {os.path.basename(path)}.", file=sys.stderr)
        sys.exit(1)

    print_evaluation_stats(statistics)


if __name__ == "__main__":
    main()
