"""Pre-download the HuggingFace assets needed for OFFLINE vLLM generation:
model weights, the input dataset, and the `xlm-roberta-base` span-matching
tokenizer used by find_invalid_samples. Run this on a machine WITH internet
(e.g. an HPC login node) so the GPU job can then run fully offline
(HF_HUB_OFFLINE=1) on air-gapped compute nodes.

Not cluster-specific: the cache location is taken from $HF_HOME (falling back to
the standard HF default). Select what to fetch with flags; the default model set
mirrors LARGE_MODELS in scripts/generate/local/run_generate_vllm.sh. Set $HF_TOKEN for gated models.

For MS MARCO this does more than warm the cache: it MATERIALIZES the joined
(query, passage) pairs to $FGR_PAIRS_DIR/msmarco/pairs-<split>.jsonl (default
~/.cache/fgr-generator/pairs; --pairs-dir overrides). The BEIR corpus is 8.8M
passages, so re-joining it inside every sharded generation job would cost
several GB of RAM and minutes each time. Generation then reads slices of the
jsonl instead. The pairs are an input-dataset cache, which is why they live
beside the HF cache and NOT under data/: data/ holds only produced span data.

Usage:
  # everything for the 3 large models, into $HF_HOME:
  HF_HOME=/path/to/cache python prefetch_hf_assets.py
  # only specific models, skip the dataset:
  python prefetch_hf_assets.py --models google/gemma-4-12B-it --no-dataset
  # only the dataset + aux tokenizer, no model weights (pass --models with no ids):
  python prefetch_hf_assets.py --models
  # materialize the MS MARCO train+validation pairs, no model weights:
  python prefetch_hf_assets.py --models --datasets msmarco
"""
import argparse
import json
import os

# Mirrors LARGE_MODELS in scripts/generate/local/run_generate_vllm.sh; override with --models.
DEFAULT_MODELS = [
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B-it",
    "Qwen/Qwen3.6-27B",
]
# vLLM needs safetensors + config + tokenizer only; skip other weight formats.
MODEL_IGNORE = ["*.pth", "*.gguf", "*.bin", "original/*", "consolidated*"]
AUX_IGNORE = ["*.h5", "*.msgpack", "*.bin", "*.onnx", "tf_model*", "flax*", "rust_model*"]

# Input datasets are cached outside the repo's data/ (which holds only span
# data). Shared with scripts/generate/karolina/run_msmarco_vllm.sh through $FGR_PAIRS_DIR.
DEFAULT_PAIRS_DIR = os.environ.get(
    "FGR_PAIRS_DIR", os.path.expanduser("~/.cache/fgr-generator/pairs"))


def materialize_msmarco(splits, out_dir, min_score):
    """Join BEIR MS MARCO once and write one jsonl per split.

    Rows keep qrels order, so a shard addressed by line index is stable across
    jobs and reruns.
    """
    from custom_utils.msmarco import load_msmarco

    # One call for every split: the 8.8M-row corpus lookup is built once and the
    # per-split records are separated afterwards via their own "split" field.
    records = load_msmarco(splits=splits, min_score=min_score)

    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        rows = [r for r in records if r["split"] == split]
        path = os.path.join(out_dir, f"pairs-{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"    -> {path} ({len(rows)} pairs)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                    help="HF model ids to download (default: the LARGE_MODELS list). "
                         "Pass with no values to skip model weights.")
    ap.add_argument("--datasets", nargs="*", choices=["longembed", "msmarco"],
                    default=["longembed"],
                    help="Input datasets to prepare (default: longembed). 'msmarco' also "
                         "materializes the joined pairs under --pairs-dir.")
    ap.add_argument("--pairs-dir", default=DEFAULT_PAIRS_DIR,
                    help="where materialized pairs go, as <pairs-dir>/<dataset>/"
                         "pairs-<split>.jsonl (default: $FGR_PAIRS_DIR or "
                         "~/.cache/fgr-generator/pairs).")
    ap.add_argument("--no-dataset", dest="datasets", action="store_const", const=[],
                    help="Prepare no dataset at all.")
    ap.add_argument("--msmarco-splits", nargs="*", default=None,
                    help="qrels splits to materialize for msmarco (default: train "
                         "validation). 'test' is TREC-DL 2019 with GRADED judgments -- "
                         "pass --min-score 2 with it or 5158 of its 9260 rows enter as "
                         "positives despite being judged non-relevant.")
    ap.add_argument("--min-score", type=int, default=1,
                    help="Minimum qrels score treated as relevant (default: 1).")
    ap.add_argument("--aux", dest="aux", action="store_true", default=True,
                    help="also cache the xlm-roberta-base span-matching tokenizer "
                         "(default: on).")
    ap.add_argument("--no-aux", dest="aux", action="store_false")
    args = ap.parse_args()

    hf_home = os.environ.get("HF_HOME", "<default HF cache>")
    print(f"HF cache: {hf_home}", flush=True)
    token = os.environ.get("HF_TOKEN")

    from huggingface_hub import snapshot_download

    for m in args.models:
        print(f"=== model: {m} ===", flush=True)
        p = snapshot_download(repo_id=m, ignore_patterns=MODEL_IGNORE, token=token)
        print(f"    -> {p}", flush=True)

    if args.aux:
        print("=== aux tokenizer: xlm-roberta-base ===", flush=True)
        p = snapshot_download(repo_id="xlm-roberta-base",
                              ignore_patterns=AUX_IGNORE, token=token)
        print(f"    -> {p}", flush=True)

    if "longembed" in args.datasets:
        print("=== dataset: LongEmbed ===", flush=True)
        from custom_utils.longembed import load_longembed
        n = len(load_longembed())
        print(f"    cached {n} LongEmbed pairs", flush=True)

    if "msmarco" in args.datasets:
        print("=== dataset: MS MARCO (BEIR) ===", flush=True)
        from custom_utils.msmarco import DEFAULT_SPLITS
        splits = args.msmarco_splits if args.msmarco_splits is not None else DEFAULT_SPLITS
        materialize_msmarco(splits, os.path.join(args.pairs_dir, "msmarco"), args.min_score)

    print("PREFETCH DONE", flush=True)


if __name__ == "__main__":
    main()
