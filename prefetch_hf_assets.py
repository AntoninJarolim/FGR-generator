"""Pre-download the HuggingFace assets needed for OFFLINE vLLM generation:
model weights, the LongEmbed dataset, and the `xlm-roberta-base` span-matching
tokenizer used by find_invalid_samples. Run this on a machine WITH internet
(e.g. an HPC login node) so the GPU job can then run fully offline
(HF_HUB_OFFLINE=1) on air-gapped compute nodes.

Not cluster-specific: the cache location is taken from $HF_HOME (falling back to
the standard HF default). Select what to fetch with flags; the default model set
mirrors LARGE_MODELS in run_generate_vllm.sh. Set $HF_TOKEN for gated models.

Usage:
  # everything for the 3 large models, into $HF_HOME:
  HF_HOME=/path/to/cache python prefetch_hf_assets.py
  # only specific models, skip the dataset:
  python prefetch_hf_assets.py --models google/gemma-4-12B-it --no-dataset
  # only the dataset + aux tokenizer, no model weights (pass --models with no ids):
  python prefetch_hf_assets.py --models
"""
import argparse
import os

# Mirrors LARGE_MODELS in run_generate_vllm.sh; override with --models.
DEFAULT_MODELS = [
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B-it",
    "Qwen/Qwen3.6-27B",
]
# vLLM needs safetensors + config + tokenizer only; skip other weight formats.
MODEL_IGNORE = ["*.pth", "*.gguf", "*.bin", "original/*", "consolidated*"]
AUX_IGNORE = ["*.h5", "*.msgpack", "*.bin", "*.onnx", "tf_model*", "flax*", "rust_model*"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                    help="HF model ids to download (default: the LARGE_MODELS list). "
                         "Pass with no values to skip model weights.")
    ap.add_argument("--dataset", dest="dataset", action="store_true", default=True,
                    help="also cache the LongEmbed dataset (default: on).")
    ap.add_argument("--no-dataset", dest="dataset", action="store_false")
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

    if args.dataset:
        print("=== dataset: LongEmbed ===", flush=True)
        from custom_utils.longembed import load_longembed
        n = len(load_longembed())
        print(f"    cached {n} LongEmbed pairs", flush=True)

    print("PREFETCH DONE", flush=True)


if __name__ == "__main__":
    main()
