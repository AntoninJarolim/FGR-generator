#!/bin/zsh
# Launch the local FGR viewer: a small stdlib-only web app that lists the
# extracted-relevancy outputs, highlights the selected spans inside each
# document, and compares models side by side.
#
# Usage:
#   ./run_visualize.sh                 # serve http://127.0.0.1:8123/
#   ./run_visualize.sh --port 9000
#   ./run_visualize.sh --data-dir data/extracted_relevancy
set -euo pipefail

# Same conda bootstrapping as run_generate_vllm.sh: non-interactive shells do
# not define the `conda` function, and `conda activate` does not reliably put
# the env's bin on PATH, so resolve the interpreter by absolute path.
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fgr-generator

cd "$(dirname "$0")"
exec "$CONDA_PREFIX/bin/python" visualize/server.py "$@"
