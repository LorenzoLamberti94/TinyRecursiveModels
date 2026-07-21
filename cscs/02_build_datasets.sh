#!/bin/bash
# ============================================================================
# 02 — Build datasets. Needs HuggingFace access (dataset download). Run it where
# there IS internet on your system — typically a login node, or the debug
# partition if compute nodes on Alps can reach HF. The venv already has the
# light deps (numpy/hf/pydantic); no GPU needed. Just activate it directly:
#
#   source cscs/config.sh
#   source $TRM_VENV/bin/activate
#   bash cscs/02_build_datasets.sh
#
# The venv links the base's libs, so torch import needs the base active. If the
# import fails on a bare login node, run this under the base instead:
#   bash cscs/_srun.sh bash cscs/02_build_datasets.sh   (debug partition)
# If neither login nor compute reaches HF, pre-download elsewhere and copy the
# built folders into $TRM_DATA.
# ============================================================================
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "${TRM_VENV}/bin/activate"
cd "${TRM_ROOT}"

# Sudoku-Extreme: 1000 examples, 1000 augments
python dataset/build_sudoku_dataset.py \
    --output-dir "${TRM_DATA}/sudoku-extreme-1k-aug-1000" \
    --subsample-size 1000 --num-aug 1000

# Maze-Hard: 1000 examples, 8 dihedral augments (README uses --aug)
python dataset/build_maze_dataset.py \
    --output-dir "${TRM_DATA}/maze-30x30-hard-1k" --aug

# ARC-AGI-1 (training+evaluation+concept). Requires kaggle/combined/*.json in the repo.
python -m dataset.build_arc_dataset \
    --input-file-prefix kaggle/combined/arc-agi \
    --output-dir "${TRM_DATA}/arc1concept-aug-1000" \
    --subsets training evaluation concept \
    --test-set-name evaluation

echo ">>> datasets built under ${TRM_DATA}"
ls -1 "${TRM_DATA}"
