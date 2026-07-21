#!/bin/bash
# ============================================================================
# 02 — Build datasets. MUST run on a LOGIN NODE (compute nodes have no internet;
# this step downloads from HuggingFace). Uses the container so numpy/hf/pydantic
# are available without polluting your login environment:
#
#   source cscs/config.sh
#   srun -A $CSCS_ACCOUNT -p debug -t 00:20:00 --nodes=1 --ntasks=1 \
#        --environment=trm bash cscs/02_build_datasets.sh
#
# NOTE: if compute nodes on your system DO reach the internet, you can run this
# under the debug partition as shown. If neither login nor compute has HF access,
# pre-download the datasets elsewhere and copy them into $TRM_DATA.
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
