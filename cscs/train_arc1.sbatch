#!/bin/bash
#SBATCH --job-name=trm-arc1
#SBATCH --account=a-XXXX
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# ARC-AGI-1 (concept-augmented). README: 4 GPUs, ~3 DAYS. This uses the config
# DEFAULTS (epochs=100000, eval_interval=10000, lr=1e-4, wd=0.1, puzzle_emb_lr=1e-2,
# evaluator arc@ARC) — matching the README ARC recipe, NOT the sudoku one.
#
# WALLTIME: CSCS 'normal' typically caps at 24h, but this run needs ~72h.
# It checkpoints every eval; resume a killed run by re-submitting with:
#   +load_checkpoint=<path-to-latest-step-file>
# (uncomment the LOAD line below). Chain 3-4 submissions, or use --dependency.
#
# Submit:  sbatch -A <account> cscs/train_arc1.sbatch
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cscs/config.sh"
source "${TRM_VENV}/bin/activate"
cd "${TRM_ROOT}"

RUN_NAME="pretrain_att_arc1concept_4"
CKPT="${TRM_RUNS}/${RUN_NAME}"
mkdir -p "${CKPT}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

# LOAD="+load_checkpoint=${CKPT}/step_XXXXX"   # <-- uncomment to resume
LOAD=""

bash "$SLURM_SUBMIT_DIR/cscs/_srun.sh" bash -c "
  source ${TRM_VENV}/bin/activate
  python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --nnodes=1 \
      --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
      pretrain.py \
      arch=trm \
      'data_paths=[${TRM_DATA}/arc1concept-aug-1000]' \
      arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
      +run_name=${RUN_NAME} ema=True \
      +checkpoint_path=${CKPT} ${LOAD}
"
echo ">>> ${RUN_NAME} segment finished at $(date)"
