#!/bin/bash
# Common training launcher, invoked inside the container by each train_*.sbatch.
# Expects env vars: RUN_NAME, DATA_SUBDIR, TRM_ARCH_ARGS (extra hydra overrides).
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "${TRM_VENV}/bin/activate"
cd "${TRM_ROOT}"

CKPT="${TRM_RUNS}/${RUN_NAME}"
mkdir -p "${CKPT}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

echo ">>> ${RUN_NAME} | ${GPUS_PER_NODE} GPUs | data=${DATA_SUBDIR} | $(date)"

# global_batch_size is left at the config default (768) -> effective batch is
# identical to the paper regardless of GPU count (split across ranks in-code).
torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" --nnodes=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    pretrain.py \
    arch=trm \
    "data_paths=[${TRM_DATA}/${DATA_SUBDIR}]" \
    "evaluators=[]" \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.L_layers=2 \
    ${TRM_ARCH_ARGS} \
    +run_name="${RUN_NAME}" ema=True \
    +checkpoint_path="${CKPT}"

echo ">>> ${RUN_NAME} finished at $(date). To upload metrics from a login node:"
echo "    wandb sync ${WANDB_DIR}/wandb/offline-run-*"
