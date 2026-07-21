#!/bin/bash
# ============================================================================
# CSCS Alps (GH200) — shared config for TRM replication.
# EDIT the values in the "EDIT ME" block, then `source` this from the others.
# ============================================================================

# ----------------------------- EDIT ME -------------------------------------
export CSCS_ACCOUNT="a-XXXX"          # your SLURM project account (sacctmgr / `id`)
export CSCS_USER="${USER}"            # your CSCS username

# Which sudo-free base provides PyTorch on the GH200 nodes. Both are rootless
# (NEITHER is docker). Pick one:
#   uenv      -> CSCS-native software stack (container-free)   [default]
#   container -> Container Engine + NGC image (needs ~/.edf/trm.toml)
export TRM_BACKEND="uenv"

# --- if TRM_BACKEND=uenv: pick a PyTorch uenv. List options with:
#       uenv image find                (then `uenv image pull <name>`)
#     Names are versioned/site-specific — verify against `uenv image find`.
export TRM_UENV="pytorch/v2.6.0"      # e.g. pytorch/v2.6.0 (check available)
export TRM_UENV_VIEW="default"        # the view that exposes torch (often 'default')

# --- if TRM_BACKEND=container: NGC PyTorch image for aarch64/GH200 (25.04 ~ torch 2.7)
export TRM_IMAGE="nvcr.io#nvidia/pytorch:25.04-py3"
# ---------------------------------------------------------------------------

# Standard CSCS paths (verify against current docs if these ever change)
export SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${CSCS_USER}}"
export TRM_ROOT="${SCRATCH}/TinyRecursiveModels"     # where you git-clone the repo
export TRM_BASE="${SCRATCH}/trm"                     # venv + datasets + runs live here
export TRM_VENV="${TRM_BASE}/venv"
export TRM_DATA="${TRM_BASE}/dataset"
export TRM_RUNS="${TRM_BASE}/runs"
export HF_HOME="${TRM_BASE}/hf-cache"

# GH200 architecture for CUDA extension builds (Hopper = sm_90)
export TORCH_CUDA_ARCH_LIST="9.0"

# Compute nodes have NO internet -> log W&B offline, sync from a login node later.
export WANDB_MODE="offline"
export WANDB_DIR="${TRM_RUNS}"
# Put your key in ~/.wandb_key (chmod 600).  Only needed for `wandb sync` on login node.
[ -f "${HOME}/.wandb_key" ] && export WANDB_API_KEY="$(cat ${HOME}/.wandb_key)"

mkdir -p "${TRM_BASE}" "${TRM_DATA}" "${TRM_RUNS}" "${HF_HOME}"
