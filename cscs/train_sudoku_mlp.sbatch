#!/bin/bash
#SBATCH --job-name=trm-sudoku-mlp
#SBATCH --account=a-XXXX
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Sudoku-Extreme, MLP variant (paper's headline ~87% exact-accuracy).
# Submit:  sbatch -A <account> cscs/train_sudoku_mlp.sbatch

export RUN_NAME="pretrain_mlp_t_sudoku"
export DATA_SUBDIR="sudoku-extreme-1k-aug-1000"
export TRM_ARCH_ARGS="arch.mlp_t=True arch.pos_encodings=none arch.H_cycles=3 arch.L_cycles=6"

srun --environment=trm bash "$(dirname "$0")/_train.sh"
