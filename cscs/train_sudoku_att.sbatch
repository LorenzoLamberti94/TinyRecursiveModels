#!/bin/bash
#SBATCH --job-name=trm-sudoku-att
#SBATCH --account=a-XXXX
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Sudoku-Extreme, attention variant (paper ~75% exact-accuracy).
# Submit:  sbatch -A <account> cscs/train_sudoku_att.sbatch

export RUN_NAME="pretrain_att_sudoku"
export DATA_SUBDIR="sudoku-extreme-1k-aug-1000"
export TRM_ARCH_ARGS="arch.H_cycles=3 arch.L_cycles=6"

srun --environment=trm bash "$(dirname "$0")/_train.sh"
