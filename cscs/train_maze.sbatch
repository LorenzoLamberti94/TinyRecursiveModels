#!/bin/bash
#SBATCH --job-name=trm-maze
#SBATCH --account=a-XXXX
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# Maze-Hard 30x30 (attention). README uses global_batch_size=768 on 4 GPUs and
# L_cycles=4. Effective batch = 768 (paper). Submit:
#   sbatch -A <account> cscs/train_maze.sbatch

export RUN_NAME="pretrain_att_maze30x30"
export DATA_SUBDIR="maze-30x30-hard-1k"
export TRM_ARCH_ARGS="arch.H_cycles=3 arch.L_cycles=4"

bash "$(dirname "$0")/_srun.sh" bash "$(dirname "$0")/_train.sh"
