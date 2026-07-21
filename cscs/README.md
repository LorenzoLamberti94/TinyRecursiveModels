# TRM replication on CSCS Alps (GH200)

Scripts to run the paper's **exact** recipe (full `global_batch_size=768`) on
Alps GH200 nodes (4× 96 GB per node) — no batch-size compromises.

> These are templates following the standard Alps **Container Engine** workflow.
> Verify partition names, image tag, and `$SCRATCH` path against current CSCS
> docs (https://docs.cscs.ch) — they change. Every site-specific value is in the
> `EDIT ME` block of `config.sh` or the `#SBATCH --account` lines.

## One-time setup

```bash
# On a CSCS login node:
cd $SCRATCH
git clone <this-repo> TinyRecursiveModels        # -> $SCRATCH/TinyRecursiveModels
cd TinyRecursiveModels

# 1. Edit the EDIT ME block:
$EDITOR cscs/config.sh                            # set CSCS_ACCOUNT, image tag
cp cscs/trm.toml ~/.edf/trm.toml                  # register the container env
echo "YOUR_WANDB_KEY" > ~/.wandb_key && chmod 600 ~/.wandb_key   # optional

source cscs/config.sh

# 2. Build the venv inside the container (compiles adam-atan2 for sm_90):
srun -A $CSCS_ACCOUNT -p debug -t 00:30:00 --nodes=1 --ntasks=1 \
     --environment=trm bash cscs/01_setup_env.sh

# 3. Build datasets (needs HuggingFace access — run where there is internet):
srun -A $CSCS_ACCOUNT -p debug -t 00:20:00 --nodes=1 --ntasks=1 \
     --environment=trm bash cscs/02_build_datasets.sh
```

## Train

```bash
sbatch -A $CSCS_ACCOUNT cscs/train_sudoku_mlp.sbatch   # ~87% target
sbatch -A $CSCS_ACCOUNT cscs/train_sudoku_att.sbatch   # ~75% target
sbatch -A $CSCS_ACCOUNT cscs/train_maze.sbatch         # Maze-Hard
sbatch -A $CSCS_ACCOUNT cscs/train_arc1.sbatch         # ARC-AGI-1 (~3 days, chain jobs)
```

Each writes checkpoints to `$TRM_BASE/runs/<run_name>/` and logs W&B **offline**
(compute nodes have no internet). Push metrics later from a login node:

```bash
source cscs/config.sh
wandb sync $TRM_RUNS/wandb/offline-run-*
```

## Notes / deviations from the badile (16 GB) runs

- **Effective batch = 768** (the paper's value). In-code, `global_batch_size` is
  split across ranks (`local = global // num_replicas`), so 4 GPUs @ 768 = 192/GPU
  and the effective batch matches the paper exactly — this is what the 16 GB cards
  could not do (they were forced to 256/64, which is why att-Sudoku and Maze failed).
- **PyTorch version**: comes from the NGC container (~2.7 in 25.04), not the exact
  `2.7.0+cu126` pin — functionally equivalent on Hopper.
- **ARC walltime**: single ARC run needs ~72 h but `normal` caps lower; `train_arc1.sbatch`
  checkpoints every eval and documents how to resume with `+load_checkpoint=`.
- **ARC-AGI-2**: clone `train_arc1.sbatch`, swap the dataset to `arc2concept-aug-1000`
  (build it with `--subsets training2 evaluation2 concept --test-set-name evaluation2`).
```
