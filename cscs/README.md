# TRM replication on CSCS Alps (GH200)

Scripts to run the paper's **exact** recipe (full `global_batch_size=768`) on
Alps GH200 nodes (4× 96 GB per node) — no batch-size compromises.

## Pick a base (no sudo, no docker)

CSCS gives you two **rootless** ways to get PyTorch on GH200. Neither is docker;
neither needs root. Choose one via `TRM_BACKEND` in `config.sh`:

| Backend | What it is | Setup |
|---|---|---|
| **`uenv`** (default) | CSCS-native software stack, container-free | pick a PyTorch uenv with `uenv image find` |
| `container` | Container Engine + NGC image | `cp cscs/trm.toml ~/.edf/trm.toml` |

> ⚠️ **You cannot fully test the GH200 stack on an x86 workstation** — GH200 is
> aarch64, so torch/CUDA/`adam-atan2` are architecture-specific and can't be
> built elsewhere and copied. What *is* pre-validated (on x86): the repo code,
> the training commands, dataset building, and short training steps. The aarch64
> torch + `adam-atan2` compile + GPU smoke test run in `01_setup_env.sh`, which
> prints an explicit OK/FAIL so you know immediately on CSCS.

> Verify partition names, the uenv/image name, and `$SCRATCH` against current
> CSCS docs (https://docs.cscs.ch) — they drift. All site-specific values live in
> the `EDIT ME` block of `config.sh` and the `#SBATCH --account` lines.

## One-time setup

```bash
# On a CSCS login node:
cd $SCRATCH
git clone https://github.com/LorenzoLamberti94/TinyRecursiveModels.git
cd TinyRecursiveModels

# 1. Edit the EDIT ME block: account, TRM_BACKEND, and the uenv (or image) name.
$EDITOR cscs/config.sh
uenv image find                         # if backend=uenv: find a pytorch uenv
# cp cscs/trm.toml ~/.edf/trm.toml      # if backend=container
echo "YOUR_WANDB_KEY" > ~/.wandb_key && chmod 600 ~/.wandb_key   # optional

source cscs/config.sh

# 2. Build the venv overlay + compile adam-atan2 for sm_90 (short debug job,
#    enters the base you selected; prints a CUDA smoke test):
bash cscs/01_setup_env.sh

# 3. Build datasets (needs HuggingFace access — run where there is internet):
source $TRM_VENV/bin/activate
bash cscs/02_build_datasets.sh
```

## Train

```bash
sbatch -A $CSCS_ACCOUNT cscs/train_sudoku_mlp.sbatch   # ~87% target
sbatch -A $CSCS_ACCOUNT cscs/train_sudoku_att.sbatch   # ~75% target
sbatch -A $CSCS_ACCOUNT cscs/train_maze.sbatch         # Maze-Hard
sbatch -A $CSCS_ACCOUNT cscs/train_arc1.sbatch         # ARC-AGI-1 (~3 days, chain jobs)
```

All train scripts go through `cscs/_srun.sh`, which enters whichever base
`TRM_BACKEND` selects — so switching uenv↔container is a **one-line edit** in
`config.sh`, no script changes.

Each writes checkpoints to `$TRM_BASE/runs/<run_name>/` and logs W&B **offline**
(compute nodes have no internet). Push metrics later from a login node:

```bash
source cscs/config.sh
wandb sync $TRM_RUNS/wandb/offline-run-*
```

## File map

| File | Role |
|---|---|
| `config.sh` | Paths + `EDIT ME` block; sets `TRM_BACKEND` (uenv/container) |
| `_srun.sh` | Dispatches `srun` into the selected base |
| `_build_venv.sh` | venv overlay + `adam-atan2` build (same for both bases) |
| `_train.sh` | Shared `torchrun` launcher (Sudoku/Maze family) |
| `01_setup_env.sh` | Runs `_build_venv.sh` inside the base (debug job) |
| `02_build_datasets.sh` | Builds Sudoku + Maze + ARC-1 datasets |
| `train_*.sbatch` | One SLURM job per experiment |
| `trm.toml` | Container EDF (only used if `TRM_BACKEND=container`) |

## Notes / deviations

- **Effective batch = 768** (the paper's value). In-code `global_batch_size` is
  split across ranks (`local = global // num_replicas`), so 4 GPUs @ 768 = 192/GPU
  and the effective batch matches the paper exactly — this is what the 16 GB
  badile cards could not do (forced to 256/64, which is why att-Sudoku and Maze
  failed to converge there).
- **PyTorch version** comes from the base (uenv or NGC ~2.7), not the exact
  `2.7.0+cu126` pin — functionally equivalent on Hopper.
- **ARC walltime**: one ARC run needs ~72 h but `normal` caps lower;
  `train_arc1.sbatch` checkpoints every eval and shows how to resume with
  `+load_checkpoint=`.
- **ARC-AGI-2**: clone `train_arc1.sbatch`, swap the dataset to
  `arc2concept-aug-1000` (build with `--subsets training2 evaluation2 concept
  --test-set-name evaluation2`).
