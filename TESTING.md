# Testing / evaluating pretrained TRM checkpoints

This explains how to reproduce the **test-set accuracy** of the pretrained
checkpoints in [`pretrained/`](pretrained/) on a single GPU (verified on a
badile13 RTX 4070 Ti SUPER, 16 GB). It rebuilds the environment + dataset and
runs one evaluation pass from a saved checkpoint.

## Pretrained checkpoints

| Dir | Variant | Test exact-acc | Paper target | Notes |
|---|---|---|---|---|
| `pretrained/sudoku_mlp/step_65100` | Sudoku-Extreme, MLP | **88.75%** | ~87% ±2% | ✅ replicates |
| `pretrained/sudoku_att/step_65100` | Sudoku-Extreme, attention | **66.43%** | ~75% ±2% | ~9 pts short |

Both models were trained on CSCS Alps GH200 with the paper's full
`global_batch_size=768`. Each dir ships `all_config.yaml` (the exact training
config).

No Maze-Hard checkpoint is published yet: our 4-GPU `bs=768` maze run hit a NCCL
desync on the cluster, and the 1-GPU `bs=128` fallback reached 96% per-cell but
~0% whole-maze accuracy, i.e. it did not converge. Maze is being retrained.

The numbers above were reproduced by evaluating these checkpoints on badile13
(RTX 4070 Ti, 16 GB) with the commands below — the MLP result matched the
training-time number exactly (0.8875).

> The weights are a plain `torch.save(model.state_dict())`. The keys carry the
> `_orig_mod.` prefix from `torch.compile`; `pretrain.py`'s loader handles this.

## 1. Environment (once)

Single GPU with CUDA. The `adam-atan2` CUDA extension needs GCC ≥ 9 and a CUDA
toolkit — on badile that's `gcc-toolset-13` + `/usr/local/cuda-12.8`.

```bash
# uv is the fastest way to get an isolated Python 3.10 env on scratch
BASE=/scratch/$USER
uv venv $BASE/envs/trm-venv --python 3.10
export VIRTUAL_ENV=$BASE/envs/trm-venv

uv pip install torch==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126
grep -vE '^(torch==|adam-atan2)' specific_requirements.txt > /tmp/reqs.txt
uv pip install -r /tmp/reqs.txt

# build adam-atan2 for this GPU (sm_89 = RTX 40-series; use 9.0 for GH200/H100)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$CUDA_HOME/bin:$PATH
export CC=$(which gcc) CXX=$(which g++) TORCH_CUDA_ARCH_LIST=8.9
uv pip install --no-cache-dir --no-build-isolation adam-atan2==0.0.3

$VIRTUAL_ENV/bin/python -c "import torch,adam_atan2; print('OK', torch.__version__, torch.cuda.is_available())"
```

(CSCS Alps users: use the `cscs/` scripts instead — uenv + venv overlay.)

## 2. Dataset (once)

Testing needs the same dataset the model was trained on. For Sudoku:

```bash
export HF_HOME=/scratch/$USER/hf-cache
python dataset/build_sudoku_dataset.py \
  --output-dir /scratch/$USER/dataset/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000
# Maze: python dataset/build_maze_dataset.py --output-dir .../maze-30x30-hard-1k --aug
```

## 3. Evaluate a checkpoint

Use `eval_only.py` (in this repo) — it loads the checkpoint, runs a single
inference-only pass over the test set, and prints the accuracy. No training, no
EMA, no W&B. The arch flags must match the checkpoint (they're in its
`all_config.yaml`); everything else uses the `trm` defaults.

```bash
export VENV=/scratch/$USER/envs/trm-venv

# Sudoku — MLP  (=> all/exact_accuracy = 0.8875)
WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$VENV/bin/python eval_only.py arch=trm \
  data_paths=[/scratch/$USER/dataset/sudoku-extreme-1k-aug-1000] evaluators=[] \
  global_batch_size=512 \
  arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  +load_checkpoint=$PWD/pretrained/sudoku_mlp/step_65100

# Sudoku — attention  (=> ~0.6643): drop mlp_t / pos_encodings (defaults: rope, mlp_t=False)
#   ... arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
#   +load_checkpoint=$PWD/pretrained/sudoku_att/step_65100
```

Once a Maze checkpoint exists, evaluate it the same way with
`data_paths=[.../maze-30x30-hard-1k]`, `arch.L_cycles=4`, and
`global_batch_size=128` (seq_len is 900).

## 4. Read the result

`eval_only.py` prints it directly at the end, e.g.:

```
==================== EVAL RESULT ====================
  all/accuracy = 0.9583
  all/exact_accuracy = 0.8875
EVAL_RESULT_JSON {"all": {"accuracy": 0.9583, "exact_accuracy": 0.8875, ...}}
```

`exact_accuracy` is the whole-puzzle test accuracy; `accuracy` is per-token.

## Notes / gotchas

- **`+` prefix**: `load_checkpoint` isn't in the base yaml, so Hydra requires
  the `+` (`+load_checkpoint=...`), else *"Could not override 'load_checkpoint'"*.
- **Match the arch**: `pos_encodings`/`mlp_t` change the model. rope vs none (and
  attention vs mlp_t) have no distinct learned params, so a mismatch loads
  *silently* and gives wrong accuracy. Copy the arch flags from the checkpoint's
  `all_config.yaml`. Defaults are `pos_encodings=rope`, `mlp_t=False`.
- **Batch size** only affects eval speed/memory, not accuracy: `512` fits 16 GB
  for Sudoku; use `128` for Maze (seq_len 900).
- **First ~1-2 min** is `torch.compile`; set `DISABLE_COMPILE=1` to skip it.
