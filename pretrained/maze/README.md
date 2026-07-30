# Maze-Hard 30×30 — attention (`pretrain_att_maze30x30_1gpu`)

Pretrained TRM checkpoint and its **replicated** test performance.

| | |
|---|---|
| Checkpoint | `step_390620` |
| Dataset | `maze-30x30-hard-1k` (1000 mazes × 8 dihedral augments; test = 1000; seq_len 900) |
| Params | ~6.8 M |
| Config | `all_config.yaml` (`mlp_t=False`, `pos_encodings=rope`; `L_layers=2`, `H_cycles=3`, `L_cycles=4`; `global_batch_size=128`) |

## Replicated evaluation

Evaluated with `eval_only.py` on badile13 (RTX 4070 Ti SUPER, 16 GB):

| Metric | Value | Paper target |
|---|---|---|
| **exact_accuracy** (whole-maze) | **0.00%** | ~85% |
| accuracy (per-cell) | 96.11% |  |
| lm_loss | 0.0651 |  |

✗ **Did not converge — negative result.** 96% of cells are correct but *no*
maze is solved end-to-end (the solution path is a tiny fraction of cells, so
high per-cell accuracy coexists with 0% exact). These weights are kept for
reproducibility of the failure, not as a working model.

**Why it failed:** the paper's Maze result uses 4 GPUs at `bs=768`. On our
cluster that config hit a NCCL rank-desync (watchdog timeout); the single-GPU
fallback `bs=128` (a documented recipe) trained to completion but never learned
full solutions — the same small-effective-batch signature seen at `bs=64` on the
16 GB cards. A working Maze replication needs the full 4-GPU `bs=768` run to be
made stable.

### Command
```bash
WANDB_MODE=disabled python eval_only.py arch=trm \
  data_paths=[/scratch/$USER/dataset/maze-30x30-hard-1k] evaluators=[] \
  global_batch_size=128 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +load_checkpoint=$PWD/pretrained/maze/step_390620
```

## Training provenance
Trained on CSCS Alps GH200, 1× GH200, `global_batch_size=128`, ~20 h.
See [`../../TESTING.md`](../../TESTING.md).
