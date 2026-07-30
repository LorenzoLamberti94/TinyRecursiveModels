# Sudoku-Extreme — MLP variant (`pretrain_mlp_t_sudoku`)

Pretrained TRM checkpoint and its **replicated** test performance.

| | |
|---|---|
| Checkpoint | `step_65100` |
| Dataset | `sudoku-extreme-1k-aug-1000` (1000 puzzles × 1000 augments; test = 422,786) |
| Params | ~5.0 M |
| Config | `all_config.yaml` (`arch.mlp_t=True`, `arch.pos_encodings=none`, `L_layers=2`, `H_cycles=3`, `L_cycles=6`) |

## Replicated evaluation

Evaluated with `eval_only.py` on badile13 (RTX 4070 Ti SUPER, 16 GB):

| Metric | Value | Paper target |
|---|---|---|
| **exact_accuracy** (whole-puzzle) | **88.75%** | ~87% ±2% |
| accuracy (per-cell) | 95.83% |  |
| lm_loss | 0.0999 |  |

✅ **Replicates the paper.** Matches the training-time number (0.8875) exactly on a
cold reload — confirming the result is real, not a logging artifact.

### Command
```bash
WANDB_MODE=disabled python eval_only.py arch=trm \
  data_paths=[/scratch/$USER/dataset/sudoku-extreme-1k-aug-1000] evaluators=[] \
  global_batch_size=512 \
  arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  +load_checkpoint=$PWD/pretrained/sudoku_mlp/step_65100
```

## Training provenance
Trained on CSCS Alps GH200 with the paper's full `global_batch_size=768`
(4× GH200, ~1.2 h). See [`../../TESTING.md`](../../TESTING.md) for the full setup.
