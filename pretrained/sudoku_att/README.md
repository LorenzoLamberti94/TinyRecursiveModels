# Sudoku-Extreme — attention variant (`pretrain_att_sudoku`)

Pretrained TRM checkpoint and its **replicated** test performance.

| | |
|---|---|
| Checkpoint | `step_65100` |
| Dataset | `sudoku-extreme-1k-aug-1000` (1000 puzzles × 1000 augments; test = 422,786) |
| Params | ~6.8 M |
| Config | `all_config.yaml` (attention: `mlp_t=False`, `pos_encodings=rope` [defaults]; `L_layers=2`, `H_cycles=3`, `L_cycles=6`) |

## Replicated evaluation

Evaluated with `eval_only.py` on badile13 (RTX 4070 Ti SUPER, 16 GB):

| Metric | Value | Paper target |
|---|---|---|
| **exact_accuracy** (whole-puzzle) | **66.42%** | ~75% ±2% |
| accuracy (per-cell) | 88.03% |  |
| lm_loss | 0.2731 |  |

⚠️ **~9 points short of the paper's ~75% target.** Consistent with the CSCS
training-time value (66.43%). This is the full `bs=768` recipe, so it is *not* a
batch-size artifact — likely seed sensitivity or a subtle recipe detail; the MLP
variant (see `../sudoku_mlp`) replicated cleanly on the same setup. A
multi-seed rerun would clarify whether the gap is variance.

### Command
```bash
WANDB_MODE=disabled python eval_only.py arch=trm \
  data_paths=[/scratch/$USER/dataset/sudoku-extreme-1k-aug-1000] evaluators=[] \
  global_batch_size=512 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  +load_checkpoint=$PWD/pretrained/sudoku_att/step_65100
```
(No `mlp_t`/`pos_encodings` overrides — the attention variant uses the `trm`
defaults `mlp_t=False`, `pos_encodings=rope`.)

## Training provenance
Trained on CSCS Alps GH200 with the paper's full `global_batch_size=768`
(4× GH200, ~1.2 h). See [`../../TESTING.md`](../../TESTING.md).
