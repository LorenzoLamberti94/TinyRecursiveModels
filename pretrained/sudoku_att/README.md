# Sudoku-Extreme — attention variant (`pretrain_att_sudoku`)

Pretrained TRM checkpoint and its **replicated** test performance.

| | |
|---|---|
| Checkpoint | `step_195300` |
| Dataset | `sudoku-extreme-1k-aug-1000` (1000 puzzles × 1000 augments; test = 422,786) |
| Params | ~6.8 M |
| Config | `all_config.yaml` (attention: `mlp_t=False`, `pos_encodings=rope` [defaults]; `L_layers=2`, `H_cycles=3`, `L_cycles=6`; **`lr_min_ratio=0.1`, `epochs=150000`**) |

## Replicated evaluation

Evaluated with `eval_only.py` on badile13 (RTX 4070 Ti SUPER, 16 GB):

| Metric | Value | Paper target |
|---|---|---|
| **exact_accuracy** (whole-puzzle) | **73.21%** | ~75% ±2% |
| accuracy (per-cell) | 91.81% |  |

✅ **Replicates** (within the paper's ~75% ±2% band).

### How this differs from the paper's published command — and why

The README recipe (`lr_min_ratio=1.0`, `epochs=50000`) does **not** replicate for
the attention variant: it gives ~66%. The default `lr_min_ratio=1.0` holds the
learning rate **constant** (no decay), which makes attention/Maze training *peak
then collapse*. Sweeping checkpoints of a longer constant-LR run showed exactly
that: 69.1% at step 130,200 → **collapsed to 32%** at step 195,300.

This checkpoint adds **cosine LR decay** (`lr_min_ratio=0.1`) and trains 3× longer
(`epochs=150000`). The LR anneals into a stable minimum instead of collapsing —
its trajectory climbs to the end: 31.5% → 35.3% → 68.8% → **73.2%** (final = best).
The MLP variant (`../sudoku_mlp`) is robust to the constant LR and replicated at
88.75% without this change.

### Command
```bash
WANDB_MODE=disabled python eval_only.py arch=trm \
  data_paths=[/scratch/$USER/dataset/sudoku-extreme-1k-aug-1000] evaluators=[] \
  global_batch_size=512 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  +load_checkpoint=$PWD/pretrained/sudoku_att/step_195300
```
(No `mlp_t`/`pos_encodings` overrides — the attention variant uses the `trm`
defaults `mlp_t=False`, `pos_encodings=rope`. `lr_min_ratio`/`epochs` are
training-time only and don't affect evaluation.)

## Training provenance
Trained on CSCS Alps GH200 (4× GH200, `global_batch_size=768`, ~2.8 h) with
cosine LR decay. See [`../../TESTING.md`](../../TESTING.md).
