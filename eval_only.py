"""Evaluate a pretrained TRM checkpoint on the test set and print the accuracy.

Reuses pretrain.py's model/data/eval code but skips all training, EMA, and
wandb. Single GPU. Same Hydra overrides as pretrain.py; point it at a checkpoint
with +load_checkpoint=... and match the arch flags to that checkpoint.

Example:
  DISABLE_COMPILE=1 python eval_only.py arch=trm \
    data_paths=[/scratch/$USER/dataset/sudoku-extreme-1k-aug-1000] evaluators=[] \
    global_batch_size=256 \
    arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
    +load_checkpoint=$PWD/pretrained/sudoku_mlp/step_65100
"""
import json
import hydra
from omegaconf import DictConfig

from pretrain import (
    create_dataloader,
    init_train_state,
    evaluate,
    create_evaluators,
    load_synced_config,
)


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    # rank 0 / world size 1 — single-GPU eval
    config = load_synced_config(hydra_config, rank=0, world_size=1)

    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )
    if eval_loader is None:
        raise SystemExit("No test split found under the given data_paths.")

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        evaluators = []

    # Builds the model and (because +load_checkpoint is set) loads the weights.
    train_state = init_train_state(config, eval_metadata, rank=0, world_size=1)
    train_state.model.eval()

    metrics = evaluate(
        config, train_state, eval_loader, eval_metadata, evaluators,
        rank=0, world_size=1, cpu_group=None,
    )

    out = {
        set_name: {k: float(v) for k, v in m.items()}
        for set_name, m in (metrics or {}).items() if isinstance(m, dict)
    }
    print("\n==================== EVAL RESULT ====================")
    for set_name, m in out.items():
        for k, v in m.items():
            print(f"  {set_name}/{k} = {v:.4f}")
    print("EVAL_RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
