"""Entry point for the procedural warm-up stage.

    python -m procedural_warmup.warmup.cli --config procedural_warmup/config/files/ca-rule110.yaml

Builds the model and the (dataset, masking) pair selected by ``cfg.data.source`` via the
registry, runs the masked-token trainer, and (unless ``--no-strip``) writes a stripped
checkpoint ready for image training. If ``cfg.curriculum`` is non-empty the multi-stage
runner is used instead (Stage-2 scaffold).
"""

from __future__ import annotations

import argparse

from procedural_warmup.config import config_to_dict, load_config
from procedural_warmup.data import build_source
from procedural_warmup.model import build_model
from procedural_warmup.utils import RunDir, set_seed
from procedural_warmup.warmup.process import clean_checkpoint
from procedural_warmup.warmup.trainer import Trainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Procedural warm-up (masked-token pretraining).")
    ap.add_argument("--config", required=True, help="path to a warm-up YAML config")
    ap.add_argument("--no-strip", action="store_true", help="skip writing a stripped ckpt")
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed (for seed sweeps)")
    ap.add_argument("--run-name", default=None, help="override cfg.run_name (isolate a seed run)")
    return ap.parse_args()


def run(cfg) -> None:
    set_seed(cfg.seed)
    run_dir = RunDir.create(cfg.results_dir, cfg.run_name)
    run_dir.save_config(config_to_dict(cfg))

    if cfg.curriculum:
        from procedural_warmup.warmup.curriculum import run_curriculum

        run_curriculum(cfg, run_dir)
        return

    model, mlm_head = build_model(cfg)
    dataset, masking = build_source(cfg)

    logger = None
    if cfg.wandb.enabled:
        try:
            from procedural_warmup.warmup.logger import WandbLogger

            logger = WandbLogger(cfg)
        except Exception as exc:  # pragma: no cover - optional dependency path
            print(f"[wandb] disabled: {exc}")

    trainer = Trainer(cfg, model, mlm_head, dataset, masking, run_dir, logger=logger)
    ckpt = trainer.train()
    if logger is not None:
        logger.finish()

    if ckpt is not None and not getattr(cfg, "_no_strip", False):
        clean_checkpoint(ckpt)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.no_strip:
        cfg._no_strip = True
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_name is not None:
        cfg.run_name = args.run_name
    run(cfg)


if __name__ == "__main__":
    main()
