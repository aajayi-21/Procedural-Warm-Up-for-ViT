"""Multi-stage curriculum runner — Stage-2 scaffold.

A curriculum is an ordered list of warm-up stages, each with its own data source, step
count and config overrides (``cfg.curriculum`` of :class:`StageConfig`). A single model is
carried across stages (its blocks keep learning); the optimizer/LR schedule restart per
stage (LR re-warming). All stages share one vocabulary, so ``cfg.vocab.K`` must cover every
source used — the shared frozen embeddings and MLM head then work unchanged.

This is intentionally a thin orchestrator over the existing :class:`Trainer` and the source
registry: adding the full curriculum (docs/curriculum.md, to be written) needs no change to
the trainer or data interfaces.
"""

from __future__ import annotations

import copy

from procedural_warmup.config import config_to_dict
from procedural_warmup.config.schema import _merge
from procedural_warmup.data import build_source
from procedural_warmup.model import build_model
from procedural_warmup.utils import RunDir, set_seed
from procedural_warmup.warmup.trainer import Trainer


def _stage_config(base_cfg, stage, index: int):
    """Build a per-stage config: base + source/steps + nested overrides."""
    cfg = copy.deepcopy(base_cfg)
    cfg.curriculum = []  # prevent recursion
    cfg.data.source = stage.source
    cfg.training.steps = stage.steps
    cfg.run_name = f"{base_cfg.run_name}-stage{index}-{stage.source}"
    if stage.overrides:
        _merge(cfg, stage.overrides)
    return cfg


def run_curriculum(base_cfg, run_dir: RunDir):
    """Run all stages in ``base_cfg.curriculum`` on one shared model."""
    set_seed(base_cfg.seed)
    model, mlm_head = build_model(base_cfg)

    last_ckpt = None
    for i, stage in enumerate(base_cfg.curriculum):
        stage_cfg = _stage_config(base_cfg, stage, i)
        stage_run = RunDir.create(base_cfg.results_dir, stage_cfg.run_name)
        stage_run.save_config(config_to_dict(stage_cfg))
        dataset, masking = build_source(stage_cfg)
        print(f"[curriculum] stage {i}: source={stage.source} steps={stage.steps}")
        trainer = Trainer(stage_cfg, model, mlm_head, dataset, masking, stage_run)
        last_ckpt = trainer.train() or last_ckpt

    return last_ckpt
