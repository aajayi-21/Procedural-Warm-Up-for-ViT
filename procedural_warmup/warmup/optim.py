"""Optimizer and LR schedule for the warm-up stage (AdamW + linear warmup -> cosine)."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def make_optimizer(cfg, params) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
    )


def make_scheduler(cfg, optimizer) -> LRScheduler | None:
    """Linear warmup for ``warmup_steps`` then cosine decay to ``min_lr`` over the rest."""
    if not cfg.scheduler.enabled:
        return None
    total = cfg.training.steps
    warmup = cfg.scheduler.warmup_steps
    if warmup > 0:
        start_factor = max(cfg.scheduler.warmup_start_lr / cfg.optimizer.lr, 1e-8)
        warm = LinearLR(
            optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(total - warmup, 1), eta_min=cfg.scheduler.min_lr
        )
        return SequentialLR(optimizer, [warm, cosine], milestones=[warmup])
    return CosineAnnealingLR(optimizer, T_max=total, eta_min=cfg.scheduler.min_lr)
