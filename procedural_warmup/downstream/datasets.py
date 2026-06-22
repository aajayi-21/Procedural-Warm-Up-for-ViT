"""CIFAR-10/100 data for downstream transfer.

Images are resized to 224 so the ViT-T/16 patch grid is 14x14 = 196 tokens, matching the
warm-up positional geometry. Training uses timm's RandAugment + random-erase pipeline and
Mixup/CutMix; evaluation uses a deterministic resize/normalize.
"""

from __future__ import annotations

import os

import torch
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets

_DATASETS = {
    "CIFAR10": (datasets.CIFAR10, 10),
    "CIFAR100": (datasets.CIFAR100, 100),
}


def num_classes(name: str) -> int:
    return _DATASETS[name][1]


def _build_transform(cfg, is_train: bool):
    if is_train:
        return create_transform(
            input_size=cfg.data.input_size,
            is_training=True,
            color_jitter=cfg.aug.color_jitter,
            auto_augment=cfg.aug.auto_augment,
            interpolation="bicubic",
            re_prob=cfg.aug.reprob,
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    return create_transform(
        input_size=cfg.data.input_size,
        is_training=False,
        interpolation="bicubic",
        crop_pct=1.0,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )


def build_dataset(cfg, is_train: bool):
    if cfg.data.dataset not in _DATASETS:
        raise ValueError(f"Unsupported dataset '{cfg.data.dataset}'")
    cls, n = _DATASETS[cfg.data.dataset]
    transform = _build_transform(cfg, is_train)
    ds = cls(root=cfg.data.data_root, train=is_train, transform=transform, download=True)
    return ds, n


def resolve_num_workers(cfg) -> int:
    """``num_workers <= 0`` means auto = CPU cores capped at 16.

    The input pipeline (resize 32->224 + RandAugment) is the bottleneck for a tiny model,
    so more workers raises throughput; the cap bounds RAM held by in-flight 224px batches.
    """
    nw = cfg.data.num_workers
    if nw is None or nw <= 0:
        return min(os.cpu_count() or 0, 16)
    return nw


def build_loaders(cfg):
    """Return ``(train_loader, val_loader, n_classes, mixup_fn)``."""
    train_ds, n = build_dataset(cfg, is_train=True)
    val_ds, _ = build_dataset(cfg, is_train=False)

    nw = resolve_num_workers(cfg)
    # persistent_workers avoids re-spawning workers each epoch (default prefetch is fine;
    # raising it multiplies RAM held by in-flight upscaled batches).
    loader_kwargs: dict = {"num_workers": nw, "pin_memory": True}
    if nw > 0:
        loader_kwargs["persistent_workers"] = True
    print(f"[data] DataLoader workers={nw}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs,
    )

    mixup_fn = None
    if cfg.aug.mixup > 0 or cfg.aug.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=cfg.aug.mixup,
            cutmix_alpha=cfg.aug.cutmix,
            label_smoothing=cfg.aug.smoothing,
            num_classes=n,
        )
    return train_loader, val_loader, n, mixup_fn
