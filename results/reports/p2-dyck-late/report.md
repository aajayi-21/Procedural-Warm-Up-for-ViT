# Downstream run: p2-dyck-late

_Generated 2026-06-25T19:01:32Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `random init (no warm-up)`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **72.42%** (final 72.42%, top-5 92.48%).

## Figures

![p2-dyck-late_downstream_curves](../../figures/p2-dyck-late_downstream_curves.png)
