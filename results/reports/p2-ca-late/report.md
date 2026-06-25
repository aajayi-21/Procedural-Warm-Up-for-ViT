# Downstream run: p2-ca-late

_Generated 2026-06-25T17:49:10Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `random init (no warm-up)`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **70.77%** (final 70.67%, top-5 91.83%).

## Figures

![p2-ca-late_downstream_curves](../../figures/p2-ca-late_downstream_curves.png)
