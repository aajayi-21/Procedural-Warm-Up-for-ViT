# Downstream run: cifar100-forward-deep-2dpos

_Generated 2026-06-27T03:38:00Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110-forward-deep-2dpos/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **67.24%** (final 67.14%, top-5 89.19%).

## Figures

![cifar100-forward-deep-2dpos_downstream_curves](..\..\figures\cifar100-forward-deep-2dpos_downstream_curves.png)
