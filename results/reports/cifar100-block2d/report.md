# Downstream run: cifar100-block2d

_Generated 2026-06-26T23:02:52Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110-block2d/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **69.42%** (final 69.29%, top-5 90.95%).

## Figures

![cifar100-block2d_downstream_curves](..\..\figures\cifar100-block2d_downstream_curves.png)
