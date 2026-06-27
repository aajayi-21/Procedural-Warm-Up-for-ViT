# Downstream run: cifar100-block2d-1dpos

_Generated 2026-06-27T00:34:48Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110-block2d-1dpos/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **69.17%** (final 69.05%, top-5 90.75%).

## Figures

![cifar100-block2d-1dpos_downstream_curves](..\..\figures\cifar100-block2d-1dpos_downstream_curves.png)
