# Downstream run: cifar100-iid-true-1dpos

_Generated 2026-06-27T09:47:54Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-iid1step-true-1dpos/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **66.33%** (final 66.08%, top-5 89.33%).

## Figures

![cifar100-iid-true-1dpos_downstream_curves](..\..\figures\cifar100-iid-true-1dpos_downstream_curves.png)
