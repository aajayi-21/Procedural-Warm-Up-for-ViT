# Downstream run: cifar100-dyck-2dpos

_Generated 2026-06-25T23:19:59Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/dyck-2dpos/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **72.19%** (final 72.02%, top-5 92.21%).

## Figures

![cifar100-dyck-2dpos_downstream_curves](../../figures/cifar100-dyck-2dpos_downstream_curves.png)
