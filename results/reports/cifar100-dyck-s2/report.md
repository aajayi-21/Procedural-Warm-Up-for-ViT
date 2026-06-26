# Downstream run: cifar100-dyck-s2

_Generated 2026-06-25T21:15:33Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/dyck-vit-t-s2/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **72.13%** (final 71.93%, top-5 92.10%).

## Figures

![cifar100-dyck-s2_downstream_curves](../../figures/cifar100-dyck-s2_downstream_curves.png)
