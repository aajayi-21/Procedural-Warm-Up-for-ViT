# Downstream run: cifar100-hard

_Generated 2026-06-24T14:08:36Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110-hard/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **71.86%** (final 71.75%, top-5 91.81%).

## Figures

![cifar100-hard_downstream_curves](../../figures/cifar100-hard_downstream_curves.png)
