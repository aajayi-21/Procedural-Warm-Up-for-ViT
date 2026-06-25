# Downstream run: cifar100-iid-true

_Generated 2026-06-24T21:53:53Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-iid1step-true/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **66.81%** (final 66.71%, top-5 89.81%).

## Figures

![cifar100-iid-true_downstream_curves](../../figures/cifar100-iid-true_downstream_curves.png)
