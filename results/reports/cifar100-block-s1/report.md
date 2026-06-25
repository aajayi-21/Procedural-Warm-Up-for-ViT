# Downstream run: cifar100-block-s1

_Generated 2026-06-24T23:23:20Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110-block-s1/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **72.22%** (final 71.92%, top-5 92.25%).

## Figures

![cifar100-block-s1_downstream_curves](../../figures/cifar100-block-s1_downstream_curves.png)
