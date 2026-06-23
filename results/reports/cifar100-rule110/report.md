# Downstream run: cifar100-rule110

_Generated 2026-06-22T22:55:57Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **68.86%** (final 68.61%, top-5 90.84%).

## Figures

![cifar100-rule110_downstream_curves](../../figures/cifar100-rule110_downstream_curves.png)
