# Downstream run: cifar100-block

_Generated 2026-06-24T15:37:15Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110-block/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **71.42%** (final 71.33%, top-5 91.96%).

## Figures

![cifar100-block_downstream_curves](../../figures/cifar100-block_downstream_curves.png)
