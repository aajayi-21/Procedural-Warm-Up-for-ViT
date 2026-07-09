# Downstream run: cifar100-dw32-shuffle

_Generated 2026-07-06T18:00:25Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/dw32-shuffle-vit-t/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **64.84%** (final 64.78%, top-5 88.45%).

## Figures

![cifar100-dw32-shuffle_downstream_curves](..\..\figures\cifar100-dw32-shuffle_downstream_curves.png)
