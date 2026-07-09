# Downstream run: cifar100-dw32

_Generated 2026-07-06T16:18:43Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/dw32-vit-t/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **67.88%** (final 67.78%, top-5 90.41%).

## Figures

![cifar100-dw32_downstream_curves](..\..\figures\cifar100-dw32_downstream_curves.png)
