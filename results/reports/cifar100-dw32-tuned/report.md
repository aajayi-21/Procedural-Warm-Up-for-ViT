# Downstream run: cifar100-dw32-tuned

_Generated 2026-07-06T19:40:31Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/dw32-tuned-vit-t/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **68.60%** (final 68.59%, top-5 91.00%).

## Figures

![cifar100-dw32-tuned_downstream_curves](..\..\figures\cifar100-dw32-tuned_downstream_curves.png)
