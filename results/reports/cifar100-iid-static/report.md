# Downstream run: cifar100-iid-static

_Generated 2026-06-25T00:09:48Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-iid-static/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **65.72%** (final 65.54%, top-5 89.50%).

## Figures

![cifar100-iid-static_downstream_curves](../../figures/cifar100-iid-static_downstream_curves.png)
