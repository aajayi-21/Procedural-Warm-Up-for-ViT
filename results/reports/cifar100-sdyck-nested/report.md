# Downstream run: cifar100-sdyck-nested

_Generated 2026-06-25T22:34:27Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/spatial-dyck-nested/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **70.42%** (final 70.36%, top-5 91.42%).

## Figures

![cifar100-sdyck-nested_downstream_curves](../../figures/cifar100-sdyck-nested_downstream_curves.png)
