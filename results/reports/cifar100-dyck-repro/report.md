# Downstream run: cifar100-dyck-repro

_Generated 2026-07-06T06:29:58Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/dyck-repro/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **72.08%** (final 71.98%, top-5 92.17%).

## Figures

![cifar100-dyck-repro_downstream_curves](..\..\figures\cifar100-dyck-repro_downstream_curves.png)
