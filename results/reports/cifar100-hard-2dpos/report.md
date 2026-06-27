# Downstream run: cifar100-hard-2dpos

_Generated 2026-06-27T08:15:52Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110-hard-2dpos/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **71.36%** (final 71.17%, top-5 91.69%).

## Figures

![cifar100-hard-2dpos_downstream_curves](..\..\figures\cifar100-hard-2dpos_downstream_curves.png)
