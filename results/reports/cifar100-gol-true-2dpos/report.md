# Downstream run: cifar100-gol-true-2dpos

_Generated 2026-06-27T12:52:08Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/gol-step-true-2dpos/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **66.54%** (final 66.38%, top-5 89.57%).

## Figures

![cifar100-gol-true-2dpos_downstream_curves](..\..\figures\cifar100-gol-true-2dpos_downstream_curves.png)
