# Downstream run: cifar100-gol-shuffled-2dpos

_Generated 2026-06-27T14:24:27Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/gol-step-shuffled-2dpos/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **65.12%** (final 64.91%, top-5 88.79%).

## Figures

![cifar100-gol-shuffled-2dpos_downstream_curves](..\..\figures\cifar100-gol-shuffled-2dpos_downstream_curves.png)
