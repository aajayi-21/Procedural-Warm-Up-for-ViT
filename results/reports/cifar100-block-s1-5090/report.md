# Downstream run: cifar100-block-s1-5090

_Generated 2026-06-24T22:39:49Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `checkpoints/ca-rule110-block-s1-5090/ckpt_step_015000_stripped.pt`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **71.55%** (final 71.40%, top-5 91.97%).

## Figures

![cifar100-block-s1-5090_downstream_curves](../../figures/cifar100-block-s1-5090_downstream_curves.png)
