# Downstream run: cifar100-random-repro

_Generated 2026-07-06T07:59:15Z_

## Configuration

- dataset: **CIFAR100** (224px)
- init: `random init (no warm-up)`
- model: vit_tiny_patch16_224, 300 epochs, AdamW lr=0.002 wd=0.05
- aug: mixup=0.8 cutmix=1.0 smoothing=0.1

## Result

Best top-1: **70.34%** (final 70.31%, top-5 91.01%).

## Figures

![cifar100-random-repro_downstream_curves](..\..\figures\cifar100-random-repro_downstream_curves.png)
