# CIFAR100 top-1: 1D k-Dyck vs 2D DW_32 (H6 head-to-head)

_Generated 2026-07-06T20:55:23Z_

## Results

| init | run | best top-1 | dataset |
|---|---|---|---|
| Random | `cifar100-random-repro` | 70.34 | CIFAR100 |
| k-Dyck 1D | `cifar100-dyck-repro` | 72.08 | CIFAR100 |
| DW32 2D | `cifar100-dw32` | 67.88 | CIFAR100 |
| DW32 shuffle | `cifar100-dw32-shuffle` | 64.84 | CIFAR100 |

## Summary

Best initialization: **k-Dyck 1D** (72.08% top-1).

H6 decision bands (design doc §4, 3-seed verdicts required before claims): SUCCESS if DW32 >= 72.0 and DW32 - shuffle >= 1.0 (the 2D constraint itself transfers -> proceed to the DN/DQ/DC hierarchy, H8). PARTIAL if 70.7 <= DW32 < 72.0 (2D Dyck transfers something but presentation or supervision density costs -> H7 + RUN_DENSE=1 arm next). FAILURE if DW32 <= 70.7 or DW32 ~= shuffle (the CA story on a second source class -> H9b projection scramble before any further 2D investment). In-repo anchors: k-Dyck 72.64, random 70.02.

## Figures

![dw32_vs_kdyck_cifar100](..\..\figures\dw32_vs_kdyck_cifar100.png)

![dw32_vs_kdyck_cifar100_warmup](..\..\figures\dw32_vs_kdyck_cifar100_warmup.png)
