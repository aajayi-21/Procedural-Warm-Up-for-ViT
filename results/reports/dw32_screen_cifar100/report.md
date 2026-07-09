# CIFAR100 top-1: DW_32 2D Dyck screen (H6)

_Generated 2026-07-06T20:55:21Z_

## Results

| init | run | best top-1 | dataset |
|---|---|---|---|
| Random | `cifar100-random-repro` | 70.34 | CIFAR100 |
| k-Dyck 1D | `cifar100-dyck-repro` | 72.08 | CIFAR100 |
| DW32 2D | `cifar100-dw32` | 67.88 | CIFAR100 |
| DW32 shuffle | `cifar100-dw32-shuffle` | 64.84 | CIFAR100 |
| DW32 2D tuned | `cifar100-dw32-tuned` | 68.60 | CIFAR100 |

## Summary

Best initialization: **k-Dyck 1D** (72.08% top-1).

H6 screen (1 seed; differences under ~0.7 pts are noise — the in-repo Dyck seed spread is 0.67). Gate to Phase 2: any DW arm > 71.0 (design doc §6).

## Figures

![dw32_screen_cifar100](..\..\figures\dw32_screen_cifar100.png)
