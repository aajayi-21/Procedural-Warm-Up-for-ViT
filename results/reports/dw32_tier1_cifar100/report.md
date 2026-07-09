# CIFAR100 top-1: DW_32 position-code factorial (Tier 1, post-H6)

_Generated 2026-07-08T20:22:36Z_

## Results

| init | run | best top-1 | dataset |
|---|---|---|---|
| Random | `cifar100-random-repro` | 70.34 | CIFAR100 |
| k-Dyck 1D (randpos) | `cifar100-dyck-repro` | 72.08 | CIFAR100 |
| DW32 randpos | `cifar100-dw32-randpos` | 69.07 | CIFAR100 |
| DW32 randpos shuffle | `cifar100-dw32-randpos-shuffle` | 67.47 | CIFAR100 |
| DW32 sincos2d | `cifar100-dw32` | 67.88 | CIFAR100 |
| DW32 sincos2d shuffle | `cifar100-dw32-shuffle` | 64.84 | CIFAR100 |

## Summary

Best initialization: **k-Dyck 1D (randpos)** (72.08% top-1).

Tier-1 pre-registered reads (1 seed, P7 noise band 0.7; screen verdict: results/reports/dw32-screen/report.md): (a) DW32-randpos >= ~70.7 -> the sincos2d presentation was the primary killer; DW is viable -> next stack harder masking (min-span filter / c+d masking) on randpos and re-screen. (b) DW32-randpos ~= 68 (== the sincos2d arm) -> presentation exonerated -> masking/task-difficulty is primary; go to H11 arms. (c) randpos treatment-minus-control is the structure signal under fixed presentation (sincos2d pair: +3.04); if it collapses at randpos, the structure transfer depended on geometry being exposed -> H7's full factorial becomes the program. Anchors this branch: random 70.34, k-Dyck 72.08.

## Figures

![dw32_tier1_cifar100](..\..\figures\dw32_tier1_cifar100.png)

![dw32_tier1_cifar100_warmup](..\..\figures\dw32_tier1_cifar100_warmup.png)
