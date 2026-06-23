# CIFAR100 top-1 by warm-up initialization

_Generated 2026-06-22T23:32:09Z_

## Results

| init | run | best top-1 | dataset |
|---|---|---|---|
| Random | `cifar100-random` | 70.02 | CIFAR100 |
| CA-Rule110 | `cifar100-rule110` | 68.86 | CIFAR100 |
| k-Dyck | `cifar100-dyck` | 72.64 | CIFAR100 |

## Summary

Best initialization: **k-Dyck** (72.64% top-1).

Success criterion (Stage 1): the CA (Rule 110) warm-up should match or exceed random initialization, ideally approaching or beating k-Dyck.

## Figures

![focused_rule110_cifar100](../../figures/focused_rule110_cifar100.png)
