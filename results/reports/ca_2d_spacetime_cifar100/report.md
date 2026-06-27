# CIFAR100 top-1: retaining CA 2-D structure (H3 test)

_Generated 2026-06-27T14:24:35Z_

## Results

| init | run | best top-1 | dataset |
|---|---|---|---|
| Random | `cifar100-random` | 70.02 | CIFAR100 |
| k-Dyck | `cifar100-dyck` | 72.64 | CIFAR100 |
| CA binary 1D | `cifar100-rule110` | 68.86 | CIFAR100 |
| CA binary 2D | `cifar100-2dpos` | 65.35 | CIFAR100 |
| CA block2d 1D | `cifar100-block2d-1dpos` | 69.17 | CIFAR100 |
| CA block2d 2D | `cifar100-block2d` | 69.42 | CIFAR100 |
| CA forward-deep 2D | `cifar100-forward-deep-2dpos` | 67.24 | CIFAR100 |
| CA next-state 2D | `cifar100-nextstate-2dpos` | 66.17 | CIFAR100 |
| CA block 1D | `cifar100-block` | 71.42 | CIFAR100 |
| CA block 2D | `cifar100-block-2dpos` | 29.75 | CIFAR100 |
| CA hard 1D | `cifar100-hard` | 71.86 | CIFAR100 |
| CA hard 2D | `cifar100-hard-2dpos` | 71.36 | CIFAR100 |
| CA iid-true 1D | `cifar100-iid-true` | 66.81 | CIFAR100 |
| CA iid-true ring | `cifar100-iid-true-1dpos` | 66.33 | CIFAR100 |
| GoL-step 2D | `cifar100-gol-true-2dpos` | 66.54 | CIFAR100 |

## Summary

Best initialization: **k-Dyck** (72.64% top-1).

Success criterion (Stage 1): the CA (Rule 110) warm-up should match or exceed random initialization, ideally approaching or beating k-Dyck.

## Figures

![ca_2d_spacetime_cifar100](..\..\figures\ca_2d_spacetime_cifar100.png)
