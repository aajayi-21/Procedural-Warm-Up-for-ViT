# Weight-over-time probe: dyck-repro

_Generated 2026-07-06T14:39:23Z_

## Setup

- source: `dyck`, pos_embed: `random`
- snapshots: steps [0, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12500, 15000]
- probe batch: 64 samples from the run's own source+masking (seed 12345, cached as `probe_batch.pt`)
- distance metric: `grid`

## Result

Final mean attention distance: **6.10** (grid units; max possible 18.4 grid / 195 seq). Short-range profiles across all blocks were the CA failure signature; compare against the k-Dyck reference run before reading anything into the downstream number.

## Figures

![dyck-repro_probe_drift](..\..\figures\dyck-repro_probe_drift.png)

![dyck-repro_probe_attn_distance](..\..\figures\dyck-repro_probe_attn_distance.png)
