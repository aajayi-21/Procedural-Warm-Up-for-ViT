# Weight-over-time probe: dw32-smoke

_Generated 2026-07-05T23:42:44Z_

## Setup

- source: `dyck2d`, pos_embed: `sincos2d`
- snapshots: steps [0, 25, 50]
- probe batch: 64 samples from the run's own source+masking (seed 12345, cached as `probe_batch.pt`)
- distance metric: `grid`

## Result

Final mean attention distance: **7.28** (grid units; max possible 18.4 grid / 195 seq). Short-range profiles across all blocks were the CA failure signature; compare against the k-Dyck reference run before reading anything into the downstream number.

## Figures

![dw32-smoke_probe_drift](..\..\figures\dw32-smoke_probe_drift.png)

![dw32-smoke_probe_attn_distance](..\..\figures\dw32-smoke_probe_attn_distance.png)
