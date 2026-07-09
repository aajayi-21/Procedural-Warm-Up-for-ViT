# Weight-over-time probe: dw32-tuned-vit-t

_Generated 2026-07-06T14:39:52Z_

## Setup

- source: `dyck2d`, pos_embed: `sincos2d_tuned`
- snapshots: steps [0, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12500, 15000]
- probe batch: 64 samples from the run's own source+masking (seed 12345, cached as `probe_batch.pt`)
- distance metric: `grid`

## Result

Final mean attention distance: **6.48** (grid units; max possible 18.4 grid / 195 seq). Short-range profiles across all blocks were the CA failure signature; compare against the k-Dyck reference run before reading anything into the downstream number.

## Figures

![dw32-tuned-vit-t_probe_drift](..\..\figures\dw32-tuned-vit-t_probe_drift.png)

![dw32-tuned-vit-t_probe_attn_distance](..\..\figures\dw32-tuned-vit-t_probe_attn_distance.png)
