# Warm-up run: spatial-dyck-nested

_Generated 2026-06-25T21:52:39Z_

## Configuration

- source: `spatial_dyck`
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **1.1527**; accuracy: **0.691**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints/spatial-dyck-nested/ckpt_step_015000.pt`.

## Figures

![spatial-dyck-nested_warmup_curves](../../figures/spatial-dyck-nested_warmup_curves.png)
