# Warm-up run: dyck-vit-t-s1

_Generated 2026-06-25T17:03:20Z_

## Configuration

- source: `dyck`
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.5331**; accuracy: **0.834**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints/dyck-vit-t-s1/ckpt_step_015000.pt`.

## Figures

![dyck-vit-t-s1_warmup_curves](../../figures/dyck-vit-t-s1_warmup_curves.png)
