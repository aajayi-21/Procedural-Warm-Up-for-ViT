# Warm-up run: dyck-vit-t

_Generated 2026-06-22T20:50:09Z_

## Configuration

- source: `dyck`
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.5340**; accuracy: **0.835**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints/dyck-vit-t/ckpt_step_015000.pt`.

## Figures

![dyck-vit-t_warmup_curves](../../figures/dyck-vit-t_warmup_curves.png)
