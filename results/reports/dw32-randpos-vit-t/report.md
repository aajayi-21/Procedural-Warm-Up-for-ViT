# Warm-up run: dw32-randpos-vit-t

_Generated 2026-07-08T16:52:39Z_

## Configuration

- source: `dyck2d`
- masking: `corner_close_only` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.0597**; accuracy: **0.984**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints\dw32-randpos-vit-t\ckpt_step_015000.pt`.

## Figures

![dw32-randpos-vit-t_warmup_curves](..\..\figures\dw32-randpos-vit-t_warmup_curves.png)
