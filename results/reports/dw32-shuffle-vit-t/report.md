# Warm-up run: dw32-shuffle-vit-t

_Generated 2026-07-06T14:19:18Z_

## Configuration

- source: `dyck2d_shuffle`
- masking: `corner_close_only` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **2.6860**; accuracy: **0.131**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints\dw32-shuffle-vit-t\ckpt_step_015000.pt`.

## Figures

![dw32-shuffle-vit-t_warmup_curves](..\..\figures\dw32-shuffle-vit-t_warmup_curves.png)
