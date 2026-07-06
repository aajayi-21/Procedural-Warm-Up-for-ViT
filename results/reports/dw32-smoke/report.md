# Warm-up run: dw32-smoke

_Generated 2026-07-05T23:42:22Z_

## Configuration

- source: `dyck2d`
- masking: `corner_close_only` @ ratio 0.5
- steps: 50, batch 8
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **3.6480**; accuracy: **0.037**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints\dw32-smoke\ckpt_step_000050.pt`.

## Figures

![dw32-smoke_warmup_curves](..\..\figures\dw32-smoke_warmup_curves.png)
