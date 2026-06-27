# Warm-up run: ca-rule110-block2d

_Generated 2026-06-26T16:42:14Z_

## Configuration

- source: `ca` (rule 110)
- masking: `block2d` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.2904**; accuracy: **0.835**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints\ca-rule110-block2d\ckpt_step_015000.pt`.

## Figures

![ca-rule110-block2d_warmup_curves](..\..\figures\ca-rule110-block2d_warmup_curves.png)
