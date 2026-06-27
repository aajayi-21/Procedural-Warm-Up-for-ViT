# Warm-up run: gol-step-true-2dpos

_Generated 2026-06-26T19:40:05Z_

## Configuration

- source: `gol_step`
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.1379**; accuracy: **0.933**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints\gol-step-true-2dpos\ckpt_step_015000.pt`.

## Figures

![gol-step-true-2dpos_warmup_curves](..\..\figures\gol-step-true-2dpos_warmup_curves.png)
