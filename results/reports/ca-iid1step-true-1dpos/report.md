# Warm-up run: ca-iid1step-true-1dpos

_Generated 2026-06-26T19:01:44Z_

## Configuration

- source: `ca_step`
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.0594**; accuracy: **0.968**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints\ca-iid1step-true-1dpos\ckpt_step_015000.pt`.

## Figures

![ca-iid1step-true-1dpos_warmup_curves](..\..\figures\ca-iid1step-true-1dpos_warmup_curves.png)
