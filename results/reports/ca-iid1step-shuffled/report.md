# Warm-up run: ca-iid1step-shuffled

_Generated 2026-06-24T22:47:57Z_

## Configuration

- source: `ca_step`
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.6620**; accuracy: **0.610**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints/ca-iid1step-shuffled/ckpt_step_015000.pt`.

## Figures

![ca-iid1step-shuffled_warmup_curves](../../figures/ca-iid1step-shuffled_warmup_curves.png)
