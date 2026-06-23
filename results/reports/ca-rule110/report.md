# Warm-up run: ca-rule110

_Generated 2026-06-22T20:41:40Z_

## Configuration

- source: `ca` (rule 110)
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.1042**; accuracy: **0.946**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints/ca-rule110/ckpt_step_015000.pt`.

## Figures

![ca-rule110_warmup_curves](../../figures/ca-rule110_warmup_curves.png)
