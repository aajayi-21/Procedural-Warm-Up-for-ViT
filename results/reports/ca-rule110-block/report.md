# Warm-up run: ca-rule110-block

_Generated 2026-06-24T13:43:42Z_

## Configuration

- source: `ca` (rule 110)
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.2110**; accuracy: **0.936**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints/ca-rule110-block/ckpt_step_015000.pt`.

## Figures

![ca-rule110-block_warmup_curves](../../figures/ca-rule110-block_warmup_curves.png)
