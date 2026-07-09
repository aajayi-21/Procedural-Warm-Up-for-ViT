# Warm-up run: dyck-repro

_Generated 2026-07-06T05:00:43Z_

## Configuration

- source: `dyck`
- masking: `random` @ ratio 0.5
- steps: 15000, batch 256
- optimizer: AdamW lr=0.002 wd=0.05

## Result

Final masked-token loss (running avg): **0.4542**; accuracy: **0.864**.

Stripped checkpoint for transfer: run `process.py` on `checkpoints\dyck-repro\ckpt_step_015000.pt`.

## Figures

![dyck-repro_warmup_curves](..\..\figures\dyck-repro_warmup_curves.png)
