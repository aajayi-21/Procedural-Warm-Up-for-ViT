# Procedural Warm-Up for Vision Transformers — Cellular-Automata Extension

A research codebase for **procedural warm-up**: a brief masked-token pretraining stage that
exposes a Vision Transformer to procedurally generated, non-visual symbolic data before
standard image training. It extends Shinnick et al., *"Can You Learn to See Without Images?
Procedural Warm-Up for Vision Transformers"* (`docs/2511.13945v2.pdf`) with a **cellular-automata
(CA) data source** alongside the original formal grammars.

**Hypothesis (Stage 1).** CA spacetime diagrams carry rich local-to-global ("light-cone")
structure and sit at/above the top of the Chomsky hierarchy (Rule 110 is Turing-complete).
Warming a ViT on CA data should instil useful computational inductive biases that match or
beat the k-Dyck grammar warm-up on downstream image accuracy. See `docs/cellular-automata.md`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # see requirements.txt for the CUDA wheel note
```

Targets CUDA GPUs (5090 / 4080 / A1000); falls back to CPU automatically (for tests/smoke runs).

## Quickstart

```bash
# 0. Validate the CA generator (spacetime figures + complexity table -> results/)
python -m procedural_warmup.analysis.report

# 1. Procedural warm-up on ECA Rule 110 (writes a stripped checkpoint for transfer)
python -m procedural_warmup.warmup.cli --config procedural_warmup/config/files/ca-rule110.yaml

# 2. Downstream transfer to CIFAR-100
python -m procedural_warmup.downstream.main \
  --config procedural_warmup/config/files/downstream-cifar.yaml \
  --run-name cifar100-rule110 \
  --init checkpoints/ca-rule110/ckpt_step_015000_stripped.pt

# Or run the whole focused experiment (Rule 110 vs random vs k-Dyck) end to end:
bash scripts/run_focused_experiment.sh CIFAR100
```

## Pipeline

```
data source ──▶ masked-token warm-up ──▶ strip checkpoint ──▶ image training ──▶ analysis
(ca | dyck | gol)   (frozen embeddings,      (keep blocks+norm)   (timm ViT-T,       (figures +
                     ViT blocks trained)                           CIFAR-10/100)      reports)
```

The warm-up bypasses the patch embedding and feeds tokens through **frozen** random
token + positional embeddings, so only the transformer blocks learn structure. Those blocks
are the weights carried into image training; the embeddings and MLM head are discarded.

## Repository layout

```
procedural_warmup/
  config/      dataclass schema + YAML configs (config/files/*.yaml)
  data/        source registry; data/dyck (grammar) and data/ca (ECA + Game of Life)
  model/       frozen embeddings, ProceduralViT wrapper, model factory
  warmup/      step-based trainer, optim, checkpoint stripping, curriculum scaffold
  downstream/  timm-based CIFAR trainer, weight transfer, engine
  analysis/    figures, Lempel-Ziv complexity, layerwise probe, report CLIs
tests/         pytest unit + smoke tests
scripts/       run_warmup.sh, run_downstream.sh, run_focused_experiment.sh
results/       figures/ and reports/<run>/ (config, log.csv, metrics.json, report.md)
docs/          the paper, research notes, and the CA design doc
```

## Adding a new procedural data source

Implement the two tiny contracts in `data/base.py` and register a builder — **no change to
the trainer**:

```python
from procedural_warmup.data import register_source

@register_source("my_source")
def build_my_source(cfg):
    return MyDataset(cfg), MyMasking(cfg)   # __getitem__ -> LongTensor(N,);
                                            # masking -> (masked_input, targets, mask)
```

The same seam powers the **Stage-2 curriculum** (`warmup/curriculum.py`): an ordered list of
source stages on one shared model, configured entirely in YAML (`cfg.curriculum`).

## Results & reproducibility

Every major step and experimental run writes to `results/reports/<run>/`: the frozen
`config.yaml`, a streaming `log.csv`, a `metrics.json`, and a markdown `report.md` linking
figures under `results/figures/`. All figures are produced in code and regenerable from
saved results.

## Testing

```bash
python -m pytest            # ECA correctness, tokenize, masking, registry, warm-up smoke test
```
