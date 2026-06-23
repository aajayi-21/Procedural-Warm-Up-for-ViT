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

Warm up on a different source by swapping the config — e.g. **2-D Game of Life**, or the
harder CA that matches k-Dyck difficulty — then transfer the stripped checkpoint as in step 2:

```bash
python -m procedural_warmup.warmup.cli --config procedural_warmup/config/files/gol.yaml
python -m procedural_warmup.warmup.cli --config procedural_warmup/config/files/ca-rule110-block.yaml
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

## Warm-up data sources & task difficulty

Sources plug in behind the registry and are chosen by the warm-up config. A warm-up only
transfers if its masked-token task is **hard but learnable**: a too-easy task (one a trivial
predictor solves) teaches shallow features that don't help — the paper's WW result. Inspect
any task's intrinsic difficulty (entropy, distinct targets, trivial-baseline accuracy):

```bash
python -m procedural_warmup.analysis.task_difficulty
```

| source | config | structure |
|---|---|---|
| `dyck` | `dyck-vit-t.yaml` | context-free, nested (reference; strongest grammar) |
| `dyck_shuffle` | `dyck-shuffle.yaml` | context-sensitive, crossing dependencies |
| `ww` | `ww.yaml` | regular copy-language (negative control — hurts) |
| `ca` | `ca-rule110.yaml` | 1-D ECA Rule 110, **binary** (baseline — too easy) |
| `ca` | `ca-rule110-block.yaml` | Rule 110, **block** tokens (128 symbols ≈ k-Dyck difficulty) — **verified to learn** |
| `ca` | `ca-rule110-forward.yaml` | Rule 110, forward masking — *ablation; did not learn in probes* |
| `ca` | `ca-rule110-hard.yaml` | Rule 110, block + forward — *ablation; did not learn in probes* |
| `gol` | `gol.yaml` | **2-D Game of Life**, next-state — *Stage-4 scaffold; not yet verified* |

**What actually learns (from CPU learning-curve probes).** Two factors decide whether a
warm-up task is learnable in practice, beyond the `task_difficulty` floor:
- **block tokenization** — the default binary Rule-110 task is near-trivial (2 symbols, ~1
  bit, majority-guess 56%), so it underperforms k-Dyck. `ca-rule110-block.yaml` raises the
  per-token vocabulary to 128 symbols, matching k-Dyck's ~6-bit difficulty (raising `vocab.K`
  alone does **not** help — binary CA emits 2 states regardless). **ca-block learns strongly.**
- **random vs forward masking** — scattered random/close masking (k-Dyck, ca-block) learns;
  **`forward` masking** (predict a whole contiguous masked region with no interspersed visible
  context) did **not** learn in the probe budget. So `ca-rule110-forward`/`-hard` are kept only
  as ablations. The verified-effective recipe is **block tokenization + random masking**.

**Game of Life** (`gol.yaml`) stacks two consecutive frames (state *t*, *t+1*) and predicts
the future frame (B3/S23). It is a **Stage-4 research scaffold, not yet verified effective**:
across CPU probes this task — and every variant tried (binary/block, forward/random masking,
frozen/learnable positions, single-frame inpainting) — did not learn within the budget where
ECA-block and k-Dyck do. The likely reason is that GoL's rule-bearing dependency is *2-D and
cross-frame*, which the 1-D-token + random-positional ViT does not crack quickly (the ECA
spacetime's dependency is local and 1-D). It is provided runnable for GPU validation or
redesign (e.g. a 2-D positional encoding); the verified 2-D-structured source is ECA-block.

## Repository layout

```
procedural_warmup/
  config/      dataclass schema + YAML configs (config/files/*.yaml)
  data/        source registry; grammars (dyck, dyck_shuffle, ww) + ca (ECA + Game of Life)
  model/       frozen embeddings, ProceduralViT wrapper, model factory
  warmup/      step-based trainer, optim, checkpoint stripping, curriculum scaffold
  downstream/  timm-based CIFAR trainer, weight transfer, engine
  analysis/    figures, complexity, task-difficulty diagnostic, comparison + report CLIs
  experiments/ multi-run drivers (comparison.py: additive + substitutive)
tests/         pytest unit + smoke tests
scripts/       run_warmup.sh, run_downstream.sh, run_focused_experiment.sh, run_comparison.sh
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

## Complete method comparison (additive + substitutive)

Reproduces the paper's two analyses across all warm-up methods (random / CA / k-Dyck /
k-Dyck-Shuffle), training on the **real** dataset after each warm-up:

- **Additive** — every method warmed up then trained on the *full* dataset; the gain over
  random init is the additive benefit (paper Table 2/4).
- **Substitutive** — methods trained on *fractions* of the images; how much data each
  warm-up makes up for vs random-init at full data (paper Fig 3, "X% fewer images").

```bash
python -m procedural_warmup.experiments.comparison \
  --config procedural_warmup/config/files/comparison.yaml --dry-run   # preview the matrix
bash scripts/run_comparison.sh                                        # run it (resumable)
```

The driver runs each warm-up (skipping cached ones), trains every (method × data-fraction)
run on real CIFAR (skipping finished ones — fully resumable), then writes
`results/reports/comparison/report.md` with an **additive** table + bar chart, a
**substitutive** table + data-efficiency curves, and the per-method "data saved" estimate.
Edit `comparison.yaml` to change methods, fractions, dataset, or epochs. Rebuild just the
report from finished runs with `--only-report`.

## Monitoring progress

Training shows live **tqdm progress bars** (ETA + running loss/accuracy) for both the
warm-up step loop and the downstream epoch/batch loops. The training-curve figure under
`results/figures/` also **refreshes during the run** (warm-up: every `logging.figure_every`
steps; downstream: every eval), so an open PNG stays current. Set `logging.progress: false`
for plain line logs (CI / redirected output).

From a second terminal (or over SSH on the GPU box), watch any in-flight run — a text
dashboard with ASCII sparklines that needs no GUI:

```bash
python -m procedural_warmup.analysis.watch ca-rule110 --live          # warm-up run
python -m procedural_warmup.analysis.watch cifar100-rule110 --live    # downstream run
python -m procedural_warmup.analysis.watch ca-rule110 --figure        # regenerate the PNG now
```

## Performance

Downstream training is input-bound (tiny model, upscaled CIFAR), so the data pipeline is
tuned to keep the GPU fed: the CPU augments at native `data.cpu_size` (32px) and the batch is
**upscaled to 224 on the GPU** (`downstream/engine.py`), shrinking the host→device transfer
~50×; loaders use auto worker count + persistent workers; loss is accumulated on-GPU
(one sync/epoch); cuDNN benchmark + TF32 are enabled. A clear `[device] training on GPU: ...`
banner prints at startup (or a loud CPU-fallback warning). The warm-up stage is already light
(token data). Set `train.eval_interval > 1` to validate less often.

## Testing

```bash
python -m pytest            # ECA/GoL correctness, tokenize, masking, registry, warm-up smoke, comparison
```
