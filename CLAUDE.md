# Procedural Warm-Up for Vision Transformers — Experimentation and Research Guide

High-level guidance for working in this repository. Read .claude/ for detailed
specs and docs/ for primary sources before making non-trivial changes.

## Project Overview

This project studies procedural warm-up for Vision Transformers (ViTs): a brief
pretraining stage that exposes a ViT to procedurally generated, non-visual symbolic
data before standard image-based training. The approach builds on and extends "Can You
Learn to See Without Images? Procedural Warm-Up for Vision Transformers" (Shinnick et
al.), which shows that masked-token pretraining on formal-grammar sequences (e.g.
k-DYCK) instils generic computational inductive biases that improve convergence, data
efficiency, and downstream accuracy on image classification.

Project page: https://zlshinnick.github.io/procedural-pretraining-page/
Reference implementation (used as the design template): https://github.com/zlshinnick/procedural-warmup-vit
Parent paper: `docs/2511.13945v2.pdf` (Shinnick et al., arXiv:2511.13945v2)
2D Dyck paper: `docs/2307.16522.pdf` (Crespi Reghizzi et al., arXiv:2307.16522)

## Project History: the CA Program is Concluded (Negative Result)

The first phase of the project tested cellular-automata (CA) data as a warm-up source
(the former Hypothesis 2, plus the H1–H5 failure post-mortem and the H3 2D-spacetime
test). **The CA direction is finished and its verdict is negative — do not reopen it
without new motivation.** Key conclusions, with the evidence in `results/reports/`:

- **CA is a worse warm-up substrate than k-Dyck for this transfer objective, and the
  limitation is intrinsic to the CA operator** (local + deterministic), not to how it
  was encoded. Best engineered CA: 72.22 (block, s1) / 71.86 (hard) vs k-Dyck 72.64
  (canonical) on CIFAR-100; most CA variants land *below* random init (70.02).
- **The CA operator itself transfers ~nothing.** The decisive true-vs-shuffled operator
  controls came back ≈ 0 with the geometry hidden *and* exposed (66.81 ≈ 66.90;
  66.33 ≈ 66.36). CA's only gains came from generic, non-CA-specific levers:
  vocabulary (block tokenization → ~130 types) and a non-local objective (forward
  masking).
- **H3 (the flatten/position-code hypothesis) is rejected.** Exposing the 2D spacetime
  geometry via `sincos2d` made transfer *worse* (68.86 → 65.35 in the single-factor
  test). See `results/reports/ca-2d-spacetime/` (design) and
  `results/reports/ca-2d-spacetime-verdict/` (results + verdict), plus
  `results/reports/ca-failure-analysis/` (post-mortem).
- **What k-Dyck has and CA lacks — the "transferable triad":** (a) long-range binding
  (stack tracking under close-only masking), (b) a rich type space (K = 130 tokens),
  (c) a constraint *distribution* rather than a deterministic map. This triad is the
  working theory of why warm-up transfers, and it drives the current program.
- **A cautionary result on presentation:** re-laying-out 1D Dyck onto a 2D grid
  (spatial-dyck) *hurt* (nested 70.42 / permuted 66.93 vs 72.64), so "2D helps" is a
  hypothesis to test, never an assumption.

## Current Goal: Explicit 2D k-Dyck Pretraining (Hypotheses H6–H11)

The active research program is warm-up on **genuinely two-dimensional Dyck languages**
(Crespi Reghizzi et al., `docs/2307.16522.pdf`) — the first candidate source that keeps
the full transferable triad *and* is natively 2D rather than a re-layout: corner-quadruple
matching gives long-range binding in two axes, the alphabet Δ_k = {aᵢ, bᵢ, cᵢ, dᵢ}
gives 4k content symbols (k = 32 → K = 130, matching k-Dyck's vocabulary), and
generation is stochastic under hard global constraints. The published hierarchy
**DW_k ⊊ DN_k ⊊ DQ_k ⊊ DC_k** provides a built-in constraint/determinacy gradient,
mirroring the parent paper's Chomsky-hierarchy sweep.

**An inspirational/working experiment design is `docs/2d-dyck-experiment-design.md`.** Read it
before touching this program. In brief:

- **H6** (headline): well-nested DW_32 warm-up, presented natively on the 14×14 token
  grid with `sincos2d` positions and determinacy-audited masking, reaches the k-Dyck
  band on CIFAR-100.
- **H7**: native 2D beats rasterized 1D *iff* the position code exposes geometry.
- **H8**: the DW ⊊ DN ⊊ DQ ⊊ DC chain is a usable determinacy/constraint gradient.
- **H9**: operator controls — the 2D constraint, not its texture, carries the gain.
- **H10**: the 4k corner alphabet reproduces the k-Dyck vocabulary curve.
- **H11**: structure-targeted, determinacy-audited masking is necessary
  (corner-close-only is the 2D analog of close-only masking).



### Anchor numbers (CIFAR-100 top-1, in-repo unless noted)

| Anchor | Top-1 |
|---|---:|
| k-Dyck (best of 3 in-repo runs) | 72.80 |
| k-Dyck (canonical run) | 72.64 |
| Best CA after engineering (block, s1) | 72.22 |
| Random init | 70.02 |
| Spatial-Dyck nested / permuted | 70.42 / 66.93 |
| Parent paper ViT-T C100: warm-up / random (3 seeds) | 71.98 ± 0.74 / 68.52 ± 0.27 |
| Parent paper ImageNet-1K ViT-B delta | +1.72 |

All verdicts use **in-repo** anchors (both arms of any comparison run through identical
local infrastructure).

### Deferred: curriculum pretraining

Curriculum warm-up (multiple rounds of pretraining on different data types) remains on
the roadmap but is explicitly deferred until after the 2D Dyck hierarchy sweep (design
doc §8). Keep the pipeline's scaffolding curriculum-compatible, but do not build
curriculum features ahead of the H6–H9 verdicts.

## Environment and Tooling

All code implementation should be using Python. For machine learning, make sure to
allow GPU acceleration using CUDA (target systems will be NVIDIA 5090, 4080, A1000).
Heavy runs (15k-step warm-ups, 300-epoch downstream trainings) execute on the GPU
boxes via the `scripts/run_*.sh` entry points; validate new code with the CPU test
suite first.

Use pip for dependency management (not conda or poetry). Keep dependencies pinned
in requirements.txt. Create a virtual environment for local work using venv.
Install with pip install -r requirements.txt. When adding a dependency, add it to
requirements.txt in the same change.

## Code Principles

- Modular. Adding a new pretraining data source (a new grammar family, etc.) or a new
model/training variant should require implementing a small, well-defined interface —
not editing the core training loop. Keep data generation, the warm-up stage,
image-based training, and evaluation as separate, swappable components.
- Interpretable. Favour clear, readable code over cleverness. Each module should be
understandable in isolation, with docstrings explaining what a component does and
why it exists in the context of the research question.
- Extensible by default. New procedural data generators should plug in behind the
common generator interface so the rest of the pipeline (embedding bypass, masking,
warm-up, transfer to vision) works unchanged. New 2D Dyck sources follow the existing
per-source pattern under `procedural_warmup/data/` (as `ca/`, `dyck/`, `dyck_shuffle/`,
`ww/` do); see the design doc §5 for the intended module contents (samplers per family,
independent validators, the determinacy auditor, controls, tokenization).
- Validation before GPU time: property tests for every sampler (outputs pass the
family's membership checker), inclusion sanity across the hierarchy, auditor
correctness cross-checked by brute force on small grids, and a throughput check
against the 256 × 15k-step budget (design doc §5 gates).
- Try to follow good style, such as the style specified here: https://google.github.io/styleguide/pyguide.html

## Repository Layout

- `docs/` — primary sources and design notes. Key files: the parent paper
  (`2511.13945v2.pdf`), the 2D Dyck paper (`2307.16522.pdf`), the active experiment
  design (`2d-dyck-experiment-design.md`), the concluded CA program spec
  (`cellular-automata.md`, historical), and external research reviews
  (`compass_artifact_*.md`).
- `procedural_warmup/` — the source package: `data/` (per-source generators and
  masking), `model/` (ViT, frozen embeddings, factory), `warmup/` (masked-token
  pretraining + weight strip/transfer), `downstream/` (image training), `analysis/`
  (comparison figures), `config/`, `experiments/`.
- `scripts/` — shell entry points for warm-up, downstream, and experiment sweeps.
- `tests/` — CPU-runnable test suite; keep it passing.
- `results/reports/<run>/` — per-run configs, `metrics.json`, and `report.md`;
  `results/figures/` — generated figures.
- `checkpoints/`, `data/` — model weights and datasets (not committed).

## Figures

- Experiments should produce reproducible figures (training curves, ablations,
layerwise analyses, data-efficiency plots, etc.).
- Implement figure generation in code (not by hand), parameterised so a figure can be
regenerated from saved results. Save outputs to results/.

## Conventions
- Use the original structure and conventions of the reference repo as inspiration when reasonable.
- When introducing a new experiment, record its configuration and write its outputs
(figures + a short report) to results/reports/<run>/, following the existing per-run
pattern (`metrics.json` + `report.md`).
- Verdict/analysis reports should read numbers verbatim from the committed
`metrics.json` files and link their companion reports (see
`results/reports/ca-2d-spacetime-verdict/report.md` as the template).
- Keep changes small and composable so new pretraining data and functionality are easy
to add later.
