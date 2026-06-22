# Procedural Warm-Up for Visition Transformers Experimentation and Research Guide

High-level guidance for working in this repository. Read .claude/ for detailed
specs and docs/ for primary sources before making non-trivial changes.

## Project Overview

This project studies procedural warm-up for Vision Transformers (ViTs): a brief
pretraining stage that exposes a ViT to procedurally generated, non-visual symbolic
data before standard image-based training. The approach builds on and extends "Can You Learn
to See Without Images? Procedural Warm-Up for Vision Transformers" (Shinnick et al.),
which shows that masked-token pretraining on formal-grammar sequences (e.g. k-DYCK)
instils generic computational inductive biases that improve convergence, data
efficiency, and downstream accuracy on image classification.


Project page: https://zlshinnick.github.io/procedural-pretraining-page/
Reference implementation (use as the design template): https://github.com/zlshinnick/procedural-warmup-vit
Research Paper: In the local directory at `/docs/2511.13945v2.pdf`

### Implementation Stages

1. First implement the experiment on cellular automate at `docs/cellular-automata.md`. Ensure there is scaffolding 
for the next stage
2. Implement the ability to allow for a curriculum of pretraining which is multiple round of pretraining with different types of data (see `docs/curriculum.md`)

## Core Research Hypothesis (Cellular Automata Pretraining)

Read `doc/cellular-automata.md` for more details

Cellular-automata (CA) data as a procedural warm-up source. CA-generated
sequences carry rich, structured spatiotemporal dependencies and are known to teach
generic computational mechanisms. The hypothesis is that warming up a ViT on
CA-generated data — in place of, or in addition to, formal-grammar data — will impart
useful inductive biases and improve downstream vision performance, potentially beyond
what k-DYCK alone achieves. Work should make it straightforward to:


generate CA sequences as a new procedural data source alongside the existing grammars,
run the same masked-token warm-up + standard image-training pipeline on them, and
measure downstream accuracy / convergence against the existing baselines.

Note: the hypotheses file in this project currently numbers these differently. Treat
the CA-pretraining direction above as the active Hypothesis 2 for this codebase; flag
any conflict rather than silently renumbering.

## Environment and Tooling

All code implementation should be using Python. For machine learning, make sure to 
allow GPU=acceleration using CUDA (target systems will be NVIDIA 5090, 4080, A1000). 

Use pip for dependency management (not conda or poetry). Keep dependencies pinned
in requirements.txt. Create a virtual environment for local work using venv.
Install with pip install -r requirements.txt. When adding a dependency, add it to
requirements.txt in the same change. 

## Code Principles

- Modular. Adding a new pretraining data source (a new grammar, a cellular
automaton, etc.) or a new model/training variant should require implementing a small,
well-defined interface — not editing the core training loop. Keep data generation,
the warm-up stage, image-based training, and evaluation as separate, swappable
components.
- Interpretable. Favour clear, readable code over cleverness. Each module should be
understandable in isolation, with docstrings explaining what a component does and
why it exists in the context of the research question.
- Extensible by default. New procedural data generators should plug in behind a
common generator interface so the rest of the pipeline (embedding bypass, masking,
warm-up, transfer to vision) works unchanged.
- Try to follow good style, such as the style specified here: https://google.github.io/styleguide/pyguide.html 

## Repository Layout

- docs/ — primary sources (the paper, related work) and project notes.
results/ — generated figures and reports go here.
- Source code — model, data generators, warm-up stage, image training, and evaluation,
organised modularly per the principles above.

## Figures

- Experiments should be produce reproducible figures (training curves, ablations,
layerwise analyses, data-efficiency plots, etc.).
- Implement figure generation in code (not by hand), parameterised so a figure can be
regenerated from saved results. Save outputs to results/.

## Conventions
- Use the original structure and conventions of the reference repo as inspiration when reasonable
- When introducing a new experiment, record its configuration and write its outputs
(figures + a short report) to results/.
- Keep changes small and composable so new pretraining data and functionality are easy
to add later.
