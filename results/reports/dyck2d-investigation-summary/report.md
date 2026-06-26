# 2D-Dyck Investigation — Work Summary

**Status:** consolidated work-log for the analysis/design stage (no code or model runs in this stage).
Ties together the CA-failure diagnosis and the 2D-Dyck pretraining design, and records exactly what was
done and what comes next.

## Problem statement

The project's Stage-1 hypothesis — *CA warm-up will match or beat k-DYCK on downstream CIFAR-100* — was
rejected across this branch and its siblings (`run/cifar100-block-a5000`, `run/cifar100-hard-5090`,
`results/ca-difficulty-sweep`). This stage answers two questions: **(1) why does CA underperform?**, and
**(2) the newly-added paper `docs/2307.16522.pdf` ("Two-dimensional Dyck words") — what is it, and how do
we build pretraining tasks from it?**

## Decisions taken (user-confirmed)

1. **Scope:** written analysis/design deliverables only — no code, no model runs. (This box is CPU-only;
   full CIFAR-100 training runs on the GPU branches.)
2. **Variants:** design both **DC_k** (Dyck crossword) and **DW_k** (well-nested boxes).
3. **Framing:** design the proposed experiments as **controlled tests of why CA failed**, not a bare
   head-to-head.

## Deliverables produced

- [`../ca-failure-analysis/report.md`](../ca-failure-analysis/report.md) — the diagnosis: the full
  best-to-worst results table and five mechanism-level hypotheses (H1 local-lookup, H2 vocabulary poverty,
  H3 2D→1D flattening, H4 determinism→memorization, H5 untargeted masking), each tied to specific code,
  plus a synthesis of the prior in-repo analysis.
- [`../dyck2d-pretraining-design/report.md`](../dyck2d-pretraining-design/report.md) — the paper explainer
  (four definitions, the strict hierarchy DW⊊DN⊊DQ⊊DC, the not-tiling-recognizable result, the
  matching-graph view), two concrete pretraining tasks (`dyck2d_crossword`, `dyck2d_boxes`) with generation
  and masking schemes, a seven-row experiment matrix mapping each run to a CA-failure hypothesis, and a
  file-level implementation blueprint.
- This work-log.

## What was actually done

- Read `docs/2307.16522.pdf` end-to-end (all 17 pages: Defs 1–7, Theorems 1–9, Corollary 1 hierarchy).
- Audited the data pipeline to locate the mechanisms behind the failure: `data/ca/{eca.py, dataset.py,
  masking.py, tokenize.py, iid_step.py}`, `data/dyck/{generator.py, masking.py, spatial.py}`,
  `model/embeddings.py`, `warmup/trainer.py`.
- Extracted every downstream score from `results/reports/cifar100-*/metrics.json` and ranked them; verified
  each quoted number against its source file (spot-checks: dyck 72.64, random 70.02, rule110 68.86,
  iid-static 65.72).
- Verified every code symbol cited in the reports exists (`eca.py:25 step`, `CloseOnlyMasking`,
  `FrozenTokenEmbedding`, `Frozen2DSinCosPositionalEmbedding`, `register_source`, `build_source`,
  `DyckConfig`, `run_curriculum`).
- Confirmed the extension seam (registry + config + frozen-embedding bypass) already supports a new source
  with no trainer changes, and that `pos_embed: sincos2d` is available for intrinsic-2D data.

## Key findings (one-line each)

- CA's failure is not "less signal" but the *wrong* signal — several CA variants score **below random
  init** (Rule 110 68.86, iid-static 65.72 vs random 70.02), the signature of an actively harmful prior.
- The strongest single piece of evidence is internal: re-laying *DYCK* into 2D (`spatial-dyck`) **hurt**
  (66.93–70.42 vs 72.64), so *how* structure is presented matters as much as the structure — and CA is
  inherently spatial.
- Deeper H3 pass (see failure report §3/§5): the row-major flatten (`data/ca/dataset.py:70`) is **identical
  on all branches** and CA **never** used the 2D `sincos2d` position code — so "CA + sincos2d" is an
  untested cell, added as control **E0**. The two failing re-mappings differ in *kind*: CA's 2D→1D flatten
  is a fixed bijection (structure **obscured**, not ambiguous), whereas `spatial-dyck`'s 1D→2D layout is
  **under-determined** (per-sample D4/permutation decouples position from role; permuted 66.93 < nested
  70.42). The grammar lift itself also under-determines (DW⊊DN⊊DQ⊊DC), motivating the determinacy
  requirement on 2D-Dyck masking.
- The 2D Dyck languages are the natural fix: intrinsically 2D (not a re-layout), not tiling-recognizable
  (genuinely long-range), tunable vocabulary, stochastic-under-hard-constraint — inverting H1–H4 while
  keeping DYCK's transferable traits.

## Recommended next steps (GPU/coding stage)

1. Implement `procedural_warmup/data/dyck2d/` (crossword + boxes) per the blueprint in the design report;
   add `tests/test_dyck2d.py` with the `is_dyck_word` row/column oracle.
2. Run the CPU-cheap checks first: `analysis/task_entropy.py` and `analysis/task_difficulty.py` on the new
   configs to confirm the masked-target entropy and unigram-baseline are DYCK-like (rich), not CA-like
   (near-binary), *before* spending GPU.
3. Prioritize experiment **E1** (DC_k vs spatial-dyck — the cleanest test of H3) and **E3** (vocabulary
   sweep — the cleanest test of H2) on a GPU branch (5090/A5000), scored against the existing baselines
   (1D DYCK 72.64, random 70.02, spatial-dyck 66.93/70.42). A DC_k result clearly above spatial-dyck would
   confirm the diagnosis even if it does not beat 1D DYCK.
4. Record each run under `results/reports/<run>/` (config + log.csv + metrics.json + report.md) per house
   style, and extend the comparison figure to include the dyck2d sources.
