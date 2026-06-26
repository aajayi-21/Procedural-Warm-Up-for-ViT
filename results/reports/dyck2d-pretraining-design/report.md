# 2D-Dyck Pretraining: Paper Explainer and Task Design

**Status:** design report (no new code or runs). Explains `docs/2307.16522.pdf` and proposes new
procedural warm-up tasks built from it, framed as controlled tests of the CA-failure diagnosis in
[`../ca-failure-analysis/report.md`](../ca-failure-analysis/report.md).

## 1. The paper, in plain language

**"Two-dimensional Dyck words"** (Crespi Reghizzi, Restivo, San Pietro, 2023; arXiv 2307.16522). It is
pure formal-language theory — *no machine learning* — answering one question: **what is "balanced
parentheses" in two dimensions?**

In 1D, the Dyck language `D_k` is the set of strings over `k` bracket pairs that fully cancel
(`a_i a_i' → ε`). It is the canonical context-free object (Chomsky–Schützenberger: every CF language is a
homomorphic image of a Dyck language intersected with a regular one). The paper lifts this to **pictures**
(rectangular grids of symbols). The key move: a 1D parenthesis *pair* becomes, in 2D, a **quadruple of
corner symbols** `a, b, c, d` sitting on the four vertices of a rectangle —

```
 a · · · b      a = top-left      (opens its row, opens its column)
 ·       ·      b = top-right     (closes its row, opens its column)
 ·       ·      c = bottom-left   (opens its row, closes its column)
 c · · · d      d = bottom-right  (closes its row, closes its column)
```

The paper gives **four** definitions of a 2D Dyck language, of increasing richness, forming a **strict
hierarchy** (Corollary 1):

> **DW_k ⊊ DN_k ⊊ DQ_k ⊊ DC_k**

- **DW_k — well-nested Dyck** (Def. 2). Built by *nesting accretion*: take a sub-picture and wrap it in a
  rectangle whose top/bottom borders are a row-Dyck word and whose left/right borders are a column-Dyck
  word, then tessellate (Simplot closure). The boxes are nested or disjoint, never crossing — the direct
  2D analogue of well-nested 1D brackets. **Closest to the existing 1D k-DYCK.**
- **DN_k — neutralizable Dyck** (Def. 3). Generalizes the 1D *cancellation* rule: a rectangle whose
  interior is already all "neutral" `N` can have its four corners neutralized; a picture is valid iff a
  sequence of such steps reduces it entirely to `N`. (Example 1 in the paper shows a 4×4 reducing in six
  steps.)
- **DC_k — Dyck crossword** (Def. 4–5). The cleanest and most genuinely-2D definition: a picture is valid
  iff **every row is a 1D Dyck word** (row alphabet pairs `[a,b]` and `[c,d]`) **and every column is a 1D
  Dyck word** (column alphabet pairs `[a,c]` and `[b,d]`). Pure row-and-column constraint, no nesting
  requirement.
- **DQ_k — quaternate Dyck** (Def. 7). DC_k restricted so all matching-graph circuits have length exactly
  4 — i.e. only plain rectangles, no longer closed paths.

Two theoretical facts worth carrying into the design:

1. **None of these are tiling-recognizable** (Theorems 1, 2, 4) — the 2D analogue of "Dyck is not
   regular." They are genuinely the 2D "context-free-like" object, not a dressed-up local pattern. This is
   exactly the property the project wants a warm-up source to have: structure that a finite local rule
   cannot capture (contrast the CA failure, where the signal *was* a local rule — H1).
2. **The matching graph** (Def. 6): give every cell one *row-edge* to its row-partner and one *column-edge*
   to its column-partner. The graph decomposes into disjoint **circuits** that alternate horizontal and
   vertical edges, each of length a multiple of 4 with label in `(a b d c)+`. A length-4 circuit is a
   rectangle; longer circuits (the paper proves unbounded length `4+8h`, Theorem 8) are richer closed
   paths. **This graph is what a masking objective targets:** masking one node forces the model to traverse
   its circuit using both the row and the column constraint.

## 2. Why this is the right next experiment

The companion analysis attributes CA's failure to five properties (H1–H5). A good replacement source
should *invert* them while keeping what made DYCK work. The 2D Dyck languages do exactly this:

| CA failure mode | What 2D-Dyck supplies instead |
|---|---|
| H1 local lookup | Not tiling-recognizable → genuinely long-range; predicting a corner needs the whole row *and* column circuit |
| H2 K=4 vocabulary | `k` corner-quadruples → `4k+specials` types; tunable, can match DYCK's 130 |
| H3 spatial re-layout hurts | Structure is *intrinsically* 2D — the grid *is* the object, not a flattened sequence; pairs with `model/embeddings.py:56 Frozen2DSinCosPositionalEmbedding` |
| H4 deterministic memorization | Stochastic generation under a hard global constraint (like 1D DYCK) |
| H5 untargeted masking | Natural structure-targeted mask: hide a circuit's corner, predict from its partners |

Critically, this is **not** the failed `spatial-dyck`. Spatial-DYCK took a *1D* Dyck string and merely
re-drew it onto a grid (Hilbert / permuted layout, `data/dyck/spatial.py`) — the structure stayed 1D and
the re-layout *hurt* (66.93–70.42 vs 72.64). A DC_k picture has long-range constraints **along both axes by
definition**; there is no 1D string underneath. The headline scientific question becomes:

> Does an **intrinsically 2D** grammar beat 1D k-DYCK (72.64) and clear random (70.02), in the regime where
> a **re-laid-out** 1D grammar (spatial-dyck) failed (66.93–70.42)?

## 3. Proposed pretraining tasks (both DC_k and DW_k)

All tasks emit a `14×14` grid (`N=196`, matching the ViT-T patch count), flattened row-major to a 1D token
sequence per the `ProceduralDataset` contract (`data/base.py:17`), and use `pos_embed: sincos2d` so the
blocks can see grid adjacency.

> **Determinacy requirement (from the CA diagnosis, §5 of the failure report).** Lifting Dyck to 2D
> *loosens* the constraint — the paper's strict hierarchy DW⊊DN⊊DQ⊊DC means the crossword property admits
> structurally different pictures, so partial observation under-determines a completion more than in 1D.
> Every masking scheme below is therefore designed so the **masked target is forced by the visible
> context** (e.g. hide 1 of a rectangle's 4 corners → the 4th is determined). Masking whole circuits or
> using a high mask ratio drifts toward under-determined targets — the degenerate regime that gave CA's
> worst result (`iid-static` 65.72). **Gate every config on conditional-target entropy before GPU:** run
> `analysis/task_entropy.py`/`task_difficulty.py` and confirm the masked-cell entropy is low (DYCK-like),
> not near-uniform (CA-iid-like).

### Task 1 — `dyck2d_crossword` (DC_k), the primary
- **Generation.** Build the matching graph directly rather than rejection-sampling raw grids (which is
  exponentially unlikely to be balanced). Start from **DQ_k** (rectangles only): repeatedly pick a rectangle
  and stamp its four corners `a,b,c,d`, maintaining per-row and per-column nesting validity; the laminar
  (nested/disjoint) subset is exactly DW_k, and allowing legal crossings broadens toward full DC_k. The
  number of distinct quadruple types is the vocabulary knob `k`.
- **Correctness by checker.** Implement `is_dyck_word(line, pairs)` (a one-pass stack test) and assert it
  on **every row and every column** of each emitted picture; reject and resample any invalid grid. This is
  the membership test from Def. 4 and doubles as the unit-test oracle.
- **Masking — the 2D analogue of close-only.** "Matching-corner" masking: hide one corner of a rectangle
  (e.g. the `d`/bottom-right closer) and require predicting it from the other three corners plus the row
  and column context. Model after `data/dyck/masking.py:13 CloseOnlyMasking`. A "closing-corner" variant
  masks the `c,d`/`b,d` closers only.

### Task 2 — `dyck2d_boxes` (DW_k), the nested-box analogue
- **Generation.** Recursive *nesting accretion* (Def. 2), correct by construction: start from a small
  picture, wrap it in a corner-quadruple with row-Dyck top/bottom borders and column-Dyck left/right
  borders (the bijections `h_r, h_c` give the matching opposite border), recurse, and tessellate to fill
  the grid. No rejection needed — every accretion preserves the row/column-Dyck property.
- **Masking.** Same matching-corner / closing-corner scheme; here masked closers correspond to *nesting
  depth*, making this the tightest 2D analogue of 1D k-DYCK's "predict the closer" objective.

### Optional Task 3 — `dyck2d_neutralize` (DN_k)
A reduction-style objective: present a partially-neutralized picture and predict the next neutralizable
rectangle (or its corners). Tests whether learning a *reduction order* (a different computational shape)
transfers. Lower priority; include only if Tasks 1–2 look promising.

### Curriculum option
The existing `warmup/curriculum.py` (`run_curriculum`, `_stage_config` set `cfg.data.source` per stage)
already supports multi-stage warm-up on one model. A natural stage sequence is **1D k-DYCK → DC_k**, to
test whether 2D structure adds *on top of* the 1D bias rather than merely replacing it.

## 4. Experiment design — each run tests one failure hypothesis

Every row is a warm-up → CIFAR-100 downstream run, scored on `best_top1` against the established baselines
(1D DYCK 72.64, random 70.02, CA Rule110 68.86, spatial-dyck 66.93/70.42). The design is built so the new
runs *also* confirm or refute the CA diagnosis.

| # | Tests | Setup vs control | Prediction if the diagnosis is right |
|---|---|---|---|
| E0 | **H3** on CA itself (untested cell — cheapest, no new code) | re-run `ca-rule110` with `pos_embed: sincos2d` vs the existing `random` run (68.86) | sincos2d lifts CA toward random/DYCK if flatten-obscuring is the issue; flat if H1/H2/H4 dominate |
| E1 | **H3** intrinsic-2D vs re-layout | `dyck2d_crossword` (sincos2d) vs `spatial-dyck` nested/permuted | DC_k ≫ spatial-dyck; ideally ≥ 1D DYCK |
| E2 | **H1/H5** targeted vs uniform masking | DC_k matching-corner mask vs DC_k uniform-random mask (same data) | targeted ≫ uniform |
| E3 | **H2** vocabulary | DC_k with `k ∈ {1, 4, 16, 64}` (i.e. `4k` corner types) | accuracy rises with `k`, approaching DYCK |
| E4 | **H1** long-range vs local | DC_k (full circuits) vs DQ_k (length-4 rectangles only) vs CA | long-range ≥ local-only ≥ CA |
| E5 | pos-embed exposure | DC_k with `sincos2d` vs random/1D positional embedding | sincos2d *helps* here (opposite of the re-layout case) |
| E6 | DW_k vs DC_k | `dyck2d_boxes` vs `dyck2d_crossword` | both ≥ random; comparison isolates nesting vs crossword structure |
| E7 | curriculum | 1D-DYCK→DC_k vs DC_k-only vs 1D-DYCK-only | additive if 2D structure is complementary |

Interpretation guide: if **E1** shows DC_k ≫ spatial-dyck, H3 (form-of-presentation) is confirmed; if
**E2** shows targeted ≫ uniform, H1/H5 are confirmed on a source that *does* have long-range structure;
**E3** isolates H2 cleanly because everything except vocabulary is held fixed. A *null* result (DC_k ≈
spatial-dyck ≈ random) would itself be informative — it would point the blame back at the frozen-embedding
/ flattening pipeline rather than at the data source.

## 5. Implementation blueprint (for the later GPU/coding stage)

Documented now so building it is mechanical; the architecture already supports a new source behind a small
interface (verified against the current code).

1. **New package** `procedural_warmup/data/dyck2d/`, mirroring `data/dyck/` and `data/dyck/spatial.py`:
   - `generator.py` — `crossword_picture(...)` (matching-graph construction) and `boxes_picture(...)`
     (nesting accretion); plus `is_dyck_word(line, pairs)` validator.
   - `dataset.py` — `Dyck2DGrid(ProceduralDataset)` (`data/base.py:17`); `__getitem__` returns a flattened
     `LongTensor` of length `N`.
   - `masking.py` — `MatchingCornerMasking` returning `(masked_input, targets, mask)`
     (`MaskingStrategy` protocol, `data/base.py:34`); reuse `CloseOnlyMasking` semantics where ids align.
   - `__init__.py` — `@register_source("dyck2d_crossword")` and `@register_source("dyck2d_boxes")`
     (`data/__init__.py:22 register_source`).
2. **Register the import** at the bottom of `data/__init__.py` (the same place `ca`/`dyck` are imported) so
   the decorators run.
3. **Config** — add a `Dyck2DConfig` dataclass and a field on `RootConfig` in `config/schema.py` (next to
   `DyckConfig:52` / `SpatialDyckConfig:69`); the source is selected via `DataConfig.source`
   (`config/schema.py:134`). Add example yamls under `procedural_warmup/config/files/` with `grid 14×14`,
   `vocab.K` ≥ `2 + 4k`, and `model.pos_embed: sincos2d`.
4. **Reuse unchanged** — model factory and wrapper already bypass the patch embedding and accept any token
   source (`model/factory.py:16 build_model`, `model/wrapper.py forward_tokens`); the trainer, CLI, and
   downstream eval are source-agnostic; analysis tooling (`analysis/task_entropy.py`,
   `analysis/task_difficulty.py`, `analysis/figures.py plot_downstream_comparison`) works out of the box.
5. **Tests** — add `tests/test_dyck2d.py` asserting (a) `dyck2d_crossword`/`dyck2d_boxes` are registered
   (pattern from `tests/test_registry.py`), and (b) every row and every column of a sampled picture passes
   `is_dyck_word` (the generator's own oracle).

## 6. Summary

The CA experiments rejected one hypothesis but produced a sharp, reusable diagnosis: a warm-up source must
demand long-range, high-vocabulary, constraint-driven computation and must present its structure in a form
the frozen-embedding pipeline can use. The 2D Dyck languages of arXiv 2307.16522 are the natural source
that satisfies all of these *and* finally tests genuine 2D structure — distinct from the failed
re-layout-only `spatial-dyck`. The experiment matrix is built so the new runs simultaneously test whether
intrinsic-2D structure helps and confirm *why* CA did not.
