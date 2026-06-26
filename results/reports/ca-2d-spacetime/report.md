# Retaining CA's 2-D spacetime structure — a test of failure Hypothesis 3

**Status:** implementation + experiment design on branch `feat/ca-2d-spacetime-no-flatten`. Code, configs,
and tests are committed and validated on CPU; the heavy 15k-step warm-ups + 300-epoch CIFAR-100 trainings
run on a GPU box (`scripts/run_ca_2d_experiments.sh`). This report explains what was built, the experiment
matrix, and — critically — how to read the results without fooling ourselves.

Companion: [`../ca-failure-analysis/report.md`](../ca-failure-analysis/report.md) §3/§5 (H3). k-Dyck is
**not** modified (per the brief).

## 1. The hypothesis under test

H3 says CA warm-up underperforms partly because the intrinsically-2-D spacetime diagram is flattened
row-major to 1-D (`data/ca/dataset.py:70`, `p = t·W + x`) **and** given a structure-free random positional
code, so the 2-D adjacency a cell's causal parents live at (`p−W−1, p−W, p−W+1`) is hidden and the model
settles on a non-transferable local lookup. This work "un-flattens" by exposing/using the 2-D structure and
asks whether transfer recovers.

## 2. Architectural reality (what "don't flatten" can and cannot mean)

A ViT is permutation-equivariant given positions, and we only transfer the **transformer blocks + final
norm** to vision (`warmup/process.py:21` strips `tok.`, `pos`, `head.`, `cls_token`, `patch_embed.`;
`wrapper.py:44` hard-asserts `N = H·W = 196`). So 2-D structure can enter **only** through (a) the frozen
positional embedding and (b) the objective/masking geometry — *not* new attention parameters (those would
not transfer) and *not* a different token count (that breaks the patch-grid match). Every approach below
respects this; the ones that cannot (2-D RoPE, per-row BCE) are explicitly design-only (§6).

The flatten itself is kept (row-major time-major); "retaining 2-D structure" means making that layout's
geometry **usable** — `Frozen2DSinCosPositionalEmbedding` maps `p → (p//W, p%W) = (time, space)`, exactly
CA's geometry (`model/embeddings.py`, verified in `tests/test_ca_2d.py`).

## 3. What was implemented

| Lever | Mechanism | Cost | Where |
|---|---|---|---|
| `sincos2d` positional code | exposes (time, space) grid adjacency to the blocks | config | already wired (`factory.py`) |
| `block2d` masking | mask contiguous `block_h×block_w` (time×space) rectangles, not i.i.d. cells → kills the copy-a-neighbour shortcut | **new code** | `data/ca/masking.py`, schema `block_h/block_w` |
| `forward` / next-state masking | mask the last `forward_rows` time-rows = predict future state(s); `forward_rows=1` = predict the entire next state vector | config | already wired |
| `sincos1d` ring positional code | periodic sin/cos over the N-cell ring — the *correct* code for the `ca_step` board (where `sincos2d` is wrong) | **new code** | `model/embeddings.py:Frozen1DRingPositionalEmbedding`, `factory.py` |
| `gol_step` source | predict the entire next **Game-of-Life** state from the current i.i.d. 2-D board (genuinely-2-D operator) + true/shuffled control | **new code** | `data/ca/gol.py:GolStepDataset`, registered `gol_step` |

New tests: `tests/test_ca_2d.py` (block2d contiguity/coverage, single-step next-state, sincos2d/sincos1d
geometry, `gol_step` correctness `y == life_step(x)` and the shuffled decoupling). Full suite: 50 passing.

## 4. Experiment matrix

All H3 cells use **binary** tokenization (the clean test — block tokenization coarsens the space axis 7×,
putting the rule's 3-cell stencil *inside* a token, so it muddies H3; it is kept only for the H2/vocab
story). Each row is warm-up → strip → CIFAR-100; score = `best_top1` vs the existing baselines
(random 70.02, k-Dyck 72.64, CA binary-1D 68.86, CA block-1D 71.42, CA hard-1D 71.86, CA iid-true 66.81).

**Core 2×3 factorial — separates "2-D via position" from "2-D via objective":**

| binary | pos = random (1-D) | pos = sincos2d (2-D) |
|---|---|---|
| mask = random | `ca-rule110` *(exists, 68.86)* | **`ca-rule110-2dpos`** |
| mask = block2d | `ca-rule110-block2d-1dpos` | **`ca-rule110-block2d`** |
| mask = forward (deep, 12) | *(edit pos→random)* | **`ca-rule110-forward-deep-2dpos`** |

> **Two design notes on the factorial.** (1) The **forward** cells use a *self-contained* spacetime
> (`sim_width=14`, `boundary=zero`, so the 14×14 grid is one closed strip): every masked future cell is then a
> deterministic function of the visible rows (100% well-posed at any depth — verified), and both axes are
> non-periodic so `sincos2d` is exactly correct. The **random/block2d** cells keep the standard `sim_width=64`
> crop (inpainting), whose deep light-cones leave the window — that is fine for inpainting but would make a
> deep-forward target ~75% aleatoric, which is why forward uses the self-contained strip. Consequence: the
> mask-mode column also varies the data regime, so read the clean signal off the **pos×mask interaction within
> a fixed mask mode** (both pos rows of a given mask share the regime), per §5. (2) **block2d** realizes ~0.45
> coverage at `mask_ratio=0.5` (overlapping rectangles), a touch below random's ~0.50 — so the
> block2d-vs-random *main effect* mixes geometry with a slightly lower masked fraction; the pos×mask
> interaction is unaffected (both block2d cells share the coverage).

**Satellites:** `ca-rule110-forward-2dpos` (forward 7), `ca-rule110-nextstate-2dpos` (forward 1 — diagnostic,
predicted ≈ floor, see §5), `ca-rule110-block-2dpos` & `ca-rule110-hard-2dpos` (the "are the mods still
needed" runs, §7).

**Next-state operator controls (the genuinely diagnostic part):**

| Run | Operator | Pos code | Control |
|---|---|---|---|
| `ca-iid1step-true-1dpos` / `-shuffled-1dpos` | 1-D ECA next-state (existing 66.8 objective) | `sincos1d` ring | true vs shuffled |
| `gol-step-true-2dpos` / `-shuffled-2dpos` | 2-D Game-of-Life next-state | `sincos2d` | true vs shuffled |

These re-run the *failed* "predict the entire next state vector" objective with the board's adjacency now
**exposed** (the 196-ring `sincos1d` for the 1-D board; `sincos2d` for the Life board) — a sharp test of
whether the prior next-state failure was the hidden adjacency.

> **Toroidal-encoding caveat.** The Life board is toroidal but `sincos2d` is non-periodic, so the wrap
> neighbours of ~27% border cells are not encoded as adjacent (a doubly-periodic code aliases badly on a
> 14-grid, unlike the 196-ring `sincos1d`, so it is not worth building). This is *conservative* — it can only
> hide a real operator signal at the border, never fabricate one — and it cancels in the true−shuffled gap.

> **Marginal-floor caveat (measured).** One Life step on a random board is mostly deaths: the `gol_step`
> target is ~0.30 live-density, so "predict all-dead" already scores ~0.70 *warm-up* token accuracy (vs the
> balanced ~0.51 floor for ECA Rule 110). So do **not** read `gol_step` warm-up accuracy as competence — it
> is dominated by the marginal. The true−shuffled **downstream** gap is the signal, and it is marginal-matched
> (shuffled uses `life_step(z)` at the same density), so it cancels the floor cleanly.

## 5. How to read the results — the one thing that matters

**The conv-shortcut confound.** An ECA/Life step is a *fixed, deterministic, local* rule, so **every**
masking scheme here is solvable by a single translation-equivariant stencil lookup; `sincos2d`'s whole
effect is to make that stencil *cheaper to learn* (one shared offset rule instead of 196 memorized
position-pair patterns). A modest transfer bump is therefore exactly what you'd see if you merely installed
a **depthwise-conv locality prior** — a fine vision prior, but **not** evidence that CA computation
transferred. Headline accuracy alone cannot tell these apart. Two things can:

1. **The true−shuffled gap** (must be > 0). The prior `ca_step` result was load-bearing precisely because
   true ≈ shuffled (66.81 ≈ 66.90) proved *zero* operator signal. Any next-state win that does not also open
   a true−shuffled gap is a marginal/locality artifact, not operator transfer.
2. **The pos×mask interaction, not the main effect.** If the `sincos2d × random` cell alone recovers, it was
   purely the position code (the *objective* half of H3 is falsified). If only the bottom-right cells
   (2-D position **and** a non-local objective) clear random-init, H3 is supported: exposure is necessary and
   the non-local objective is what makes it bite.

**Predicted nulls (so we don't misread them):** `ca-rule110-nextstate-2dpos` (forward_rows=1) leaves the
masked row's parents fully visible → a pure parallel 1-step lookup *analogous to* (not identical to) the
`ca_step` objective that scored 66.8; run it as a labeled diagnostic and *expect ≈ floor*. It is not the same
objective — it masks only the final row of an *evolved* trajectory and conditions on 13 visible, texture-rich
rows, so a non-floor result could be texture exploitation rather than operator transfer and would not by
itself contradict `ca_step`. `sincos2d × random` is predicted small/≈0 (sincos2d even *hurt* 1-D Dyck by
−0.45, cf. `cifar100-dyck-2dpos`). The deep-forward run is deliberately deep (12 masked rows, 2 seed) on a
*self-contained* `sim_width=14` strip, so its targets are 100% determined by the 2 visible rows and the only
way to fill them is to iterate the rule — genuine composition, not a wide windowed lookup.

**Decision tree:**
- Only bottom-right (2-D pos × non-local objective) clears random meaningfully → **H3 supported**.
- A `random`-pos objective cell already recovers → the **objective** does the work; position is secondary.
- `sincos2d × random` alone recovers → it was purely the position code.
- Next-state true−shuffled gap stays ~0 even with adjacency exposed → the binding constraint is operator
  **locality/determinism**, not the flatten → H3 insufficient; **pivot to the intrinsically-2-D 2-D-Dyck
  sources** (long-range binding + rich vocab), per [`../dyck2d-pretraining-design/report.md`](../dyck2d-pretraining-design/report.md).

## 6. Other approaches considered, and why they are design-only

- **Per-row token / multi-label-BCE next-state** (one token = a whole W-bit row, predict the next row vector
  with `Linear(d, W)` + BCE). The most literal "predict the entire next state vector," but it **breaks the
  transfer interface**: N drops to H=14 (≠196 patch grid) and the frozen scaled-identity token embedding
  cannot represent an arbitrary W-bit vector (it is a lookup, not a projection). It would need a parallel
  embedding+head+loss path and would transfer from a 14-token regime to a 196-token one. *The transfer-safe
  realization of the same idea is `forward_rows=1` (implemented).* Documented, not built.
- **2-D RoPE inside attention** (fixed rotary on Q/K). **Violates the transfer constraint**: it rotates Q/K
  in every block, but the downstream ViT has no RoPE → silent train/inference mismatch (or it changes the
  block forward, which then would not be a plain timm ViT). Excluded.
- **Learned-but-frozen 2-D positional embedding.** Same additive-and-stripped slot as `sincos2d` but learned
  then frozen; dominated by the analytic version. A robustness check only.
- **Column-major / Hilbert / axial re-flatten on its own.** A no-op under any fixed positional code that
  already matches the flatten — only meaningful jointly with the positional embedding, which `sincos2d`
  already handles. Not run standalone.

## 7. Re-examining the earlier CA rule modifications

The earlier CA gains came from two modifications, both applied **only** with the random 1-D positional code:
**block tokenization** (`ca-rule110-block`, K=130, +vocab; 71.42) and **forward masking**
(`ca-rule110-forward`/`-hard`; hard = block+forward, 71.86). The question is whether they are still needed
once 2-D structure is retained, or whether they were partly *compensating* for the flatten:

- **Is the block/vocab mod still needed?** Compare `ca-rule110-2dpos` (binary + 2-D) against `ca-rule110-block`
  (block + 1-D, 71.42) and `ca-rule110-block-2dpos` (block + 2-D). If binary+2-D ≈ block+1-D, exposing the
  geometry *substitutes* for the vocabulary lever (the block mod was largely compensating for the flatten).
  If block+2-D ≫ binary+2-D, the vocabulary lever is independent (an H2 effect, orthogonal to H3).
- **Is the forward/next-state mod still needed?** Compare `ca-rule110-2dpos` (random mask) against
  `ca-rule110-forward-deep-2dpos`. If the position fix alone recovers, the forward objective was substituting
  for it; if forward still adds on top, the non-local objective is doing independent work (consistent with
  H1, the local-shortcut hypothesis).
- **Ceiling.** `ca-rule110-hard-2dpos` (block + forward + 2-D) vs `ca-rule110-hard` (same minus 2-D, 71.86)
  isolates the marginal value of the 2-D code on top of the best 1-D recipe.

## 8. Running it

```bash
scripts/run_ca_2d_experiments.sh CIFAR100      # warm-up -> strip -> downstream for all rows + figure
```
CPU-only here (per the project setup), so only the unit tests and config/forward validation run locally;
the warm-ups/trainings are for the 5090/A5000 boxes. Baselines (`cifar100-{random,dyck,rule110,block,hard,
iid-true}`) must already exist for their bars to appear in the comparison figure.

## 9. Files changed

- `data/ca/masking.py` — new `block2d` mode (vectorized, inclusion-exclusion block count, validation).
- `analysis/figures.py` + `analysis/compare.py` — comparison skips runs without a `metrics.json` (no crash).
- `config/schema.py` — `MaskingConfig.block_h/block_w`; `pos_embed` doc adds `sincos1d`.
- `model/embeddings.py` — `Frozen1DRingPositionalEmbedding`; `model/factory.py` — `sincos1d` branch.
- `data/ca/gol.py` — `GolStepDataset`; `data/ca/__init__.py` — registers `gol_step`.
- `config/files/` — 12 new YAMLs (8 CA-grid + 2 `ca_step`-ring + 2 `gol_step`).
- `tests/test_ca_2d.py` — 9 tests. `scripts/run_ca_2d_experiments.sh` — the runner.
