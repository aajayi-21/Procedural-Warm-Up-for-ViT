# CA vs. k‑Dyck: how the data is fed, what the network actually predicts, and whether we capture the spatial structure

**Scope.** This report does three things requested by the review of `scripts/run_ca_2d_experiments.sh`:

1. Explains the **training pipeline** end‑to‑end for every experiment in the 2‑D script — i.e. *what the
   network input is, what it outputs, and what the loss is* (the part that was confusing).
2. Explains **exactly how the spatial CA data is input into the model** for the Rule‑110 spacetime grid and
   for the two Game‑of‑Life rounds.
3. Compares the CA results to k‑Dyck and answers the suspicion that **we are not adequately capturing the
   spatial nature of the CA tasks**.

All numbers are read verbatim from `results/reports/*/metrics.json`; all code references are
`file:line`. Companion documents: [`../ca-2d-spacetime/report.md`](../ca-2d-spacetime/report.md) (the H3
design) and [`../ca-failure-analysis/report.md`](../ca-failure-analysis/report.md) (the five‑hypothesis
post‑mortem). This report does not re‑run anything; it reads the committed results.

---

## 0. TL;DR

- **The pretraining task is masked‑token prediction (MLM), not image work.** The ViT's *patch embedding is
  bypassed* during warm‑up; symbol ids go through **frozen** token + positional embeddings into the
  transformer blocks, and a throwaway `Linear(d → K)` head predicts the token id at masked positions with
  cross‑entropy. Only the **transformer blocks + final norm** are kept and transplanted into a fresh ViT that
  is then trained on CIFAR‑100 (`model/wrapper.py`, `model/factory.py:49`, `warmup/process.py:21`).
- **The 2‑D structure of CA can enter the model through exactly two channels:** the **frozen positional
  embedding** (`sincos2d`/`sincos1d`) and the **masking geometry** (`block2d`/`forward`). It *cannot* enter
  through the token content (each binary cell is **1 bit**, a 4‑row scaled‑identity embedding) or through any
  spatial mixing operator, because anything that changed the token count or the attention math would not
  survive transfer (`wrapper.py:44` hard‑asserts `N = H·W = 196`).
- **k‑Dyck wins (72.64%) and CA never overtakes it.** Plain Rule‑110 (68.86%) is *below random init*
  (70.02%). Crucially, when we *did* expose the geometry — `sincos2d`, `block2d`, deep‑forward, a genuinely
  2‑D Game‑of‑Life operator — transfer **did not improve; in the cleanest cell it got worse** (CA binary
  `sincos2d` = 65.35% vs the 1‑D random‑pos baseline 68.86%).
- **So the suspicion is half right, in an important way.** It is *not* that the code forgot to expose the
  geometry — the 2‑D script was built precisely to expose it. It is that **the transfer interface forbids the
  genuinely spatial encodings** (per‑row tokens, 2‑D RoPE, conv stems all break transfer — see
  `../ca-2d-spacetime/report.md` §6), and the one spatial channel we *can* use (a positional code) is not
  enough to make a local, deterministic, 1‑bit‑per‑token operator transfer to vision. The binding
  constraints are operator **locality + determinism + vocabulary poverty**, not (only) the flatten.
- **A diagnostic worth internalizing:** the easier the warm‑up task is to *fit*, the *worse* it transfers.
  The next‑state operators are learned almost perfectly during warm‑up (acc 0.93–0.97) and transfer
  ~nothing; k‑Dyck never gets above 0.835 warm‑up accuracy and transfers best.

---

## 1. The training pipeline (what is the network's input/output?)

Every row of `scripts/run_ca_2d_experiments.sh` is a three‑stage chain:

```
   STAGE A  warm-up (symbolic MLM)          STAGE B  strip            STAGE C  vision transfer
 ┌───────────────────────────────┐      ┌──────────────────┐      ┌────────────────────────────┐
 │ token ids (B,N)                │      │ keep blocks+norm │      │ fresh timm ViT             │
 │   │ FROZEN token emb (K→d)     │      │ drop tok/pos/MLM │      │  patch_embed (NEW, random) │
 │   │ + FROZEN pos emb (N→d)     │  ──▶ │ drop patch_embed │  ──▶ │  + blocks/norm ⬅ warm-up   │
 │   │ CLS ▸ transformer blocks   │      │ drop cls/posembed│      │  + classifier head (NEW)   │
 │   │ final norm                 │      │ → *_stripped.pt  │      │ train on CIFAR-100 (DeiT)  │
 │   ▼ MLM head Linear(d→K)       │      └──────────────────┘      │ → best_top1                │
 │   CE loss on MASKED positions  │                                └────────────────────────────┘
 └───────────────────────────────┘
```

### Stage A — warm‑up = masked‑token prediction on a symbol grid

This is the part the script's `python -m procedural_warmup.warmup.cli` calls do, and it is **identical in
mechanism for CA, Game‑of‑Life, and k‑Dyck** — only the *data source* and *masking strategy* differ
(`warmup/trainer.py` is deliberately source‑agnostic).

- **Input:** a batch of integer token grids `(B, N)` with `N = H·W = 14·14 = 196`. Tokens are *symbols*, not
  pixels. For binary CA the vocabulary is `K = 4`: `0=PAD, 1=MASK, 2=cell‑off, 3=cell‑on`
  (`data/ca/tokenize.py:18,21`). For k‑Dyck `K = 130`: `0=PAD, 1=MASK, 2..65 = 64 open brackets,
  66..129 = 64 close brackets` (`config/files/dyck-vit-t.yaml:8`, `data/dyck/generator.py`).
- **Masking** corrupts the input: selected positions are overwritten with `MASK_ID=1` and remembered as
  targets (`data/ca/masking.py:99‑105`). The strategy is what makes the objective *reconstruction* vs
  *prediction* (see §1.1).
- **Embedding (frozen!):** the masked ids are mapped to vectors by `FrozenTokenEmbedding` — a `K×d`
  scaled‑identity matrix, **not trainable** (`model/embeddings.py:17‑39`) — and a **frozen** positional
  vector is *added* (`wrapper.py:46‑47`). Freezing both is deliberate: it stops the model from solving the
  task inside the embeddings and forces the **transformer blocks** to learn the data's structure
  (`model/embeddings.py:1‑7`). These embeddings are thrown away at transfer, so only the blocks carry signal.
- **Backbone:** prepend the backbone's CLS token, run `vit.blocks`, then `vit.norm` (`wrapper.py:31‑39`).
  The **patch‑embedding projection is never touched** — there are no images here.
- **Output / "the prediction":** per‑token features `(B, N, d)` → `mlm_head: Linear(d, K)` → logits
  `(B, N, K)`. **Loss = cross‑entropy over masked positions only**, `argmax` over the `K` vocabulary
  (`warmup/trainer.py:77‑83`). In words: *"for each masked cell, classify which symbol it should be."* The
  reported warm‑up `acc` is the fraction of masked tokens whose argmax matches the true symbol.

So the network output during warm‑up is **a categorical distribution over the K symbols at each masked grid
position** — never a class label, never a pixel. The "task" is reconstruct/predict the missing symbols.

### 1.1 Two flavours of the warm‑up objective (this is the source of confusion)

The script mixes **two genuinely different objectives** behind the same trainer, controlled by the masking
strategy. They answer different questions:

| Objective | Masking | What is visible / predicted | Used by (script rows) |
|---|---|---|---|
| **Reconstruction / inpainting** (MAE‑/BERT‑style) | `random`, `block2d`, `forward`, `lightcone` (`data/ca/masking.py`) | ~50% of the grid is visible; predict the masked cells *of the same grid* | `ca-rule110-2dpos`, `…-block2d`, `…-block2d-1dpos`, `…-forward-2dpos`, `…-forward-deep-2dpos`, `…-nextstate-2dpos`, `…-block-2dpos`, `…-hard-2dpos` |
| **Transduction / next‑state** (full‑mask) | `TransductionMasking` (`data/ca/iid_step.py:98`) | the **entire current state x is visible**, predict the **entire next state y** — *no target token is ever fed* | `ca-iid1step-true/shuffled-1dpos`, `gol-step-true/shuffled-2dpos` |

For the transduction rows the dataset emits a length‑`2N` pair `[x | y]`; `TransductionMasking` slices it into
`input = x` (all visible) and `target = y` (all masked), so the model always *sees 196 tokens and predicts
196 tokens* (`data/ca/iid_step.py:109‑113`). The `masking:` block in those YAMLs is **ignored** (the configs
say so). This is the "learn the update operator" objective; reconstruction is the "fill in the texture"
objective.

### Stage B — strip (`warmup/process.py`)

`clean_checkpoint` deletes everything warm‑up‑specific and everything that will be re‑initialised for vision:
`tok.`, `pos`, `head.` (MLM), and `patch_embed.`, `pos_embed`, `cls_token` (`process.py:21‑22`). **What
remains is exactly `blocks.*` + `norm.*`** — the only weights that transfer. (This is why no positional
choice can leak into vision: the frozen pos embedding is stripped; it only shaped *what the blocks learned*.)

### Stage C — downstream vision transfer (`downstream/main.py`)

A **fresh** `vit_tiny_patch16_224` is built with a random patch embedding, random positional embedding, random
CLS and a real classifier head; the stripped `blocks+norm` are loaded **non‑strict** into it
(`downstream/init_weights.py:118‑146`). It is then trained on CIFAR‑100 with the DeiT recipe — 300 epochs,
AdamW lr 2e‑3 / wd 0.05, cosine schedule, RandAugment + Mixup 0.8 + CutMix 1.0 + label smoothing 0.1, AMP, at
224 px / a 14×14 patch grid (`config/files/downstream-cifar.yaml`). The headline metric is **best top‑1**.
The 14×14 patch grid is *deliberately* the same `N=196` as the warm‑up token grid, so the transferred blocks
operate on the same sequence length they were trained on.

---

## 2. How the spatial CA data is input — Rule‑110 spacetime grid

Take `ca-rule110-2dpos` (the pure H3 cell: binary tokens, random masking, `sincos2d`). The pipeline from
rule to model tensor:

1. **Simulate a spacetime diagram.** `simulate_spacetime(rule=110, width=sim_width=64, n_rows=H=14,
   burn_in=32, boundary=periodic, init_density=0.5)` evolves a random 1‑D ECA ring and **stacks successive
   time steps as rows** → a `(14, 64)` array of `{0,1}` cells, **rows = time, columns = space**
   (`data/ca/eca.py:43‑73`). An ECA step is the local 3‑cell rule `idx=(left<<2)|(row<<1)|right;
   table[idx]` (`eca.py:25‑40`). The simulation is *wider* than the eventual window (64 ≫ 14) so the
   in‑window cells have genuine out‑of‑window causal history (real light cones, not boundary wrap)
   (`data/ca/dataset.py:1‑9`).
2. **Crop a window.** A random horizontal offset selects a `(H=14, cell_W=14)` window (for `binary`,
   `cells_per_token = 1` so `cell_W = W = 14`) (`dataset.py:35‑39, 66‑68`).
3. **Tokenize.** `binary_tokens` adds the 2 reserved ids → cells `{0,1}` become token ids `{2,3}`. Vocabulary
   is `K=4` (`tokenize.py:21‑23`). *(In `block` mode instead, 7 consecutive cells are packed MSB‑first into
   one of `2^7=128` symbols, `K=130` — this is the vocabulary lever, but it coarsens the space axis 7× and so
   is **not** used in the clean H3 cells.)*
4. **Flatten row‑major to length `N=196`.** `ids.reshape(self.N)` (NumPy C‑order), so grid cell `(t, x)`
   lands at sequence position **`p = t·14 + x`** (`dataset.py:70`). *This is the flatten.* A cell's causal
   parents `(t−1, x−1..x+1)` now sit at positions `p−15, p−14, p−13` — a stride‑14 stencil in the 1‑D
   sequence.
5. **Mask.** With `random` masking each position is hidden i.i.d. at ratio 0.5; with `block2d` a few
   contiguous `4×4` (time×space) rectangles are stamped so a masked cell's neighbours are usually *also*
   masked and cannot be copied (`data/ca/masking.py:72‑96`).
6. **Embed.** Each id → frozen 192‑d vector (4‑row scaled‑identity, scale 0.02), **plus** the frozen
   positional vector for index `p`. With `pos_embed: sincos2d`, that vector is
   `Frozen2DSinCosPositionalEmbedding`, which maps `p → (row, col) = (p//14, p%14) = (time, space)` and
   encodes each axis with multi‑frequency sin/cos — **this is the only place the (time, space) adjacency is
   handed to the model** (`model/embeddings.py:87‑120`, `factory.py:40‑41`). The 1‑D baseline `ca-rule110`
   instead uses `random` frozen vectors that expose *no* distance at all.
7. **Transformer + head.** CLS ▸ blocks ▸ norm ▸ `Linear(192→4)`; cross‑entropy on the masked cells.

### The crux for "are we capturing the spatial nature?"

Because the patch embedding is bypassed and the token embedding is a frozen 4‑row identity, **each Rule‑110
cell reaches the transformer as a single token carrying one bit of content.** There is *no convolution, no
patch projection, no 2‑D mixing in the input representation.* The entire "spatial" signal available to the
blocks is:

- **(a)** the additive positional code (random ⇒ geometry hidden; `sincos2d` ⇒ (time,space) adjacency
  exposed), and
- **(b)** the *shape of the mask* (i.i.d. cells vs `block2d` rectangles vs `forward` rows).

That is by design and by necessity: anything richer (a per‑row "whole next state" token, 2‑D RoPE in
attention, a conv stem) **changes `N` or the attention math and therefore would not transfer** — the wrapper
hard‑asserts `N=196` and only blocks+norm survive the strip (`../ca-2d-spacetime/report.md` §2, §6).

---

## 3. How the spatial CA data is input — the two Game‑of‑Life rounds

The script's GoL rows use `source: gol_step` (`config/files/gol-step-true-2dpos.yaml:16`), which is **not**
the spacetime‑stacking path; it is the **next‑state transduction** path. Two differences from Rule‑110
matter:

- **What is "2‑D" here is different.** For Rule‑110 the 2‑D grid is a *spacetime diagram* (time × space). For
  Game of Life the 14×14 grid is a *single spatial snapshot* (`space_y × space_x`); time appears as the
  **input→output relation** between the current board `x` and the next board `y`. GoL is the genuinely‑2‑D
  operator: its Moore‑8 neighbourhood is 2‑D‑local, so `sincos2d` exposes exactly the adjacency the operator
  needs (`data/ca/gol.py:52‑73`).

GoL input path (`GolStepDataset`, `data/ca/gol.py:75‑99`):

1. Sample a density `p` from `{0.2, 0.3, 0.4, 0.5}` (Life is sparse) and draw an i.i.d. Bernoulli board
   `x ∈ {0,1}^{14×14}` (`gol.py:91‑92`). *Structure‑free input — no evolved texture to memorise.*
2. Compute the next state with the real B3/S23 rule: `y = life_step(x)` (toroidal Moore‑8;
   `gol.py:23‑33, 95`). For the **shuffled control** (`mode: shuffled`) `y = life_step(z)` for an *unrelated*
   board `z` at the same density — same output marginal, **no causal link** to `x`.
3. Tokenize both boards binary and concatenate **`[x | y]`** → length `2N = 392` (`gol.py:96‑98`).
4. `TransductionMasking` splits it: the model sees **all of `x`** (196 tokens) and must predict **all of
   `y`** (196 targets), full mask, no target token fed (`data/ca/iid_step.py:98‑113`).
5. Positions are `sincos2d` over the 14×14 board, so the model is told where each cell sits.

The 1‑D analogue (`ca-iid1step-true/shuffled-1dpos`, `source: ca_step`, `data/ca/iid_step.py:39‑72`) is
identical except `x`/`y` are a **196‑cell ECA ring** (Rule 110), and the positional code is `sincos1d` — a
*periodic* ring sin/cos, which is the correct geometry for a ring (a 14×14 `sincos2d` would wrongly fragment
the ring every 14 cells; `model/embeddings.py:58‑84`).

> **Why the `true` vs `shuffled` pairs exist.** `true` and `shuffled` have *identical* geometry, tokenization
> and target marginals; the only difference is whether `y` is causally produced from `x`. **`true` beating
> `shuffled` downstream is the operator‑learning signal** (`data/ca/iid_step.py:15‑19`). This is the clean
> test of "did the model learn the CA *operator*, or just its output statistics?"

---

## 4. Results vs. k‑Dyck

All CIFAR‑100, ViT‑tiny, 300 epochs, best top‑1, from `results/reports/*/metrics.json`. Warm‑up
acc/loss are the running averages from each warm‑up `metrics.json` (`data/ca/...` → `warmup/trainer.py`).

| Init source | Run | Pos code | Mask / objective | Vocab K | Warm‑up acc | Warm‑up loss | **Best top‑1** | Δ vs random |
|---|---|---|---|---:|---:|---:|---:|---:|
| **k‑Dyck** | `cifar100-dyck` | random | close‑only recon | 130 | 0.835 | 0.534 | **72.64** | **+2.62** |
| k‑Dyck (2‑D pos) | `cifar100-dyck-2dpos` | sincos2d | recon | 130 | — | — | 72.19 | +2.17 |
| CA hard (1‑D) | `cifar100-hard` | random | block + forward | 130 | — | — | 71.86 | +1.84 |
| CA hard (2‑D) | `cifar100-hard-2dpos` | sincos2d | block + forward | 130 | — | — | 71.36 | +1.34 |
| CA block (1‑D) | `cifar100-block` | random | random recon | 130 | — | — | 71.42 | +1.40 |
| CA block2d (2‑D) | `cifar100-block2d` | sincos2d | block2d recon | 4 | 0.835 | 0.290 | 69.42 | −0.60 |
| CA block2d (1‑D) | `cifar100-block2d-1dpos` | random | block2d recon | 4 | — | — | 69.17 | −0.85 |
| CA binary (1‑D) | `cifar100-rule110` | random | random recon | 4 | **0.946** | 0.104 | 68.86 | −1.16 |
| CA forward‑deep (2‑D) | `cifar100-forward-deep-2dpos` | sincos2d | forward‑12 recon | 4 | — | — | 67.24 | −2.78 |
| GoL next‑state **true** (2‑D) | `cifar100-gol-true-2dpos` | sincos2d | transduction | 4 | 0.933 | 0.138 | 66.54 | −3.48 |
| CA next‑state (2‑D) | `cifar100-nextstate-2dpos` | sincos2d | forward‑1 recon | 4 | — | — | 66.17 | −3.85 |
| CA next‑state **true** ring | `cifar100-iid-true-1dpos` | sincos1d | transduction | 4 | **0.968** | 0.059 | 66.33 | −3.69 |
| CA next‑state **shuffled** ring | `cifar100-iid-shuffled-1dpos` | sincos1d | transduction | 4 | 0.609 | 0.663 | 66.36 | −3.66 |
| **Random init (baseline)** | `cifar100-random` | — | — | — | — | — | **70.02** | 0.00 |
| CA binary (2‑D) | `cifar100-2dpos` | sincos2d | random recon | 4 | 0.562 | 0.686 | 65.35 | −4.67 |
| GoL next‑state **shuffled** (2‑D) | `cifar100-gol-shuffled-2dpos` | sincos2d | transduction | 4 | 0.704 | 0.603 | 65.12 | −4.90 |
| CA forward shallow (2‑D) | `cifar100-forward-2dpos` | sincos2d | forward‑7 recon | 4 | — | — | 65.08 | −4.94 |
| ⚠ CA block (2‑D) | `cifar100-block-2dpos` | sincos2d | random recon | 130 | 0.346 | 2.500 | **29.75** | −40.27 |

(`cifar100-dyck-s1` reaches 72.80 — the global best — per `../ca-failure-analysis/report.md` §1.)

### 4.1 What the comparison says

1. **k‑Dyck is the ceiling and CA never reaches it.** Best CA = 71.86 (`hard`, 1‑D); best k‑Dyck = 72.80.
   The headline gap at the top is ~1 pt, but the more damning fact is the **bottom**: plain Rule‑110 (68.86),
   CA binary 2‑D (65.35), and every next‑state run (65.1–66.5) sit **below random init (70.02)**. CA can
   install an *actively harmful* prior, not merely a weak one.

2. **Exposing the geometry did not help — in the cleanest test it hurt.** The single‑factor H3 cell is
   `ca-rule110` (random pos) → `ca-rule110-2dpos` (`sincos2d`), changing *only* the positional code. Result:
   **68.86 → 65.35 (−3.5 pt).** Adding `sincos2d` to k‑Dyck also *cost* 0.45 pt (72.64 → 72.19). So the
   "expose 2‑D adjacency" lever, on its own, is not the missing ingredient — it is mildly harmful here.

3. **Block masking (the other spatial lever) barely moves with the position code.** `block2d-1dpos` 69.17 vs
   `block2d` (2‑D) 69.42 = **+0.25 pt** — within noise. The 2‑D *objective* and the 2‑D *position* together
   still land below random.

4. **The genuinely‑2‑D operator (Game of Life) transfers a hair, then stalls.** GoL `true` 66.54 vs
   `shuffled` 65.12 = **+1.42 pt** — a *real* operator‑transfer signal (the only positive true−shuffled gap
   in the set). But 66.54 is still **3.5 pt below random**. The 1‑D ECA next‑state shows **zero** operator
   transfer: `true` 66.33 vs `shuffled` 66.36 = **−0.03 pt**. So the more spatial operator helps *relatively*
   but not enough to matter *absolutely*.

5. **The "hard" recipe (vocab + forward) is what closes most of the gap, and it does so in 1‑D.** `hard`
   (block tokenization K=130 + forward masking) reaches 71.86 with a *random* 1‑D code; adding `sincos2d`
   (`hard-2dpos`) *removes* 0.5 pt. I.e. the gains came from **vocabulary and a non‑local objective, not from
   2‑D geometry.**

---

## 5. The warm‑up‑accuracy paradox (a key diagnostic)

Reading the warm‑up acc/loss column alongside transfer is illuminating:

| Run | Warm‑up acc | Best top‑1 |
|---|---:|---:|
| `ca-iid1step-true-1dpos` (next‑state, 1‑D) | **0.968** | 66.33 |
| `ca-rule110` (binary recon, 1‑D) | **0.946** | 68.86 |
| `gol-step-true-2dpos` (next‑state, 2‑D) | 0.933 | 66.54 |
| `ca-rule110-block2d` (block2d recon) | 0.835 | 69.42 |
| `dyck-vit-t` (close‑only recon) | 0.835 | **72.64** |
| `ca-rule110-2dpos` (binary recon, `sincos2d`) | 0.562 | 65.35 |

**The runs the model solves most easily transfer the worst.** A 1‑bit, deterministic, local rule under
i.i.d. random masking is solved to 94–97% accuracy by **memorising the 8‑entry lookup table and reading the
visible neighbours** — no long‑range computation, nothing a vision model wants. k‑Dyck's close‑only objective
*cannot* be solved that way (a masked closer needs the matching opener from the stack), so it tops out lower
in warm‑up accuracy and learns transferable machinery (long‑range binding, a 128‑way type space). This is
direct support for hypotheses **H1 (local solvability)** and **H4 (determinism invites memorisation)** in
`../ca-failure-analysis/report.md`.

Two further reads:

- **Operator *learnable* ≠ operator *transferable*.** In warm‑up the next‑state operators *are* learned —
  ECA `true` 0.968 vs `shuffled` 0.609 (a huge warm‑up gap), GoL `true` 0.933 vs `shuffled` 0.704. Yet
  downstream the ECA gap collapses to ~0 and the GoL gap is +1.42. The model genuinely internalised the rule;
  the rule just isn't a useful vision prior.
- **`ca-rule110-2dpos` is the exception that proves the rule.** `sincos2d` *removes* the unique per‑position
  fingerprints that made the lookup memorisable, so warm‑up accuracy crashes to 0.562 (near the binary floor)
  — the model is *forced* toward a translation‑equivariant stencil. That should be the "right" computation,
  yet transfer still *dropped* to 65.35. Forcing the better computation via geometry did not rescue transfer
  — strong evidence the flatten/position code is **not** the binding constraint for the binary task.

---

## 6. The `cifar100-block-2dpos` catastrophe (flag — likely a training failure, not a finding)

`cifar100-block-2dpos` collapses to **29.75% top‑1** (top‑5 59.07), 40 pt below random. Its warm‑up did not
converge: final warm‑up **loss 2.50, acc 0.346** (vs 0.10–0.29 for the others). This run is `block`
tokenization (`K=130`) + random recon + `sincos2d`. The combination of a 128‑way categorical reconstruction
target with the smooth/low‑rank `sincos2d` code and the default LR (2e‑3) appears to have produced a
degenerate optimisation, and the broken blocks then poisoned transfer. **Recommendation: treat this cell as a
failed run, not as evidence about geometry** — re‑run with a lower warm‑up LR / longer LR warm‑up and confirm
before citing it. It is the one entry in the matrix that is an artefact rather than a result.

---

## 7. So — are we adequately capturing the spatial nature of the CA tasks?

**Short answer: the experiments *did* test spatial capture along every axis the transfer interface allows,
and capturing it more did not help. But there is a legitimate, deeper sense in which the interface itself
caps how spatial the task can be — and that, plus 1‑bit tokens, is the real limitation.**

**What *was* captured (and tested):**
- 2‑D adjacency via `sincos2d` / ring adjacency via `sincos1d` (positional channel).
- 2‑D objective geometry via `block2d` (can't copy a neighbour) and `forward`/deep‑forward (must iterate the
  rule), on a self‑contained well‑posed strip for the deep case (`forward-deep-2dpos`).
- A genuinely‑2‑D operator with its geometry exposed (`gol-step-true-2dpos`).
All of these are the *correct* ways to inject spatial structure given the constraints, and they are exactly
the cells that failed to beat random/k‑Dyck.

**Where the suspicion is genuinely right — the spatial nature is *structurally* under‑captured:**

1. **Each cell is one bit.** With binary tokenization the token embedding is a frozen 4‑row identity, so a
   cell contributes ~1 bit of content; *all* of the "what is where" must be reconstructed by the blocks from
   position alone. k‑Dyck hands the blocks a 128‑way type at every position. The spatial richness of a CA
   pattern (a glider, a triangle) is never presented as *content* — only as a configuration of 1‑bit tokens.
   `block` tokenization adds vocabulary but *destroys spatial resolution* (7 cells → 1 token), so it trades
   the very thing we want to capture.
2. **The transfer interface forbids the most natural spatial encodings.** A "predict the whole next state as
   a spatial object" head (per‑row token + `Linear(d→W)` + BCE), 2‑D RoPE in attention, or a conv stem are
   the obvious ways to make the task *spatial in the architecture* — and **all three break transfer** (they
   change `N≠196` or add ops the downstream timm ViT doesn't have; `../ca-2d-spacetime/report.md` §6). So the
   only spatial channels we *can* use are a positional code and a mask shape — both of which we tried.
3. **A frozen additive positional code is a weak way to inject geometry.** It tells the model *where* a token
   is but does not *bias attention toward neighbours*; the blocks still have to discover the stencil. And
   because it is stripped at transfer, even a perfect spatial code only shapes *what the blocks learned*, not
   what vision sees.
4. **`sincos2d` mis‑encodes the torus.** Both Rule‑110 (periodic ring) and GoL (toroidal board) wrap, but
   `sincos2d` is non‑periodic, so ~27% border cells' wrap‑neighbours are not exposed as adjacent
   (`data/ca/gol.py:60‑66`). Conservative (it can hide a real signal, not fabricate one) but it does mean the
   2‑D geometry we hand the model is *imperfect* for these specific automata.

**But "capture the geometry better" is unlikely to be the fix**, because the failures are over‑determined by
non‑spatial causes that the data confirms: the task is **locally solvable** (warm‑up acc 0.95–0.97 by lookup,
§5), **deterministic** (memorisation beats distribution‑learning, H4; the ECA true−shuffled gap is ~0), and
**vocabulary‑poor** (the only lever that reliably helped — `block`/`hard` — was vocabulary + a non‑local
objective, *in 1‑D*). Even the genuinely‑2‑D, geometry‑exposed GoL operator only bought +1.42 pt and stayed
below random.

---

## 8. Recommendations

1. **Stop treating geometry as the lever.** The 2×3 factorial in `../ca-2d-spacetime/report.md` is now
   answered: `sincos2d × random` did *not* recover (it dropped), so the *objective*, not the position code, is
   what matters — and the objective levers that help (vocabulary, non‑locality) are the `hard` recipe, which
   needs no 2‑D code.
2. **If pursuing spatial capture further, change the token, not the position.** The interface‑safe way to
   raise per‑cell information *and* keep 2‑D resolution is a **patch‑of‑cells token that preserves locality**
   (e.g. 2×2 cell blocks → a 16‑way symbol on a 7×7 grid) rather than the 1×7 row blocks of `block` mode that
   flatten the space axis. This adds vocabulary without collapsing one spatial dimension. (Still bounded by
   determinism/locality, so expect modest gains.)
3. **Force non‑locality, since that is what transferred for k‑Dyck.** The deep‑forward strip
   (`forward-deep-2dpos`) is the right idea but under‑powered at K=4; combine it with the patch‑of‑cells
   vocabulary above and a *masking that targets light‑cone tips* (the `lightcone` mode exists,
   `data/ca/masking.py:66‑71`, but was never run) so the objective *demands* multi‑step composition.
4. **Re‑run `block-2dpos`** with a gentler warm‑up LR before drawing any conclusion from its 29.75.
5. **Pivot per the failure analysis.** Both `../ca-failure-analysis/report.md` §4 and `../ca-2d-spacetime/`
   §5's decision tree point the same way: the next experiment should keep k‑Dyck's transferable properties
   (long‑range binding, rich vocabulary, a *constraint distribution* rather than a deterministic map) while
   adding *genuine* 2‑D structure — the **2‑D Dyck languages** (`docs/2307.16522.pdf`,
   `../dyck2d-pretraining-design/report.md`) — instead of "more complex CA."

---

## 9. Provenance & exact configs

- **Pipeline code:** warm‑up `warmup/trainer.py:72‑93` (the `_step`/loss); model `model/wrapper.py`,
  `model/factory.py`, `model/embeddings.py`; strip `warmup/process.py:21`; transfer `downstream/main.py`,
  `downstream/init_weights.py:118`.
- **Data sources:** Rule‑110 spacetime `data/ca/dataset.py` + `data/ca/eca.py`; tokenization
  `data/ca/tokenize.py`; masking `data/ca/masking.py`; next‑state `data/ca/iid_step.py`; Game of Life
  `data/ca/gol.py`; k‑Dyck `data/dyck/generator.py` + `data/dyck/dataset.py`.
- **Configs (all warm‑up):** 15 000 steps, batch 256, AdamW lr 2e‑3 wd 0.05, 1 000‑step LR warm‑up, 14×14
  grid. **Downstream:** `config/files/downstream-cifar.yaml` — 300 epochs, batch 512, AdamW lr 2e‑3 wd 0.05,
  50‑epoch warm‑up, RandAugment/Mixup/CutMix, 224 px.
- **Metrics:** `results/reports/<run>/metrics.json` (downstream keys: `best_top1`, `final_top1`,
  `final_top5`, `init_checkpoint`, `init_summary`; warm‑up keys: `final_avg_loss`, `final_avg_acc`, `source`,
  `masking`).
- **Figures:** comparison `results/figures/ca_2d_spacetime_cifar100.png`; per‑run warm‑up/downstream curves
  `results/figures/<run>_warmup_curves.png` / `_downstream_curves.png`.

_Generated as an analysis report (no new runs) on branch `feat/ca-2d-spacetime-no-flatten`._
