# Why Cellular-Automata Warm-Up Underperforms k-DYCK

**Status:** analysis report (no new code or runs). Diagnoses the rejected hypothesis that CA warm-up
would match or beat k-DYCK for downstream CIFAR-100, using the existing code and results.

## 1. The result

Every DYCK warm-up lands at **72.0–72.8%** top-1 on CIFAR-100. The best CA variants — after substantial
engineering (block tokenization, "hard" rules, spatial block structure) — reach only **71.4–72.2%** and
never overtake DYCK. Plain Rule 110 (**68.86%**) finishes *below* random initialization (**70.02%**), and
the purely-local 1-step CA tasks (**65.5–66.9%**) are the worst sources in the whole study. Numbers are
read verbatim from `results/reports/cifar100-*/metrics.json`.

| Rank | Warm-up source | Run | best top-1 | final top-1 | top-5 | Δ vs random |
|---:|---|---|---:|---:|---:|---:|
| 1 | k-DYCK (seed 1) | cifar100-dyck-s1 | **72.80** | 72.77 | 92.66 | **+2.78** |
| 2 | k-DYCK | cifar100-dyck | **72.64** | 72.62 | 92.89 | +2.62 |
| 3 | CA Rule 110 (block, s1) | cifar100-block-s1 | 72.22 | 71.92 | 92.25 | +2.20 |
| 4 | k-DYCK (2D pos-embed) | cifar100-dyck-2dpos | 72.19 | 72.02 | 92.21 | +2.17 |
| 5 | k-DYCK (seed 2) | cifar100-dyck-s2 | 72.13 | 71.93 | 92.10 | +2.11 |
| 6 | CA Rule 110 (hard) | cifar100-hard | 71.86 | 71.75 | 91.81 | +1.84 |
| 7 | CA Rule 110 (block, s1-5090) | cifar100-block-s1-5090 | 71.55 | 71.40 | 91.97 | +1.53 |
| 8 | CA Rule 110 (block) | cifar100-block | 71.42 | 71.33 | 91.96 | +1.40 |
| 9 | Spatial-DYCK (nested) | cifar100-sdyck-nested | 70.42 | 70.36 | 91.42 | +0.40 |
| 10 | **Random init (baseline)** | cifar100-random | **70.02** | 69.87 | 90.76 | 0.00 |
| 11 | CA Rule 110 (standard) | cifar100-rule110 | 68.86 | 68.61 | 90.84 | **−1.16** |
| 12 | Spatial-DYCK (permuted) | cifar100-sdyck-permuted | 66.93 | 66.62 | 89.55 | −3.09 |
| 13 | CA 1-step iid (shuffled) | cifar100-iid-shuffled | 66.90 | 66.81 | 89.67 | −3.12 |
| 14 | CA 1-step iid (true) | cifar100-iid-true | 66.81 | 66.71 | 89.81 | −3.21 |
| 15 | CA 1-step iid (static) | cifar100-iid-static | 65.72 | 65.54 | 89.50 | −4.30 |

The crucial reading: **several CA variants are below random init.** The failure is not "CA helps less than
DYCK" — it is "CA can instill an *actively harmful* inductive bias." A good theory of the failure must
explain not just the ~0.5 pt gap at the top but the 4+ pt deficit at the bottom.

## 2. Five hypotheses for the gap

Each is grounded in a concrete mechanism in the data pipeline. The warm-up objective is masked-token
prediction: features `→` MLM head `→` cross-entropy on masked positions only
(`procedural_warmup/warmup/trainer.py`). What the model is *forced to compute* to minimize that loss is
what (if anything) transfers. The hypotheses say DYCK forces a transferable computation and CA does not.

### H1 — The CA task is locally solvable, so no long-range computation is learned
An elementary CA step is a deterministic 3-cell → 1-cell lookup table
(`data/ca/eca.py:25 def step(...)`: `idx = (left<<2)|(row<<1)|right; return table[idx]`). Under the
**default** masking — i.i.d. random at `mask_ratio=0.5` (`data/ca/masking.py:36`) — a masked cell usually
still has visible neighbours, so the optimum is to *memorize the 8-entry rule table and read off the
neighbourhood*. No attention over distance, no state. By contrast DYCK uses **close-only masking**
(`data/dyck/masking.py:13 CloseOnlyMasking`, `:23` selects only closing-bracket ids): predicting a masked
closer requires knowing which opener is still on the stack — an unbounded-distance dependency. DYCK forces
stack-tracking via attention; CA rewards a local lookup.

### H2 — Vocabulary poverty starves the representation
CA binary tokenization has `K = N_SPECIAL + 2 = 4` (two content tokens; `data/ca/tokenize.py:18,21`),
whereas DYCK uses `K = 2 + 64 + 64 = 130` bracket types. Token embeddings are **frozen orthogonal**
(`model/embeddings.py:15 FrozenTokenEmbedding` — `weight[:K,:K] = eye(K)*scale`), so the only way the
blocks can exploit token identity is to build machinery that separates many near-orthogonal types. DYCK
forces a rich 128-way type space; CA's near-binary signal barely exercises it. Notably the repo *already
tried to fix this*: `block` tokenization coarse-grains `block_size` cells into one of `2**block_size`
ids — explicitly described in `data/ca/tokenize.py:9-11` as "the CA analog of the paper's vocabulary-size
sweep." It helped (block variants rose to ~71.4–72.2) but never closed the gap, which tells us vocabulary
is *a* factor but not the whole story.

### H3 — 2D→1D flattening + a structure-free position code hides CA's geometry (the strongest evidence)
CA's signal is genuinely 2D — a time×space spacetime diagram — but it is flattened **row-major** into a 1D
token sequence at `data/ca/dataset.py:70` (`ids.reshape(self.N)`; NumPy C-order, so grid cell `(t,x)` →
sequence position `p = t·W + x`). A cell's causal parents `(t−1, x−1..x+1)` then land at positions
`p−W−1, p−W, p−W+1` — stride ≈ W=14 apart — and the widening light-cone becomes a non-contiguous stencil
in 1D. Worse, the position code is **random** frozen vectors (`model/embeddings.py:40
FrozenPositionalEmbedding`; `cfg.model.pos_embed` defaults to `"random"`, `config/schema.py:31`), which
expose neither 1D nor 2D distance — so "p−14 is my vertical neighbour" is never handed to the model.

This is a CA-*specific* penalty: random position codes are fine for 1D DYCK (matches are content/order-based,
resolved by attention) but bad for CA, whose rule **is** a spatial-locality operation. Same pipeline,
opposite effect.

Two facts sharpen this (verified across `main`, `feat/ca-iid-operator-test`, `results/ca-difficulty-sweep`,
`run/cifar100-block-a5000`, `run/cifar100-hard-5090`):
- The row-major flatten (`dataset.py:70`) is **identical on every branch**, and **CA was never paired with
  the 2D-structured `sincos2d` code** (it appears only on `dyck-2dpos`/`spatial-dyck`). So **CA + sincos2d
  is an untested cell** — the one configuration that could undo the flatten's damage cheaply.
- Determinacy distinction (§5): CA's flatten is a *fixed bijection* — obscured, **not** ambiguous — whereas
  the failed `spatial-dyck` probe is genuinely under-determined.

The cleanest internal diagnostic is still `spatial-dyck`: re-laying *DYCK itself* into 2D made transfer
**drop** (nested 70.42, permuted **66.93** vs 1D 72.64; `results/reports/spatial-dyck-screen/`). The form in
which structure is presented matters as much as the structure — and CA is *inherently* spatialized.

### H4 — Determinism invites memorization instead of distribution-learning
A CA spacetime diagram is a deterministic function of its seed (`data/ca/dataset.py:simulate_spacetime`):
for any masked cell there is exactly one correct answer derivable from the rule. DYCK sequences are
*stochastically generated under a hard global constraint* (`data/dyck/generator.py:23 dyck_ids`, stochastic
opens/closes with a stack that must flush — `:40-45`). The model must internalize the *constraint
distribution*, not a fixed map. The repo itself encodes this suspicion: `data/ca/iid_step.py` builds a
"true vs shuffled" control precisely to detect whether the model learns the CA operator or just its
marginal statistics — and `cifar100-iid-true` (66.81) barely differs from `cifar100-iid-shuffled` (66.90),
i.e. the causal operator added almost nothing.

### H5 — The masking objective is not structure-targeted
DYCK masks exactly the tokens whose prediction needs global state (closers). CA inherits MAE/BERT-style
uniform masking by default. More principled CA modes exist — `forward` (predict the future rows) and
`lightcone` (mask later-time cells with rising probability), `data/ca/masking.py:6-11,38-47` — but the
default is `random`, and even the principled modes did not lift CA above DYCK. So even where CA *has*
long-range structure, the objective rarely demands it.

## 3. What the prior in-repo analysis already established

- **`phase1-operator-vs-texture`** — the headline operator-vs-texture screen: DYCK 72.64 vs CA 68.86; CA
  fails its own success bar ("match or exceed random").
- **`phase2a-layer-localization`** — the warm-up benefit is *distributed across all blocks*, not localized
  to early or late layers, and grafting CA-early/DYCK-late (or vice-versa) degrades performance. CA cannot
  supply the distributed representation DYCK does.
- **`spatial-dyck-screen`** — 1D temporal order is critical; 2D re-layout of the same grammar hurts
  (supports H3).
- **`ca-generator`** — Rule 110 is objectively complex (Lempel-Ziv 2564, Wolfram Class IV, Turing-complete)
  yet that algorithmic complexity does **not** translate into a useful warm-up signal. High complexity ≠
  useful bias.
- **`focused_rule110_cifar100`** — the clean three-way Random / CA / DYCK comparison confirming CA loses to
  random.

## 4. Synthesis

CA does not fail by teaching the model *nothing*; it fails by teaching the *wrong thing*. The CA warm-up,
as instantiated here, is a **local, low-vocabulary, deterministic, spatially-flattened texture predictor**
— precisely the bias transfer-to-vision does not want. The sub-random scores (iid-static 65.72, plain
Rule 110 68.86) are the signature of an actively harmful prior: the model spends warm-up specializing its
blocks for a cheap local lookup, then has to unlearn it during image training, ending below a fresh
initialization. DYCK wins because its objective forces the three transferable computations CA's does not:
long-range binding (stack tracking), a rich type space (128 brackets), and a constraint distribution rather
than a memorized map.

The actionable consequence — pursued in the companion report — is that the right next experiment is **not**
"more complex CA," but a procedural source that keeps DYCK's transferable properties while adding *genuine*
2D structure (not a re-layout): the 2D Dyck languages of `docs/2307.16522.pdf`.

## 5. Dimensional re-mapping and under-determination (1D↔2D)

Two of the failing sources re-map dimensionality in opposite directions; comparing them isolates *what*
about re-mapping hurts. The shared variable is the **alignment between where the data's dependencies live
and what the position code reveals about a token's place**.

| Source | Native | Transform | Pos code | Dependency | Failure mode | best top-1 |
|---|---|---|---|---|---|---:|
| 1D DYCK | 1D | none | random | order/stack | — (aligned enough) | 72.64 |
| **CA** | 2D | 2D→1D, **fixed bijection** | random | 2D spatial-local | structure **obscured** + local shortcut | 68.86 |
| spatial-dyck nested | 1D | 1D→2D, **per-sample D4** (`spatial.py:128`) | sincos2d | order/stack | **partly under-determined** | 70.42 |
| spatial-dyck permuted | 1D | 1D→2D, **per-sample perm** (`spatial.py:126`) | sincos2d | order/stack | **fully under-determined** | 66.93 |

**The two failures rhyme but are not identical:**

- **2D→1D (CA) is NOT under-determined.** The flatten is a fixed bijection — grid cell `p` *always* means
  `(t,x) = (p//14, p%14)`. Nothing is ambiguous; the geometry is merely **obscured** (flattened + a
  structure-free position code), which lets the model settle on a position-indexed *local lookup* — the
  cheapest, least-transferable solution (H3 compounding H1). In principle recoverable by exposing 2D
  adjacency (`sincos2d`) — untested for CA — though H1/H2/H4 would likely still cap it.
- **1D→2D (spatial-dyck) IS under-determined**, by construction: per-sample randomization makes the *same
  grid cell hold different roles in different samples*, so even a clean `sincos2d` code can't be tied to
  structure. The *amount* of randomization predicts the *amount* of damage (permuted 66.93 < nested 70.42).

**Verdict:** "going to 2D caused the failure" is too coarse. Re-mapping dimensionality fails when it
**decouples position from role**. Spatial-dyck does that (per-sample randomness → under-determination);
CA's flatten does not, yet CA still fails for the adjacent reason that it never *exposes* the
role-determining geometry.

**Is there under-determination going 1D→2D? Yes, in two senses.**
1. *Layout (avoidable).* The per-sample D4/permutation injects ambiguity; a fixed layout removes it — but
   then absolute position becomes a type shortcut (the reason they randomized). That tension is the lesson.
2. *The grammar lift (unavoidable; central to the DC_k design).* The paper proves **DW_k ⊊ DN_k ⊊ DQ_k ⊊
   DC_k** — "every row and column is a Dyck word" is strictly *weaker* than well-nested boxes, so the same
   row/column projections admit structurally different pictures. Where 1D Dyck's stack gives a *unique*
   parse (a masked closer is *forced* by its one matching opener), a 2D crossword under-determines a
   completion more. **Design consequence:** the masking objective must keep each masked target *determined
   by the visible context*, else the model fits only the marginal — the degenerate regime that produced
   CA's worst score (`iid-static` 65.72, an unconstrained Bernoulli cell). This is the determinacy
   requirement carried into the companion design report.

## 6. Figures referenced (already in `results/figures/`)

- `../../figures/phase1-operator-vs-texture.png` — operator-vs-texture headline.
- `../../figures/spatial-dyck-screen.png` — 1D vs spatialized DYCK (evidence for H3).
- `../../figures/phase2a-layer-localization.png` — distributed-signal ablation.
- `../../figures/ca_spacetime_panel.png` — CA spacetime examples / complexity panel.
- `../../figures/focused_rule110_cifar100.png` — Random / CA / DYCK three-way.
