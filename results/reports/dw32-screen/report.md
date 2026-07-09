# H6 screen verdict — DW_32 structure transfers (largest operator signal yet measured), but the sincos2d presentation drowns it below random init

**Status:** analysis report (no new runs). Numbers verbatim from the committed
`results/reports/*/metrics.json` of the screen launched by `scripts/run_dw2d_experiments.sh`
(1 seed each; P7: single-run differences < 0.7 are noise). Companion reports:
design [`../../../docs/2d-dyck-experiment-design.md`](../../../docs/2d-dyck-experiment-design.md) (§4 H6 bands),
sampler gate [`../dw32-stats/report.md`](../dw32-stats/report.md),
position-code audits [`../pos-embed-audit-sincos2d/report.md`](../pos-embed-audit-sincos2d/report.md) /
[`../pos-embed-audit-sincos2d_tuned/report.md`](../pos-embed-audit-sincos2d_tuned/report.md),
probes [`../dw32-vit-t-probe/report.md`](../dw32-vit-t-probe/report.md) et al.,
head-to-head [`../dw32_vs_kdyck_cifar100/report.md`](../dw32_vs_kdyck_cifar100/report.md).

## 0. Bottom line

- **H6 verdict: FAILURE band** (A = 67.88 ≤ 70.7) — but **not the CA null**. The
  treatment−control gap is **+3.04** (67.88 vs 64.84), the largest true-vs-shuffled
  operator signal measured in this project (CA's best: +1.42, GoL; CA's decisive 1-D
  nulls: −0.09/−0.03). **The 2D matching structure itself transfers.**
- **The failure is presentation-dominated.** The structure-free control locates the
  sincos2d + corner-objective *baseline* at 64.84 — ~5.5 pts below random init
  (70.34). The +3.04 of real structure sits on top of that hole and cannot climb out.
  This is the 4th and 5th in-repo data point on one line: every `sincos2d` warm-up arm
  has transferred worse than its random-position counterpart (k-Dyck 72.64→72.19; CA
  68.86→65.35; hard 71.86→71.36; now both DW arms sub-random).
- **The grid-matched band is not the fix.** `sincos2d_tuned` (48/48 effective
  frequencies vs 15/48) improved transfer by only +0.72 (68.60) — still 1.7 below
  random. The problem is not *which* frozen structured code, but that a structured
  frozen code is stripped at transfer and replaced downstream by a randomly-initialized
  learned one: block circuitry keyed to the sinusoid channels must be unlearned.
- **The task is far too easy at 14×14 — a second, real but secondary, drag.** DW32
  warm-up shows a phase transition to ~95% masked accuracy at ~step 1,000 (99% by
  3,400; tuned: 99% by **450**), then coasts: mean final weight drift 1.63 vs k-Dyck's
  2.28 (k-Dyck climbs all 15k steps, reaching 99% only at ~9,650). Roughly 3/4 of the
  DW training budget carries near-zero gradient.

## 1. The numbers (CIFAR-100 best top-1, this branch, seed 42)

| Arm | Run | Warm-up final acc (sat. step to 0.99) | **Top-1** | Δ vs random |
|---|---|---|---:|---:|
| B  k-Dyck 1D (canonical recipe, random pos) | `cifar100-dyck-repro` | 0.864 (9,650) | **72.08** | +1.74 |
| C  random init | `cifar100-random-repro` | — | **70.34** | 0 |
| A′ DW_32 + sincos2d_tuned | `cifar100-dw32-tuned` | 0.988 (450) | 68.60 | −1.74 |
| A  DW_32 + sincos2d (treatment) | `cifar100-dw32` | 0.934 (3,400) | 67.88 | −2.46 |
| D  DW_32 marginal-shuffle (control) | `cifar100-dw32-shuffle` | 0.131 (never) | 64.84 | −5.50 |

Phase-0 note: the k-Dyck reproduction (72.08) sits −0.56 from the historical 72.64 —
outside the ±0.4 gate, inside the 0.67 in-repo seed spread; the random anchor
reproduced (+0.32). All contrasts here are within-branch, per the in-repo-anchors rule.

**Decision bands (design doc §4):** success (A ≥ 72.0 ∧ A−D ≥ 1.0) — no. Partial
(70.7 ≤ A < 72.0) — no. **Failure (A ≤ 70.7) — yes**, with the *predicted-outcome*
pattern matching the presentation branch, not the operator-null branch: A − D = +3.04
clears the ≥1.0 operator bar decisively while the whole 2D-presented family sits below
random. (Cf. spatial-dyck: nested 70.42 vs permuted 66.93 — same shape: structure gap
inside a depressed presentation band.)

## 2. Mechanistic reads (probes)

Uniform-attention baseline on the 14×14 grid = 7.28 units (max 18.4).

| Warm-up | final mean attn distance | note |
|---|---:|---|
| dyck-repro | 6.10 | focused below uniform; drift 2.28 (learning all run) |
| dw32-vit-t | 7.09 | ≈uniform; mid-blocks *above* uniform (7.8) — long-range binding present, no short-range CA collapse |
| dw32-tuned | 6.48 | slightly focused; least drift (1.30) — easiest task |
| dw32-shuffle | (NaN rows) | **CLS attention sink**: up to 67% of grid-query mass parked on CLS (structure-free training has nothing to attend to); probe renormalization now clamped |

The shuffle arm's 13.1% warm-up accuracy (≫ the 3.1% marginal floor) is explained by a
*counting* solution: the per-picture index multiset is preserved by the shuffle, so the
visible corners' index histogram narrows the masked-d distribution — global bookkeeping,
no matching. It transferred worst of all arms (64.84, cf. `iid-static` 65.72).

## 3. Suspect-by-suspect assessment

1. **sincos2d presentation — primary cause (strong evidence).** Control at −5.5 vs
   random with zero structure; consistent direction across 5 in-repo arm pairs; tuned
   band fixes the code's geometry but recovers only +0.72 ⇒ the mismatch is
   *structural*: the frozen positional frame the blocks specialised to is deleted at
   transfer (downstream `pos_embed` is fresh, learned, random-init).
2. **Masking too easy — real, secondary.** Saturation at 1/10th of the budget wastes
   most training pressure and the ~50% adjacent-partner shortcut is measured
   (`dw32-stats`). But within the sincos family, the *easiest* arm (tuned, sat. 450)
   transferred *best* — easiness alone does not order these outcomes.
3. **"DW is the wrong problem" — not supported.** +3.04 over its own matched control is
   the strongest structure-transfer signal this project has produced. The language is
   fine; the wrapper (position code, and secondarily task difficulty at 14×14) is
   what's broken.

## 4. Licensed next steps (in order)

1. **DW_32 + `random` frozen positions, plus its shuffle control at random positions**
   (H7 arm B; two YAMLs, no new code, 1 seed). The single decisive experiment: removes
   the only factor with strong evidence while keeping data + masking fixed.
   Pre-registered read: ≥ ~70.7 ⇒ presentation was the killer, DW viable → combine with
   harder masking; ≈ 68 ⇒ presentation exonerated → masking/task-difficulty becomes primary.
2. **Harder masking arms** (attack saturation + shortcut): `min(row_span, col_span) ≥ 2`
   eligibility (one line), c+d masking (both closing roles — forces both axes), H11
   audited-random (needs pre-generation for throughput).
3. **Rasterized-1D DW** (H7 C/D): the "2D language, 1D presentation" pivot if (1) lands partial.
4. **Downstream-side alignment** (MAE-style fixed sincos2d in the vision model):
   principled test of "transfer the positional frame too", but changes the downstream
   recipe — requires re-anchoring every arm; defer.
5. **No 3-seed spend on this configuration**; seeds go to the first arm that clears random.

## 5. Tier-1 addendum: the position-code factorial (run 2026-07-06+)

Completed factorial (`scripts/run_dw2d_tier1.sh`; see `../dw32_tier1_cifar100/report.md`
and `../dw32-tier1-probe-compare/report.md`):

| | sincos2d | randpos | pos-code effect |
|---|---:|---:|---:|
| DW_32 structure | 67.88 | **69.07** | +1.19 |
| DW_32 shuffle | 64.84 | 67.47 | +2.63 |
| **structure gap** | **+3.04** | **+1.60** | |

Anchors: random 70.34, k-Dyck 72.08.

- **The presentation hypothesis is confirmed as a real tax (+1.2 to +2.6) but rejected
  as the primary cause**: DW+randpos still lands 1.27 *below* random init. The
  position-code question is measured and closed — `random` is the default henceforth.
- **The primary cause is insufficient sustained difficulty.** Saturation to 99% masked
  accuracy: dyck 9,650 steps (never fully solved, final avg 0.864); DW sincos2d 3,400;
  DW randpos **1,150**. Block-weight-norm trajectories (probe snapshots, vs init):
  dyck grows continuously 1.0→3.15 (machinery built all run); DW sincos2d spikes to
  2.91 by 2k then **decays to 2.30** (post-saturation weight-decay erosion: no gradient
  opposes wd=0.05 for ~12k steps); DW randpos barely builds (1.85). Transfer tracks
  built-and-retained machinery. The structure lift DW can generate before exhausting
  (+1.6 to +3.0) is smaller than the fixed specialization cost of symbolic warm-up
  (structure-free arms: 64.8–67.5, cf. parent-paper dyck-shuffle 67.22 < random) —
  k-Dyck wins because its lift (~+4.8 over its shuffled variant) exceeds that cost.
- **Licensed next steps:** (1) raise sustained difficulty on the randpos base — strict
  `min(row_span, col_span)` filter + c+d masking (hours; saturation step must move
  right, then transfer); (2) free erosion test: strip the 2k/4k probe snapshots of
  `dw32-vit-t` and transfer them (early-stop vs 15k); (3) begin the H8 ladder with
  DN_32 — the "too rigid/easy" failure mode is exactly what the hierarchy's
  less-constrained families vary. NOT licensed: further embedding work, downstream-side
  positional alignment, 3-seed spends on any current arm.

_Generated as an analysis report (no new runs) on branch `feat/2d-dyck`._
