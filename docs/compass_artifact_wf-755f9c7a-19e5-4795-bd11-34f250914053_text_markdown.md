# Procedural Warm-Up for ViTs: Validating the CA Verdict and Designing 2D-Dyck Experiments

## TL;DR
- **The conclusion "CA pretraining is worse than k-Dyck warm-up" is supported as a bounded, mechanism-level claim.** The operator-shuffle control (near-zero delta between true and scrambled CA rules) and the fact that CA only nears k-Dyck after importing two non-CA-specific fixes (block tokenization ≈130-way vocab; forward masking) are strong evidence that raw CA under masked-state prediction lacks the transferable properties k-Dyck has. It does **not** refute Zhang et al.'s edge-of-chaos result, which used a different objective (autoregressive next-state prediction of CA time-evolution) and a different model (a GPT-2 LLM, not a ViT).
- **Explicit 2D Dyck (Crespi Reghizzi et al.) is a well-motivated, medium-risk next step and is the right one to pursue.** The strict inclusion chain DW_k ⊊ DN_k ⊊ DQ_k ⊊ DC_k is a genuine constraint/determinacy gradient analogous to the parent paper's Chomsky sweep, and it preserves all three transferable properties (long-range binding, rich vocabulary via 4k corner symbols, constraint-based rather than deterministic mapping) while adding 2D structure that matches a ViT's native inductive bias. Proceed — but gate it behind a reproduction check and lead with the native-vs-rasterized comparison rather than DW_k alone.
- **A concrete design (H6–H10) is specified below.** It branches from the existing repo, holds infrastructure fixed across 1D/2D, adapts masked-state prediction to 2D (carrying the forward-masking lesson), includes the decisive shuffle control, and ties decision thresholds to the parent-paper anchors (+1.72% ImageNet-1K; CIFAR-100 71.98% for k-Dyck vs. 68.52% random).

---

## Sourcing note
I could not open the uploaded project files (`report.md`, `Current_Hypotheses_and_Ideas`, and the two PDFs) directly through the available tools. Claims attributed to the user's experiments are therefore based on the descriptions in the task brief and are flagged "per the project brief"; the uploaded reports are authoritative wherever they differ. Claims about the two papers are based on the actual arXiv versions retrieved (2511.13945v2 and 2307.16522). External-literature claims are labeled as such and separated from the project's own evidence.

---

## Part 1 — Is "CA is worse than k-Dyck" a valid conclusion?

### 1.1 What the parent paper actually established
Shinnick et al., *Can You Learn to See Without Images? Procedural Warm-Up for Vision Transformers* (arXiv:2511.13945v2, CVPR 2026) is the reference frame for "what good looks like." Directly-quoted results:
- **k-Dyck (k=64, 128-token vocab) is the winning grammar.** On ViT-T/16, CIFAR-100 top-1 goes from **68.52% (random init) → 71.98% (k-Dyck)**; regular WW gives **66.44%** (below random) and context-sensitive k-Dyck-Shuffle gives **70.11%** (Table 4). On ViT-B/16, k-Dyck warm-up yields **+1.72% on ImageNet-1K** (77.49 → 79.21), "equivalent to 28% of the ImageNet-1K data."
- **Structure, not statistics, drives the gain.** Shuffling token order within Dyck sequences drops CIFAR-100 to **67.22%**, *below* random init (Table 5). This is the parent paper's own decisive control and the template for the user's operator-shuffle control.
- **The signal is distributed and lives in late layers.** Shuffling attention-only or MLP-only weights both degrade performance (71.98 → 70.57 / 70.71, Table 7); transferring only the final 4 layers recovers most of the gain (71.66 vs. 71.98, Table 8). Warm-up acts on both attention and MLP, unlike Mimetic init — and mainly on *late* layers, opposite to standard visual pretraining.
- **Vocabulary has an interior optimum.** CIFAR-100 by vocabulary: k=16→69.05, k=32→69.78, k=64→71.98, k=80→71.83 (Table 11).
- **Objective:** masked-token prediction, 50% mask ratio on structurally-informative tokens (closing brackets), frozen random embeddings, 15k steps, patch-bypass interface.
- **Seed rigor:** 3-seed mean±std reported on representative sets (e.g., CIFAR-100: k-Dyck 71.98 ± 0.74; random 68.52 ± 0.27; Mimetic 70.72 ± 0.39; FractalDB 64.61 ± 0.51).

From these, the project brief identifies **three transferable properties** that explain k-Dyck's success, all well-grounded in the paper: **(1) long-range binding** (stack-based hierarchical dependencies; shuffling kills it), **(2) rich vocabulary** (interior optimum near k=64/128 tokens), and **(3) constraint distribution** — the mapping is constraint-based (many valid completions, especially for Shuffle) rather than a deterministic function of a local window.

### 1.2 The user's CA evidence, as described
Per the project brief, the uploaded reports contain: (a) a **post-mortem failure analysis** of CA warm-up; (b) an **operator control** comparing true vs. shuffled/scrambled CA rules that shows a **near-zero delta**; (c) a finding that **CA only approaches k-Dyck once two generic, non-CA-specific modifications are bolted on** — block tokenization importing ~130-way vocabulary richness, and forward masking importing a non-local objective; (d) an attribution of failure to CA's **locality and determinism**, which lets a ViT solve masked-state prediction by shallow rule-table lookup rather than distributed attention/computation; plus an "H3-verdict" and a CIFAR-100 summary.

### 1.3 Assessment of validity

**The internal logic is sound and, crucially, mechanism-first rather than score-first.** The single most persuasive piece of evidence is the **operator-shuffle control returning a near-zero delta** — the exact analogue of the parent paper's order-shuffle control, and diagnostic in a way a raw accuracy number is not. Interpretation: if replacing the *true* CA update rule with a *scrambled* rule does not change downstream accuracy, then the ViT is **not** extracting anything specific to the CA's computational structure; it is learning something rule-agnostic (marginal statistics, local co-occurrence, or a trivial copy/lookup). For k-Dyck the corresponding manipulation (order shuffle) *destroys* the benefit, proving the model used the structure. The contrast is the crux, and it is valid.

**The "two modifications" finding is the second strong pillar, correctly interpreted as a bound rather than a rescue.** CA reaching k-Dyck-like performance *only* after (i) block tokenization importing ≈130-way vocabulary richness and (ii) forward masking importing a non-local objective is effectively a decomposition proof: the benefit, when it finally appears, is attributable to **the imported generic properties (rich vocabulary + non-local objective), not to CA itself.** Both are two of the three transferable properties k-Dyck already has natively. "CA works if you make it behave like Dyck" is evidence *for* the properties hypothesis and *against* CA having independent value under this pipeline.

**The mechanistic attribution (locality + determinism → shallow lookup) is well-supported by external theory.** Elementary/Neural CA are by construction **local and (for classical ECA) deterministic**: the next state of a cell is a fixed function of a small fixed neighborhood. Under a *masked-state infilling* objective on a *single* configuration, recovering a masked cell is a bounded-window lookup — no long-range binding to learn, no distribution over valid completions to represent. Liu, Ash, Goel, Krishnamurthy & Zhang, *Transformers Learn Shortcuts to Automata* (arXiv:2210.10749v2, ICLR 2023), show transformers preferentially learn low-depth "shortcut" solutions to automata: their abstract states "a Transformer with o(T) layers can exactly replicate the computation of an automaton… polynomial-sized O(log T)-depth solutions always exist; furthermore, O(1)-depth simulators are surprisingly common," and the paper documents that such shortcuts are **statistically brittle** and generalize poorly out-of-distribution. A CA masked-infill task invites exactly this shortcut — the opposite of the "distributed across attention + MLP, concentrated in late layers" signature the parent paper found for k-Dyck. This explains why CA is not intrinsically useless but is *mismatched to this objective* (see §1.5).

**Statistical rigor — the main caveat to close.** The k-Dyck-vs-random gap (~3.5 points) is several times the ~0.3–0.7 seed std, so it is real. For the CA verdict to be equally airtight, the uploaded reports should show — symmetrically, across the same 3+ seeds and a pre-registered decision threshold — that the **CA-vs-random delta is within noise (or negative)** and that the **true-CA-vs-shuffled-CA delta is within noise**. "Near-zero delta" is consistent with this, but the published verdict should state seed count, std, and threshold explicitly, and should not rest on CA runs with fewer seeds than the k-Dyck runs.

**Fairness of the comparison — mostly satisfied, one confound to name.** The conclusion is only meaningful if CA and k-Dyck ran through **identical infrastructure, compute, and tokenization budget.** The project's design (same repo, same ViT-T/16, same masked objective, same step budget) satisfies this in principle, and the operator-shuffle control is *internally* matched by construction, which makes it decisive regardless of cross-condition drift. The one genuine confound to flag is **vocabulary/tokenization budget**: k-Dyck natively uses ~128 tokens, whereas a naive CA encoding uses ~2–3 symbols, so part of a raw CA deficit could be a vocabulary-budget artifact — which is exactly why the block-tokenization (≈130-way) manipulation matters and partly closes the gap. The correct framing (which the brief appears to adopt): the operator-shuffle control neutralizes this confound because true-CA and shuffled-CA share the same vocabulary and objective, so their near-zero delta cannot be a tokenization artifact.

### 1.4 Does it refute Zhang et al. (edge of chaos)? No — and the report should say so explicitly.
Zhang, Patel, Rizvi et al., *Intelligence at the Edge of Chaos* (arXiv:2410.02536, ICLR 2025), found that rule complexity correlates with downstream reasoning/chess performance, with an intermediate sweet spot; verbatim: *"Both uniform and periodic systems, and often also highly chaotic systems, resulted in poorer downstream performance, highlighting a sweet spot of complexity conducive to intelligence."* Three differences bound its applicability to the user's setting:
1. **Model:** a GPT-2 LLM with token embeddings replaced by a linear projection for binary vectors (per §3.2–3.3), *not* a ViT; downstream evaluation freezes the pretrained GPT-2 layers and trains only input/output projections.
2. **Objective:** predicting the **future states of the automaton** — verbatim (§3.1): *"we sample subsequences by selecting random windows of 60 time steps and 100 spatial dimensions… we train models to predict either 1 or 5 steps in the future."* This models the CA's **time-evolution**, a genuinely non-local, multi-step computation, whereas the user's ViT pipeline uses **masked-state infilling of a single configuration**, a fundamentally local task.
3. **Complexity control:** Zhang et al. deliberately select rules by complexity (Lempel-Ziv, Wolfram class; ECAs have 256 rules, 88 unique under symmetry, including Rule 110 which is Turing-complete and Rule 90 which draws the Sierpinski triangle). The user's CA condition does not appear to sweep this edge-of-chaos axis.

So the user's result is **not** a counterexample to Zhang et al.; it concerns a different objective on a different architecture. Defensible phrasing: *"CA data, encoded as static configurations under masked-state prediction and fed to a ViT with frozen embeddings, does not confer the transferable inductive bias that k-Dyck does. This neither confirms nor refutes the edge-of-chaos finding, which concerns autoregressive next-state prediction of CA time-evolution in LLMs."* Avoid the overclaim "CA is bad for pretraining."

### 1.5 The decisive mechanistic insight (external corroboration)
Reconciling the user's failure with the *successes* of CA elsewhere yields a clean, publishable explanation: **in every documented case where CA data helped a transformer, the objective forced modeling of the CA's time evolution, not infilling of a static state.**
- **Zhang et al. 2024:** GPT-2 predicting future ECA states 1 or 5 steps ahead over 60-timestep × 100-cell space-time windows.
- **Lee, Han, Kumar & Agrawal, *Training Language Models via Neural Cellular Automata* (arXiv:2603.10055):** a transformer LLM trained with next-token prediction on **rolled-out NCA spatiotemporal trajectories**; reports up to **6% LM improvement and up to 1.6× faster convergence** from 164M NCA tokens (even beating 1.6B tokens of Common Crawl), and finds "optimal NCA complexity varies by domain."
- **Jiang/Shinnick et al., *Procedural Pretraining* (arXiv:2601.21725, ICML 2026):** their CA condition uses ECA Rule 110 where "the model must predict the next state," and they characterize Zhang et al.'s CA gains as "marginal but consistent improvements."

The time-evolution rollout is what makes CA non-local and computationally rich; masked infilling of a single frame discards exactly that. **This is the single most important open question the CA post-mortem should register:** the verdict "CA is worse than k-Dyck" is valid *for the masked-static-configuration objective*, but a CA warm-up built on **next-state / trajectory prediction** was not tested and remains a live (if lower-priority) alternative. Note also that all three CA successes are on **LLMs**; the only ViT procedural-warm-up paper (the parent) found the winning signal to be k-Dyck, with structural order (not token distribution) driving gains — consistent with the user's negative CA result on ViTs.

### 1.6 Verdict (Part 1)
**Supported, with scope.** The conclusion "CA pretraining (as implemented: static configurations, masked-state prediction, frozen embeddings, ViT-T) is worse than k-Dyck warm-up" is valid and unusually well-controlled for a negative result, because the operator-shuffle control is internally matched and diagnostic. Residual caveats / open questions to record: (a) publish seed count, variance, and decision threshold symmetrically for CA and k-Dyck; (b) state that the CA baseline was not merely vocabulary-starved (the shuffle control mitigates this, but say so); (c) bound the claim so it does not read as refuting Zhang et al.; (d) flag that a **time-evolution/next-state CA objective** was never tested and is the most defensible way CA could still work.

---

## Part 2 — Is explicit 2D Dyck the right next step?

### 2.1 What the 2D Dyck paper provides
Crespi Reghizzi, Restivo & San Pietro, *Two-dimensional Dyck words* (arXiv:2307.16522), lift Dyck from strings to **pictures** (2D symbol arrays) with a strict inclusion hierarchy of four families:
- **DW_k (well-nested):** rephrases "any two matching pairs are well-nested or disjoint" for rectangular boxes; a generalization of the Chinese-box language, proven **not tiling-system recognizable** (unlike Chinese boxes) — genuinely beyond the 2D "regular" class.
- **DN_k (neutralizable):** rephrases Dyck cancellation as a **neutralization rule** on quadruples of corner symbols, iterated from 2×2 subpictures until the picture is wholly neutralized.
- **DQ_k (quaternate):** the subset of Dyck crosswords whose matching circuits are all length-4 (rectangles).
- **DC_k (Dyck crossword):** the row-column combination — **every row and every column is a 1D Dyck word** over an alphabet of **size 4k** (the 4k corner symbols). Matching relations form circuits of length a multiple of 4; length-4 circuits are rectangles, longer ones (the paper illustrates length 12 and 36) give rich non-rectangular patterns. Also proven not tiling-recognizable.
- **Strict chain:** DW_k ⊊ DN_k ⊊ DQ_k ⊊ DC_k (the paper proves DN_k coincides with quaternate DC_k where neutralizability induces a partial order, and the four are ordered by strict inclusion).

Note: this is a **formal-language-theory paper with no ML experiments** — every ML prediction below is an inference, not an established result.

### 2.2 Mapping to the three transferable properties
| Property (from k-Dyck) | Does 2D Dyck preserve/extend it? |
|---|---|
| **Long-range binding** | **Extended.** In DC_k every row *and* every column must independently balance, so a symbol participates in **two orthogonal long-range matching constraints** at once. Circuits of unbounded length (multiples of 4) create dependencies spanning the whole picture — strictly richer than 1D nesting, and a natural match for ViT self-attention, which is 2D/global from layer 1. |
| **Rich vocabulary** | **Native.** DC_k uses **4k corner symbols**; k is a direct knob. This is the cleanest match to the parent paper's vocabulary finding (interior optimum near 128 tokens / k=64): setting k≈32 gives 128 symbols, aligning the 2D vocab budget with the proven 1D sweet spot and with the ≈130-way block-tokenization result from the CA post-mortem. |
| **Constraint distribution (non-deterministic)** | **Preserved and strengthened.** Like k-Dyck-Shuffle, a masked cell in a DC_k picture generally has **multiple valid fillings** consistent with row+column constraints — constraint satisfaction, not a deterministic local lookup. This is precisely the property CA lacked. |

**This is the strongest argument for 2D Dyck:** it keeps everything that made k-Dyck work (hierarchical binding, rich vocab, constraint-based completion) while adding the 2D structure that matches a ViT's native inductive bias — without reintroducing CA's fatal locality+determinism.

### 2.3 The inclusion chain as a determinacy/ablation ladder
DW_k ⊊ DN_k ⊊ DQ_k ⊊ DC_k is a genuine, publishable analogue of the parent paper's Chomsky sweep (Regular WW → CF k-Dyck → CS Dyck-Shuffle). It is a **constraint-richness / determinacy gradient**: DW_k is most constrained (rigid rectangular nesting), DC_k least constrained (arbitrary circuits, most valid configurations, most entangled). This lets the user re-run the parent paper's central question — *does an interior sweet spot exist?* — in 2D. The parent paper found CF k-Dyck beat both its simpler (WW) and more-entangled (Shuffle) neighbors; the analogous 2D prediction is that an **intermediate** family (likely DN_k/DQ_k) beats both the most rigid (DW_k) and the most entangled (DC_k).

### 2.4 Should we start with DW_k?
**Partly. Use DW_k as an anchored floor, not the headline bet.** The brief's rationale ("most constrained → most learnable") is reasonable for establishing that the 2D pipeline works at all. But the parent paper's own result is a caution: the *most* constrained 1D language (regular WW) **hurt** performance (66.44 < 68.52). By analogy, DW_k — the most rigid, most nearly-deterministic 2D family — is the family most at risk of the **CA-style failure mode**, where rigid nesting collapses toward a near-deterministic local pattern a ViT solves by shortcut. Recommendation: **run the whole ladder**, bet the primary hypothesis on an intermediate family (DN_k or DQ_k), and treat DW_k as the control that tests whether "too rigid" reproduces the WW/CA failure. Start *building* with the DC_k generator (the row-column crossword is the most direct to implement) and derive subfamilies by filtering circuit length.

### 2.5 Recommendation (Part 2)
**Proceed with 2D Dyck — it is better-motivated than any CA variant under the current objective — but gate it behind a reproduction check and design it as a ladder with the native-vs-rasterized comparison as a first-class question.** It tests a novel, defensible hypothesis (2D constraint structure matches ViT inductive bias) rather than re-litigating CA. Prioritize it **above** the complexity-and-modality curriculum (§4.5).

---

## Part 3 — Experiment design (H6–H10)

**Conventions:** branch from `github.com/zlshinnick/procedural-warmup-vit`; identical infrastructure/optimizer/step-budget across all conditions (AdamW, lr 2e-3, cosine decay, 15k warm-up steps, batch 256, 50% mask, 1k warm-up steps in schedule); ViT-T/16 for CIFAR-10/100, FOOD-101, STL-10, Tiny-ImageNet; ViT-B/16 for ImageNet-1K; standard Dosovitskiy architecture, 16×16 patches; frozen random symbol embeddings; sincos-2D positional encoding; patch-bypass interface. Report 3+ seeds, mean±std. **Primary metric:** CIFAR-100 top-1 (the parent paper's ablation workhorse). **Anchor thresholds:** random init 68.52; k-Dyck 71.98 (+3.46); ImageNet-1K k-Dyck +1.72%.

### Global controls (apply to every hypothesis)
- **C-shuffle (decisive):** scramble the 2D matching structure while preserving symbol marginals (permute cells within each picture, or use a "scrambled operator" that violates row/column balance). Analogue of the CA operator-shuffle and the parent paper's order-shuffle. **Prediction:** if 2D Dyck works for the right reason, C-shuffle collapses the gain to ≤ random init.
- **C-parity (infrastructure):** 1D k-Dyck (k=64) run through the *identical* branched pipeline, to confirm the branch reproduces the parent number before any 2D claim (§4.1).
- **C-vocab-matched:** hold total symbol vocabulary equal across 1D and 2D (4k≈128 ⇒ k=32) so no condition wins on vocabulary budget alone.

### H6 — 2D structure transfers to ViTs (existence)
- **Claim:** Warm-up on 2D Dyck pictures (primary family DN_k or DQ_k, k=32) improves downstream ViT accuracy over random init, by a margin comparable to 1D k-Dyck.
- **Conditions:** {random init; 1D k-Dyck (C-parity); 2D Dyck primary; 2D Dyck + C-shuffle}.
- **Objective:** masked-state prediction of picture cells (§4.3).
- **Decision threshold:** success if CIFAR-100 ≥ ~71.5 (within noise of the 71.98 anchor) and C-shuffle ≤ ~69 (near random). Partial success if > ~69.5 but < k-Dyck.
- **Predicted outcomes:** *Properties hypothesis* → 2D ≈ or > 1D, C-shuffle collapses. *Null/CA-style* → 2D ≈ random and/or C-shuffle ≈ 2D (model ignored 2D structure).

### H7 — Determinacy-gradient sweet spot (the ladder)
- **Claim:** Downstream accuracy is non-monotonic across DW_k ⊊ DN_k ⊊ DQ_k ⊊ DC_k, peaking at an intermediate family; the most rigid (DW_k) underperforms (WW/CA-style), and the most entangled (DC_k) also underperforms (Dyck-Shuffle-style).
- **Conditions:** all four families at matched k and matched vocabulary; + C-shuffle for the two extremes.
- **Decision threshold:** "sweet spot confirmed" if an interior family beats both DW_k and DC_k by > 1 std — mirroring the parent paper's k-Dyck > {WW, Shuffle}.
- **Predicted outcomes:** *Constraint-distribution hypothesis* → inverted-U. *Pure-richness hypothesis* → monotonic increase toward DC_k. *Rigidity-fails hypothesis* → DW_k ≈ random, everything else > random.

### H8 — Native 2D vs. rasterized 1D (the key comparison)
- **Claim:** Feeding 2D Dyck pictures with their **native 2D token layout + sincos-2D positional encoding** outperforms feeding the **same pictures rasterized into 1D sequences** (row-major) with 1D positional encoding — i.e., the 2D structure, not merely the extra symbols, is what helps a ViT.
- **Conditions:** {native 2D + sincos-2D PE; rasterized 1D + 1D PE; rasterized 1D + sincos-2D PE (isolates PE from layout)}; all from the same DN_k/DQ_k data.
- **Decision threshold:** native 2D beats rasterized 1D by > 1 std on CIFAR-100 and holds on ≥1 larger dataset (Tiny-ImageNet).
- **Predicted outcomes:** *2D-inductive-match hypothesis* → native > rasterized. *"It's just vocabulary/constraint count" hypothesis* → native ≈ rasterized (which would demote 2D Dyck to "a fancier 1D Dyck" and argue for cheaper 1D variants).

### H9 — Vocabulary richness in 2D (interior optimum)
- **Claim:** Downstream accuracy is non-monotonic in k (⇒ 4k symbols), peaking near 4k ≈ 128 (k≈32), mirroring the parent paper's k=16<32<64>80 curve.
- **Conditions:** k ∈ {4, 8, 32, 64} (⇒ 16, 32, 128, 256 symbols), primary family fixed.
- **Decision threshold:** interior peak within the swept range; peak ≥ k-Dyck anchor.
- **Predicted outcomes:** interior optimum (supports richness-with-saturation) vs. monotonic (supports "more symbols always better," reweighting the k choice).

### H10 — Complementarity & scale (does it survive ImageNet?)
- **Claim:** 2D Dyck warm-up on ViT-B/16 improves ImageNet-1K top-1, and the gain persists through fine-tuning on the smaller datasets (additive setting), like the parent paper's +1.72%.
- **Conditions:** {random init + ImageNet; 1D k-Dyck + ImageNet (anchor); 2D Dyck + ImageNet}; then fine-tune on the 5 downstream sets.
- **Decision threshold:** ImageNet-1K ≥ +1.5% over random (comparable to k-Dyck's +1.72%) and positive persistence after fine-tuning (cf. parent Table 3: k-Dyck CIFAR-100 +1.66 after ImageNet).
- **Predicted outcomes:** persistence ⇒ 2D provides complementary signal (publishable); wash-out at scale ⇒ 2D is a small-data-only effect.

### Summary decision table
| Result pattern | Interpretation | Action |
|---|---|---|
| H6 pass + H8 native>raster + H7 inverted-U | 2D structure is a genuine, ViT-matched inductive bias | Scale to H10, write up |
| H6 pass but H8 native≈raster | Benefit is vocab/constraint, not 2D geometry | Pivot to cheap 1D multi-constraint grammars |
| H6 ≈ random, C-shuffle ≈ H6 | 2D infilling is a shortcut (CA redux) | Switch objective (§4.3) or abandon |
| H7 monotonic to DC_k | No sweet spot; entanglement helps in 2D | Re-examine parent-paper Shuffle result |

---

## Part 4 — Implementation plan, practical considerations, risks

### 4.1 Reproduce a paper number first (hard gate)
Before writing any 2D generator, the branched repo must **reproduce one existing parent-paper table result** — the natural choice is 1D k-Dyck → CIFAR-100 ≈ 71.98 (±0.74) on ViT-T/16. Only once the branch reproduces this within noise should new generator code be trusted; otherwise a null 2D result is uninterpretable (could be an infrastructure regression). This is condition **C-parity** and should be logged as experiment #1.

### 4.2 2D generator (feasibility)
- Implement under `generators/grammars_2d/` alongside existing generators, pip-based env, matching the existing generator interface (fixed output size N=H×W to fill the ViT token grid).
- **Most tractable construction: DC_k via row-column crossword.** Sampling a valid Dyck crossword is non-trivial because row and column constraints are coupled, but feasible strategies include: (a) start from a known-valid template (nested rectangles) and apply structure-preserving edits; (b) constraint-propagation / backtracking sampler over the 4k corner alphabet; (c) sample compatible row-Dyck and column-Dyck words via the paper's "alphabetic graph" of compatible couplings. Derive DQ_k (keep only length-4 circuits), DN_k (neutralizable), and DW_k (well-nested rectangles) by filtering/constraining the DC_k sampler — one code path yields the whole ladder.
- **Cost:** like 1D Dyck, generation is cheap and can run on the fly; no GPU needed for data. Validate every generated picture with a membership checker (rows/columns balanced; circuit structure correct) — cheap and essential to avoid training on malformed pictures; unit-test against the paper's figures (length-12 and length-36 circuits).

### 4.3 Objective: masked-state prediction in 2D + the forward-masking lesson
- **Default:** mask ~50% of *structurally-informative* cells (closing corners) and predict them, mirroring the parent's close-only masking. In 2D, "structurally informative" = corner symbols that close a row or column match.
- **Forward-masking lesson (from the CA post-mortem):** forward masking helped CA by importing a **non-local objective** — the very thing that saved CA is native to Dyck. The 2D analogue is a **directional/causal reveal** (e.g., predict the bottom-right corner given top-left context), forcing genuinely long-range 2D binding rather than local completion. Include a **masking-scheme ablation** {random-cell mask, close-only mask, directional/forward mask} because the CA experience shows the objective can matter as much as the data. Explicit tell: 2D Dyck should *not* need CA-style rescue tricks — if it only works under forward masking, that is evidence it sits closer to CA than to Dyck on the transferable-properties axis.
- **Non-determinism is a feature:** multiple valid completions per mask (like Dyck-Shuffle) → train with teacher-forcing on one valid completion; the model cannot reach 100% accuracy but is driven to track row/column stack state — the exact mechanism the parent paper relied on.

### 4.4 Mapping pictures to patch tokens
- **Native 2D (primary):** one Dyck-picture cell ↔ one ViT token position; H×W picture fills the token grid; sincos-2D PE. The cleanest test of the 2D hypothesis (H8).
- **Rasterized 1D (control):** row-major flatten to a sequence, 1D PE — literally the parent paper's setup with a 2D-derived vocabulary. The native-vs-rasterized gap is the scientific payload of H8.
- **Vocabulary:** 4k corner symbols vs. 1D k-Dyck's bracket vocab vs. the ≈130-way block-tokenization result — hold total symbols matched (k=32 ⇒ 128) so comparisons are not confounded by vocab budget (C-vocab-matched).

### 4.5 Lower-priority alternative: complexity-and-modality curriculum
The curriculum idea (Type-2 k-Dyck → Type-1 Dyck-Shuffle → abstract procedural images like FractalDB → real images) is reasonable but **lower priority**:
- The parent paper found **Dyck-Shuffle < Dyck** (70.11 < 71.98), so a curriculum moving *toward* more entanglement runs against the paper's own gradient result and would need justification.
- **FractalDB is the weakest link:** in the parent paper, FractalDB warm-up (sample-matched) *underperformed random init* on several sets — Table 3/Table 2 confirm CIFAR-10 88.98 (−2.31) and CIFAR-100 64.61 (−3.91), versus procedural warm-up's ImageNet-1K 79.21 (+1.72) and CIFAR-100 71.98 (+3.46). Bolting FractalDB into a curriculum risks importing a known-negative component.
- Curricula add many degrees of freedom (stage lengths, transitions) that make negative results hard to attribute — poor experimental hygiene relative to the clean 2D-Dyck ladder.
- **Verdict:** keep it as a single exploratory arm *after* H6–H8 resolve, and include only stages with positive evidence (k-Dyck, possibly 2D Dyck) rather than FractalDB. The 2D-Dyck ladder dominates it on both novelty and interpretability.

### 4.6 Risks and mitigations
| Risk | Likelihood | Mitigation |
|---|---|---|
| 2D generator produces malformed/edge-case pictures | Med | Membership checker on every sample; unit-test against paper's length-12/36 circuits |
| DC_k sampling too slow / distributionally biased | Med | Pre-generate a large validated pool; monitor symbol/structure marginals; template+edit sampling |
| 2D infilling becomes a shortcut (CA redux) | Med | C-shuffle control (must collapse); masking-scheme ablation; H8 native-vs-raster |
| Gains don't survive ImageNet-B scale | Med | H10 gate; if wash-out, report as a small-data method (still useful) |
| Branch fails to reproduce k-Dyck anchor | Low-Med | Reproduction gate (§4.1) before any 2D coding |
| Vocabulary confound inflates/deflates 2D | Med | C-vocab-matched (4k=128 ⇒ k=32) |
| Over-claiming vs. Zhang et al. edge-of-chaos | Low | Scope language fixed per §1.4 |

### 4.7 Suggested sequence
1. **Gate:** reproduce k-Dyck CIFAR-100 anchor on the branch (§4.1).
2. Build `generators/grammars_2d/` DC_k sampler + membership checker; derive DW_k/DN_k/DQ_k.
3. **H6** (existence + C-shuffle) on CIFAR-100 — go/no-go.
4. **H8** (native vs. rasterized) — the decisive scientific test.
5. **H7** (ladder) and **H9** (vocab) in parallel once H6/H8 pass.
6. **H10** (ImageNet-B + fine-tune persistence) only if small-scale passes.
7. Optional: single curriculum arm (§4.5), positive components only.

---

## Recommendations (staged, with thresholds that change them)
1. **Finalize the CA post-mortem as a bounded negative result now.** Publish it with symmetric seed statistics and an explicit scope statement (§1.4/§1.6). *Threshold to revisit:* if a re-run shows true-CA − shuffled-CA > 1 std, the "rule-agnostic learning" claim weakens and CA deserves another look.
2. **Do not abandon CA entirely — register the untested objective.** A CA warm-up on **next-state / time-evolution prediction** (à la Zhang et al. / the NCA LLM paper) is the one defensible CA variant. Keep it as a low-priority backlog item; *threshold to promote it:* if 2D Dyck (H6) fails the C-shuffle test, pivot budget here.
3. **Green-light 2D Dyck as the primary next direction**, executing §4.7 in order. *Go/no-go at H6:* CIFAR-100 ≥ ~71.5 with C-shuffle collapse. *Kill/pivot at H8:* if native ≈ rasterized, drop the 2D-geometry framing and test cheap 1D multi-constraint grammars instead.
4. **Lead with DN_k/DQ_k, keep DW_k as the rigidity control**, and build from the DC_k generator downward.
5. **Deprioritize the FractalDB curriculum**; if pursued later, drop the FractalDB stage given its negative parent-paper numbers.
6. **Two non-negotiable methodological invariants:** the reproduction gate (§4.1) and the shuffle-style control (C-shuffle). These are what made the CA verdict trustworthy and are what will make the 2D verdict trustworthy.

---

## Caveats
- I could not open the uploaded files; Part 1's characterization of the CA experiments relies on the task brief's description of `report.md` / `Current_Hypotheses_and_Ideas`. The uploaded reports are authoritative where they differ.
- Quoted paper numbers are the authors' own reported results (parent paper 2511.13945v2 tables; 2307.16522 theorems). The 2D Dyck paper contains **no ML experiments** — all ML predictions here are inferences.
- Recent companion preprints (2601.21725, 2603.10055) are not independently replicated; treat their magnitudes (e.g., "up to 6%," "1.6×") as indicative.
- The strongest single takeaway is methodological: the operator-shuffle-style control and the reproduction gate are what make both the CA verdict and the forthcoming 2D verdict credible — keep them central.