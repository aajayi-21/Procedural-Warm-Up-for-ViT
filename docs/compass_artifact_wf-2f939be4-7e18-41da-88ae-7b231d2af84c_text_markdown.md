# Critical Analysis of Four Research Hypotheses Extending "Procedural Warm-Up for Vision Transformers"

## TL;DR
- **Hypothesis 4 (2D spatial/logic pretraining) is the most promising direction** — it is genuinely novel (no one has pretrained a ViT on 2D cellular automata or grid-reasoning tasks and shown transfer to natural-image classification), and it is grounded by direct evidence that vanilla ViTs are 2D-representation-deficient. **Hypothesis 1 (Type-0 grammar) is the weakest** — it is theoretically incoherent (Type-0 languages are undecidable/unsamplable in a controlled way) and is directly contradicted by your own non-monotonic result that context-free beats context-sensitive.
- **Hypothesis 2 (Chomsky-hierarchy curriculum) is viable but risky**, threatened by catastrophic forgetting/overtraining and by your own finding that WW (regular) is useless-to-harmful; the most defensible version is a narrow easy→hard curriculum that excludes WW. **Hypothesis 3 (grammar→abstract-images→real-images) is viable but incremental** and risks redundancy, since your paper shows symbolic procedural data already beats and is complementary to FractalDB.
- Ranking by promise: **H4 > H2 ≈ H3 > H1.** All four should be reframed around your paper's true causal claim — that gains come from *learnable structural dependencies concentrated in late layers*, not from Chomsky-rank per se.

## Key Findings

**Your paper's own evidence is the binding constraint on all four hypotheses.** The central empirical facts are: (a) gains are non-monotonic in Chomsky complexity — context-free k-Dyck > context-sensitive Dyck-Shuffle > regular WW, with WW actively hurting CIFAR-100 (66.44 vs 68.52 baseline); (b) benefits come from structural *order*, not token distribution; (c) information is distributed across both attention and MLP and concentrated in *late* layers; (d) downstream accuracy peaks at *intermediate* warm-up length and intermediate vocabulary (k=64); (e) your method already beats FractalDB and is complementary to it. Any extension must be argued against these facts, not against a generic "more complexity = more transfer" intuition that your data refutes.

**A critical cross-modal tension exists between your vision result and the LLM literature.** In language, Hu, Petty, Shi, Merrill & Linzen ("Between Circuits and Chomsky," ACL 2025, aclanthology 2025.acl-long.478) found that context-*sensitive* k-Shuffle Dyck transfers *best* to natural language, while copy-language ww "is unhelpful at all durations." Your vision paper found the opposite ordering at the top (context-free k-Dyck best, context-sensitive Dyck-Shuffle worse) but the *same* result at the bottom (WW/ww useless). This convergence on "the right *intermediate* learnable structure wins, and pure repetition fails" — rather than "higher Chomsky rank wins" — is the key organizing principle for your follow-up work.

**The unifying explanation is learnability, not hierarchy position.** Hu et al.'s thesis is that effective transfer requires a language that both captures hierarchical/crossing dependencies *and* remains within the computational limits of the architecture (their best language, k-Shuffle Dyck, is the one definable in C-RASP); they report that for a 1B-parameter model trained on ~1.6B tokens of natural language, "pre-pretraining achieves the same loss and better linguistic generalization with a 33% smaller token budget." Deletang et al. ("Neural Networks and the Chomsky Hierarchy," ICLR 2023) showed transformers fail to generalize on many non-regular tasks at all. This "learnability sweet spot" framing — also seen in "Intelligence at the Edge of Chaos" (Zhang et al., ICLR 2025), where "Models trained on Class IV ECA rules, which exhibit structured yet complex behaviors, perform optimally" while "both uniform and periodic systems, and often also highly chaotic systems, resulted in poorer downstream performance, highlighting a sweet spot of complexity conducive to intelligence" — is the correct lens, and it predicts your non-monotonicity directly.

## Details

### Hypothesis 1 — Unrestricted / recursively enumerable (Type-0) grammar will increase performance

**Theoretical viability: Very low. This hypothesis is close to incoherent as stated.** Three independent problems:

1. **Sampling/decidability is fatal.** Type-0 (recursively enumerable) languages are exactly those recognized by a Turing machine that may *loop forever* on non-members; membership is only semi-decidable and the halting problem is embedded in them. There is no general procedure to generate well-formed Type-0 strings in a controlled, terminating way, and crucially no way to build the *masked-prediction supervision signal* your method relies on (your method needs a defined "valid completion" to mask and predict). Your current grammars work precisely because Dyck/Dyck-Shuffle have tractable generators and definable completions. A genuine Type-0 grammar does not. Any "Type-0" data one could actually generate would in practice be a *decidable* subset — i.e., not really Type-0, just a more complex context-sensitive or recursive language.

2. **Your own data refutes the premise.** The hypothesis assumes performance increases with Chomsky rank. But your results are *non-monotonic and peak at context-free*: k-Dyck (CF) > Dyck-Shuffle (CS) > WW (regular). Moving *up* from context-free to context-sensitive already *hurt* you. Extrapolating further up to Type-0 has no empirical support and contradicts the trend. The mechanism is learnability: Deletang et al. show transformers cannot even generalize context-sensitive tasks reliably; an undecidable target would be unlearnable, producing noise rather than structure.

3. **It contradicts the "intermediate complexity" sweet spot** seen in your warm-up-length and vocabulary-size ablations, in "Intelligence at the Edge of Chaos," and in the LLM curriculum literature.

**Novelty:** The *idea* is novel only because no one has done it — but that is because it is ill-posed, not because it is unexplored. The coherent neighbor is Peter Bloem's "Universal pre-training by iterated random computation" (Vrije Universiteit Amsterdam, arXiv:2506.20057, dated August 18 2025), which pretrains on the output of *random Turing-complete computations* justified "from the perspective of algorithmic complexity, building on recent research that shows that sequence models can be trained to approximate Solomonoff induction," and shows that "finetuning a model after pre-training offers faster convergence and better generalization." This is arguably the *coherent* version of "Type-0-like" pretraining: instead of sampling a specific undecidable language, you sample over a distribution of programs.

**Verdict: Not viable as literally stated; do not pursue "Type-0 grammar." Reframe as "algorithmic-complexity-maximal but learnable" data** — e.g., Bloem-style random-computation outputs, or richer recursive (Type-1/decidable) grammars with controllable complexity — and test whether they extend the *intermediate* sweet spot rather than exceed it. Predicted outcome: flat-to-negative beyond the k-Dyck optimum.

### Hypothesis 2 — Curriculum moving up/down the Chomsky hierarchy

**Theoretical viability: Moderate, with serious confounds.** This is the most "researchy" of the four because the direction question (easy→hard vs hard→easy) is genuinely open and testable.

Arguments *for*: Curriculum learning by complexity has precedent and the easy→hard direction is the classical default (Bengio et al. 2009). The LLM curriculum literature shows staged easy→hard training can reallocate specialized attention heads into deeper layers — which resonates with your late-layer finding. Papadimitriou & Jurafsky ("Injecting structural hints," Findings of EMNLP 2023, pp. 8402–8413) found that "non-context-free relationships form the best inductive biases" — and that mixing crossing (non-context-free) structure into nested data significantly improved downstream transfer across English, Japanese, and Basque — evidence that combining structure types helps. (Verify the precise mix-in fraction against their figures before citing a specific percentage.)

Arguments *against* (strong):
- **The WW problem.** Your data shows regular WW is useless-to-harmful. A curriculum that includes WW as its "easy" stage may waste budget or actively damage initialization. Hu et al. independently confirm ww "is unhelpful at all durations." So a hierarchy-spanning curriculum that *starts at Type-3* is starting with a poison stage.
- **Catastrophic forgetting / interference.** Multi-stage sequential training without replay causes the later stage to overwrite earlier representations (well-documented in sequential fine-tuning). If k-Dyck (your best single grammar) is stage 1 and a worse grammar is stage 2, stage 2 may erase the very structure that helped.
- **Catastrophic overtraining.** Springer et al. (2025, "Overtrained Language Models Are Harder to Fine-Tune," ICML 2025) show extended pretraining increases parameter sensitivity and *degrades* downstream adaptability — directly mirroring your "intermediate warm-up length is best" finding. A multi-stage curriculum lengthens total warm-up, pushing you past the sweet spot.

**Direction prediction:** Easy→hard (regular→CF→CS), *excluding* WW, ending on your best single grammar (k-Dyck), is most defensible. Hard→easy risks ending on a low-value grammar. But the cleanest scientific framing is *not* "traverse the hierarchy" but "does any ordering beat the best single grammar (k-Dyck) at matched total budget?"

**Novelty:** Curriculum-by-formal-complexity *for vision pretraining* is novel. Curriculum learning in vision exists but has barely been applied to ViTs, and never (per my search) by ordering *non-visual formal-grammar* pretraining stages by Chomsky rank. Moderately novel.

**Experimental design:**
- **Baselines (critical):** (i) best single grammar (k-Dyck) at *total* matched token budget — this is the baseline to beat; (ii) random-order mixture of all grammars (interleaved, not staged) to separate "ordering" from "diversity"; (iii) each single grammar alone.
- **Conditions:** easy→hard (excl. WW), hard→easy, easy→hard (incl. WW), and a "mix-in" condition à la Papadimitriou & Jurafsky (k-Dyck with a small fraction of Dyck-Shuffle).
- **Controls:** hold total warm-up tokens fixed (guards against catastrophic overtraining as a confound); use identical LR schedule; consider LR re-warming between stages and small replay buffers to isolate forgetting.
- **Metrics:** downstream top-1 on ImageNet-1k + the smaller datasets; convergence speed; and your layerwise/component probes to test whether curricula push useful information even later/deeper.
- **Ablation:** measure forgetting explicitly (probe stage-1 grammar accuracy after stage 2).

**Verdict: Viable and worth one well-controlled study, but likely to yield a null-or-modest result** unless ordering beats the single-grammar baseline at matched budget. The highest-value version is the "mix-in" framing, not the full hierarchy traversal.

### Hypothesis 3 — Three-stage: formal grammar → abstract images (FractalDB) → real images

**Theoretical viability: Moderate. The composition logic is the crux.** Your paper makes two facts that pull in opposite directions: (a) you frame symbolic procedural data as providing a *complementary* signal to visual data, which predicts additive stacking *should* help; but (b) you show your method *outperforms* FractalDB head-to-head and FractalDB actively *hurts* several datasets (CIFAR-10 −2.31, CIFAR-100 −3.91 in your Table 2). If FractalDB is a *worse* warm-up than yours, inserting it as a *later* stage (closer to the real-image task, hence more influential on final weights) risks overwriting your symbolic gains with a lower-quality signal — a catastrophic-forgetting failure mode.

The ordering matters enormously and the hypothesis under-specifies it. The "complementary signals compose" intuition is plausible but unproven; the more likely outcome given your FractalDB numbers is *interference* or *redundancy* (FractalDB and your method may teach overlapping 2D/structural priors, so stacking yields diminishing returns).

**Novelty:** Three-stage *pre-pre-training → pre-training → fine-tuning* pipelines exist conceptually (the Han et al. neural-cellular-automata LLM work explicitly adopts a three-stage paradigm; your own LLM paper takes "steps toward combining multiple types of procedural data"). But the specific *symbolic→procedural-visual→real* vision stack is novel. Moderately novel, somewhat incremental (it combines two known-good ingredients rather than introducing a new mechanism).

**Experimental design:**
- **Baselines:** (i) your 2-stage (grammar→real); (ii) FractalDB→real (2-stage); (iii) grammar+FractalDB *interleaved* then real; (iv) both orderings of the 3-stage (grammar→FractalDB→real vs FractalDB→grammar→real).
- **Key metric:** does 3-stage beat the *better* of the two 2-stage pipelines (yours), at matched total compute? Anything less is not "most performance."
- **Confounds:** total compute/data budget must be matched; the third real-image stage dominates final weights, so report layerwise probes to see which stage's structure survives.
- **Diagnostic:** representational-similarity (CKA) between stages to test redundancy vs complementarity directly.

**Verdict: Viable but the weakest *novelty-to-effort* ratio. Likely modest or null gains** given that FractalDB underperforms your method. Pursue only if reframed as a *scientific* question ("do symbolic and visual procedural priors compose, or are they redundant?") rather than an engineering claim of "most performance." Use representation-similarity analysis as the core deliverable.

### Hypothesis 4 — Pretraining on 2D logic / 2D spatial-reasoning problems

**Theoretical viability: Highest of the four — but it cuts against your paper's headline thesis, which is a feature for novelty and a tension to address.** Your paper deliberately used data with "no 2D structure or explicit correspondence with image properties" and argued reasoning-over-images is "primarily a reasoning problem, not an image problem." H4 proposes the opposite: that *matching* the 2D token-grid geometry of the ViT will transfer better. There is strong independent evidence for the 2D-matters view:

- **ViTARC (Li, Xu, Khalil & Sanner, TMLR 05/2025, arXiv:2410.06405)** shows a vanilla ViT "fails dramatically on most ARC tasks even when trained on one million examples per task," reflecting "an inherent representational deficiency" — it "cannot accurately model spatial relationships between the objects in an ARC grid and the grid boundaries" (overall test accuracy ~18%). Adding explicit 2D positional/object encodings yields "a total improvement of 57.36% over the baseline ViT-Vanilla" and "a test solve rate close to 100% on more than half of the 400 public ARC tasks." This is direct evidence that ViTs have a 2D-spatial *deficiency* that targeted pretraining could plausibly address — and that 1D positional schemes (which your current method freezes) are precisely the weak point.
- **"Intelligence at the Edge of Chaos"** shows intermediate-complexity (Class IV) cellular automata produce the best downstream reasoning — and 2D Life-like CA are a natural 2D analogue of your 1D Dyck.
- The 2D structure could let you *unfreeze and train the positional encoding* meaningfully during warm-up (your current method freezes it because 1D sequences have no meaningful 2D positions) — potentially fixing the exact deficiency ViTARC identifies.

**The tension to confront head-on:** If H4 works *better* than your 1D method, it partially undercuts your "it's a reasoning problem, not an image problem" claim — suggesting modality-matched geometry matters after all. If it works *worse* or equal, it *strengthens* your thesis. Either result is publishable and scientifically informative. Frame H4 explicitly as a test of your own central claim.

**Novelty: High and clean.** Per targeted search, *no one* has pretrained a ViT on 2D cellular automata, mazes, sudoku, or ARC-style grid transforms and shown transfer to natural-image classification. "Intelligence at the Edge of Chaos" used 1D elementary CA on GPT-2 LLMs (downstream = reasoning + chess move prediction), not ViTs for image classification. ViTARC does task-specific ARC supervision, not classification transfer. FractalDB and "Scaling Backwards" use 2D *images* but not 2D *reasoning tasks*, and run no controlled 2D-spatial-vs-1D-sequence comparison. This is the clearest open gap of the four.

**Experimental design:**
- **The decisive experiment:** hold the data-generating *grammar/rule* fixed where possible and vary only 1D-sequence vs 2D-grid arrangement of the same tokens, to isolate "2D geometry" from "task content." E.g., render Dyck-like or CA dynamics in a 2D grid vs flattened 1D.
- **Candidate 2D tasks:** 2D Life-like cellular automata (Game of Life) next-state prediction; ARC-style grid transformations; maze/connectivity; sudoku constraint completion.
- **Baselines:** your 1D k-Dyck method (the incumbent to beat); FractalDB; random init; Mimetic init.
- **Key design choice:** test both *frozen* (as in your current method) and *trainable* 2D positional encodings — the 2D case is where unfreezing positions should help, per ViTARC.
- **Metrics:** downstream top-1; convergence; and especially *layerwise* probes — does 2D pretraining shift useful information to *early* layers (matching standard visual pretraining) rather than your observed *late* layers? That would be a striking, thesis-relevant result.
- **Confounds:** match token budget and intermediate-complexity (per "edge of chaos," pick CA rules of intermediate/Class-IV complexity, not chaotic ones); control for the masked-prediction objective.

**Verdict: Pursue this first. Highest novelty, strongest theoretical motivation, and it doubles as a direct test of your paper's core claim.** The main risk is that 2D tasks are harder to control for complexity and learnability than formal grammars.

## Recommendations

**Stage 1 (do now): Pursue Hypothesis 4 as the flagship follow-up.** Run the decisive 1D-vs-2D controlled experiment on the same token content, plus 2D cellular-automata warm-up at intermediate complexity, with both frozen and trainable positional encodings. *Benchmark that changes the plan:* if 2D matches but does not beat your 1D k-Dyck at matched budget, pivot to framing it as confirmation of your "reasoning not image" thesis; if it clearly beats 1D, that is a new headline result and you should scale to ImageNet-1k and add layerwise probes showing whether the early-vs-late layer story flips.

**Stage 2 (parallel, lower cost): Run Hypothesis 2 in its "mix-in" form, not full hierarchy traversal.** Test k-Dyck with a small fraction of Dyck-Shuffle mixed in (à la Papadimitriou & Jurafsky) and an easy→hard curriculum *excluding WW*, always against the single-grammar k-Dyck baseline at matched total budget, with explicit forgetting probes. *Threshold:* if no ordering beats single-grammar k-Dyck at matched budget, report the null and stop — do not over-invest.

**Stage 3 (only if Stage 1/2 leave budget): Hypothesis 3 as a redundancy study.** Run all orderings of grammar/FractalDB/real with CKA representational-similarity analysis. *Threshold:* proceed to a full engineering pipeline only if 3-stage beats your 2-stage at matched compute AND CKA shows the signals are complementary rather than redundant.

**Do not pursue Hypothesis 1 as stated.** If you want the "maximal complexity" flavor, implement it as Bloem-style random-computation pretraining (a coherent, samplable surrogate for algorithmic-complexity-maximal data) and test whether it extends or merely matches the k-Dyck sweet spot. *Threshold to revisit:* only if a decidable, controllably-complex recursive grammar can be built with a well-defined masking signal.

**Cross-cutting methodological musts:** (1) Always match *total* warm-up budget to control for catastrophic overtraining, which your own intermediate-length finding predicts. (2) Always include the single-best-grammar (k-Dyck) baseline — many of these hypotheses must beat it, not just beat random init. (3) Report layerwise/component probes for every variant; the early-vs-late-layer signature is your most distinctive diagnostic and the most likely place to find a publishable mechanistic story. (4) Separate "diversity/mixture" from "ordering/curriculum" with an interleaved-mixture control.

## Caveats
- Your paper is a very recent preprint (arXiv:2511.13945, v2 dated March 2026) and not yet peer-reviewed; treat its quantitative claims (e.g., +1.7% ImageNet, 28%-data-equivalence) as preprint-stage when building on them.
- The cross-modal comparison to Hu et al. is suggestive, not dispositive: their best language (k-Shuffle Dyck, context-sensitive) differs from yours (k-Dyck, context-free), so vision and language may genuinely differ in *which* intermediate structure is optimal. Do not assume LLM curriculum findings transfer directly to ViTs.
- "Intelligence at the Edge of Chaos" and the neural-cellular-automata transfer results are LLM/sequence results; their relevance to ViT image classification is an inference, not an established fact — which is exactly why H4 needs to be tested rather than assumed.
- Catastrophic-forgetting and catastrophic-overtraining evidence comes largely from the LLM literature; the magnitude of these effects in short ViT warm-ups is unknown and should be measured, not assumed.
- Several "novelty" claims rest on absence-of-evidence from literature search; a determined search may surface niche workshop papers, particularly for jigsaw/rotation self-supervised pretext tasks adjacent to H4 (e.g., Jigsaw-ViT), so confirm before claiming strict first-of-kind status.