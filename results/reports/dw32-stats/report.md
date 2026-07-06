# DW_32 sampler gate (20000 samples)

_Generated 2026-07-06T00:14:25Z_

## Gates

- membership (Tier 1 + Tier 2): **PASS** over 20000 samples
- corner-mask determinacy: 200/200 audits fully determined
- throughput: 12262 samples/s/core (dataset path 11104/s) — on-the-fly OK

## Structure

- mean eligible d-cells/picture: 34.5 (zero-eligible: 0.03%)
- supervision density: 17.3 masked targets/sample = 8.8% of tokens (1D close-only anchor ~25% — report beside any A-vs-B verdict; the dw32-dense arm at mask_ratio 1.0 probes this axis)
- adjacent-partner fraction among eligible d's: 50.0% (min-span-1 border pairs — index leaks locally; read the H7 masked-accuracy-vs-distance diagnostic before crediting long-range binding)
- 2x2-rectangle fraction: 29.58%
- generator actions: {'h_splits': 28658, 'accretions': 85929, 'v_splits': 28724}

Known support bias: binary guillotine splits realize only the guillotine subset of the Simplot closure; non-guillotine tessellations are in DW_k but never sampled (see generator docstring).

## Figures

![dw32-stats_structure](..\..\figures\dw32-stats_structure.png)
