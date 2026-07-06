# Positional-embedding audit: sincos2d (14x14, d=192)

_Generated 2026-07-05T23:27:39Z_

## Verdict

- per-position norm: 0.0200 (spread 6.29e-10 — uniform rescale confirmed)
- min pairwise distance: 3.53e-03 (all 196 positions distinct)
- nearest embedding neighbor is a grid neighbor: 100.0% of interior cells
- effective frequencies/axis: 15/48
- far-pair (distance >= 8) cosine floor: 0.748
- linear probe (row, col) max error: 1.20e-12 cells
- one-grid-step contrast / token-embedding norm: 0.176

## Figures

![pos_embed_audit_sincos2d](..\..\figures\pos_embed_audit_sincos2d.png)
