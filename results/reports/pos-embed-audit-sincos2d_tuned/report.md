# Positional-embedding audit: sincos2d_tuned (14x14, d=192)

_Generated 2026-07-06T00:14:05Z_

## Verdict

- per-position norm: 0.0200 (spread 2.17e-09 — uniform rescale confirmed)
- min pairwise distance: 1.55e-02 (all 196 positions distinct)
- nearest embedding neighbor is a grid neighbor: 100.0% of interior cells
- effective frequencies/axis: 48/48
- far-pair (distance >= 8) cosine floor: -0.049
- linear probe (row, col) max error: 6.75e-14 cells
- one-grid-step contrast / token-embedding norm: 0.776

## Figures

![pos_embed_audit_sincos2d_tuned](..\..\figures\pos_embed_audit_sincos2d_tuned.png)
