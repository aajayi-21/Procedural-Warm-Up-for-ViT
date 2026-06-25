"""Spatial-Dyck: render the *same* latent k-Dyck tree into 1-D or 2-D token layouts.

This is the minimal-change spatial-hierarchy probe. We generate an ordinary k-Dyck string
with the validated generator (so the parse-tree / depth / vocabulary / target distributions
are *identical* to the 1-D baseline), then only change **where** its tokens sit in the
14x14 = 196 grid. Because every Dyck string is exactly N tokens and perfectly balanced, it
fills the grid completely — no filler or boundary tokens are invented.

Renderers (``cfg.spatial_dyck.mode``):

- ``1d``       : row-major identity — reproduces the k-Dyck baseline exactly (validation).
- ``nested``   : lay the sequence along a **Hilbert curve**. A subtree is a contiguous span,
                 and the curve maps contiguous spans to compact 2-D regions, so each subtree
                 occupies a compact region with its open/close at the region's two ends and
                 children nested inside — the candidate. A random D4 symmetry (one of 8
                 rotations/reflections) is applied per example so absolute position / corner
                 cannot be a type-specific shortcut, while the *relational* geometry (nested =>
                 nearby) stays stable.
- ``permuted`` : nested token multiset scattered to a fresh random coordinate permutation per
                 example — destroys stable 2-D organization while preserving vocabulary,
                 marginals, #open/#close and target count. The spatial analogue of shuffled
                 Dyck (the geometry-destroyed control).

Masking is the unchanged close-only objective (``CloseOnlyMasking``): it masks closing
brackets wherever they land, so predicting them still requires finding the hierarchical
partner — now by routing across the 2-D region instead of along the 1-D string.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from procedural_warmup.data.base import ProceduralDataset
from procedural_warmup.data.dyck.generator import dyck_ids


def _hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    """Map Hilbert distance ``d`` to ``(x, y)`` on an ``n x n`` grid (``n`` a power of two)."""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_order(side: int) -> list[tuple[int, int]]:
    """Return the ``side*side`` cells as ``(row, col)`` in Hilbert-curve order.

    For non-power-of-two ``side`` we walk the next power-of-two Hilbert curve and keep the
    in-bounds cells in order — locality is preserved across the crop.
    """
    p = 1
    while p < side:
        p *= 2
    order = []
    for d in range(p * p):
        x, y = _hilbert_d2xy(p, d)
        if x < side and y < side:
            order.append((y, x))
    return order


def _d4(r: int, c: int, s: int, k: int) -> tuple[int, int]:
    """One of the 8 square symmetries (k in 0..7) applied to ``(r, c)`` on an ``s x s`` grid."""
    if k & 1:
        r, c = c, r            # transpose
    if k & 2:
        c = s - 1 - c          # flip horizontal
    if k & 4:
        r = s - 1 - r          # flip vertical
    return r, c


class SpatialDyckGrid(ProceduralDataset):
    """Same k-Dyck tree, laid out 1-D / nested-2-D / permuted per ``cfg.spatial_dyck.mode``."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W
        self.mode = str(cfg.spatial_dyck.mode).lower()
        if self.mode not in ("1d", "nested", "permuted"):
            raise ValueError(f"spatial_dyck.mode must be 1d|nested|permuted, got {self.mode!r}")
        if self.H != self.W:
            raise ValueError("spatial_dyck assumes a square grid (H==W)")
        needed = 2 + cfg.dyck.k_open + cfg.dyck.k_close
        assert cfg.vocab.K >= needed, f"vocab.K={cfg.vocab.K} < needed={needed}"
        # Precompute the Hilbert order as flat row-major indices.
        self._curve = np.array([r * self.W + c for r, c in hilbert_order(self.H)], dtype=np.int64)
        self._rc = np.array(hilbert_order(self.H), dtype=np.int64)  # (N, 2) row,col

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def _generate(self) -> np.ndarray:
        seq = dyck_ids(
            k_open=self.cfg.dyck.k_open,
            k_close=self.cfg.dyck.k_close,
            max_length=self.N,
            open_prob=self.cfg.dyck.open_prob,
            min_pairs=self.cfg.dyck.min_pairs,
            rng=random.Random(),
        )
        return np.asarray(seq[: self.N], dtype=np.int64)

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        seq = self._generate()  # (N,) the latent tree as a 1-D Dyck string
        if self.mode == "1d":
            out = seq  # row-major identity == k-Dyck baseline
        elif self.mode == "permuted":
            out = np.empty(self.N, dtype=np.int64)
            out[np.random.permutation(self.N)] = seq
        else:  # nested: along the Hilbert curve, with a random D4 symmetry
            k = np.random.randint(8)
            rc = self._rc
            rr, cc = _d4(rc[:, 0], rc[:, 1], self.H, k)
            flat = rr * self.W + cc
            out = np.empty(self.N, dtype=np.int64)
            out[flat] = seq  # i-th tree token -> i-th (transformed) Hilbert cell
        return torch.tensor(out, dtype=torch.long)
