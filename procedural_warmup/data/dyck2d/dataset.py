"""Dataset wrapper for the DW_k sampler — the H6 treatment source.

A sample is a **two-channel** ``(2, N)`` LongTensor:

    channel 0: token ids of the 14x14 picture, flattened row-major
               (fixed layout: cell (r, c) -> token r*W + c, design principle P6 —
               matching ``Frozen2DSinCosPositionalEmbedding``'s ``p -> (p//W, p%W)``)
    channel 1: maskable-d eligibility (0/1) — 1 iff the cell is a d-corner whose
               rectangle passes the min-match-distance filter

Eligibility rides along from the generator's rectangle list for free, so the masking
strategy never has to re-parse pictures inside the training loop. The precedent for
richer-than-``(N,)`` samples consumed by a paired masking strategy is
``data/ca/iid_step.py`` (``(2N,)`` + ``TransductionMasking``).

Min-match-distance filter: a d-cell is eligible iff ``max(row_span, col_span) >=
cfg.dyck2d.min_match_distance``. Spans are odd, so the default ``2`` excludes exactly
the 2x2 rectangles — the fully-local quadruples whose three partners are all adjacent
(the corner-close analog of masking a trivially-close bracket).  Values <= 1 disable
the filter.

Residual caveat — MEASURED, not hypothetical (deferred to the H7 distance-vs-accuracy
diagnostic, tracked by ``stats.py``'s ``adjacent_partner_frac``): at defaults ~50% of
*eligible* d-cells still have ``min(row_span, col_span) == 1`` — border-word pairs
whose c- (or b-) partner is 4-adjacent along one axis, so their index leaks locally
even though the other partner is far.  For context the 1D anchor has a comparable
close-pair fraction at distance 1, so this is a shared property of both arms, but any
H6 verdict must read the masked-accuracy-vs-distance diagnostic before crediting
long-range binding.  The stricter ``min(row_span, col_span)`` filter is a one-line
change here.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from procedural_warmup.data.base import ProceduralDataset
from procedural_warmup.data.dyck2d.alphabet import vocab_size
from procedural_warmup.data.dyck2d.generator import dw_picture


class Dyck2DGrid(ProceduralDataset):
    """Yields (2, N) two-channel DW_k samples for an H x W token grid."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W
        self.k = cfg.dyck2d.k
        self.p_acc = cfg.dyck2d.p_acc
        self.open_prob = cfg.dyck2d.open_prob
        self.min_match_distance = cfg.dyck2d.min_match_distance
        self.filter_mode = cfg.dyck2d.filter_mode
        self.mask_roles = cfg.dyck2d.mask_roles
        if self.filter_mode not in ("max", "min"):
            raise ValueError(f"dyck2d.filter_mode must be 'max'|'min', got {self.filter_mode!r}")
        if self.mask_roles not in ("d", "cd"):
            raise ValueError(f"dyck2d.mask_roles must be 'd'|'cd', got {self.mask_roles!r}")
        needed = vocab_size(self.k)
        assert cfg.vocab.K >= needed, f"vocab.K={cfg.vocab.K} < needed={needed} for k={self.k}"
        assert self.H % 2 == 0 and self.W % 2 == 0, (
            f"DW pictures need even grid dims, got ({self.H}, {self.W})"
        )

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def _rect_eligible(self, rect) -> bool:
        agg = max if self.filter_mode == "max" else min
        return agg(rect.row_span, rect.col_span) >= self.min_match_distance

    def _sample(self, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
        """One (ids_grid, eligibility_grid) pair as numpy arrays."""
        grid, rects = dw_picture(
            self.H, self.W, self.k, p_acc=self.p_acc, open_prob=self.open_prob, rng=rng
        )
        elig = np.zeros((self.H, self.W), dtype=np.int64)
        for rect in rects:
            if self._rect_eligible(rect):
                elig[rect.d_pos] = 1
                if self.mask_roles == "cd":
                    elig[rect.r2, rect.c1] = 1  # the c (bottom-left) corner
        return grid, elig

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        grid, elig = self._sample()
        sample = np.stack([grid.reshape(-1), elig.reshape(-1)])  # (2, N), row-major
        return torch.from_numpy(sample)
