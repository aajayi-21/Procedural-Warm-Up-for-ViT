"""Marginal-shuffle operator control for DW_k (H6 arm D / design-doc H9a).

Sample a valid DW_k picture, then permute cell contents uniformly at random across the
196 positions. Symbol marginals and picture size are preserved exactly; every matching
constraint is destroyed. This is the pair to the decisive CA ``iid-true``/``iid-shuffled``
contrast: if the treatment beats this control downstream, the 2D constraint structure —
not vocabulary or token statistics — carries the gain (design principle P4).

Both channels are permuted with **one shared permutation**, so the eligibility bit
travels with its symbol. Post-shuffle "eligibility" is a carried label (the structural
property it encoded is undefined once matching is destroyed), which gives the control
exact per-sample parity with the treatment in masked-cell count distribution and
masked-target id distribution — the minimal-difference control. The same
``CornerCloseOnlyMasking`` config is used: it masks d-symbols wherever they landed.
"""

from __future__ import annotations

import numpy as np
import torch

from procedural_warmup.data.dyck2d.dataset import Dyck2DGrid


def marginal_shuffle(sample: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Permute the (2, N) sample's positions with one shared random permutation."""
    perm = torch.from_numpy(rng.permutation(sample.shape[1]))
    return sample[:, perm]


class Dyck2DShuffleGrid(Dyck2DGrid):
    """DW_k pictures with per-sample content shuffling (matched-marginals control)."""

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        sample = super().__getitem__(_idx)
        return marginal_shuffle(sample, np.random.default_rng())
