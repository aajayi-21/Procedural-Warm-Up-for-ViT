"""Well-nested 2D Dyck (DW_k) procedural sources — the H6 program's treatment arm.

Registers two sources behind the common interface:
- ``dyck2d``         : DW_k pictures + corner-close-only masking (H6 arm A).
- ``dyck2d_shuffle`` : marginal-shuffle operator control, same masking (H6 arm D / H9a).

See ``docs/2d-dyck-experiment-design.md`` and the 2D Dyck paper (``docs/2307.16522.pdf``).
"""

from __future__ import annotations

from procedural_warmup.data import register_source
from procedural_warmup.data.dyck2d.controls import Dyck2DShuffleGrid, marginal_shuffle
from procedural_warmup.data.dyck2d.dataset import Dyck2DGrid
from procedural_warmup.data.dyck2d.masking import CornerCloseOnlyMasking


@register_source("dyck2d")
def build_dyck2d(cfg):
    """DW_k well-nested 2D Dyck pictures + corner-close-only masking."""
    return Dyck2DGrid(cfg), CornerCloseOnlyMasking(cfg)


@register_source("dyck2d_shuffle")
def build_dyck2d_shuffle(cfg):
    """Marginal-shuffle control: same marginals/masking, matching structure destroyed."""
    return Dyck2DShuffleGrid(cfg), CornerCloseOnlyMasking(cfg)


__all__ = [
    "Dyck2DGrid",
    "Dyck2DShuffleGrid",
    "CornerCloseOnlyMasking",
    "marginal_shuffle",
    "build_dyck2d",
    "build_dyck2d_shuffle",
]
