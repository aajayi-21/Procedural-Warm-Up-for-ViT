"""k-Dyck-Shuffle source: a context-sensitive language with crossing dependencies.

Relaxes k-Dyck by dropping the well-nesting constraint while still requiring every opening
bracket to be closed — closes may occur in any order, producing interleaved/crossing
structure. Shares the Dyck token layout and config, so the close-only masking is reused.
"""

from __future__ import annotations

from procedural_warmup.data import register_source
from procedural_warmup.data.dyck.masking import CloseOnlyMasking
from procedural_warmup.data.dyck_shuffle.dataset import DyckShuffleGrid


@register_source("dyck_shuffle")
def build_dyck_shuffle(cfg):
    return DyckShuffleGrid(cfg), CloseOnlyMasking(cfg)


__all__ = ["DyckShuffleGrid", "build_dyck_shuffle"]
