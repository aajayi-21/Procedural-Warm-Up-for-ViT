"""Cellular-automata procedural sources.

Registers two sources behind the common interface:
- ``ca``  : 1-D elementary CA spacetime diagrams (the Stage-1 focus).
- ``gol`` : 2-D Game of Life snapshots (Stage-4 scaffold).
"""

from __future__ import annotations

from procedural_warmup.data import register_source
from procedural_warmup.data.ca.dataset import CASpacetimeGrid
from procedural_warmup.data.ca.gol import GameOfLifeGrid
from procedural_warmup.data.ca.masking import CAMasking


@register_source("ca")
def build_ca(cfg):
    """1-D elementary CA spacetime source."""
    return CASpacetimeGrid(cfg), CAMasking(cfg)


@register_source("gol")
def build_gol(cfg):
    """2-D Game of Life next-state source. Requires ``forward`` masking aligned to the
    two-frame split, i.e. ``masking.forward_rows == grid.H // 2``."""
    if cfg.masking.mode != "forward" or cfg.masking.forward_rows != cfg.grid.H // 2:
        raise ValueError(
            "GoL next-state requires masking.mode='forward' and "
            f"masking.forward_rows == grid.H//2 ({cfg.grid.H // 2}); got "
            f"mode={cfg.masking.mode!r}, forward_rows={cfg.masking.forward_rows}"
        )
    return GameOfLifeGrid(cfg), CAMasking(cfg)


__all__ = ["CASpacetimeGrid", "GameOfLifeGrid", "CAMasking", "build_ca", "build_gol"]
