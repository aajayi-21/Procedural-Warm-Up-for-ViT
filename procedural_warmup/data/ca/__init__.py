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
    """2-D Game of Life next-state source. With ``forward`` masking, forward_rows must
    equal grid.H//2 so the masked region is exactly the future frame."""
    if cfg.masking.mode == "forward" and cfg.masking.forward_rows != cfg.grid.H // 2:
        raise ValueError(
            f"GoL forward masking needs masking.forward_rows == grid.H//2 "
            f"({cfg.grid.H // 2}); got {cfg.masking.forward_rows}"
        )
    return GameOfLifeGrid(cfg), CAMasking(cfg)


__all__ = ["CASpacetimeGrid", "GameOfLifeGrid", "CAMasking", "build_ca", "build_gol"]
