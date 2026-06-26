"""Cellular-automata procedural sources.

Registers two sources behind the common interface:
- ``ca``  : 1-D elementary CA spacetime diagrams (the Stage-1 focus).
- ``gol`` : 2-D Game of Life snapshots (Stage-4 scaffold).
"""

from __future__ import annotations

from procedural_warmup.data import register_source
from procedural_warmup.data.ca.dataset import CASpacetimeGrid
from procedural_warmup.data.ca.gol import GameOfLifeGrid, GolStepDataset
from procedural_warmup.data.ca.iid_step import (
    CaStepDataset,
    IidBoardDataset,
    TransductionMasking,
)
from procedural_warmup.data.ca.masking import CAMasking


@register_source("ca")
def build_ca(cfg):
    """1-D elementary CA spacetime source."""
    return CASpacetimeGrid(cfg), CAMasking(cfg)


@register_source("gol")
def build_gol(cfg):
    """2-D Game of Life source (Stage-4 scaffold)."""
    return GameOfLifeGrid(cfg), CAMasking(cfg)


@register_source("gol_step")
def build_gol_step(cfg):
    """2-D Game-of-Life next-state transduction (``ca_step.mode`` true|shuffled)."""
    return GolStepDataset(cfg), TransductionMasking(cfg)


@register_source("ca_step")
def build_ca_step(cfg):
    """IID single-step ECA transduction (operator vs texture; ``ca_step.mode`` true|shuffled)."""
    return CaStepDataset(cfg), TransductionMasking(cfg)


@register_source("iid_board")
def build_iid_board(cfg):
    """IID Bernoulli board + random masking — structure-free reconstruction floor."""
    return IidBoardDataset(cfg), CAMasking(cfg)


__all__ = [
    "CASpacetimeGrid",
    "GameOfLifeGrid",
    "GolStepDataset",
    "CAMasking",
    "CaStepDataset",
    "IidBoardDataset",
    "TransductionMasking",
    "build_ca",
    "build_gol",
    "build_gol_step",
    "build_ca_step",
    "build_iid_board",
]
