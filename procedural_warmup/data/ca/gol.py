"""2-D Conway's Game of Life (B3/S23) — Stage-4 scaffold.

Outer-totalistic binary CA on a toroidal square lattice with the 8-cell Moore
neighborhood: a dead cell is born on exactly 3 live neighbors; a live cell survives on 2
or 3. Its native 2-D ``(y, x)`` grid maps directly onto ViT patch geometry.

This scaffold exposes Game of Life behind the same registry/dataset interface as the ECA
source: a sample is a random post-burn-in life configuration, tokenized and masked exactly
like the ECA grids (spatial inpainting of a life state). The "predict state t+1 from t"
variant noted in docs/cellular-automata.md (Stage 4) is a future extension that would swap
in a next-state masking strategy; the simulator below already supports it.
"""

from __future__ import annotations

import numpy as np
import torch

from procedural_warmup.data.base import ProceduralDataset
from procedural_warmup.data.ca import tokenize as tok


def life_step(grid: np.ndarray) -> np.ndarray:
    """Advance one Game-of-Life step on a toroidal grid (B3/S23)."""
    neighbors = sum(
        np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    )
    born = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return (born | survive).astype(np.uint8)


def simulate_life(
    height: int,
    width: int,
    burn_in: int,
    init_density: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a life configuration after ``burn_in`` steps from a random start."""
    if rng is None:
        rng = np.random.default_rng()
    grid = (rng.random((height, width)) < init_density).astype(np.uint8)
    for _ in range(burn_in):
        grid = life_step(grid)
    return grid


class GameOfLifeGrid(ProceduralDataset):
    """Random Game-of-Life snapshots tokenized to a length-N grid (binary tokens)."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W
        self.burn_in = cfg.ca.burn_in
        self.init_density = cfg.ca.init_density
        # Simulate on a larger torus, then crop, so the window has off-grid neighbors.
        self.sim_H = max(cfg.ca.sim_width, self.H)
        self.sim_W = max(cfg.ca.sim_width, self.W)
        assert cfg.vocab.K >= tok.vocab_size_binary()

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        rng = np.random.default_rng()
        grid = simulate_life(self.sim_H, self.sim_W, self.burn_in, self.init_density, rng)
        y = int(rng.integers(0, self.sim_H - self.H + 1))
        x = int(rng.integers(0, self.sim_W - self.W + 1))
        window = grid[y : y + self.H, x : x + self.W]
        ids = tok.binary_tokens(window)
        return torch.tensor(ids.reshape(self.N), dtype=torch.long)
