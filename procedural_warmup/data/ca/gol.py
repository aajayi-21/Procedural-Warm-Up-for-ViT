"""2-D Conway's Game of Life (B3/S23) warm-up source — next-state prediction.

Outer-totalistic binary CA on a toroidal square lattice (8-cell Moore neighborhood): a dead
cell is born on exactly 3 live neighbors; a live cell survives on 2 or 3.

**Task.** Each sample stacks two consecutive frames in the token grid — state *t* (top half)
and state *t+gol_steps* (bottom half) — block-tokenized. With ``forward`` masking the model
predicts the future frame from the past one, forcing it to apply the Life rule; on a torus
the target is fully determined by the visible frame (clean, noise-free). Block tokenization
keeps the target non-trivial (a binary frame is mostly dead -> degenerate).

**Status — not yet verified effective.** Across CPU learning-curve probes this GoL task (and
every variant tried: binary/block tokenization, forward vs random masking, frozen vs
learnable positions, single-frame inpainting) did NOT learn within the step budget where the
1-D ECA-block and k-Dyck sources clearly do. The likely cause: the rule-bearing dependency is
*cross-frame* and 2-D, which the 1-D-token + random-positional ViT does not crack quickly,
whereas the ECA spacetime's dependency is local and 1-D. This source is provided as a
Stage-4 research scaffold — validate on the full GPU run or redesign (e.g. a 2-D positional
encoding / larger token budget). The verified-effective source is ECA Rule-110 ``block``.
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


def simulate_life(height: int, width: int, burn_in: int, init_density: float,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a life configuration after ``burn_in`` steps from a random start."""
    if rng is None:
        rng = np.random.default_rng()
    grid = (rng.random((height, width)) < init_density).astype(np.uint8)
    for _ in range(burn_in):
        grid = life_step(grid)
    return grid


class GameOfLifeGrid(ProceduralDataset):
    """Stacks state t (top half) and state t+gol_steps (bottom half) for next-state prediction."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W
        if self.H % 2 != 0:
            raise ValueError("GoL needs an even grid.H (two stacked frames)")
        self.fH = self.H // 2  # per-frame token rows
        self.burn_in = cfg.ca.burn_in
        self.init_density = cfg.ca.init_density
        self.gol_steps = max(1, cfg.ca.gol_steps)
        self.tok_mode = cfg.ca.tokenize.mode
        self.block_size = cfg.ca.tokenize.block_size
        self.cell_W = self.W * tok.cells_per_token(self.tok_mode, self.block_size)
        required_K = tok.required_vocab(self.tok_mode, self.block_size)
        assert cfg.vocab.K >= required_K, (
            f"vocab.K={cfg.vocab.K} < required {required_K} for tokenize '{self.tok_mode}'"
        )

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def _tokenize(self, frame: np.ndarray) -> np.ndarray:
        if self.tok_mode == "binary":
            return tok.binary_tokens(frame)
        return tok.block_tokens(frame, self.block_size)

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        rng = np.random.default_rng()
        t0 = simulate_life(self.fH, self.cell_W, self.burn_in, self.init_density, rng)
        t1 = t0.copy()
        for _ in range(self.gol_steps):
            t1 = life_step(t1)
        ids = np.concatenate([self._tokenize(t0), self._tokenize(t1)], axis=0)  # (H, W)
        return torch.tensor(ids.reshape(self.N), dtype=torch.long)
