"""2-D Conway's Game of Life (B3/S23) as a next-state-prediction warm-up source.

Outer-totalistic binary CA on a toroidal square lattice with the 8-cell Moore
neighborhood: a dead cell is born on exactly 3 live neighbors; a live cell survives on 2
or 3. Its native 2-D ``(y, x)`` grid maps directly onto ViT patch geometry.

**Task (why it is effective).** A naive "inpaint a single Life snapshot" objective is
degenerate — Life is mostly dead cells, so predicting "dead" everywhere already scores
~90%+. Instead each sample stacks two consecutive frames in the token grid: the top half is
state *t* (visible), the bottom half is state *t+gol_steps* (masked via ``forward`` masking).
Predicting the future frame requires applying the Life rule to every cell. Because the frame
is a torus and is fully visible, the target is **fully determined and noise-free**, so the
masked-token accuracy can climb from the dead-cell prior all the way to ~1.0 purely by
learning the rule — a clean, well-posed, genuinely structured objective.
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
    """Stacks ``state_t`` (top half) and ``state_{t+gol_steps}`` (bottom half) of a Life
    grid into one length-N token sequence for next-state prediction."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W
        if self.H % 2 != 0:
            raise ValueError("GoL next-state needs an even grid.H (two stacked frames)")
        self.fH, self.fW = self.H // 2, self.W  # per-frame token dimensions
        self.burn_in = cfg.ca.burn_in
        self.init_density = cfg.ca.init_density
        self.steps = max(1, cfg.ca.gol_steps)
        # Block tokenization (7 cells -> one of 128 symbols) collapses the trivial
        # "predict mostly-dead" floor (a 7-cell block is almost never all-dead), forcing
        # the model to predict the full next-state pattern. See ca/tokenize.py.
        self.tok_mode = cfg.ca.tokenize.mode
        self.block_size = cfg.ca.tokenize.block_size
        self.cell_W = self.fW * tok.cells_per_token(self.tok_mode, self.block_size)
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
        for _ in range(self.steps):
            t1 = life_step(t1)
        ids = np.concatenate([self._tokenize(t0), self._tokenize(t1)], axis=0)  # (H, W)
        return torch.tensor(ids.reshape(self.N), dtype=torch.long)
