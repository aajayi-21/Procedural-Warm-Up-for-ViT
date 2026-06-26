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


class GolStepDataset(ProceduralDataset):
    """One Game-of-Life step on an i.i.d. 2-D board; yields ``[x | y]`` (length ``2N``).

    The genuinely-2-D analogue of :class:`~procedural_warmup.data.ca.iid_step.CaStepDataset`:
    predict the *entire next Life state* from the current state. The Moore-8 neighborhood is
    2-D-local, so the ``sincos2d`` frozen positional embedding exposes the grid adjacency the
    operator needs — a test of "does an entire-next-state objective transfer when the operator
    is genuinely 2-D AND its geometry is exposed" (failure Hypothesis 3).

    Caveat: the board is *toroidal* (``life_step`` wraps via ``np.roll``) but ``sincos2d`` is
    non-periodic, so the wrap-around neighbours of the ~27% border cells are NOT encoded as
    adjacent (a doubly-periodic code aliases badly on a 14-grid, unlike the 196-ring ``sincos1d``
    used for ``ca_step``, so it is not worth building). This is *conservative*: it makes the true
    operator slightly harder to learn at the border (risking a false negative), never a false
    positive; and it cancels in the true-minus-shuffled gap (both arms share the same code).

    ``y = life_step(x)`` for ``mode="true"``; ``y = life_step(z)`` for an unrelated board ``z``
    at the same density for ``mode="shuffled"`` (the operator-vs-marginal control: ``true``
    beating ``shuffled`` downstream is the operator-learning signal). Density is sampled per
    example from ``cfg.ca_step.densities`` so the target marginal is not a single fixed bias.
    Pairs with :class:`~procedural_warmup.data.ca.iid_step.TransductionMasking` (full mask).
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W
        self.densities = list(cfg.ca_step.densities)
        self.mode = str(cfg.ca_step.mode).lower()  # reuse ca_step.mode for the true|shuffled control
        if self.mode not in ("true", "shuffled"):
            raise ValueError(f"ca_step.mode must be 'true'|'shuffled', got {self.mode!r}")
        if cfg.vocab.K < tok.vocab_size_binary():
            raise ValueError(f"vocab.K={cfg.vocab.K} < {tok.vocab_size_binary()} for binary CA")

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        rng = np.random.default_rng()
        p = float(rng.choice(self.densities))
        x = (rng.random((self.H, self.W)) < p).astype(np.uint8)
        # "true": evolve the input itself; "shuffled": evolve an unrelated board at same density.
        src = x if self.mode == "true" else (rng.random((self.H, self.W)) < p).astype(np.uint8)
        y = life_step(src)
        pair = np.concatenate(
            [tok.binary_tokens(x).reshape(self.N), tok.binary_tokens(y).reshape(self.N)]
        )
        return torch.tensor(pair, dtype=torch.long)


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
