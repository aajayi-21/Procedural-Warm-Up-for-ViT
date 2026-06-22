"""Dataset that turns ECA spacetime diagrams into length-N token grids.

Each sample evolves a fresh random ECA on a torus *wider* than the token window, discards
a burn-in, then crops a random ``H x W_cell`` window (rows = time, columns = space) and
tokenizes it to an ``H x W`` token grid flattened row-major to length ``N = H*W``.

Simulating wider than the window (plus burn-in) means cells inside the window are
correctly influenced by history outside it, preserving genuine light-cone dependencies
rather than trivial boundary wrap-around.
"""

from __future__ import annotations

import numpy as np
import torch

from procedural_warmup.data.base import ProceduralDataset
from procedural_warmup.data.ca import tokenize as tok
from procedural_warmup.data.ca.eca import simulate_spacetime


class CASpacetimeGrid(ProceduralDataset):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.H, self.W = cfg.grid.H, cfg.grid.W
        self.N = self.H * self.W

        self.rule = cfg.ca.rule
        self.burn_in = cfg.ca.burn_in
        self.boundary = cfg.ca.boundary
        self.init_density = cfg.ca.init_density

        self.tok_mode = cfg.ca.tokenize.mode
        self.block_size = cfg.ca.tokenize.block_size
        self.cell_per_tok = tok.cells_per_token(self.tok_mode, self.block_size)
        self.cell_W = self.W * self.cell_per_tok  # CA cells needed per row before tokenizing

        # Simulate at least as wide as the window; wider gives real out-of-window history.
        self.sim_width = max(cfg.ca.sim_width, self.cell_W)

        required_K = tok.required_vocab(self.tok_mode, self.block_size)
        assert cfg.vocab.K >= required_K, (
            f"vocab.K={cfg.vocab.K} < required {required_K} for "
            f"tokenize mode '{self.tok_mode}'"
        )

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def _tokenize(self, window: np.ndarray) -> np.ndarray:
        if self.tok_mode == "binary":
            return tok.binary_tokens(window)
        return tok.block_tokens(window, self.block_size)

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        rng = np.random.default_rng()
        spacetime = simulate_spacetime(
            rule=self.rule,
            width=self.sim_width,
            n_rows=self.H,
            burn_in=self.burn_in,
            init_density=self.init_density,
            boundary=self.boundary,
            rng=rng,
        )
        max_off = self.sim_width - self.cell_W
        col = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
        window = spacetime[:, col : col + self.cell_W]  # (H, cell_W)
        ids = self._tokenize(window)  # (H, W)
        return torch.tensor(ids.reshape(self.N), dtype=torch.long)
