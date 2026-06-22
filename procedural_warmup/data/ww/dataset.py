"""Dataset wrapper for the WW generator."""

from __future__ import annotations

import torch

from procedural_warmup.data.base import ProceduralDataset
from procedural_warmup.data.ww.generator import ww_ids


class WWGrid(ProceduralDataset):
    """Yields length-N WW (copy-language) token sequences."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.N = cfg.grid.H * cfg.grid.W
        self.n_symbols = cfg.ww.n_symbols
        assert cfg.vocab.K >= 2 + self.n_symbols, (
            f"vocab.K={cfg.vocab.K} < required {2 + self.n_symbols} for ww.n_symbols"
        )

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        seq = ww_ids(self.n_symbols, self.N)
        return torch.tensor(seq, dtype=torch.long)
