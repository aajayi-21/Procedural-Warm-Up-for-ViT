"""Dataset wrapper for the k-Dyck-Shuffle generator (reuses the Dyck config block)."""

from __future__ import annotations

import torch

from procedural_warmup.data.base import ProceduralDataset
from procedural_warmup.data.dyck_shuffle.generator import dyck_shuffle_ids


class DyckShuffleGrid(ProceduralDataset):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.N = cfg.grid.H * cfg.grid.W
        needed = 2 + cfg.dyck.k_open + cfg.dyck.k_close
        assert cfg.vocab.K >= needed, f"vocab.K={cfg.vocab.K} < needed={needed}"

    def __len__(self) -> int:
        return self.cfg.dataset.n_samples

    def __getitem__(self, _idx: int) -> torch.LongTensor:
        seq = dyck_shuffle_ids(
            k_open=self.cfg.dyck.k_open,
            k_close=self.cfg.dyck.k_close,
            max_length=self.N,
            open_prob=self.cfg.dyck.open_prob,
        )
        seq = seq[: self.N]
        if len(seq) < self.N:
            seq += [self.cfg.vocab.PAD_ID] * (self.N - len(seq))
        return torch.tensor(seq, dtype=torch.long)
