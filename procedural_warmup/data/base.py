"""Common interface for procedural data sources.

A *source* is the pair (dataset, masking strategy) the warm-up trainer consumes. Adding a
new source (a grammar, a cellular automaton, ...) means implementing these two tiny
contracts and registering a builder — never editing the trainer (see CLAUDE.md
"Extensible by default"). This is also the seam the Stage-2 curriculum builds on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset


class ProceduralDataset(Dataset):
    """Base class for procedural datasets.

    A sample is a 1-D ``LongTensor`` of shape ``(N,)`` holding token ids in ``[0, K)``,
    where ``N = H * W`` matches the ViT visual-token count. Subclasses generate data on
    the fly in ``__getitem__`` (generation is cheap, so ``__len__`` is just a virtual
    epoch length and the trainer cycles the loader by step count).
    """

    def __len__(self) -> int:  # pragma: no cover - trivial
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.LongTensor:  # pragma: no cover - interface
        raise NotImplementedError


@runtime_checkable
class MaskingStrategy(Protocol):
    """Maps a batch of token ids to a masked-prediction training triple.

    Returns ``(masked_input, targets, mask)`` where ``mask`` is a boolean tensor; the
    cross-entropy loss is taken only at ``mask == True`` positions. ``targets`` holds the
    original (unmasked) ids so the same tensor doubles as the supervision signal.
    """

    def __call__(
        self, batch_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...
