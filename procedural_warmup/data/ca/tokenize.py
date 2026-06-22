"""Map cellular-automaton cells to vocabulary token ids.

Ids 0/1 are reserved for PAD/MASK (``N_SPECIAL``), so cell-derived ids start at 2.

- ``binary``: one token per cell state -> vocabulary of ``N_SPECIAL + 2``. Faithful to the
  edge-of-chaos binary I/O; positional information comes from the frozen positional
  embedding rather than the token itself.
- ``block``: coarse-grain ``block_size`` consecutive cells (MSB-first) into a single symbol
  in ``[0, 2**block_size)`` -> vocabulary of ``N_SPECIAL + 2**block_size``. This exercises a
  larger vocabulary the way k=64 does for k-Dyck and is the CA analog of the paper's
  vocabulary-size sweep.
"""

from __future__ import annotations

import numpy as np

N_SPECIAL = 2  # PAD (0), MASK (1)


def binary_tokens(window: np.ndarray) -> np.ndarray:
    """``(H, W)`` cells in {0,1} -> ``(H, W)`` token ids in {2, 3}."""
    return window.astype(np.int64) + N_SPECIAL


def vocab_size_binary() -> int:
    return N_SPECIAL + 2


def block_tokens(window: np.ndarray, block_size: int) -> np.ndarray:
    """``(H, cell_W)`` cells -> ``(H, cell_W // block_size)`` token ids.

    ``cell_W`` must be a multiple of ``block_size``.
    """
    H, cell_W = window.shape
    if cell_W % block_size != 0:
        raise ValueError(f"cell width {cell_W} not divisible by block_size {block_size}")
    token_W = cell_W // block_size
    blocks = window.reshape(H, token_W, block_size).astype(np.int64)
    weights = (1 << np.arange(block_size - 1, -1, -1)).astype(np.int64)  # MSB-first
    ids = (blocks * weights).sum(axis=2)
    return ids + N_SPECIAL


def vocab_size_block(block_size: int) -> int:
    return N_SPECIAL + (1 << block_size)


def cells_per_token(mode: str, block_size: int) -> int:
    """How many CA cells map to one token column for the given tokenization mode."""
    if mode == "binary":
        return 1
    if mode == "block":
        return block_size
    raise ValueError(f"Unknown tokenize mode '{mode}'")


def required_vocab(mode: str, block_size: int) -> int:
    return vocab_size_binary() if mode == "binary" else vocab_size_block(block_size)
