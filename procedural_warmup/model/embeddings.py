"""Frozen embeddings used during procedural warm-up.

Both the token lookup and the positional encoding are random/fixed and **frozen**: this
prevents the model from solving the masked-token objective via the embeddings, forcing the
attention and MLP layers to learn the data's structure instead. Both are discarded before
image training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenTokenEmbedding(nn.Module):
    """Scaled-identity token embedding, frozen after init.

    For ``K <= d`` the rows are exactly orthogonal (a scaled identity padded with zeros),
    realizing the paper's "approximately orthogonal random vectors" with no learnable
    parameters. Cellular-automata vocabularies are small (binary ``K=4``), so this holds.
    """

    def __init__(self, K: int, d: int, scale: float = 0.02) -> None:
        super().__init__()
        if K > d:
            raise ValueError(
                f"FrozenTokenEmbedding requires K<=d for orthogonal rows (K={K}, d={d})"
            )
        self.emb = nn.Embedding(K, d)
        with torch.no_grad():
            weight = torch.zeros(K, d)
            weight[:K, :K] = torch.eye(K) * scale
            self.emb.weight.copy_(weight)
        self.emb.weight.requires_grad_(False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


class FrozenPositionalEmbedding(nn.Module):
    """Random unit-vector positional embedding (scaled), frozen after init."""

    def __init__(self, N: int, d: int, scale: float = 0.02) -> None:
        super().__init__()
        vecs = torch.randn(N, d)
        vecs = vecs / vecs.norm(dim=1, keepdim=True) * scale
        self.emb = nn.Embedding(N, d)
        with torch.no_grad():
            self.emb.weight.copy_(vecs)
        self.emb.weight.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class Frozen2DSinCosPositionalEmbedding(nn.Module):
    """Fixed 2-D sin/cos positional embedding for an ``H x W`` token grid, frozen.

    Each position ``p`` maps to ``(row, col) = (p // W, p % W)``; half the channels encode the
    row and half the column with multi-frequency sin/cos (the standard ViT/MAE 2-D encoding).
    Unlike the random encoding, this exposes **2-D adjacency** — what a *spatial* layout (e.g.
    nested Hilbert Dyck) needs for the model to use its geometry. Row-major order maps a 1-D
    sequence onto the same grid, so it is a fair shared positional policy for 1-D vs 2-D.
    """

    def __init__(self, N: int, d: int, H: int, W: int, scale: float = 0.02) -> None:
        super().__init__()
        if d % 4 != 0:
            raise ValueError(f"2D sin/cos needs d divisible by 4, got {d}")
        if H * W != N:
            raise ValueError(f"H*W={H * W} != N={N}")
        rows = torch.arange(H).repeat_interleave(W).float()  # (N,)
        cols = torch.arange(W).repeat(H).float()             # (N,)
        dq = d // 4
        omega = 1.0 / (10000.0 ** (torch.arange(dq).float() / dq))

        def _enc(pos):  # (N,) -> (N, d/2)
            ang = pos[:, None] * omega[None, :]
            return torch.cat([ang.sin(), ang.cos()], dim=1)

        vecs = torch.cat([_enc(rows), _enc(cols)], dim=1)     # (N, d)
        vecs = vecs / vecs.norm(dim=1, keepdim=True) * scale  # match the random-encoding scale
        self.emb = nn.Embedding(N, d)
        with torch.no_grad():
            self.emb.weight.copy_(vecs)
        self.emb.weight.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)
