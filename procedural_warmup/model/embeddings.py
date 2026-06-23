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
    """Random unit-vector positional embedding (scaled).

    Frozen by default (the paper's setup — forces structure into the transformer blocks).
    Set ``trainable=True`` to let the model learn positions: this matters for 2-D tasks
    (e.g. Game of Life) where the random frozen positions give no geometric prior for the
    spatial stencil. Positions are discarded at transfer regardless, so the method's premise
    (gains live in the blocks) is preserved.
    """

    def __init__(self, N: int, d: int, scale: float = 0.02, trainable: bool = False) -> None:
        super().__init__()
        vecs = torch.randn(N, d)
        vecs = vecs / vecs.norm(dim=1, keepdim=True) * scale
        self.emb = nn.Embedding(N, d)
        with torch.no_grad():
            self.emb.weight.copy_(vecs)
        self.emb.weight.requires_grad_(trainable)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)
