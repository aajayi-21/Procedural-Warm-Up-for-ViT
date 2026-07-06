"""Frozen embeddings used during procedural warm-up.

Both the token lookup and the positional encoding are random/fixed and **frozen**: this
prevents the model from solving the masked-token objective via the embeddings, forcing the
attention and MLP layers to learn the data's structure instead. Both are discarded before
image training.
"""

from __future__ import annotations

import math

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


class Frozen1DRingPositionalEmbedding(nn.Module):
    """Fixed sin/cos positional embedding over an ``N``-cell *ring* (periodic), frozen.

    Each position ``i`` is encoded with harmonics of the ring angle ``2*pi*i/N`` (a Fourier
    basis on the cycle), so adjacency — including the ``i = N-1 <-> i = 0`` wrap — is exposed.
    This matches the **periodic 1-D boundary** of the ``ca_step`` board (an N-cell ECA ring),
    for which the 2-D ``(p//W, p%W)`` encoding is wrong (it fragments the ring every W cells).
    Like the other positional embeddings it is frozen and stripped before image transfer, so
    it only shapes what the blocks learn. Use via ``model.pos_embed: sincos1d``.
    """

    def __init__(self, N: int, d: int, scale: float = 0.02) -> None:
        super().__init__()
        if d % 2 != 0:
            raise ValueError(f"1D ring sin/cos needs even d, got {d}")
        idx = torch.arange(N).float()
        harmonics = torch.arange(1, d // 2 + 1).float()  # (d/2,) integer harmonics => periodic in N
        ang = (2.0 * math.pi / N) * idx[:, None] * harmonics[None, :]  # (N, d/2)
        vecs = torch.cat([ang.sin(), ang.cos()], dim=1)  # (N, d)
        vecs = vecs / vecs.norm(dim=1, keepdim=True) * scale  # match the random-encoding scale
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

    Audit note (see ``analysis/pos_embed_audit.py``): the base-10000 frequency schedule is
    matched to long token sequences, not a 14-wide grid — only ~15 of the d/4 = 48
    frequencies per axis vary meaningfully across 14 cells (angle span >= pi/4); the rest
    are near-constant channels that consume norm budget, floor-ing far-pair cosine
    similarity around ~0.7. The construction is still geometrically sound (every
    pre-normalization row norm is exactly sqrt(d/2), so the 0.02 rescale is uniform and
    preserves geometry), and it is kept unchanged as the H6 primary arm for comparability
    with all existing ``sincos2d`` results; the grid-matched variant below is the
    screening-arm alternative.
    """

    def __init__(self, N: int, d: int, H: int, W: int, scale: float = 0.02) -> None:
        super().__init__()
        if d % 4 != 0:
            raise ValueError(f"2D sin/cos needs d divisible by 4, got {d}")
        dq = d // 4
        omega = 1.0 / (10000.0 ** (torch.arange(dq).float() / dq))
        vecs = _sincos2d_table(N, H, W, omega, scale)
        self.emb = nn.Embedding(N, d)
        with torch.no_grad():
            self.emb.weight.copy_(vecs)
        self.emb.weight.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


# Finest wavelength (cells) of the tuned band. Strictly above the Nyquist wavelength 2:
# at exactly 2, omega = pi and the sin channel is identically 0 on integer coordinates
# (a dead channel per axis). 2.1 keeps every channel alive on the grid.
TUNED_LAMBDA_MIN = 2.1


class FrozenTuned2DSinCosPositionalEmbedding(nn.Module):
    """Grid-matched 2-D sin/cos positional embedding (``pos_embed: sincos2d_tuned``).

    Identical to :class:`Frozen2DSinCosPositionalEmbedding` except the frequency band:
    wavelengths are geometrically spaced in ``[lambda_min, lambda_max]`` **cells**
    (defaults :data:`TUNED_LAMBDA_MIN` and ``2 * max(H, W)``), so every one of the d/4
    frequencies per axis varies meaningfully across the grid — no dead channels
    (``lambda_min`` sits strictly above the Nyquist wavelength 2, at which the sin
    channel would be identically zero on integer coordinates), maximal inter-position
    contrast at the same 0.02 norm. The coarsest channel's span ``(H-1) * 2pi /
    lambda_max < 2pi`` keeps the coarse code injective per axis (no positional
    aliasing). Per-position pre-normalization norm is still exactly ``sqrt(d/2)`` (the
    sin^2 + cos^2 argument is frequency-independent), so the rescale stays uniform.

    Used as an extra 1-seed screening arm in the H6 program — NOT the primary
    treatment, which keeps the standard ``sincos2d`` for comparability.
    """

    def __init__(
        self,
        N: int,
        d: int,
        H: int,
        W: int,
        scale: float = 0.02,
        lambda_min: float = TUNED_LAMBDA_MIN,
        lambda_max: float | None = None,
    ) -> None:
        super().__init__()
        if d % 4 != 0:
            raise ValueError(f"2D sin/cos needs d divisible by 4, got {d}")
        if lambda_max is None:
            lambda_max = 2.0 * max(H, W)
        dq = d // 4
        lam = lambda_min * (lambda_max / lambda_min) ** (
            torch.arange(dq).float() / max(dq - 1, 1)
        )
        omega = 2.0 * math.pi / lam
        vecs = _sincos2d_table(N, H, W, omega, scale)
        self.emb = nn.Embedding(N, d)
        with torch.no_grad():
            self.emb.weight.copy_(vecs)
        self.emb.weight.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


def _sincos2d_table(
    N: int, H: int, W: int, omega: torch.Tensor, scale: float
) -> torch.Tensor:
    """Shared 2-D sin/cos table builder: rows/cols encoded separately, then rescaled.

    ``omega`` holds the d/4 per-axis angular frequencies; position ``p`` maps to
    ``(row, col) = (p // W, p % W)`` (row-major, the repo's fixed cell->token layout).
    """
    if H * W != N:
        raise ValueError(f"H*W={H * W} != N={N}")
    rows = torch.arange(H).repeat_interleave(W).float()  # (N,)
    cols = torch.arange(W).repeat(H).float()             # (N,)

    def _enc(pos):  # (N,) -> (N, d/2)
        ang = pos[:, None] * omega[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=1)

    vecs = torch.cat([_enc(rows), _enc(cols)], dim=1)     # (N, d)
    return vecs / vecs.norm(dim=1, keepdim=True) * scale  # match the random-encoding scale
