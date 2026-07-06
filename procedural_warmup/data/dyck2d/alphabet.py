"""Token-id scheme for the 2D Dyck corner alphabet Δ_k (Crespi Reghizzi et al., 2307.16522).

Δ_k = {a_i, b_i, c_i, d_i | 1 <= i <= k}: each quadruple labels the four corners of a
matched rectangle (a = top-left, b = top-right, c = bottom-left, d = bottom-right).
Matching pairs (Definition 5 of the paper): rows balance [a_i, b_i] and [c_i, d_i];
columns balance [a_i, c_i] and [b_i, d_i].

Ids are laid out role-major so that role tests are single range comparisons, mirroring
the 1D layout's open-block/close-block split (``a`` is the doubly-open corner, ``d`` the
doubly-close corner — the 2D analog of a closing bracket):

    0                  -> PAD
    1                  -> MASK
    [2,      2 +  k)   -> a_i (top-left)
    [2 +  k, 2 + 2k)   -> b_i (top-right)
    [2 + 2k, 2 + 3k)   -> c_i (bottom-left)
    [2 + 3k, 2 + 4k)   -> d_i (bottom-right)

so ``K = 4k + 2``; the default k = 32 gives K = 130, exactly the 1D k-Dyck vocabulary.

This module holds constants and pure id arithmetic only — no generation or parsing — so
the independent validator (:mod:`validate`) shares nothing algorithmic with the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

N_SPECIAL = 2  # PAD, MASK (shared reserved ids across all sources)


class Role(IntEnum):
    """Corner role. Values index the role-major id blocks."""

    A = 0  # top-left
    B = 1  # top-right
    C = 2  # bottom-left
    D = 3  # bottom-right


def vocab_size(k: int) -> int:
    """Minimum vocabulary for Δ_k: 2 specials + 4k content symbols."""
    return N_SPECIAL + 4 * k


def corner_id(role: Role, index: int, k: int) -> int:
    """Token id of corner symbol ``role_index`` (index in [0, k))."""
    return N_SPECIAL + int(role) * k + index


def role_of(token_id: int, k: int) -> Role:
    """Role of a content token id (raises for PAD/MASK/out-of-range)."""
    off = token_id - N_SPECIAL
    if not 0 <= off < 4 * k:
        raise ValueError(f"id {token_id} is not a Δ_{k} content symbol")
    return Role(off // k)


def index_of(token_id: int, k: int) -> int:
    """Quadruple index i of a content token id."""
    off = token_id - N_SPECIAL
    if not 0 <= off < 4 * k:
        raise ValueError(f"id {token_id} is not a Δ_{k} content symbol")
    return off % k


def d_base(k: int) -> int:
    """First id of the d-block; d ids are ``[d_base, d_base + k)``."""
    return N_SPECIAL + 3 * k


@dataclass(frozen=True)
class Rect:
    """A matched corner quadruple: a@(r1,c1), b@(r1,c2), c@(r2,c1), d@(r2,c2).

    Spans ``r2-r1`` / ``c2-c1`` are always odd (pictures and their nested frames have
    even side lengths, so paired corners sit an odd offset apart).
    """

    index: int
    r1: int
    c1: int
    r2: int
    c2: int

    @property
    def row_span(self) -> int:
        return self.r2 - self.r1

    @property
    def col_span(self) -> int:
        return self.c2 - self.c1

    @property
    def d_pos(self) -> tuple[int, int]:
        """Grid position of the d (bottom-right) corner — the maskable cell."""
        return self.r2, self.c2
