"""Size-conditioned stochastic sampler for the well-nested 2D Dyck language DW_k.

Implements Definition 2 of Crespi Reghizzi, Restivo & San Pietro, "Two-dimensional Dyck
words" (arXiv:2307.16522, p. 6-7), quoted verbatim (typography simplified; the paper's
``h_c(a_i) = c_i`` is a typo for ``h_c(a_i) = b_i`` — the codomain {b_i, d_i} and Fig. 1
force the a->b, c->d reading):

    Definition 2 (well-nested Dyck picture language). Let Δ_k = {a_i, b_i, c_i, d_i |
    1 <= i <= k}. Define two bijections: h_r : {a_i, b_i} -> {c_i, d_i}, h_c : {a_i, c_i}
    -> {b_i, d_i} with h_r(a_i) = c_i, h_r(b_i) = d_i and h_c(a_i) = b_i, h_c(c_i) = d_i.
    For every picture p ∈ Δ_k^++, for all rows w_r in the (word) Dyck language over the
    parentheses [a_i, b_i], and for all columns w_c in the Dyck language over the
    parentheses [a_i, c_i], such that |w_r| = |p|_col, |w_c| = |p|_row, the nesting
    accretion of p within w_r, w_c is the picture:

        (a_i · w_r · b_i)  over  (w_c · p · h_c(w_c))  over  (c_i · h_r(w_r) · d_i).

    The language DW_k is the smallest set including the empty picture and closed under
    nesting accretion and Simplot closure.

Two structural consequences the sampler enforces (both verified against the paper):

1. **No flat-interior accretion.** The accreted interior must be non-empty (p ∈ Δ_k^++);
   accretion "of the empty picture" degenerates to the bare 2x2 quadruple. Theorem 3
   (p. 8) exhibits the nested 2x4 ``[a a b b; c c d d]`` as a witness of DN_1 \\ DW_1
   ("cannot be obtained using nesting accretion"). Hence every height-2 (width-2) DW
   picture is a horizontal (vertical) tiling of 2x2 quadruples, and the sampler only
   accretes when both target dims are >= 4.
2. **Guillotine support bias (documented, measured — not fixed).** The Simplot closure
   admits arbitrary tessellations; this sampler realizes concatenation as recursive
   binary guillotine splits, so non-guillotine ("pinwheel") tessellations of DW tiles
   are in DW_k but never sampled. All outputs are certified DW members (every step is a
   DW constructor); the support restriction is reported via :mod:`stats` structure
   statistics alongside results, per the design doc's sampler-bias policy.

``p_acc`` (accretion vs. split) is the depth knob analogous to the 1D generator's
``open_prob``; ``open_prob`` here shapes the border Dyck words themselves.
"""

from __future__ import annotations

import random

import numpy as np

from procedural_warmup.data.dyck2d.alphabet import Rect, Role, corner_id


def _dyck_word(
    length: int, k: int, open_prob: float, rng: random.Random
) -> list[tuple[int, int, int]]:
    """One balanced Dyck word over k bracket types as ``(type, open_pos, close_pos)`` triples.

    Same stochastic open/close-with-flush scheme as the 1D generator
    (``data/dyck/generator.py:dyck_ids``), but returns matched *positions* rather than
    1D-layout token ids so the caller can place any corner-symbol pair on them.
    ``length`` must be even (may be 0 -> empty word).
    """
    if length % 2 != 0:
        raise ValueError(f"Dyck word length must be even, got {length}")
    pairs: list[tuple[int, int, int]] = []
    stack: list[tuple[int, int]] = []  # (type, open_pos)
    pos = 0
    while pos < length:
        remaining = length - pos
        if remaining <= len(stack):
            # Must close now to flush the stack within the remaining capacity.
            t, open_pos = stack.pop()
            pairs.append((t, open_pos, pos))
        elif not stack:
            stack.append((rng.randrange(k), pos))
        else:
            can_open = remaining >= len(stack) + 2
            if can_open and rng.random() < open_prob:
                stack.append((rng.randrange(k), pos))
            else:
                t, open_pos = stack.pop()
                pairs.append((t, open_pos, pos))
        pos += 1
    return pairs


def _quadruple(
    buf: np.ndarray, r: int, c: int, k: int, rng: random.Random, rects: list[Rect]
) -> None:
    """Write one 2x2 quadruple [a_i b_i; c_i d_i] at (r, c), i ~ U(k)."""
    i = rng.randrange(k)
    buf[r, c] = corner_id(Role.A, i, k)
    buf[r, c + 1] = corner_id(Role.B, i, k)
    buf[r + 1, c] = corner_id(Role.C, i, k)
    buf[r + 1, c + 1] = corner_id(Role.D, i, k)
    rects.append(Rect(i, r, c, r + 1, c + 1))


def _fill(
    buf: np.ndarray,
    r0: int,
    c0: int,
    m: int,
    n: int,
    k: int,
    p_acc: float,
    open_prob: float,
    rng: random.Random,
    rects: list[Rect],
    trace: dict | None,
) -> None:
    """Fill the m x n subpicture at (r0, c0) with a DW_k member (recursion body)."""
    if m == 2 and n == 2:
        _quadruple(buf, r0, c0, k, rng, rects)
        return
    if m == 2:
        # Correction #1: the only height-2 DW pictures are tilings of 2x2 quadruples.
        for j in range(n // 2):
            _quadruple(buf, r0, c0 + 2 * j, k, rng, rects)
        return
    if n == 2:
        for i in range(m // 2):
            _quadruple(buf, r0 + 2 * i, c0, k, rng, rects)
        return

    # m >= 4 and n >= 4: accretion or guillotine split.
    if rng.random() < p_acc:
        if trace is not None:
            trace["accretions"] = trace.get("accretions", 0) + 1
        i = rng.randrange(k)
        r1, c1 = r0 + m - 1, c0 + n - 1  # bottom-right frame corner
        buf[r0, c0] = corner_id(Role.A, i, k)
        buf[r0, c1] = corner_id(Role.B, i, k)
        buf[r1, c0] = corner_id(Role.C, i, k)
        buf[r1, c1] = corner_id(Role.D, i, k)
        rects.append(Rect(i, r0, c0, r1, c1))
        # Top border w_r over [a_j, b_j]; bottom border is its h_r image position-wise,
        # so each border pair's four cells form a full-height rectangle.
        for j, po, pc in _dyck_word(n - 2, k, open_prob, rng):
            buf[r0, c0 + 1 + po] = corner_id(Role.A, j, k)
            buf[r0, c0 + 1 + pc] = corner_id(Role.B, j, k)
            buf[r1, c0 + 1 + po] = corner_id(Role.C, j, k)  # h_r(a_j)
            buf[r1, c0 + 1 + pc] = corner_id(Role.D, j, k)  # h_r(b_j)
            rects.append(Rect(j, r0, c0 + 1 + po, r1, c0 + 1 + pc))
        # Left border w_c over [a_j, c_j]; right border is its h_c image (full-width rects).
        for j, po, pc in _dyck_word(m - 2, k, open_prob, rng):
            buf[r0 + 1 + po, c0] = corner_id(Role.A, j, k)
            buf[r0 + 1 + pc, c0] = corner_id(Role.C, j, k)
            buf[r0 + 1 + po, c1] = corner_id(Role.B, j, k)  # h_c(a_j)
            buf[r0 + 1 + pc, c1] = corner_id(Role.D, j, k)  # h_c(c_j)
            rects.append(Rect(j, r0 + 1 + po, c0, r0 + 1 + pc, c1))
        # Non-empty interior (m-2, n-2) >= (2, 2): p ∈ Δ_k^++ as Definition 2 requires.
        _fill(buf, r0 + 1, c0 + 1, m - 2, n - 2, k, p_acc, open_prob, rng, rects, trace)
    else:
        # Guillotine split into two even-sized DW pictures (Simplot-closure subset).
        vertical_cut = rng.random() < 0.5  # cut between columns vs between rows
        if vertical_cut:
            if trace is not None:
                trace["v_splits"] = trace.get("v_splits", 0) + 1
            cut = 2 * rng.randint(1, n // 2 - 1)  # even offset in [2, n-2]
            _fill(buf, r0, c0, m, cut, k, p_acc, open_prob, rng, rects, trace)
            _fill(buf, r0, c0 + cut, m, n - cut, k, p_acc, open_prob, rng, rects, trace)
        else:
            if trace is not None:
                trace["h_splits"] = trace.get("h_splits", 0) + 1
            cut = 2 * rng.randint(1, m // 2 - 1)
            _fill(buf, r0, c0, cut, n, k, p_acc, open_prob, rng, rects, trace)
            _fill(buf, r0 + cut, c0, m - cut, n, k, p_acc, open_prob, rng, rects, trace)


def dw_picture(
    m: int,
    n: int,
    k: int,
    p_acc: float = 0.6,
    open_prob: float = 0.6,
    rng: random.Random | None = None,
    trace: dict | None = None,
) -> tuple[np.ndarray, list[Rect]]:
    """Sample one m x n DW_k picture; returns ``(int64 grid of token ids, rectangles)``.

    Exact size by construction (no rejection): every recursion step strictly reduces the
    area and bottoms out at 2x2 quadruples. ``len(rects) == m*n/4`` — every cell is a
    corner of exactly one matched quadruple. ``trace`` (optional dict) accumulates
    accretion/split action counts for :mod:`stats`.
    """
    if m % 2 != 0 or n % 2 != 0:
        raise ValueError(f"DW pictures have even side lengths, got ({m}, {n})")
    if m < 2 or n < 2:
        raise ValueError(f"picture dims must be >= 2, got ({m}, {n})")
    if rng is None:
        rng = random.Random()
    buf = np.zeros((m, n), dtype=np.int64)
    rects: list[Rect] = []
    _fill(buf, 0, 0, m, n, k, p_acc, open_prob, rng, rects, trace)
    return buf, rects
