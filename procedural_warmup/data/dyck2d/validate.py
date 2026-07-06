"""Independent membership checking for 2D Dyck pictures (shares only :mod:`alphabet`).

Two tiers, written from the paper's definitions rather than from the sampler:

- **Tier 1** (:func:`recover_rectangles` / :func:`check_picture`): LIFO-parse every row
  over the row pairs [a_i,b_i], [c_i,d_i] and every column over the column pairs
  [a_i,c_i], [b_i,d_i], then check *rectangle closure* — each a's row-matched b and
  column-matched c must close on one common d. This is a **necessary** condition for
  DW_k membership (DW ⊆ DC, whose rows/columns are Dyck words; well-nesting makes the
  LIFO matching the true matching) and recovers the unique rectangle partition that the
  masking filter, auditor and statistics consume. It deliberately does **not** reject
  partially-overlapping bounding boxes — those are legal in DW (paper p. 7: two boxes
  are disjoint, nested, "or they overlap and a third box exists that minimally bounds
  both", e.g. the plus-sign overlap of an accretion's border-word rectangles). Tier 1
  alone over-accepts: the Theorem-3 witness ``[a a b b; c c d d]`` ∈ DN_1 \\ DW_1 passes.

- **Tier 2** (:func:`is_dw_guillotine`): memoized recursive decomposition — a subpicture
  is accepted iff it is a 2x2 quadruple, a Definition-2 accretion frame around an
  accepted interior (only when both dims >= 4), or an even guillotine split into two
  accepted halves. Acceptance **proves** DW_k membership (every step is a DW
  constructor). Rejection does not disprove it (non-guillotine Simplot tessellations are
  outside this tier's reach) — exact on the sampler's support, which is what the
  property-test gate needs.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from procedural_warmup.data.dyck2d.alphabet import Rect, Role, corner_id, index_of, role_of


class MembershipError(ValueError):
    """A picture failed a membership check; the message says where and why."""


def _decode(grid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (roles, indices) int arrays; raises MembershipError on non-content ids."""
    off = grid.astype(np.int64) - 2
    if (off < 0).any() or (off >= 4 * k).any():
        r, c = np.argwhere((off < 0) | (off >= 4 * k))[0]
        raise MembershipError(
            f"cell ({r},{c}) holds id {int(grid[r, c])}, not a Δ_{k} content symbol"
        )
    return off // k, off % k


def _parse_line(
    roles: np.ndarray,
    idxs: np.ndarray,
    openers: tuple[Role, Role],
    closer_of: dict[Role, Role],
    what: str,
) -> dict[int, int]:
    """LIFO-parse one line; returns {close_pos: open_pos}. Raises MembershipError.

    ``openers`` are the two roles that push; ``closer_of`` maps closing role -> the
    opening role it must pop (same quadruple index required).
    """
    matches: dict[int, int] = {}
    stack: list[tuple[Role, int, int]] = []  # (role, index, pos)
    for pos in range(len(roles)):
        role, idx = Role(int(roles[pos])), int(idxs[pos])
        if role in openers:
            stack.append((role, idx, pos))
        else:
            need = closer_of.get(role)
            if need is None:  # pragma: no cover - closer_of covers both closing roles
                raise MembershipError(f"{what}: unexpected role {role.name} at {pos}")
            if not stack:
                raise MembershipError(f"{what}: {role.name}_{idx} at {pos} closes an empty stack")
            top_role, top_idx, top_pos = stack.pop()
            if top_role != need or top_idx != idx:
                raise MembershipError(
                    f"{what}: {role.name}_{idx} at {pos} closes {top_role.name}_{top_idx} "
                    f"from {top_pos} (need {need.name}_{idx})"
                )
            matches[pos] = top_pos
    if stack:
        role, idx, pos = stack[-1]
        raise MembershipError(f"{what}: {role.name}_{idx} at {pos} never closed")
    return matches


def recover_rectangles(grid: np.ndarray, k: int) -> list[Rect]:
    """Tier 1: parse rows and columns, enforce rectangle closure, return the quadruples."""
    m, n = grid.shape
    roles, idxs = _decode(grid, k)

    # Row parses: a/c push; b pops a; d pops c.
    row_match: dict[tuple[int, int], tuple[int, int]] = {}  # closer (r,c) -> opener (r,c)
    for r in range(m):
        for pos, open_pos in _parse_line(
            roles[r], idxs[r], (Role.A, Role.C), {Role.B: Role.A, Role.D: Role.C}, f"row {r}"
        ).items():
            row_match[(r, pos)] = (r, open_pos)

    # Column parses: a/b push; c pops a; d pops b.
    col_match: dict[tuple[int, int], tuple[int, int]] = {}
    for c in range(n):
        for pos, open_pos in _parse_line(
            roles[:, c], idxs[:, c], (Role.A, Role.B), {Role.C: Role.A, Role.D: Role.B}, f"col {c}"
        ).items():
            col_match[(pos, c)] = (open_pos, c)

    # Rectangle closure: index rectangles by their a-corner.
    b_of = {a_pos: b_pos for b_pos, a_pos in row_match.items() if roles[b_pos] == Role.B}
    c_of = {a_pos: c_pos for c_pos, a_pos in col_match.items() if roles[c_pos] == Role.C}
    rects: list[Rect] = []
    for (r1, c1) in np.argwhere(roles == Role.A):
        a_pos = (int(r1), int(c1))
        b_pos, c_pos = b_of.get(a_pos), c_of.get(a_pos)
        if b_pos is None or c_pos is None:  # pragma: no cover - parses guarantee matches
            raise MembershipError(f"a at {a_pos} has no row/column match")
        d_expect = (c_pos[0], b_pos[1])
        # The d that closes c in its row, and the d that closes b in its column,
        # must both be that one cell.
        if row_match.get(d_expect) != c_pos:
            raise MembershipError(
                f"quadruple at a={a_pos}: row-mate of d-cell {d_expect} is "
                f"{row_match.get(d_expect)}, expected c at {c_pos}"
            )
        if col_match.get(d_expect) != b_pos:
            raise MembershipError(
                f"quadruple at a={a_pos}: column-mate of d-cell {d_expect} is "
                f"{col_match.get(d_expect)}, expected b at {b_pos}"
            )
        if int(idxs[d_expect]) != int(idxs[a_pos]):  # pragma: no cover - parses enforce it
            raise MembershipError(f"index mismatch in quadruple at a={a_pos}")
        rects.append(Rect(int(idxs[a_pos]), a_pos[0], a_pos[1], d_expect[0], d_expect[1]))

    if len(rects) != (m * n) // 4:
        raise MembershipError(
            f"recovered {len(rects)} quadruples, expected {m * n // 4}"
        )
    return rects


def check_picture(grid: np.ndarray, k: int) -> None:
    """Raise :class:`MembershipError` unless Tier 1 accepts ``grid``."""
    recover_rectangles(grid, k)


def is_dc_member(grid: np.ndarray, k: int) -> bool:
    """Boolean wrapper over the Tier-1 check (necessary condition; DW ⊆ ... ⊆ DC)."""
    try:
        check_picture(grid, k)
        return True
    except MembershipError:
        return False


def is_dw_guillotine(grid: np.ndarray, k: int) -> bool:
    """Tier 2: certify DW_k membership via accretion/guillotine decomposition.

    Accept ⇒ ``grid`` ∈ DW_k. Reject ⇏ ∉ DW_k (non-guillotine tessellations are not
    searched); exact on the guillotine sampler's support.
    """
    m, n = grid.shape
    if m % 2 or n % 2 or m < 2 or n < 2:
        return False
    g = grid.astype(np.int64)

    def _is_quadruple(r0: int, c0: int) -> bool:
        idx = int(g[r0, c0]) - 2
        if not 0 <= idx < k:  # top-left must be an a
            return False
        return (
            int(g[r0, c0 + 1]) == corner_id(Role.B, idx, k)
            and int(g[r0 + 1, c0]) == corner_id(Role.C, idx, k)
            and int(g[r0 + 1, c0 + 1]) == corner_id(Role.D, idx, k)
        )

    def _is_border_dyck(cells: np.ndarray, open_role: Role, close_role: Role) -> bool:
        """Is this 1D border segment a Dyck word over [open_role_j, close_role_j]?"""
        stack: list[int] = []
        for tid in cells:
            tid = int(tid)
            try:
                role, idx = role_of(tid, k), index_of(tid, k)
            except ValueError:
                return False
            if role == open_role:
                stack.append(idx)
            elif role == close_role:
                if not stack or stack.pop() != idx:
                    return False
            else:
                return False
        return not stack

    def _frame_ok(r1: int, r2: int, c1: int, c2: int) -> bool:
        """Definition-2 frame test on the half-open box [r1,r2) x [c1,c2), dims >= 4."""
        rb, cb = r2 - 1, c2 - 1
        i = int(g[r1, c1]) - 2
        if not 0 <= i < k:
            return False
        if (
            int(g[r1, cb]) != corner_id(Role.B, i, k)
            or int(g[rb, c1]) != corner_id(Role.C, i, k)
            or int(g[rb, cb]) != corner_id(Role.D, i, k)
        ):
            return False
        top, bottom = g[r1, c1 + 1 : cb], g[rb, c1 + 1 : cb]
        left, right = g[r1 + 1 : rb, c1], g[r1 + 1 : rb, cb]
        if not _is_border_dyck(top, Role.A, Role.B):
            return False
        if not _is_border_dyck(left, Role.A, Role.C):
            return False
        # bottom must be h_r(top) (a->c, b->d) and right must be h_c(left) (a->b, c->d):
        # role block shifts by +2k (A->C, B->D) resp. +k (A->B, C->D), same index.
        if not np.array_equal(bottom, top + 2 * k):
            return False
        if not np.array_equal(right, left + k):
            return False
        return True

    @lru_cache(maxsize=None)
    def _ok(r1: int, r2: int, c1: int, c2: int) -> bool:
        h, w = r2 - r1, c2 - c1
        if h == 2 and w == 2:
            return _is_quadruple(r1, c1)
        if h >= 4 and w >= 4 and _frame_ok(r1, r2, c1, c2):
            if _ok(r1 + 1, r2 - 1, c1 + 1, c2 - 1):
                return True
        for cut in range(2, w, 2):  # vertical cuts at even offsets
            if _ok(r1, r2, c1, c1 + cut) and _ok(r1, r2, c1 + cut, c2):
                return True
        for cut in range(2, h, 2):  # horizontal cuts
            if _ok(r1, r1 + cut, c1, c2) and _ok(r1 + cut, r2, c1, c2):
                return True
        return False

    return _ok(0, m, 0, n)
