"""Determinacy auditor for masked 2D Dyck pictures (design principle P5).

The masked-token objective is only well-posed when each masked target is determined by
the visible context (the CA study's ``iid-static`` regime, 65.72 on CIFAR-100, is what
under-determination produces). This module decides, per masked cell, whether its value
is forced — and cross-checks itself by brute force on small grids.

Three modes:

- ``mode="closing"`` — exact under the ``mask_roles="cd"`` policy (masked cells are
  closing corners, c or d). Both roles are closers of the COLUMN pair system, and the
  column openers (a, b) are never masked, so a top-down column parse forces every
  hole's role and index from the stack top alone; the completion is re-checked with
  Tier 1. Also exact for d-only masks (a sub-policy).

- ``mode="corner"`` — exact under the corner-close-only policy (only d-cells masked).
  All a/b/c symbols are visible, and projections of a Dyck line onto a subset of its
  pair types remain Dyck, so LIFO-parsing the visible {a,b} row projections and {a,c}
  column projections recovers every rectangle; each a's row-mate b and column-mate c
  imply the position *and* index of its d. Every hole must be exactly one implied
  d-position; the completed picture is then re-checked end-to-end with Tier 1.
  Determinacy here is w.r.t. the masking policy the model is trained under (masks are
  always d's) — the exact notion the objective needs; ``brute_force_forced`` with
  ``restrict_to_d=True`` verifies it, and with ``restrict_to_d=False`` measures the
  stronger unrestricted-completion uniqueness.

- ``mode="general"`` — sound-but-incomplete constraint propagation for arbitrary masks
  (the H11 audited-random scaffold and the H8 determinacy-rate covariate). Per-hole
  candidate sets are pruned to values feasible for every containing line (row/column
  Dyck-with-wildcards, decided by interval DP) until fixpoint; a hole is labeled forced
  iff a single candidate survives, and when *every* hole is singleton the completion is
  additionally verified end-to-end with Tier 1. Line feasibility is a relaxation of DW
  membership, so pruning never removes a truly possible value — on a **valid** picture
  a surviving singleton is therefore truly forced even when other holes stay ambiguous.
  Any contradiction (including a corrupt hole-free line) proves no completion exists
  and clears all forced labels. The forced/soundness guarantee presumes a valid input
  picture; completeness is measured against brute force, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from procedural_warmup.data.dyck2d.alphabet import Role, corner_id, d_base, index_of, role_of
from procedural_warmup.data.dyck2d.validate import MembershipError, check_picture

# Line pair systems: (opener_role, closer_role) pairs active in rows vs columns.
_ROW_PAIRS = ((Role.A, Role.B), (Role.C, Role.D))
_COL_PAIRS = ((Role.A, Role.C), (Role.B, Role.D))


@dataclass
class AuditReport:
    """Per-mask determinacy verdict.

    ``forced``/``values`` are (H, W) arrays: ``forced[r, c]`` is True iff the masked
    cell at (r, c) is provably determined, in which case ``values[r, c]`` holds its
    token id (0 elsewhere). ``determinacy_rate`` = forced fraction among masked cells
    (1.0 when nothing is masked); ``ok`` = every masked cell forced and the completion
    validates. ``reasons`` collects human-readable failure notes.
    """

    ok: bool
    forced: np.ndarray
    values: np.ndarray
    determinacy_rate: float
    reasons: list[str]


# ------------------------------------------------------------------------------------
# corner mode
# ------------------------------------------------------------------------------------


def _parse_projection(
    line_vals: list[tuple[int, int]], open_role: Role, close_role: Role, k: int
) -> tuple[dict[int, int], bool]:
    """LIFO-match ``(pos, id)`` cells of one pair type; returns ({close_pos: open_pos}, ok)."""
    matches: dict[int, int] = {}
    stack: list[tuple[int, int]] = []  # (index, pos)
    for pos, tid in line_vals:
        role = Role((tid - 2) // k)
        if role == open_role:
            stack.append(((tid - 2) % k, pos))
        elif role == close_role:
            if not stack or stack[-1][0] != (tid - 2) % k:
                return matches, False
            matches[pos] = stack.pop()[1]
    return matches, (len(stack) == 0)


def _audit_corner(grid: np.ndarray, mask: np.ndarray, k: int) -> AuditReport:
    m, n = grid.shape
    forced = np.zeros((m, n), dtype=bool)
    values = np.zeros((m, n), dtype=np.int64)
    reasons: list[str] = []
    n_masked = int(mask.sum())
    if n_masked == 0:
        return AuditReport(True, forced, values, 1.0, reasons)

    lo, hi = 2, 2 + 4 * k
    visible = ~mask
    bad = visible & ((grid < lo) | (grid >= hi))
    if bad.any():
        r, c = map(int, np.argwhere(bad)[0])
        reasons.append(f"visible cell ({r},{c}) holds non-content id {int(grid[r, c])}")
        return AuditReport(False, forced, values, 0.0, reasons)

    roles = np.where(visible, (grid - 2) // k, -1)

    # Row {a,b} projections -> each a's row-mate b; column {a,c} -> each a's column-mate c.
    b_of: dict[tuple[int, int], tuple[int, int]] = {}
    c_of: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(m):
        cells = [(c, int(grid[r, c])) for c in range(n) if visible[r, c] and roles[r, c] in (Role.A, Role.B)]
        matches, ok = _parse_projection(cells, Role.A, Role.B, k)
        if not ok:
            reasons.append(f"row {r}: visible {{a,b}} projection is not Dyck (a policy violation "
                           f"or corrupt picture)")
            return AuditReport(False, forced, values, 0.0, reasons)
        for close_pos, open_pos in matches.items():
            b_of[(r, open_pos)] = (r, close_pos)
    for c in range(n):
        cells = [(r, int(grid[r, c])) for r in range(m) if visible[r, c] and roles[r, c] in (Role.A, Role.C)]
        matches, ok = _parse_projection(cells, Role.A, Role.C, k)
        if not ok:
            reasons.append(f"col {c}: visible {{a,c}} projection is not Dyck")
            return AuditReport(False, forced, values, 0.0, reasons)
        for close_pos, open_pos in matches.items():
            c_of[(open_pos, c)] = (close_pos, c)

    # Implied d-positions from every visible a.
    implied: dict[tuple[int, int], int] = {}
    for (ar, ac) in map(tuple, np.argwhere(roles == Role.A)):
        a_pos = (int(ar), int(ac))
        b_pos, cc_pos = b_of.get(a_pos), c_of.get(a_pos)
        if b_pos is None or cc_pos is None:
            reasons.append(f"a at {a_pos} lacks a visible row or column mate")
            return AuditReport(False, forced, values, 0.0, reasons)
        idx = (int(grid[a_pos]) - 2) % k
        implied[(cc_pos[0], b_pos[1])] = corner_id(Role.D, idx, k)

    ok = True
    for pos, want in implied.items():
        if mask[pos]:
            forced[pos] = True
            values[pos] = want
        elif int(grid[pos]) != want:
            reasons.append(f"visible cell {pos} holds {int(grid[pos])}, implied d is {want}")
            ok = False
    stray = mask & ~forced
    if stray.any():
        r, c = map(int, np.argwhere(stray)[0])
        reasons.append(
            f"masked cell ({r},{c}) is not an implied d-position (policy violation: "
            f"an a/b/c was masked, or the picture is corrupt)"
        )
        ok = False

    if ok:
        completed = grid.copy()
        completed[forced] = values[forced]
        try:
            check_picture(completed, k)
        except MembershipError as exc:  # pragma: no cover - defensive end-to-end check
            reasons.append(f"completed picture failed Tier 1: {exc}")
            ok = False

    rate = float(forced[mask].mean()) if n_masked else 1.0
    return AuditReport(ok, forced, values, rate, reasons)


# ------------------------------------------------------------------------------------
# general mode
# ------------------------------------------------------------------------------------


def _line_feasible(ids: list[int | None], cands: list[set[int] | None],
                   pairs, k: int) -> bool:
    """Can this line be completed to a Dyck word over ``pairs``? (interval DP)

    ``ids[i]`` is the visible token id or None for a hole; ``cands[i]`` is the hole's
    candidate set (None for visible cells). Both pair types share one nesting order
    (single-stack Dyck over the union alphabet), so the classic interval DP applies:
    a segment is balanced iff its first cell opens some pair whose closer appears at a
    position j with balanced interior/suffix, for some admissible (open, close) values.
    """
    length = len(ids)
    if length % 2:
        return False

    def _can_be(i: int, role: Role, idx: int) -> bool:
        tid = corner_id(role, idx, k)
        if ids[i] is not None:
            return ids[i] == tid
        return tid in cands[i]  # type: ignore[operator]

    @lru_cache(maxsize=None)
    def _bal(lo: int, hi: int) -> bool:  # half-open [lo, hi)
        if lo == hi:
            return True
        for j in range(lo + 1, hi, 2):  # candidate closer position for cell lo
            pair_ok = any(
                _can_be(lo, open_role, idx) and _can_be(j, close_role, idx)
                for open_role, close_role in pairs
                for idx in range(k)
            )
            if pair_ok and _bal(lo + 1, j) and _bal(j + 1, hi):
                return True
        return False

    return _bal(0, length)


def _audit_general(grid: np.ndarray, mask: np.ndarray, k: int) -> AuditReport:
    m, n = grid.shape
    forced = np.zeros((m, n), dtype=bool)
    values = np.zeros((m, n), dtype=np.int64)
    reasons: list[str] = []
    holes = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not holes:
        return AuditReport(True, forced, values, 1.0, reasons)

    all_ids = set(range(2, 2 + 4 * k))
    cands: dict[tuple[int, int], set[int]] = {h: set(all_ids) for h in holes}

    class _Contradiction(Exception):
        pass

    def _prune_line(cells: list[tuple[int, int]], pairs) -> bool:
        """Prune candidates of holes on one line; True if anything changed."""
        ids = [None if mask[p] else int(grid[p]) for p in cells]
        csets = [cands[p] if mask[p] else None for p in cells]
        # Whole-line feasibility first — this also catches corrupt hole-free lines,
        # which the per-hole loop below would never examine.
        if not _line_feasible(ids, csets, pairs, k):
            reasons.append(f"line through {cells[0]}..{cells[-1]}: no feasible completion")
            raise _Contradiction
        changed = False
        for i, p in enumerate(cells):
            if not mask[p]:
                continue
            keep = set()
            for s in list(cands[p]):
                trial = list(csets)
                trial[i] = {s}
                if _line_feasible(ids, trial, pairs, k):
                    keep.add(s)
            if keep != cands[p]:
                cands[p] = keep
                csets[i] = keep
                changed = True
            if not keep:
                reasons.append(f"hole {p}: no line-feasible value")
                raise _Contradiction
        return changed

    # Fixpoint over row/column pruning (each pass only shrinks candidate sets).
    try:
        for _ in range(4 * len(holes) + 4):
            changed = False
            for r in range(m):
                changed |= _prune_line([(r, c) for c in range(n)], _ROW_PAIRS)
            for c in range(n):
                changed |= _prune_line([(r, c) for r in range(m)], _COL_PAIRS)
            if not changed:
                break
    except _Contradiction:
        # A contradiction proves NO valid completion exists (pruning is sound), so
        # every intermediate singleton is vacuous — never report it as forced.
        return AuditReport(False, forced, values, 0.0, reasons)

    ok = True
    for h in holes:
        if len(cands[h]) == 1:
            forced[h] = True
            values[h] = next(iter(cands[h]))
        else:
            ok = False
    if ok:
        completed = grid.copy()
        completed[forced] = values[forced]
        try:
            check_picture(completed, k)
        except MembershipError as exc:
            reasons.append(f"singleton completion failed Tier 1: {exc}")
            forced[:] = False
            ok = False

    rate = float(forced[mask].mean())
    return AuditReport(ok, forced, values, rate, reasons)


def _audit_closing(grid: np.ndarray, mask: np.ndarray, k: int) -> AuditReport:
    """Exact audit for the ``mask_roles="cd"`` policy (masked cells are c's or d's).

    Both c and d are CLOSERS in the column pair system ([a,c] and [b,d]): parse each
    column top-down over visible openers (a, b) and closers, treating each hole as a
    closer of the current stack top — (A, i) on top forces c_i, (B, i) forces d_i.
    Every hole is therefore determined by its column alone (a/b are never masked under
    the policy, so the opener context is always complete); row parses of the completed
    picture are re-checked end-to-end with Tier 1. Also exact for d-only masks (a
    sub-policy). A hole whose stack top is missing or that leaves openers unclosed is
    a policy violation -> unforced, ok=False.
    """
    m, n = grid.shape
    forced = np.zeros((m, n), dtype=bool)
    values = np.zeros((m, n), dtype=np.int64)
    reasons: list[str] = []
    n_masked = int(mask.sum())
    if n_masked == 0:
        return AuditReport(True, forced, values, 1.0, reasons)

    lo, hi = 2, 2 + 4 * k
    visible = ~mask
    bad = visible & ((grid < lo) | (grid >= hi))
    if bad.any():
        r, c = map(int, np.argwhere(bad)[0])
        reasons.append(f"visible cell ({r},{c}) holds non-content id {int(grid[r, c])}")
        return AuditReport(False, forced, values, 0.0, reasons)

    ok = True
    for c in range(n):
        stack: list[tuple[Role, int, int]] = []  # (role, index, row)
        for r in range(m):
            if mask[r, c]:
                if not stack:
                    reasons.append(f"hole ({r},{c}): column stack empty (policy violation)")
                    ok = False
                    continue
                top_role, top_idx, _ = stack.pop()
                closer = Role.C if top_role == Role.A else Role.D
                forced[r, c] = True
                values[r, c] = corner_id(closer, top_idx, k)
                continue
            role, idx = role_of(int(grid[r, c]), k), index_of(int(grid[r, c]), k)
            if role in (Role.A, Role.B):
                stack.append((role, idx, r))
            else:
                need = Role.A if role == Role.C else Role.B
                if not stack or stack[-1][0] != need or stack[-1][1] != idx:
                    reasons.append(f"col {c}: visible {role.name}_{idx} at row {r} "
                                   f"does not close the stack top")
                    ok = False
                    if stack:
                        stack.pop()
                    continue
                stack.pop()
        if stack:
            reasons.append(f"col {c}: {len(stack)} opener(s) never closed")
            ok = False

    if ok:
        completed = grid.copy()
        completed[forced] = values[forced]
        try:
            check_picture(completed, k)
        except MembershipError as exc:
            reasons.append(f"completed picture failed Tier 1: {exc}")
            ok = False

    rate = float(forced[mask].mean()) if n_masked else 1.0
    if not ok:
        forced[:] = False
        rate = 0.0
    return AuditReport(ok, forced, values, rate, reasons)


def audit_mask(grid: np.ndarray, mask: np.ndarray, k: int, mode: str = "corner") -> AuditReport:
    """Audit one (picture, mask) pair; see the module docstring for mode semantics."""
    if mode == "corner":
        return _audit_corner(grid, mask, k)
    if mode == "closing":
        return _audit_closing(grid, mask, k)
    if mode == "general":
        return _audit_general(grid, mask, k)
    raise ValueError(f"mode must be 'corner'|'general'|'closing', got {mode!r}")


# ------------------------------------------------------------------------------------
# brute force (test-time ground truth on small grids) + H11 scaffold
# ------------------------------------------------------------------------------------


def brute_force_forced(
    grid: np.ndarray,
    mask: np.ndarray,
    k: int,
    restrict_to_d: bool = False,
    restrict_roles: str | None = None,
    cap: int = 2 ** 22,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate completions; a hole is forced iff all Tier-1-valid completions agree.

    ``restrict_roles`` limits hole values to a policy's pool: ``"d"`` (the
    corner-close-only policy — what corner mode must match exactly), ``"cd"`` (the
    closing-corners policy — what closing mode must match), or ``None`` for
    unrestricted-completion uniqueness. ``restrict_to_d=True`` is the legacy spelling
    of ``restrict_roles="d"``. Only for small grids/masks: the candidate count
    ``(pool)^holes`` must stay under ``cap``.
    """
    holes = [tuple(map(int, p)) for p in np.argwhere(mask)]
    forced = np.zeros(grid.shape, dtype=bool)
    values = np.zeros(grid.shape, dtype=np.int64)
    if not holes:
        return forced, values
    if restrict_to_d and restrict_roles is None:
        restrict_roles = "d"
    if restrict_roles == "d":
        pool = list(range(d_base(k), d_base(k) + k))
    elif restrict_roles == "cd":
        pool = list(range(2 + 2 * k, 2 + 4 * k))  # c-block + d-block (contiguous)
    elif restrict_roles is None:
        pool = list(range(2, 2 + 4 * k))
    else:
        raise ValueError(f"restrict_roles must be 'd'|'cd'|None, got {restrict_roles!r}")
    if len(pool) ** len(holes) > cap:
        raise ValueError(
            f"{len(pool)}^{len(holes)} completions exceed cap={cap}; shrink the grid/mask"
        )
    survivors: list[tuple[int, ...]] = []
    work = grid.copy()
    for combo in product(pool, repeat=len(holes)):
        for h, v in zip(holes, combo):
            work[h] = v
        try:
            check_picture(work, k)
            survivors.append(combo)
        except MembershipError:
            pass
    for h in holes:
        work[h] = grid[h]
    if not survivors:
        return forced, values
    arr = np.array(survivors)
    for j, h in enumerate(holes):
        col = arr[:, j]
        if (col == col[0]).all():
            forced[h] = True
            values[h] = int(col[0])
    return forced, values


def audited_random_mask(
    grid: np.ndarray, k: int, mask_ratio: float, rng: np.random.Generator
) -> np.ndarray:
    """H11 scaffold: i.i.d. mask draw, then unmask every cell not provably forced.

    The picture must be valid: general-mode forced labels are only meaningful when at
    least one completion exists, so an invalid grid (e.g. a shuffle-control sample) is
    rejected loudly rather than yielding an unsound "audited" mask.
    """
    check_picture(grid, k)  # raises MembershipError on invalid input
    mask = rng.random(grid.shape) < mask_ratio
    report = audit_mask(grid, mask, k, mode="general")
    return mask & report.forced
