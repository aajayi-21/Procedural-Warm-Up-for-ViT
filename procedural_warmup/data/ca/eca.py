"""Elementary cellular automata (1-D, binary, 3-cell neighborhood).

An ECA *rule* is an 8-bit lookup table over the 2**3 = 8 neighborhood configurations.
Following Wolfram's convention the neighborhood ``(left, center, right)`` indexes the
table as ``4*left + 2*center + right`` and ``table[i] = (rule >> i) & 1``. Stacking
successive 1-D states over time produces a *spacetime diagram* (rows = time, columns =
space) with rich local-to-global ("light-cone") structure — the data we warm up on.

Generation is vectorized with NumPy and is computationally negligible, like the grammar
generators it replaces.
"""

from __future__ import annotations

import numpy as np


def rule_table(rule: int) -> np.ndarray:
    """Return the length-8 output lookup table for an ECA ``rule`` in ``[0, 255]``."""
    if not 0 <= rule <= 255:
        raise ValueError(f"ECA rule must be in [0, 255], got {rule}")
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)


def step(row: np.ndarray, table: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """Advance one ECA time step. ``row`` is a 1-D uint8 array of 0/1 cell states."""
    if boundary == "periodic":
        left = np.roll(row, 1)
        right = np.roll(row, -1)
    elif boundary == "zero":
        left = np.empty_like(row)
        left[0] = 0
        left[1:] = row[:-1]
        right = np.empty_like(row)
        right[-1] = 0
        right[:-1] = row[1:]
    else:
        raise ValueError(f"Unknown boundary '{boundary}'")
    idx = (left << 2) | (row << 1) | right
    return table[idx]


def simulate_spacetime(
    rule: int,
    width: int,
    n_rows: int,
    burn_in: int = 0,
    init_density: float = 0.5,
    boundary: str = "periodic",
    init: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Evolve an ECA and return its spacetime diagram of shape ``(n_rows, width)``.

    The first ``burn_in`` rows (after the random initial condition) are discarded so the
    returned window reflects the rule's asymptotic dynamics rather than startup transients.
    """
    if rng is None:
        rng = np.random.default_rng()
    table = rule_table(rule)
    if init is None:
        row = (rng.random(width) < init_density).astype(np.uint8)
    else:
        row = init.astype(np.uint8).copy()

    for _ in range(burn_in):
        row = step(row, table, boundary)

    rows = np.empty((n_rows, width), dtype=np.uint8)
    for t in range(n_rows):
        rows[t] = row
        row = step(row, table, boundary)
    return rows
