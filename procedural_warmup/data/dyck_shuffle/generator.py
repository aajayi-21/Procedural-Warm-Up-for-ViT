"""k-Dyck-Shuffle generator: balanced but not well-nested (crossing dependencies).

Same token layout as k-Dyck (opens ``[2, 2+k_open)``, type-matched closes
``[2+k_open, ...)``). Unlike Dyck, a close may target *any* currently-open type rather than
the most-recent one (non-LIFO), which produces crossing/interleaved structure. The sequence
is still balanced: every opened type is eventually closed within ``max_length``.
"""

from __future__ import annotations

import random

OPEN_BASE = 2


def dyck_shuffle_ids(
    k_open: int,
    k_close: int,
    max_length: int,
    open_prob: float = 0.6,
    rng: random.Random | None = None,
) -> list[int]:
    if rng is None:
        rng = random.Random()
    close_base = OPEN_BASE + k_open

    seq: list[int] = []
    open_types: list[int] = []  # multiset of currently-open types (order != nesting)
    while len(seq) < max_length:
        remaining = max_length - len(seq)
        if remaining <= len(open_types):
            # Must start closing to balance within the remaining capacity.
            t = open_types.pop(rng.randrange(len(open_types)))
            seq.append(close_base + t)
        elif not open_types:
            t = rng.randrange(k_open)
            open_types.append(t)
            seq.append(OPEN_BASE + t)
        else:
            can_open = remaining >= len(open_types) + 2
            if can_open and rng.random() < open_prob:
                t = rng.randrange(k_open)
                open_types.append(t)
                seq.append(OPEN_BASE + t)
            else:
                # Close ANY currently-open type -> crossing/interleaved dependencies.
                t = open_types.pop(rng.randrange(len(open_types)))
                seq.append(close_base + t)
    return seq
