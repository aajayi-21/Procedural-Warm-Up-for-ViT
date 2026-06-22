"""k-Dyck sequence generator (context-free formal grammar).

Reproduces the reference repo's stack-based generator. Token layout (shared reserved ids):

    0                    -> PAD
    1                    -> MASK
    [2, 2 + k_open)      -> opening brackets (one id per bracket *type*)
    [2 + k_open, 2 + k_open + k_close) -> closing brackets (type-matched to opens)

A valid k-Dyck string is properly nested: every open of type ``t`` is matched by a close
of the same type. The generator guarantees a balanced sequence of exactly ``max_length``
tokens (``max_length`` should be even) by flushing the stack once the remaining capacity
can only just accommodate the open brackets.
"""

from __future__ import annotations

import random

OPEN_BASE = 2


def dyck_ids(
    k_open: int,
    k_close: int,
    max_length: int,
    open_prob: float = 0.6,
    min_pairs: int = 1,
    rng: random.Random | None = None,
) -> list[int]:
    """Generate one balanced k-Dyck sequence as a list of token ids of length ``max_length``."""
    if rng is None:
        rng = random.Random()
    close_base = OPEN_BASE + k_open

    seq: list[int] = []
    stack: list[int] = []  # bracket-type indices currently open
    while len(seq) < max_length:
        remaining = max_length - len(seq)
        if remaining <= len(stack):
            # Must close now to flush the stack within the remaining capacity.
            t = stack.pop()
            seq.append(close_base + t)
        elif not stack:
            # Cannot close an empty stack -> must open.
            t = rng.randrange(k_open)
            stack.append(t)
            seq.append(OPEN_BASE + t)
        else:
            can_open = remaining >= len(stack) + 2  # leave room to close everything later
            if can_open and rng.random() < open_prob:
                t = rng.randrange(k_open)
                stack.append(t)
                seq.append(OPEN_BASE + t)
            else:
                t = stack.pop()
                seq.append(close_base + t)

    # min_pairs is satisfied by construction for any reasonable (even) max_length >= 2.
    return seq
