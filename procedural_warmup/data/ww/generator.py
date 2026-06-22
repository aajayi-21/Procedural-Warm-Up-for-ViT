"""WW generator: sample a random substring and concatenate its exact copy.

Token layout: ids 0/1 are PAD/MASK; symbols use ids ``[2, 2 + n_symbols)``. A length-N
sequence is ``base + base`` where ``base`` has ``N // 2`` random symbols (N is even = 196).
"""

from __future__ import annotations

import random


def ww_ids(n_symbols: int, length: int, rng: random.Random | None = None) -> list[int]:
    if rng is None:
        rng = random.Random()
    half = length // 2
    base = [2 + rng.randrange(n_symbols) for _ in range(half)]
    seq = base + base
    return seq[:length]
