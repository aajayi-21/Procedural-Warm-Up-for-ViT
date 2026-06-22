"""Layerwise-transfer analysis — Stage-5 scaffold.

The parent paper finds warm-up gains concentrate in the *late* transformer layers. This
helper selects a contiguous block range from a stripped warm-up checkpoint so a downstream
run can transfer only those blocks (first-k / middle-k / final-k), reproducing the paper's
layerwise probe. Blocks not selected stay randomly initialized in the downstream model.
"""

from __future__ import annotations

import re

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def num_blocks(state_dict: dict) -> int:
    idxs = {int(m.group(1)) for k in state_dict if (m := _BLOCK_RE.match(k))}
    return (max(idxs) + 1) if idxs else 0


def select_blocks(state_dict: dict, which: str, k: int = 4, depth: int | None = None) -> dict:
    """Return a sub-state-dict keeping only blocks in the chosen range.

    ``which`` is one of ``first`` | ``middle`` | ``final`` | ``all``; ``k`` is the range
    width (ignored for ``all``). Non-block tensors (e.g. ``norm.*``) are always kept.
    """
    depth = depth or num_blocks(state_dict)
    if which == "all":
        keep = set(range(depth))
    elif which == "first":
        keep = set(range(0, k))
    elif which == "final":
        keep = set(range(depth - k, depth))
    elif which == "middle":
        start = (depth - k) // 2
        keep = set(range(start, start + k))
    else:
        raise ValueError(f"Unknown selection '{which}'")

    out: dict = {}
    for key, val in state_dict.items():
        m = _BLOCK_RE.match(key)
        if m is None:
            out[key] = val  # keep norm etc.
        elif int(m.group(1)) in keep:
            out[key] = val
    return out
