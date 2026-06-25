"""Spatial-Dyck renderer: Hilbert layout, multiset preservation, close-only maskability."""

import numpy as np
import torch

from procedural_warmup.config import load_config
from procedural_warmup.data import available_sources, build_source
from procedural_warmup.data.dyck.spatial import hilbert_order


def _cfg(mode):
    cfg = load_config(None)
    cfg.data.source = "spatial_dyck"
    cfg.vocab.K = 130
    cfg.dataset.n_samples = 64
    cfg.spatial_dyck.mode = mode
    return cfg


def _valid_rowmajor_dyck(x) -> bool:
    stack = []
    for t in x:
        if 2 <= t < 66:
            stack.append(int(t) - 2)
        elif 66 <= t < 130:
            if not stack or stack.pop() != int(t) - 66:
                return False
    return len(stack) == 0


def test_registered():
    assert "spatial_dyck" in available_sources()


def test_hilbert_order_is_a_permutation():
    o = hilbert_order(14)
    assert len(o) == 196
    assert len({(r, c) for r, c in o}) == 196
    assert all(0 <= r < 14 and 0 <= c < 14 for r, c in o)
    assert o != [(r, c) for r in range(14) for c in range(14)]  # not row-major identity


def test_hilbert_locality_unit_steps():
    # On a power-of-two grid every consecutive curve cell is a grid neighbour (the locality
    # property that maps contiguous spans -> compact regions).
    o = hilbert_order(16)
    steps = [abs(o[i][0] - o[i + 1][0]) + abs(o[i][1] - o[i + 1][1]) for i in range(len(o) - 1)]
    assert all(s == 1 for s in steps)


def test_modes_preserve_multiset_and_fill_grid():
    for mode in ("1d", "nested", "permuted"):
        ds, _ = build_source(_cfg(mode))
        x = ds[0].numpy()
        assert x.shape == (196,)
        assert (x == 0).sum() == 0  # grid fully filled, no PAD
        opens = ((x >= 2) & (x < 66)).sum()
        closes = ((x >= 66) & (x < 130)).sum()
        assert opens == closes == 98  # balanced, same token budget as the 1-D baseline


def test_1d_is_valid_dyck_but_nested_is_scrambled():
    x1 = build_source(_cfg("1d"))[0][0].numpy()
    assert _valid_rowmajor_dyck(x1)  # 1d == baseline: a proper left-to-right Dyck string
    # nested relocates tokens by the Hilbert curve -> almost never a valid row-major Dyck
    nested = build_source(_cfg("nested"))[0]
    assert sum(_valid_rowmajor_dyck(nested[i].numpy()) for i in range(8)) == 0


def test_closers_are_maskable_in_2d():
    from procedural_warmup.data.dyck.masking import CloseOnlyMasking

    cfg = _cfg("nested")
    ds, masking = build_source(cfg)
    batch = torch.stack([ds[i] for i in range(8)])
    masked_input, target, mask = masking(batch)
    assert mask.any()  # some closers got masked, wherever they landed in 2-D
    # every masked position's target is a closing bracket id
    assert bool(((target[mask] >= 66) & (target[mask] < 130)).all())
