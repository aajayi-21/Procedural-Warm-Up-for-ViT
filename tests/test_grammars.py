"""WW and k-Dyck-Shuffle generators, masking, and registration."""

import torch

from procedural_warmup.data import available_sources, build_source
from procedural_warmup.data.dyck_shuffle.generator import dyck_shuffle_ids
from procedural_warmup.data.ww.generator import ww_ids
from procedural_warmup.data.ww.masking import WWMasking


def test_sources_registered():
    for name in ("ww", "dyck_shuffle"):
        assert name in available_sources()


def test_ww_is_a_copy():
    seq = ww_ids(n_symbols=8, length=10)
    assert len(seq) == 10
    assert seq[:5] == seq[5:]  # second half is an exact copy of the first
    assert all(2 <= t < 10 for t in seq)


def test_ww_masking_first_half_only(base_cfg):
    base_cfg.data.source = "ww"
    masking = WWMasking(base_cfg)
    original = torch.full((8, base_cfg.N), 5, dtype=torch.long)
    masked, targets, mask = masking(original)
    assert torch.equal(targets, original)
    assert (masked[mask] == base_cfg.vocab.MASK_ID).all()
    half = base_cfg.N // 2
    grid = mask.reshape(8, base_cfg.N)
    assert grid[:, half:].sum() == 0  # nothing masked in the second half (the copy)


def _per_type_balanced(seq, k_open):
    """Every opened type is closed exactly once and never closed before opened."""
    open_ct = [0] * k_open
    close_ct = [0] * k_open
    for t in seq:
        if 2 <= t < 2 + k_open:
            open_ct[t - 2] += 1
        elif 2 + k_open <= t < 2 + 2 * k_open:
            ty = t - (2 + k_open)
            close_ct[ty] += 1
            if close_ct[ty] > open_ct[ty]:
                return False
    return open_ct == close_ct


def _is_well_nested(seq, k_open):
    stack = []
    for t in seq:
        if 2 <= t < 2 + k_open:
            stack.append(t - 2)
        elif 2 + k_open <= t < 2 + 2 * k_open:
            if not stack or stack[-1] != t - (2 + k_open):
                return False
            stack.pop()
    return not stack


def test_dyck_shuffle_balanced_and_crossing():
    import random
    rng = random.Random(0)
    crossing_seen = 0
    for _ in range(50):
        seq = dyck_shuffle_ids(8, 8, 64, open_prob=0.6, rng=rng)
        assert len(seq) == 64
        assert _per_type_balanced(seq, 8)  # every open closed, none closed early
    # Over many samples at least one should be non-well-nested (genuine crossing).
    for _ in range(100):
        seq = dyck_shuffle_ids(8, 8, 64, open_prob=0.6, rng=rng)
        if not _is_well_nested(seq, 8):
            crossing_seen += 1
    assert crossing_seen > 0, "dyck_shuffle never produced crossing dependencies"


def test_build_grammar_sources(base_cfg):
    for src in ("ww", "dyck_shuffle"):
        base_cfg.data.source = src
        dataset, masking = build_source(base_cfg)
        sample = dataset[0]
        assert sample.shape == (base_cfg.N,)
        assert int(sample.min()) >= 2 and int(sample.max()) < base_cfg.vocab.K
