"""IID single-step CA transduction sources (Phase-1 operator-vs-texture test)."""

import numpy as np
import torch

from procedural_warmup.config import load_config
from procedural_warmup.data import available_sources, build_source
from procedural_warmup.data.ca import tokenize as tok
from procedural_warmup.data.ca.eca import rule_table, step


def _cfg(source="ca_step", **ca_step):
    cfg = load_config(None)
    cfg.data.source = source
    cfg.vocab.K = 4
    cfg.dataset.n_samples = 64
    for k, v in ca_step.items():
        setattr(cfg.ca_step, k, v)
    return cfg


def test_sources_registered():
    assert {"ca_step", "iid_board"} <= set(available_sources())


def test_ca_step_true_target_is_rule_applied_to_input():
    cfg = _cfg(mode="true", rule=110, boundary="periodic")
    ds, _ = build_source(cfg)
    N = cfg.grid.H * cfg.grid.W
    table = rule_table(110)
    item = ds[0]
    assert item.shape == (2 * N,)
    x = (item[:N].numpy() - tok.N_SPECIAL).astype(np.uint8)
    y = item[N:].numpy() - tok.N_SPECIAL
    assert set(np.unique(x)).issubset({0, 1})
    np.testing.assert_array_equal(y, step(x, table, "periodic"))  # target == rule(input)


def test_ca_step_shuffled_breaks_the_causal_link():
    cfg = _cfg(mode="shuffled", rule=110, boundary="periodic")
    ds, _ = build_source(cfg)
    N = cfg.grid.H * cfg.grid.W
    table = rule_table(110)
    unrelated = 0
    for i in range(32):
        item = ds[i]
        x = (item[:N].numpy() - tok.N_SPECIAL).astype(np.uint8)
        y = item[N:].numpy() - tok.N_SPECIAL
        if not np.array_equal(y, step(x, table, "periodic")):
            unrelated += 1
    assert unrelated >= 31  # target = rule(z), z independent of x -> essentially never == rule(x)


def test_transduction_masking_is_full_mask_with_no_visible_target():
    cfg = _cfg(mode="true")
    ds, masking = build_source(cfg)
    N = cfg.grid.H * cfg.grid.W
    batch = torch.stack([ds[i] for i in range(8)])
    x, y, mask = masking(batch)
    assert x.shape == (8, N) and y.shape == (8, N)
    assert mask.dtype == torch.bool and bool(mask.all())  # loss on every position
    assert torch.equal(x, batch[:, :N]) and torch.equal(y, batch[:, N:])


def test_iid_board_floor_is_binary_single_frame():
    cfg = _cfg(source="iid_board")
    ds, _ = build_source(cfg)
    N = cfg.grid.H * cfg.grid.W
    item = ds[0]
    assert item.shape == (N,)
    assert set(np.unique(item.numpy())).issubset({2, 3})
