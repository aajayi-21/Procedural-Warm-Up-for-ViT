"""Game of Life rule correctness and the next-state dataset."""

import numpy as np
import torch

from procedural_warmup.config import load_config
from procedural_warmup.data import available_sources, build_source
from procedural_warmup.data.ca.gol import GameOfLifeGrid, life_step


def test_gol_registered():
    assert "gol" in available_sources()


def test_block_still_life_is_stable():
    # A 2x2 block is a still life: unchanged by a Life step.
    grid = np.zeros((6, 6), dtype=np.uint8)
    grid[2:4, 2:4] = 1
    np.testing.assert_array_equal(life_step(grid), grid)


def test_blinker_has_period_two():
    # A 3-cell row (blinker) oscillates with period 2.
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, 2:5] = 1
    one = life_step(grid)
    assert not np.array_equal(one, grid)            # it changed (vertical <-> horizontal)
    np.testing.assert_array_equal(life_step(one), grid)  # back to start after 2 steps


def _cfg():
    return load_config("procedural_warmup/config/files/gol.yaml")


def test_gol_dataset_is_a_genuine_next_state_pair():
    cfg = _cfg()
    cfg.ca.tokenize.mode = "binary"  # binary makes the token<->cell mapping trivially checkable
    cfg.vocab.K = 4
    ds = GameOfLifeGrid(cfg)
    sample = ds[0]
    assert sample.shape == (cfg.N,)
    grid = sample.numpy().reshape(cfg.grid.H, cfg.grid.W) - 2  # undo binary token offset
    fH = cfg.grid.H // 2
    t0, t1 = grid[:fH], grid[fH:]
    expected = t0.copy()
    for _ in range(cfg.ca.gol_steps):
        expected = life_step(expected)
    np.testing.assert_array_equal(t1, expected)  # bottom frame == Life(top frame)


def test_gol_block_mode_shape_and_range():
    cfg = _cfg()  # default config uses block tokenization (K=130)
    assert cfg.ca.tokenize.mode == "block"
    ds = GameOfLifeGrid(cfg)
    sample = ds[0]
    assert sample.shape == (cfg.N,)
    assert int(sample.min()) >= 2 and int(sample.max()) < cfg.vocab.K


def test_gol_forward_masking_targets_future_frame():
    cfg = _cfg()
    _, masking = build_source(cfg)
    batch = torch.full((4, cfg.N), 2, dtype=torch.long)
    masked, targets, mask = masking(batch)
    grid_mask = mask[0].reshape(cfg.grid.H, cfg.grid.W)
    fH = cfg.grid.H // 2
    assert grid_mask[:fH].sum() == 0     # top (state t) is visible
    assert grid_mask[fH:].all()          # bottom (state t+1) is fully masked


def test_gol_rejects_mismatched_masking():
    cfg = _cfg()
    cfg.masking.mode = "random"          # wrong for GoL next-state
    try:
        build_source(cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
