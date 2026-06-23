"""Game of Life rule correctness and the next-state dataset."""

import numpy as np
import torch

from procedural_warmup.config import load_config
from procedural_warmup.data import available_sources, build_source
from procedural_warmup.data.ca.gol import GameOfLifeGrid, life_step


def test_gol_registered():
    assert "gol" in available_sources()


def test_block_still_life_is_stable():
    grid = np.zeros((6, 6), dtype=np.uint8)
    grid[2:4, 2:4] = 1  # 2x2 block — a still life
    np.testing.assert_array_equal(life_step(grid), grid)


def test_blinker_has_period_two():
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, 2:5] = 1  # 3-cell blinker
    one = life_step(grid)
    assert not np.array_equal(one, grid)
    np.testing.assert_array_equal(life_step(one), grid)  # period 2


def _cfg():
    return load_config("procedural_warmup/config/files/gol.yaml")


def test_gol_block_dataset_shape_and_range():
    cfg = _cfg()  # default: block tokenization (K=130)
    assert cfg.ca.tokenize.mode == "block"
    sample = GameOfLifeGrid(cfg)[0]
    assert sample.shape == (cfg.N,)
    assert int(sample.min()) >= 2 and int(sample.max()) < cfg.vocab.K


def test_gol_dataset_is_a_genuine_next_state_pair():
    cfg = _cfg()
    cfg.ca.tokenize.mode = "binary"  # binary makes token<->cell trivially checkable
    cfg.vocab.K = 4
    grid = GameOfLifeGrid(cfg)[0].numpy().reshape(cfg.grid.H, cfg.grid.W) - 2
    fH = cfg.grid.H // 2
    t0, t1 = grid[:fH], grid[fH:]
    expected = t0.copy()
    for _ in range(cfg.ca.gol_steps):
        expected = life_step(expected)
    np.testing.assert_array_equal(t1, expected)  # bottom frame == Life(top frame)


def test_gol_forward_masking_targets_future_frame():
    cfg = _cfg()
    _, masking = build_source(cfg)
    batch = torch.full((4, cfg.N), 2, dtype=torch.long)
    masked, targets, mask = masking(batch)
    grid_mask = mask[0].reshape(cfg.grid.H, cfg.grid.W)
    fH = cfg.grid.H // 2
    assert grid_mask[:fH].sum() == 0   # state t visible
    assert grid_mask[fH:].all()        # state t+1 fully masked


def test_gol_rejects_misaligned_forward_masking():
    cfg = _cfg()
    cfg.masking.forward_rows = 5  # != grid.H//2
    try:
        build_source(cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
