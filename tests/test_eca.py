"""Correctness of the elementary CA simulator."""

import numpy as np
import pytest

from procedural_warmup.data.ca.eca import rule_table, simulate_spacetime, step


def test_rule_table_known_values():
    assert list(rule_table(30)) == [0, 1, 1, 1, 1, 0, 0, 0]
    assert list(rule_table(90)) == [0, 1, 0, 1, 1, 0, 1, 0]
    assert list(rule_table(110)) == [0, 1, 1, 1, 0, 1, 1, 0]
    assert list(rule_table(0)) == [0] * 8
    assert list(rule_table(255)) == [1] * 8


def test_rule_table_out_of_range():
    with pytest.raises(ValueError):
        rule_table(256)


def test_rule90_is_xor_of_neighbors():
    rng = np.random.default_rng(0)
    row = (rng.random(64) < 0.5).astype(np.uint8)
    expected = np.roll(row, 1) ^ np.roll(row, -1)  # left XOR right (periodic)
    np.testing.assert_array_equal(step(row, rule_table(90)), expected)


def test_rule0_and_rule255_extremes():
    row = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    np.testing.assert_array_equal(step(row, rule_table(0)), np.zeros_like(row))
    np.testing.assert_array_equal(step(row, rule_table(255)), np.ones_like(row))


def test_rule90_sierpinski_from_single_seed():
    # Rule 90 from a single live cell on a zero background draws Pascal's triangle mod 2.
    width = 17
    init = np.zeros(width, dtype=np.uint8)
    init[width // 2] = 1
    st = simulate_spacetime(90, width=width, n_rows=3, boundary="zero", init=init)
    np.testing.assert_array_equal(st[0], init)
    # Row 1: the two cells adjacent to the seed are live.
    expected_row1 = np.zeros(width, dtype=np.uint8)
    expected_row1[[width // 2 - 1, width // 2 + 1]] = 1
    np.testing.assert_array_equal(st[1], expected_row1)


def test_simulate_shape_and_determinism():
    a = simulate_spacetime(110, width=40, n_rows=14, burn_in=8,
                           rng=np.random.default_rng(123))
    b = simulate_spacetime(110, width=40, n_rows=14, burn_in=8,
                           rng=np.random.default_rng(123))
    assert a.shape == (14, 40)
    assert a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)  # same seed -> identical
