"""Cellular-automaton tokenization."""

import numpy as np

from procedural_warmup.data.ca import tokenize as tok


def test_binary_tokens_offset_by_specials():
    window = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    ids = tok.binary_tokens(window)
    np.testing.assert_array_equal(ids, window + tok.N_SPECIAL)
    assert ids.min() >= tok.N_SPECIAL
    assert tok.vocab_size_binary() == tok.N_SPECIAL + 2


def test_block_tokens_msb_first():
    # block_size=3, bits 1,0,1 -> 0b101 = 5, plus specials.
    window = np.array([[1, 0, 1, 0, 1, 1]], dtype=np.uint8)
    ids = tok.block_tokens(window, block_size=3)
    assert ids.shape == (1, 2)
    assert ids[0, 0] == 5 + tok.N_SPECIAL  # 101
    assert ids[0, 1] == 3 + tok.N_SPECIAL  # 011
    assert tok.vocab_size_block(3) == tok.N_SPECIAL + 8


def test_cells_per_token_and_required_vocab():
    assert tok.cells_per_token("binary", 7) == 1
    assert tok.cells_per_token("block", 7) == 7
    assert tok.required_vocab("binary", 7) == 4
    assert tok.required_vocab("block", 7) == tok.N_SPECIAL + 128
