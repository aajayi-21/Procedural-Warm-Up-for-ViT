"""Masking strategies satisfy the (masked_input, targets, mask) contract."""

import torch

from procedural_warmup.data.ca.masking import CAMasking
from procedural_warmup.data.dyck.masking import CloseOnlyMasking


def _triple_invariants(masked, targets, mask, original, mask_id):
    # targets are the originals unchanged; masked positions hold MASK_ID; others unchanged.
    assert torch.equal(targets, original)
    assert (masked[mask] == mask_id).all()
    assert torch.equal(masked[~mask], original[~mask])


def test_ca_random_masking(base_cfg):
    base_cfg.masking.mode = "random"
    base_cfg.masking.mask_ratio = 0.5
    masking = CAMasking(base_cfg)
    original = torch.full((16, base_cfg.N), 3, dtype=torch.long)  # state-1 tokens
    masked, targets, mask = masking(original)
    _triple_invariants(masked, targets, mask, original, base_cfg.vocab.MASK_ID)
    frac = mask.float().mean().item()
    assert 0.4 < frac < 0.6  # ~0.5 in expectation


def test_ca_forward_masking_masks_trailing_rows(base_cfg):
    base_cfg.masking.mode = "forward"
    base_cfg.masking.forward_rows = 4
    masking = CAMasking(base_cfg)
    original = torch.full((8, base_cfg.N), 2, dtype=torch.long)
    masked, targets, mask = masking(original)
    _triple_invariants(masked, targets, mask, original, base_cfg.vocab.MASK_ID)
    # Exactly the last 4 of 14 rows are masked, identically across the batch.
    grid = mask[0].reshape(base_cfg.grid.H, base_cfg.grid.W)
    assert grid[:-4].sum() == 0
    assert grid[-4:].all()
    # exact masked-cell count: last 4 rows * W cols, identical across the batch.
    assert int(mask.sum().item()) == 4 * base_cfg.grid.W * original.shape[0]


def test_dyck_close_only_masking(base_cfg):
    base_cfg.masking.mask_ratio = 1.0  # mask every eligible (closing) token
    masking = CloseOnlyMasking(base_cfg)
    close_base = 2 + base_cfg.dyck.k_open
    # one open (id 2), one close (id close_base) repeated.
    original = torch.tensor([[2, close_base, 2, close_base]], dtype=torch.long)
    masked, targets, mask = masking(original)
    _triple_invariants(masked, targets, mask, original, base_cfg.vocab.MASK_ID)
    # Only the closing-bracket positions are masked.
    assert mask.tolist() == [[False, True, False, True]]
