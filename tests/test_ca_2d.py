"""2-D-retaining CA warm-up: block2d masking, single-step next-state, sincos2d geometry.

These cover the H3 ("don't flatten away the 2-D structure") experiments: the block2d masking
mode, the forward_rows=1 next-state-vector objective, and that the sincos2d positional
embedding encodes CA's (time, space) grid. See results/reports/ca-2d-spacetime/report.md.
"""

import pytest
import torch

from procedural_warmup.config import load_config
from procedural_warmup.data.ca.masking import CAMasking
from procedural_warmup.model.embeddings import Frozen2DSinCosPositionalEmbedding


def _triple_invariants(masked, targets, mask, original, mask_id):
    assert torch.equal(targets, original)
    assert (masked[mask] == mask_id).all()
    assert torch.equal(masked[~mask], original[~mask])


def test_block2d_masks_contiguous_rectangles(base_cfg):
    base_cfg.masking.mode = "block2d"
    base_cfg.masking.mask_ratio = 0.5
    base_cfg.masking.block_h = 4
    base_cfg.masking.block_w = 4
    masking = CAMasking(base_cfg)
    original = torch.full((8, base_cfg.N), 3, dtype=torch.long)
    masked, targets, mask = masking(original)
    _triple_invariants(masked, targets, mask, original, base_cfg.vocab.MASK_ID)

    cov = mask.float().mean().item()
    # inclusion-exclusion block count targets mask_ratio; overlaps keep realized coverage near
    # (and not far above) it. ~0.45 in practice for 4x4 blocks on a 14x14 grid at ratio 0.5.
    assert 0.30 < cov <= 0.55

    H, W, bh, bw = base_cfg.grid.H, base_cfg.grid.W, 4, 4
    grid = mask.view(8, H, W)
    for b in range(8):
        # We stamp >= 1 full block, so a fully-masked bh x bw window must exist (contiguity).
        found = any(
            bool(grid[b, t : t + bh, l : l + bw].all())
            for t in range(H - bh + 1)
            for l in range(W - bw + 1)
        )
        assert found, "block2d should leave at least one fully-masked rectangle"


def test_forward_single_step_masks_only_next_state(base_cfg):
    base_cfg.masking.mode = "forward"
    base_cfg.masking.forward_rows = 1  # predict the entire next state vector from the past
    masking = CAMasking(base_cfg)
    original = torch.full((8, base_cfg.N), 2, dtype=torch.long)
    masked, targets, mask = masking(original)
    _triple_invariants(masked, targets, mask, original, base_cfg.vocab.MASK_ID)
    grid = mask[0].reshape(base_cfg.grid.H, base_cfg.grid.W)
    assert grid[:-1].sum() == 0  # all but the last time-row visible
    assert grid[-1:].all()       # the whole final state vector is the target
    assert int(mask.sum().item()) == base_cfg.grid.W * original.shape[0]


def test_block2d_rejects_oversized_block(base_cfg):
    base_cfg.masking.mode = "block2d"
    base_cfg.masking.block_h = base_cfg.grid.H + 1
    with pytest.raises(ValueError):
        CAMasking(base_cfg)


def test_sincos2d_encodes_time_and_space():
    H, W, d = 14, 14, 192
    pe = Frozen2DSinCosPositionalEmbedding(H * W, d, H, W)
    w = pe.emb.weight  # (N, d); first half encodes the row(=time), second half the col(=space)
    half = d // 2
    time_half = w[:, :half].view(H, W, half)
    space_half = w[:, half:].view(H, W, half)
    # Within one time-row the time-half is constant across space ...
    assert torch.allclose(time_half, time_half[:, :1, :].expand(-1, W, -1), atol=1e-6)
    # ... and the space-half is constant down a column across time.
    assert torch.allclose(space_half, space_half[:1, :, :].expand(H, -1, -1), atol=1e-6)
    # Different time-rows / space-cols genuinely differ.
    assert not torch.allclose(time_half[0, 0], time_half[1, 0], atol=1e-6)
    assert not torch.allclose(space_half[0, 0], space_half[0, 1], atol=1e-6)


def test_ca_sincos2d_model_builds_and_runs():
    cfg = load_config(None)
    cfg.vocab.K = 4  # CA binary
    cfg.model.pos_embed = "sincos2d"
    from procedural_warmup.model import build_model

    model, mlm_head = build_model(cfg)
    model.eval()
    batch = torch.randint(2, 4, (2, cfg.grid.H * cfg.grid.W))
    logits = mlm_head(model.forward_tokens(batch))
    assert logits.shape == (2, cfg.grid.H * cfg.grid.W, cfg.vocab.K)


def test_gol_step_true_target_is_one_life_step(base_cfg):
    from procedural_warmup.data import build_source
    from procedural_warmup.data.ca.gol import life_step

    base_cfg.data.source = "gol_step"
    base_cfg.vocab.K = 4
    base_cfg.ca_step.mode = "true"
    dataset, masking = build_source(base_cfg)
    H, W, N = base_cfg.grid.H, base_cfg.grid.W, base_cfg.N
    pair = dataset[0].unsqueeze(0)  # (1, 2N)
    x, y, mask = masking(pair)
    assert x.shape == y.shape == (1, N) and mask.all()  # full-mask transduction
    # The target is exactly one Life step on the (row-major) input board.
    xb = (x[0].view(H, W) - 2).numpy().astype("uint8")
    yb = (y[0].view(H, W) - 2).numpy().astype("uint8")
    assert (life_step(xb) == yb).all()


def test_gol_step_shuffled_decouples_target(base_cfg):
    from procedural_warmup.data import build_source
    from procedural_warmup.data.ca.gol import life_step

    base_cfg.data.source = "gol_step"
    base_cfg.vocab.K = 4
    base_cfg.ca_step.mode = "shuffled"
    dataset, masking = build_source(base_cfg)
    H, W = base_cfg.grid.H, base_cfg.grid.W
    # Over several samples, the target should NOT generally equal life_step(input).
    matches = 0
    for _ in range(8):
        x, y, _ = masking(dataset[0].unsqueeze(0))
        xb = (x[0].view(H, W) - 2).numpy().astype("uint8")
        yb = (y[0].view(H, W) - 2).numpy().astype("uint8")
        matches += int((life_step(xb) == yb).all())
    assert matches == 0  # shuffled target is causally unrelated to the input board


def test_ring_pos_embedding_is_periodic_and_local():
    from procedural_warmup.model.embeddings import Frozen1DRingPositionalEmbedding

    N, d = 196, 192
    w = Frozen1DRingPositionalEmbedding(N, d).emb.weight  # (N, d), unit-scaled rows
    w = torch.nn.functional.normalize(w, dim=1)
    sim = lambda a, b: float((w[a] * w[b]).sum())
    # Adjacent ring positions are more similar than far ones ...
    assert sim(0, 1) > sim(0, N // 2)
    # ... and the ring wraps: position 0 is adjacent to N-1 (as similar as 0 vs 1).
    assert sim(0, N - 1) > sim(0, N // 2)


def test_ca_step_sincos1d_model_builds_and_runs():
    cfg = load_config(None)
    cfg.vocab.K = 4
    cfg.model.pos_embed = "sincos1d"
    from procedural_warmup.model import build_model

    model, mlm_head = build_model(cfg)
    model.eval()
    batch = torch.randint(2, 4, (2, cfg.grid.H * cfg.grid.W))
    assert mlm_head(model.forward_tokens(batch)).shape == (2, cfg.N, cfg.vocab.K)
