"""DW_k 2D Dyck source: sampler validity, masking contract, determinacy audit, controls.

Regression anchors (both verified verbatim against docs/2307.16522.pdf):
- Definition 2 forbids flat-interior accretion, so the nested 2x4 ``[a a b b; c c d d]``
  is the Theorem-3 witness of DN_1 \\ DW_1 — Tier 1 must accept it (it is in DN ⊆ DC)
  and Tier 2 must reject it.
- Partially-overlapping bounding boxes are legal in DW (p. 7), so the plus-sign 4x4
  accretion with non-empty border words must pass both tiers.
"""

import random

import numpy as np
import pytest
import torch

from procedural_warmup.data import build_source
from procedural_warmup.data.base import MaskingStrategy, ProceduralDataset
from procedural_warmup.data.dyck2d.alphabet import (
    Role,
    corner_id,
    d_base,
    index_of,
    role_of,
    vocab_size,
)
from procedural_warmup.data.dyck2d.audit import audit_mask, brute_force_forced
from procedural_warmup.data.dyck2d.controls import marginal_shuffle
from procedural_warmup.data.dyck2d.dataset import Dyck2DGrid
from procedural_warmup.data.dyck2d.generator import dw_picture
from procedural_warmup.data.dyck2d.masking import CornerCloseOnlyMasking
from procedural_warmup.data.dyck2d.validate import (
    MembershipError,
    check_picture,
    is_dw_guillotine,
    recover_rectangles,
)


def _dyck2d_cfg(cfg, **overrides):
    cfg.data.source = "dyck2d"
    for key, val in overrides.items():
        setattr(cfg.dyck2d, key, val)
    return cfg


def _witness_pictures(k=1):
    a, b, c, d = (corner_id(r, 0, k) for r in (Role.A, Role.B, Role.C, Role.D))
    nested_2x4 = np.array([[a, a, b, b], [c, c, d, d]])  # Thm-3 witness: DN \ DW
    tiled_2x4 = np.array([[a, b, a, b], [c, d, c, d]])  # in DW (two quadruples)
    # A 4x4 accretion with non-empty border words: border boxes overlap in a plus sign.
    plus_sign = np.array(
        [
            [a, a, b, b],
            [a, a, b, b],
            [c, c, d, d],
            [c, c, d, d],
        ]
    )
    return nested_2x4, tiled_2x4, plus_sign


# ------------------------------------------------------------------------------------
# alphabet
# ------------------------------------------------------------------------------------


def test_alphabet_roundtrip():
    for k in (2, 32):
        seen = set()
        for role in Role:
            for i in range(k):
                tid = corner_id(role, i, k)
                assert role_of(tid, k) == role and index_of(tid, k) == i
                seen.add(tid)
        assert seen == set(range(2, 2 + 4 * k))
        assert vocab_size(k) == 4 * k + 2
        assert d_base(k) == 2 + 3 * k
    with pytest.raises(ValueError):
        role_of(0, 32)  # PAD is not a content symbol


# ------------------------------------------------------------------------------------
# sampler
# ------------------------------------------------------------------------------------


def test_sampler_exact_size_and_no_rejection():
    rng = random.Random(0)
    for m, n in [(2, 2), (2, 8), (8, 2), (4, 4), (6, 10), (14, 14)]:
        grid, rects = dw_picture(m, n, 32, rng=rng)
        assert grid.shape == (m, n)
        assert len(rects) == m * n // 4  # every cell is a corner of exactly one quadruple
        assert grid.min() >= 2 and grid.max() < 2 + 4 * 32
    for bad in [(3, 4), (4, 7), (0, 4), (1, 1)]:
        with pytest.raises(ValueError):
            dw_picture(*bad, 32)


@pytest.mark.parametrize("size", [(2, 6), (4, 4), (6, 4), (6, 6)])
def test_sampler_validity_small(size):
    rng = random.Random(1)
    for _ in range(200):
        grid, _ = dw_picture(*size, k=3, rng=rng)
        check_picture(grid, 3)  # Tier 1 (raises on failure)
        assert is_dw_guillotine(grid, 3)  # Tier 2 certifies true DW membership


def test_sampler_validity_full_size():
    rng = random.Random(2)
    for _ in range(100):
        grid, rects = dw_picture(14, 14, 32, rng=rng)
        recovered = recover_rectangles(grid, 32)
        assert is_dw_guillotine(grid, 32)
        # Independent Tier-1 recovery agrees with the generator's own rectangle list.
        assert sorted((r.r1, r.c1, r.r2, r.c2, r.index) for r in recovered) == sorted(
            (r.r1, r.c1, r.r2, r.c2, r.index) for r in rects
        )


def test_rectangle_stats_sane():
    rng = random.Random(3)
    spans, depths, elig = [], [], []
    for _ in range(100):
        grid, rects = dw_picture(14, 14, 32, rng=rng)
        boxes = [(r.r1, r.c1, r.r2, r.c2) for r in rects]
        for r in rects:
            assert r.row_span % 2 == 1 and r.col_span % 2 == 1  # spans are odd
            spans.append(max(r.row_span, r.col_span))
            depths.append(
                sum(
                    1
                    for (s1, t1, s2, t2) in boxes
                    if s1 <= r.r1 and t1 <= r.c1 and s2 >= r.r2 and t2 >= r.c2
                    and (s1, t1, s2, t2) != (r.r1, r.c1, r.r2, r.c2)
                )
            )
        elig.append(sum(1 for r in rects if max(r.row_span, r.col_span) >= 2))
    assert min(spans) == 1 and max(spans) > 1  # both 2x2 and larger rects occur
    # Nesting exists. Depth counts *all* containing boxes (frames + border-word pairs
    # that span the cell), so it can grow well past the accretion-level count; the only
    # hard bound is the number of other rectangles in the picture.
    assert 1 <= max(depths) < 14 * 14 // 4
    assert np.mean(elig) > 0  # the default filter leaves maskable cells


# ------------------------------------------------------------------------------------
# validator
# ------------------------------------------------------------------------------------


def test_validator_rejects_corruptions():
    grid, rects = dw_picture(14, 14, 32, rng=random.Random(4))
    k = 32

    swapped = grid.copy()  # swap two cells with different symbols
    (r1, c1), (r2, c2) = (0, 0), rects[-1].d_pos
    swapped[r1, c1], swapped[r2, c2] = grid[r2, c2], grid[r1, c1]
    changed_index = grid.copy()  # change one d's quadruple index
    dr, dc = rects[0].d_pos
    changed_index[dr, dc] = corner_id(Role.D, (rects[0].index + 1) % k, k)
    changed_role = grid.copy()  # overwrite a b-corner with an a
    br, bc = rects[0].r1, rects[0].c2
    changed_role[br, bc] = corner_id(Role.A, rects[0].index, k)
    padded = grid.copy()
    padded[0, 0] = 0  # PAD is illegal in a complete picture

    for corrupt in (swapped, changed_index, changed_role, padded):
        with pytest.raises(MembershipError):
            check_picture(corrupt, k)


def test_validator_correctness_witnesses():
    nested, tiled, plus = _witness_pictures(k=1)
    # Thm-3 witness: in DN \ DW -> Tier 1 accepts, Tier 2 rejects.
    check_picture(nested, 1)
    assert not is_dw_guillotine(nested, 1)
    # Tiled 2x4 is in DW.
    check_picture(tiled, 1)
    assert is_dw_guillotine(tiled, 1)
    # Plus-sign accretion (overlapping border-word boxes) is in DW: both tiers accept.
    check_picture(plus, 1)
    assert is_dw_guillotine(plus, 1)
    # Its Tier-1 recovery has partially-overlapping boxes — the paper's third case.
    rects = recover_rectangles(plus, 1)
    boxes = {(r.r1, r.c1, r.r2, r.c2) for r in rects}
    assert (0, 0, 3, 3) in boxes and (0, 1, 3, 2) in boxes and (1, 0, 2, 3) in boxes


# ------------------------------------------------------------------------------------
# dataset + masking
# ------------------------------------------------------------------------------------


def test_dataset_sample_contract(base_cfg):
    cfg = _dyck2d_cfg(base_cfg)
    ds = Dyck2DGrid(cfg)
    sample = ds[0]
    assert isinstance(ds, ProceduralDataset)
    assert sample.shape == (2, cfg.N) and sample.dtype == torch.long
    ids, elig = sample[0], sample[1]
    assert set(elig.tolist()) <= {0, 1}
    lo = d_base(cfg.dyck2d.k)
    assert ((ids[elig.bool()] >= lo) & (ids[elig.bool()] < lo + cfg.dyck2d.k)).all()
    # Eligibility agrees with an independent re-parse of the picture.
    grid = ids.reshape(cfg.grid.H, cfg.grid.W).numpy()
    expect = np.zeros((cfg.grid.H, cfg.grid.W), dtype=np.int64)
    for r in recover_rectangles(grid, cfg.dyck2d.k):
        if max(r.row_span, r.col_span) >= cfg.dyck2d.min_match_distance:
            expect[r.d_pos] = 1
    assert np.array_equal(elig.reshape(cfg.grid.H, cfg.grid.W).numpy(), expect)


def test_dataset_vocab_guard(base_cfg):
    cfg = _dyck2d_cfg(base_cfg)
    cfg.vocab.K = 100  # < 4*32 + 2
    with pytest.raises(AssertionError):
        Dyck2DGrid(cfg)


def test_masking_triple_invariants(base_cfg):
    cfg = _dyck2d_cfg(base_cfg)
    ds, masking = build_source(cfg)
    batch = torch.stack([ds[i] for i in range(16)])
    assert batch.shape == (16, 2, cfg.N)
    masked, targets, mask = masking(batch)
    assert masked.shape == targets.shape == mask.shape == (16, cfg.N)
    assert torch.equal(targets, batch[:, 0])
    assert (masked[mask] == cfg.vocab.MASK_ID).all()
    assert torch.equal(masked[~mask], batch[:, 0][~mask])


def test_masking_only_eligible_d(base_cfg):
    cfg = _dyck2d_cfg(base_cfg)
    cfg.masking.mask_ratio = 1.0
    ds, masking = build_source(cfg)
    batch = torch.stack([ds[i] for i in range(8)])
    _masked, targets, mask = masking(batch)
    assert torch.equal(mask, batch[:, 1].bool())  # exactly the eligible set
    lo = d_base(cfg.dyck2d.k)
    assert ((targets[mask] >= lo) & (targets[mask] < lo + cfg.dyck2d.k)).all()


def test_masking_filter_respected(base_cfg):
    cfg = _dyck2d_cfg(base_cfg, min_match_distance=99)
    cfg.masking.mask_ratio = 1.0
    ds, masking = build_source(cfg)
    batch = torch.stack([ds[i] for i in range(4)])
    assert masking(batch)[2].sum() == 0  # nothing eligible

    cfg2 = _dyck2d_cfg(base_cfg, min_match_distance=1)
    ds2, masking2 = build_source(cfg2)
    batch2 = torch.stack([ds2[i] for i in range(4)])
    _m, targets, mask = masking2(batch2)
    lo = d_base(cfg2.dyck2d.k)
    n_d = ((batch2[:, 0] >= lo) & (batch2[:, 0] < lo + cfg2.dyck2d.k)).sum()
    assert mask.sum() == n_d  # filter disabled: every d is eligible (ratio 1.0)


def test_masking_rejects_plain_ids(base_cfg):
    cfg = _dyck2d_cfg(base_cfg)
    masking = CornerCloseOnlyMasking(cfg)
    with pytest.raises(ValueError, match="two-channel"):
        masking(torch.zeros(4, cfg.N, dtype=torch.long))


def test_mask_ratio_expectation(base_cfg):
    cfg = _dyck2d_cfg(base_cfg)
    cfg.masking.mask_ratio = 0.5
    ds, masking = build_source(cfg)
    batch = torch.stack([ds[i] for i in range(64)])
    _m, _t, mask = masking(batch)
    frac = mask.sum().item() / batch[:, 1].sum().item()
    assert 0.4 < frac < 0.6  # ~0.5 of the eligible set in expectation


# ------------------------------------------------------------------------------------
# determinacy audit
# ------------------------------------------------------------------------------------


def test_audit_corner_all_forced():
    rng = random.Random(5)
    mask_rng = np.random.default_rng(5)
    for _ in range(50):
        grid, rects = dw_picture(14, 14, 32, rng=rng)
        mask = np.zeros((14, 14), dtype=bool)
        for r in rects:
            if max(r.row_span, r.col_span) >= 2 and mask_rng.random() < 0.5:
                mask[r.d_pos] = True
        report = audit_mask(grid, mask, 32, mode="corner")
        assert report.ok, report.reasons
        assert report.determinacy_rate == 1.0
        assert (report.values[mask] == grid[mask]).all()


def test_audit_detects_policy_violation():
    grid, rects = dw_picture(6, 6, 3, rng=random.Random(6))
    for pos in [(rects[0].r1, rects[0].c1), (rects[0].r2, rects[0].c1)]:  # an a, a c
        mask = np.zeros((6, 6), dtype=bool)
        mask[pos] = True
        report = audit_mask(grid, mask, 3, mode="corner")
        assert not report.ok


def test_audit_vs_brute_force():
    rng = random.Random(7)
    mask_rng = np.random.default_rng(7)
    for size in [(4, 4), (6, 6)]:
        for _ in range(10):
            grid, rects = dw_picture(*size, k=2, rng=rng)
            d_positions = [r.d_pos for r in rects]
            mask = np.zeros(size, dtype=bool)
            for pos in d_positions[:3]:
                mask[pos] = True
            report = audit_mask(grid, mask, 2, mode="corner")
            bf_forced, bf_vals = brute_force_forced(grid, mask, 2, restrict_to_d=True)
            # Corner mode must equal policy-restricted brute force exactly.
            assert (report.forced == bf_forced).all()
            assert (report.values[mask] == bf_vals[mask]).all()
            # General mode must be sound: forced ⊆ brute-forced (unrestricted).
            gen = audit_mask(grid, mask, 2, mode="general")
            bf2_forced, bf2_vals = brute_force_forced(grid, mask, 2, restrict_to_d=False)
            assert (~gen.forced | bf2_forced).all()
            if gen.forced.any():
                assert (gen.values[gen.forced] == bf2_vals[gen.forced]).all()
    # Adversarial mask (an a-corner): brute force and corner mode agree it's not clean.
    grid, rects = dw_picture(4, 4, 2, rng=rng)
    mask = np.zeros((4, 4), dtype=bool)
    mask[rects[0].r1, rects[0].c1] = True
    assert not audit_mask(grid, mask, 2, mode="corner").ok


# ------------------------------------------------------------------------------------
# shuffle control
# ------------------------------------------------------------------------------------


def test_shuffle_control(base_cfg):
    cfg = base_cfg
    cfg.data.source = "dyck2d_shuffle"
    ds, masking = build_source(cfg)
    destroyed = 0
    for i in range(50):
        sample = ds[i]
        assert sample.shape == (2, cfg.N)
        grid = sample[0].reshape(cfg.grid.H, cfg.grid.W).numpy()
        try:
            check_picture(grid, cfg.dyck2d.k)
        except MembershipError:
            destroyed += 1
    assert destroyed == 50  # matching structure destroyed in practice

    # The shared permutation preserves the (id, eligibility) multiset exactly.
    base = Dyck2DGrid(_dyck2d_cfg(base_cfg))[0]
    shuffled = marginal_shuffle(base.clone(), np.random.default_rng(0))
    orig_pairs = sorted(map(tuple, base.T.tolist()))
    shuf_pairs = sorted(map(tuple, shuffled.T.tolist()))
    assert orig_pairs == shuf_pairs

    # Same masking class runs on the control and masks only d-symbols.
    batch = torch.stack([ds[i] for i in range(8)])
    _m, targets, mask = masking(batch)
    lo = d_base(cfg.dyck2d.k)
    if mask.any():
        assert ((targets[mask] >= lo) & (targets[mask] < lo + cfg.dyck2d.k)).all()


# ------------------------------------------------------------------------------------
# registry + trainer integration
# ------------------------------------------------------------------------------------


def test_registry_contract(base_cfg):
    for source in ("dyck2d", "dyck2d_shuffle"):
        base_cfg.data.source = source
        ds, masking = build_source(base_cfg)
        assert isinstance(ds, ProceduralDataset)
        assert isinstance(masking, MaskingStrategy)


def test_strict_filter_mode(base_cfg):
    """filter_mode=min: both partners of every eligible d are non-adjacent."""
    cfg = _dyck2d_cfg(base_cfg, filter_mode="min")
    ds = Dyck2DGrid(cfg)
    for i in range(20):
        sample = ds[i]
        ids = sample[0].reshape(cfg.grid.H, cfg.grid.W).numpy()
        elig = sample[1].reshape(cfg.grid.H, cfg.grid.W).numpy()
        by_d = {r.d_pos: r for r in recover_rectangles(ids, cfg.dyck2d.k)}
        for pos in map(tuple, np.argwhere(elig == 1)):
            r = by_d[pos]
            assert min(r.row_span, r.col_span) >= 2  # no adjacent partner on either axis
    # The strict set is a strict subset of the default set on average.
    n_strict = sum(int(ds[i][1].sum()) for i in range(20))
    n_default = sum(int(Dyck2DGrid(_dyck2d_cfg(base_cfg, filter_mode="max"))[i][1].sum())
                    for i in range(20))
    assert 0 < n_strict < n_default


def test_cd_masking_roles(base_cfg):
    """mask_roles=cd: eligibility marks c and d corners; masking hides only those."""
    cfg = _dyck2d_cfg(base_cfg, mask_roles="cd")
    cfg.masking.mask_ratio = 1.0
    ds, masking = build_source(cfg)
    batch = torch.stack([ds[i] for i in range(8)])
    masked, targets, mask = masking(batch)
    assert torch.equal(mask, batch[:, 1].bool())  # exactly the eligible set at ratio 1.0
    k = cfg.dyck2d.k
    c_lo, d_hi = 2 + 2 * k, 2 + 4 * k
    assert ((targets[mask] >= c_lo) & (targets[mask] < d_hi)).all()  # only c/d ids
    # Both roles are actually present among the masked targets.
    roles = (targets[mask] - 2) // k
    assert set(roles.unique().tolist()) == {int(Role.C), int(Role.D)}
    # Eligibility marks the c and d of each passing rect and nothing else.
    ids = batch[0, 0].reshape(cfg.grid.H, cfg.grid.W).numpy()
    elig = batch[0, 1].reshape(cfg.grid.H, cfg.grid.W).numpy()
    expect = np.zeros_like(elig)
    for r in recover_rectangles(ids, k):
        if max(r.row_span, r.col_span) >= cfg.dyck2d.min_match_distance:
            expect[r.d_pos] = 1
            expect[r.r2, r.c1] = 1
    assert np.array_equal(elig, expect)


def test_closing_audit_exact_vs_brute_force():
    """closing mode == policy-restricted brute force, incl. jointly-masked c+d pairs."""
    rng = random.Random(11)
    mask_rng = np.random.default_rng(11)
    for size in [(4, 4), (6, 6)]:
        for _ in range(10):
            grid, rects = dw_picture(*size, k=2, rng=rng)
            mask = np.zeros(size, dtype=bool)
            planted = 0
            for r in rects:
                if planted >= 2:  # keep the brute-force pool (4^holes) tiny
                    break
                mask[r.d_pos] = True
                mask[r.r2, r.c1] = True  # the row-mate c too: single-axis insufficient
                planted += 1
            report = audit_mask(grid, mask, 2, mode="closing")
            assert report.ok, report.reasons
            assert report.determinacy_rate == 1.0
            assert (report.values[mask] == grid[mask]).all()
            bf_forced, bf_vals = brute_force_forced(grid, mask, 2, restrict_roles="cd")
            assert (report.forced == bf_forced).all()
            assert (report.values[mask] == bf_vals[mask]).all()
    # Policy violation: masking an opener (a or b) must not audit clean.
    grid, rects = dw_picture(4, 4, 2, rng=rng)
    for pos in [(rects[0].r1, rects[0].c1), (rects[0].r1, rects[0].c2)]:
        mask = np.zeros((4, 4), dtype=bool)
        mask[pos] = True
        assert not audit_mask(grid, mask, 2, mode="closing").ok


def test_invalid_filter_and_roles_raise():
    from procedural_warmup.config import load_config

    # Fresh config per case: _dyck2d_cfg mutates in place, so reusing one config
    # would trip the first case's invalid filter_mode in every later case.
    with pytest.raises(ValueError, match="filter_mode"):
        Dyck2DGrid(_dyck2d_cfg(load_config(None), filter_mode="median"))
    with pytest.raises(ValueError, match="mask_roles"):
        Dyck2DGrid(_dyck2d_cfg(load_config(None), mask_roles="abcd"))
    with pytest.raises(ValueError, match="mask_roles"):
        CornerCloseOnlyMasking(_dyck2d_cfg(load_config(None), mask_roles="abcd"))


@pytest.mark.parametrize("name", [
    "dw32-vit-t", "dw32-shuffle-vit-t", "dw32-tuned-vit-t", "dw32-dense-vit-t",
    "dw32-randpos-vit-t", "dw32-randpos-shuffle-vit-t", "dw32-smoke", "dyck-repro",
    "dw32-randpos-strict-vit-t", "dw32-randpos-cd-vit-t", "dw32-randpos-hard-vit-t",
])
def test_experiment_configs_load_and_build(name):
    """Every experiment YAML loads, satisfies the vocab bound, and builds its source."""
    from procedural_warmup.config import load_config

    cfg = load_config(f"procedural_warmup/config/files/{name}.yaml")
    assert cfg.run_name == name
    if cfg.data.source.startswith("dyck2d"):
        assert cfg.vocab.K >= vocab_size(cfg.dyck2d.k)
        assert cfg.model.pos_embed in ("random", "sincos2d", "sincos2d_tuned")
    ds, masking = build_source(cfg)
    assert isinstance(ds, ProceduralDataset)
    assert isinstance(masking, MaskingStrategy)


def test_trainer_smoke_dyck2d(base_cfg, tmp_path):
    """(B, 2, N) batches flow through Trainer -> model -> checkpoint -> strip."""
    import torch as _torch

    from procedural_warmup.utils import RunDir
    from procedural_warmup.warmup.process import clean_checkpoint
    from procedural_warmup.warmup.trainer import Trainer
    from procedural_warmup.model import build_model

    cfg = _dyck2d_cfg(base_cfg)
    cfg.model.pos_embed = "sincos2d"
    cfg.training.steps = 2
    cfg.training.device = "cpu"
    cfg.dataset.batch_size = 8
    cfg.dataset.num_workers = 0
    cfg.dataset.n_samples = 64
    cfg.logging.print_freq = 1
    cfg.logging.figure_every = 0
    cfg.scheduler.warmup_steps = 1
    cfg.checkpoint.save_steps = [2]
    cfg.checkpoint.out_dir = str(tmp_path / "checkpoints")
    cfg.results_dir = str(tmp_path / "results")

    model, mlm_head = build_model(cfg)
    dataset, masking = build_source(cfg)
    run_dir = RunDir.create(cfg.results_dir, cfg.run_name)
    ckpt = Trainer(cfg, model, mlm_head, dataset, masking, run_dir).train()
    assert ckpt is not None and ckpt.exists()

    stripped = clean_checkpoint(ckpt)
    state = _torch.load(stripped, map_location="cpu", weights_only=False)["model"]
    assert state and all(k.startswith(("blocks.", "norm.")) for k in state)


def test_throughput_smoke(base_cfg):
    import time

    ds = Dyck2DGrid(_dyck2d_cfg(base_cfg))
    t0 = time.perf_counter()
    for i in range(200):
        ds[i]
    rate = 200 / (time.perf_counter() - t0)
    assert rate > 500, f"{rate:.0f} samples/s — below the loose CI floor"
