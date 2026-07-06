"""Weight-over-time probe: trainer snapshots, drift math, attention recomputation."""

import numpy as np
import pytest
import torch

from procedural_warmup.analysis.weight_probe import (
    attention_distance_profile,
    load_checkpoint_series,
    relative_drift,
)
from procedural_warmup.data import build_source
from procedural_warmup.model import build_model
from procedural_warmup.utils import RunDir
from procedural_warmup.warmup.trainer import Trainer


def test_trainer_saves_probe_snapshots(base_cfg, tmp_path):
    cfg = base_cfg
    cfg.data.source = "dyck2d"
    cfg.training.steps = 2
    cfg.training.device = "cpu"
    cfg.dataset.batch_size = 8
    cfg.dataset.num_workers = 0
    cfg.dataset.n_samples = 64
    cfg.logging.print_freq = 1
    cfg.logging.figure_every = 0
    cfg.scheduler.warmup_steps = 1
    cfg.checkpoint.save_steps = [2]
    cfg.checkpoint.probe_steps = [0, 2]
    cfg.checkpoint.out_dir = str(tmp_path / "checkpoints")
    cfg.results_dir = str(tmp_path / "results")

    model, mlm_head = build_model(cfg)
    dataset, masking = build_source(cfg)
    run_dir = RunDir.create(cfg.results_dir, cfg.run_name)
    Trainer(cfg, model, mlm_head, dataset, masking, run_dir).train()

    ckpt_dir = tmp_path / "checkpoints" / cfg.run_name
    p0, p2 = ckpt_dir / "probe_step_000000.pt", ckpt_dir / "probe_step_000002.pt"
    assert p0.exists() and p2.exists()
    payload = torch.load(p0, map_location="cpu", weights_only=False)
    assert set(payload) == {"step", "model_state"}  # no optimizer/head bloat

    series = load_checkpoint_series(ckpt_dir)
    assert [s for s, _ in series] == [0, 2]
    # Training actually moved the weights between the two snapshots.
    drift = relative_drift(series)
    final = [drift["groups"][g][b][-1] for g in drift["groups"] for b in drift["groups"][g]]
    assert all(np.isfinite(final)) and max(final) > 0


def test_relative_drift_known_value():
    w = torch.randn(4, 4)
    state0 = {"blocks.0.attn.qkv.weight": w}
    state1 = {"blocks.0.attn.qkv.weight": 2.0 * w}
    for g in ("attn.proj", "mlp.fc1", "mlp.fc2", "norm1", "norm2"):
        state0[f"blocks.0.{g}.weight"] = w
        state1[f"blocks.0.{g}.weight"] = w
    drift = relative_drift([(0, state0), (100, state1)])
    # ||2W - W|| / ||W|| == 1 exactly for the doubled group, 0 for the others.
    assert drift["groups"]["attn.qkv"][0] == pytest.approx([0.0, 1.0])
    assert drift["groups"]["mlp.fc1"][0] == pytest.approx([0.0, 0.0])


def test_drift_requires_step0():
    state = {"blocks.0.attn.qkv.weight": torch.ones(2, 2)}
    with pytest.raises(ValueError, match="not 0"):
        relative_drift([(500, state), (1000, state)])


def test_attention_profile_shape_and_range(base_cfg):
    model, _ = build_model(base_cfg)
    ids = torch.randint(0, base_cfg.vocab.K, (2, base_cfg.N))
    prof = attention_distance_profile(model, ids, base_cfg.grid.W, "grid")
    assert prof.shape == (12, 3)
    assert (prof > 0).all() and (prof <= 13 * 2 ** 0.5).all()
    prof_seq = attention_distance_profile(model, ids, base_cfg.grid.W, "seq")
    assert (prof_seq > 0).all() and (prof_seq <= 195).all()


def test_attention_recompute_matches_unfused(base_cfg):
    """The hook-based softmax(qk^T/sqrt(dh)) must equal timm's own unfused attention."""
    model, _ = build_model(base_cfg)
    model.eval()
    blk = model.vit.blocks[0]
    ids = torch.randint(0, base_cfg.vocab.K, (2, base_cfg.N))

    # Capture the block's input and qkv output.
    grabbed = {}
    h1 = blk.register_forward_pre_hook(lambda _m, i: grabbed.__setitem__("x", i[0].detach()))
    h2 = blk.attn.qkv.register_forward_hook(
        lambda _m, _i, out: grabbed.__setitem__("qkv", out.detach())
    )
    with torch.no_grad():
        model.forward_tokens(ids)
    h1.remove(), h2.remove()

    attn_mod = blk.attn
    B, T, _ = grabbed["qkv"].shape
    qkv = grabbed["qkv"].reshape(B, T, 3, attn_mod.num_heads, attn_mod.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    ours = ((q * attn_mod.scale) @ k.transpose(-2, -1)).softmax(dim=-1)

    # timm's unfused path on the normed block input.
    x_normed = blk.norm1(grabbed["x"])
    attn_mod.fused_attn = False
    qkv2 = attn_mod.qkv(x_normed).reshape(B, T, 3, attn_mod.num_heads, attn_mod.head_dim)
    qkv2 = qkv2.permute(2, 0, 3, 1, 4)
    q2, k2 = attn_mod.q_norm(qkv2[0]), attn_mod.k_norm(qkv2[1])
    theirs = ((q2 * attn_mod.scale) @ k2.transpose(-2, -1)).softmax(dim=-1)
    attn_mod.fused_attn = True

    assert torch.allclose(ours, theirs, atol=1e-5)
