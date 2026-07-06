"""Property tests pinning the geometric audit of the frozen 2-D positional codes.

These regression-pin the findings of ``analysis/pos_embed_audit.py`` for the 14x14 /
d=192 configuration the H6 program runs: the standard ``sincos2d`` is geometrically
sound but leaves only 15/48 frequencies per axis effective (far-pair cosine floor
~0.75), while the grid-matched ``sincos2d_tuned`` keeps all 48 alive with ~0 floor and
several-fold larger per-grid-step contrast — at identical norm and with identical
neighbor/decodability guarantees.
"""

import math

import numpy as np
import pytest
import torch

from procedural_warmup.analysis.pos_embed_audit import audit_stats, build_pos_embedding
from procedural_warmup.model.factory import build_model

H = W = 14
N = H * W
D = 192

KINDS = ["sincos2d", "sincos2d_tuned"]


@pytest.fixture(params=KINDS)
def kind_and_weight(request):
    pe = build_pos_embedding(request.param, N, D, H, W)
    return request.param, pe.emb.weight.detach()


def test_per_position_norm_uniform(kind_and_weight):
    kind, w = kind_and_weight
    norms = w.norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, 0.02), atol=1e-6)
    # The pre-normalization construction has constant norm sqrt(d/2) for every position
    # (sin^2 + cos^2 = 1 per frequency-axis pair), so normalization is a uniform rescale
    # that preserves geometry. Verify by rebuilding the raw table.
    from procedural_warmup.analysis.pos_embed_audit import _omega_of

    omega = torch.from_numpy(_omega_of(kind, D, H, W)).float()
    rows = torch.arange(H).repeat_interleave(W).float()
    cols = torch.arange(W).repeat(H).float()
    raw = torch.cat(
        [
            torch.cat([(p[:, None] * omega).sin(), (p[:, None] * omega).cos()], dim=1)
            for p in (rows, cols)
        ],
        dim=1,
    )
    assert torch.allclose(raw.norm(dim=1), torch.full((N,), math.sqrt(D / 2)), atol=1e-4)


def test_all_positions_distinct_with_margin(kind_and_weight):
    _, w = kind_and_weight
    dist = torch.cdist(w.double(), w.double())
    dist.fill_diagonal_(float("inf"))
    assert dist.min().item() > 1e-4  # absolute margin at the 0.02 scale


def test_similarity_decays_with_grid_distance(kind_and_weight):
    kind, w = kind_and_weight
    curve = audit_stats(w, H, W, kind)["sim_by_distance"]
    vals = [curve[b] for b in range(1, 7)]
    assert all(a > b + 1e-3 for a, b in zip(vals, vals[1:])), (
        f"{kind}: binned similarity not strictly decreasing over bins 1..6: {vals}"
    )


def test_nearest_neighbor_is_grid_neighbor(kind_and_weight):
    kind, w = kind_and_weight
    rate = audit_stats(w, H, W, kind)["nn_grid_neighbor_rate"]
    assert rate == 1.0, f"{kind}: NN-grid-neighbor rate {rate:.3f} < 1.0"


def test_linear_probe_recovers_coordinates(kind_and_weight):
    kind, w = kind_and_weight
    err = audit_stats(w, H, W, kind)["linear_probe_max_err"]
    assert err < 1e-2, f"{kind}: linear probe max coordinate error {err}"


def test_effective_frequency_count(kind_and_weight):
    kind, w = kind_and_weight
    eff = audit_stats(w, H, W, kind)["effective_freqs_per_axis"]
    assert eff == (15 if kind == "sincos2d" else 48)


def test_tuned_has_no_dead_channels():
    # Channel-level liveness (a frequency-span count alone would miss a Nyquist sin
    # channel that is identically zero on integer coordinates — hence lambda_min > 2).
    w = build_pos_embedding("sincos2d_tuned", N, D, H, W).emb.weight
    assert audit_stats(w, H, W, "sincos2d_tuned")["dead_channels"] == 0


def test_tuned_improves_contrast():
    base = audit_stats(
        build_pos_embedding("sincos2d", N, D, H, W).emb.weight, H, W, "sincos2d"
    )
    tuned = audit_stats(
        build_pos_embedding("sincos2d_tuned", N, D, H, W).emb.weight, H, W, "sincos2d_tuned"
    )
    assert tuned["far_pair_sim_floor"] < 0.3 < base["far_pair_sim_floor"]
    assert tuned["token_signal_ratio"] >= 2.0 * base["token_signal_ratio"]

    def contrast(s):
        return s["sim_by_distance"][1] - s["sim_by_distance"][7]

    assert contrast(tuned) >= 2.0 * contrast(base)


def test_factory_builds_tuned(base_cfg):
    base_cfg.model.pos_embed = "sincos2d_tuned"
    model, mlm_head = build_model(base_cfg)
    ids = torch.randint(0, base_cfg.vocab.K, (2, N))
    logits = mlm_head(model.forward_tokens(ids))
    assert logits.shape == (2, N, base_cfg.vocab.K)
    assert not model.pos.emb.weight.requires_grad
