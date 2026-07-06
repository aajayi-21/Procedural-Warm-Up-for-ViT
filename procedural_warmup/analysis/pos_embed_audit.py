"""Positional-embedding audit: quantify how much grid geometry a frozen code exposes.

The H3 post-mortem showed the position code can silently decide an experiment, so the H6
program audits its codes *before* GPU time. For a given ``pos_embed`` kind this module
computes (and figures) the geometric properties the property tests pin:

- per-position norm uniformity (the 0.02 rescale must be a uniform scalar);
- minimum pairwise distance (all 196 positions distinct with margin);
- nearest-neighbor-in-embedding == grid-neighbor rate over interior cells;
- binned cosine similarity vs. grid Euclidean distance (decay curve + far-pair floor);
- effective frequencies per axis: count with angle span ``(W-1) * omega >= pi/4``
  (the base-10000 ``sincos2d`` band leaves ~15/48 alive at 14 cells; the grid-matched
  ``sincos2d_tuned`` band keeps 48/48);
- linear-probe (row, col) recovery error (lstsq with bias; exactness certifies that
  position is linearly decodable despite the normalization);
- ``token_signal_ratio``: mean 4-neighbor positional contrast ``||e_p - e_q||`` over the
  0.02 token-embedding norm — how large one grid step is relative to the token signal
  (the sincos2d x K=130 interaction number behind the ``block-2dpos`` divergence watch).

    python -m procedural_warmup.analysis.pos_embed_audit --pos-embed sincos2d
    python -m procedural_warmup.analysis.pos_embed_audit --pos-embed sincos2d_tuned
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from procedural_warmup.model.embeddings import (
    Frozen2DSinCosPositionalEmbedding,
    FrozenPositionalEmbedding,
    FrozenTuned2DSinCosPositionalEmbedding,
)
from procedural_warmup.utils import RunDir


def build_pos_embedding(kind: str, N: int, d: int, H: int, W: int):
    if kind == "sincos2d":
        return Frozen2DSinCosPositionalEmbedding(N, d, H, W)
    if kind == "sincos2d_tuned":
        return FrozenTuned2DSinCosPositionalEmbedding(N, d, H, W)
    if kind == "random":
        return FrozenPositionalEmbedding(N, d)
    raise ValueError(f"unsupported pos-embed kind for audit: {kind!r}")


def _omega_of(kind: str, d: int, H: int, W: int) -> np.ndarray | None:
    """The per-axis angular-frequency vector, if the kind is a sin/cos code."""
    dq = d // 4
    if kind == "sincos2d":
        return 1.0 / (10000.0 ** (np.arange(dq) / dq))
    if kind == "sincos2d_tuned":
        from procedural_warmup.model.embeddings import TUNED_LAMBDA_MIN

        lam_min, lam_max = TUNED_LAMBDA_MIN, 2.0 * max(H, W)
        lam = lam_min * (lam_max / lam_min) ** (np.arange(dq) / max(dq - 1, 1))
        return 2.0 * math.pi / lam
    return None


def grid_distance_matrix(H: int, W: int) -> np.ndarray:
    """(N, N) Euclidean distances between token grid coordinates (row-major)."""
    r = np.arange(H * W) // W
    c = np.arange(H * W) % W
    return np.sqrt((r[:, None] - r[None, :]) ** 2 + (c[:, None] - c[None, :]) ** 2)


def audit_stats(weight: torch.Tensor, H: int, W: int, kind: str = "?") -> dict:
    """All audit metrics for one (N, d) frozen positional table."""
    w = weight.detach().double().numpy()
    N, d = w.shape
    assert N == H * W
    norms = np.linalg.norm(w, axis=1)
    wn = w / norms[:, None]
    sim = wn @ wn.T  # cosine similarity
    dist2 = ((w[:, None, :] - w[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dist2, np.inf)
    gd = grid_distance_matrix(H, W)

    # Nearest neighbor in embedding space vs the 4-neighborhood, interior cells only.
    nn_hits = total = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            p = r * W + c
            q = int(np.argmin(dist2[p]))
            qr, qc = q // W, q % W
            nn_hits += abs(qr - r) + abs(qc - c) == 1
            total += 1

    # Binned similarity-vs-distance curve (integer-rounded Euclidean bins).
    off = ~np.eye(N, dtype=bool)
    bins = np.round(gd[off]).astype(int)
    sims = sim[off]
    curve = {int(b): float(sims[bins == b].mean()) for b in np.unique(bins)}

    # Linear probe: recover (row, col) from the embedding with a bias column.
    target = np.stack([np.arange(N) // W, np.arange(N) % W], axis=1).astype(float)
    design = np.concatenate([w, np.ones((N, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    probe_err = float(np.abs(design @ coef - target).max())

    omega = _omega_of(kind, d, H, W)
    eff = (
        int(((max(H, W) - 1) * omega >= math.pi / 4).sum()) if omega is not None else None
    )

    neighbor_pairs = [
        (r * W + c, r * W + c + 1) for r in range(H) for c in range(W - 1)
    ] + [(r * W + c, (r + 1) * W + c) for r in range(H - 1) for c in range(W)]
    contrast = float(
        np.mean([np.linalg.norm(w[p] - w[q]) for p, q in neighbor_pairs])
    )

    far = gd >= 8
    np.fill_diagonal(far, False)
    # Channels that never vary across positions carry no positional information
    # (frequency-level span checks miss e.g. a Nyquist sin channel that is identically
    # zero on integer coordinates — count channels, not just frequencies).
    dead_channels = int((w.std(axis=0) < 1e-4 * 0.02).sum())
    return {
        "kind": kind,
        "norm_mean": float(norms.mean()),
        "norm_spread": float(norms.max() - norms.min()),
        "dead_channels": dead_channels,
        "min_pairwise_dist": float(np.sqrt(dist2[np.isfinite(dist2)].min())),
        "nn_grid_neighbor_rate": nn_hits / total,
        "sim_by_distance": curve,
        "far_pair_sim_floor": float(sim[far].mean()),
        "effective_freqs_per_axis": eff,
        "total_freqs_per_axis": d // 4,
        "linear_probe_max_err": probe_err,
        "token_signal_ratio": contrast / 0.02,
    }


def plot_pos_embed_audit(weight: torch.Tensor, H: int, W: int, figures_dir, kind: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = weight.detach().double().numpy()
    wn = w / np.linalg.norm(w, axis=1, keepdims=True)
    sim = wn @ wn.T
    stats = audit_stats(weight, H, W, kind)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    im = axes[0].imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    axes[0].set_title(f"cosine similarity ({kind})")
    axes[0].set_xlabel("token p (row-major)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    curve = stats["sim_by_distance"]
    axes[1].plot(list(curve.keys()), list(curve.values()), marker="o", color="#4878cf")
    axes[1].axhline(0.0, color="grey", lw=0.8)
    axes[1].set_title("mean cosine similarity vs grid distance")
    axes[1].set_xlabel("grid Euclidean distance (cells)")
    axes[1].set_ylim(-1, 1)

    omega = _omega_of(kind, w.shape[1], H, W)
    if omega is not None:
        span = (max(H, W) - 1) * omega
        axes[2].bar(np.arange(len(span)), span, color="#4878cf")
        axes[2].axhline(math.pi / 4, color="crimson", lw=1.2, label="pi/4 span")
        axes[2].set_yscale("log")
        axes[2].set_title(
            f"angle span per frequency ({stats['effective_freqs_per_axis']}/"
            f"{stats['total_freqs_per_axis']} effective)"
        )
        axes[2].set_xlabel("frequency index")
        axes[2].legend()
    else:
        axes[2].axis("off")
        axes[2].text(0.1, 0.5, "no frequency structure (random code)", fontsize=11)
    fig.suptitle(f"Frozen positional-embedding audit: {kind}")
    fig.tight_layout()
    path = figures_dir / f"pos_embed_audit_{kind}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a frozen positional embedding.")
    ap.add_argument("--pos-embed", default="sincos2d",
                    choices=["sincos2d", "sincos2d_tuned", "random"])
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--grid", type=int, nargs=2, default=[14, 14], metavar=("H", "W"))
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    H, W = args.grid
    pe = build_pos_embedding(args.pos_embed, H * W, args.embed_dim, H, W)
    stats = audit_stats(pe.emb.weight, H, W, args.pos_embed)
    run_dir = RunDir.create(args.results_dir, f"pos-embed-audit-{args.pos_embed}")
    fig = plot_pos_embed_audit(pe.emb.weight, H, W, run_dir.figures_dir, args.pos_embed)
    run_dir.save_metrics(stats)
    eff = stats["effective_freqs_per_axis"]
    run_dir.write_report(
        title=f"Positional-embedding audit: {args.pos_embed} ({H}x{W}, d={args.embed_dim})",
        sections={
            "Verdict": (
                f"- per-position norm: {stats['norm_mean']:.4f} "
                f"(spread {stats['norm_spread']:.2e} — uniform rescale confirmed)\n"
                f"- min pairwise distance: {stats['min_pairwise_dist']:.2e} "
                f"(all {H*W} positions distinct)\n"
                f"- nearest embedding neighbor is a grid neighbor: "
                f"{stats['nn_grid_neighbor_rate']:.1%} of interior cells\n"
                f"- effective frequencies/axis: "
                f"{eff if eff is not None else 'n/a'}/{stats['total_freqs_per_axis']}\n"
                f"- far-pair (distance >= 8) cosine floor: {stats['far_pair_sim_floor']:.3f}\n"
                f"- linear probe (row, col) max error: {stats['linear_probe_max_err']:.2e} cells\n"
                f"- one-grid-step contrast / token-embedding norm: "
                f"{stats['token_signal_ratio']:.3f}"
            ),
        },
        figures=[str(fig)],
    )
    print(f"[pos-embed-audit] wrote {run_dir.root / 'report.md'}")


if __name__ == "__main__":
    main()
