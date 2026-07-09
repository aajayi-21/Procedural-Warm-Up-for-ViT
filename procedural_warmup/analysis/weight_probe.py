"""Weight-over-time probe: per-layer drift + attention-distance profiles across warm-up.

Post-hoc analysis of the ``probe_step_*.pt`` snapshots a warm-up run saves when
``checkpoint.probe_steps`` is set (see ``warmup/trainer.py``). Two readouts per run:

- **Relative weight drift** ``||W_t - W_0||_F / ||W_0||_F`` per transformer block and
  parameter group (attn.qkv, attn.proj, mlp.fc1, mlp.fc2, norm1, norm2), with W_0 the
  step-0 (init) snapshot — where in the network, and when, does the warm-up write?
- **Attention distance**: the model is replayed at each snapshot on one fixed probe
  batch (drawn once from the run's own source + masking, then cached to disk so
  re-invocations reuse it), and each head's mean attention-weighted token distance is
  computed. timm fuses attention (``F.scaled_dot_product_attention`` returns no
  weights), so the probe recomputes ``softmax(q k^T / sqrt(d_h))`` from a forward hook
  on each block's ``attn.qkv`` — verified against timm's unfused path (q_norm/k_norm
  are Identity on ViT-T; asserted at runtime). The CLS row/column are dropped and rows
  renormalized, leaving the 196 grid tokens. Distances are grid-Euclidean between
  ``(p // W, p % W)`` coordinates for 2D position codes, ``|i - j|`` sequence distance
  otherwise (``--dist`` overrides). **This is the CA-failure-signature diagnostic**:
  short-range profiles were the CA post-mortem's fingerprint; k-Dyck's long-range
  profile is the reference (probe arm B to calibrate).

Outputs follow the repo conventions: ``results/reports/<run>-probe/{metrics.json,
report.md}`` + figures in ``results/figures/``.

    python -m procedural_warmup.analysis.weight_probe --run dw32-vit-t
    python -m procedural_warmup.analysis.weight_probe --compare dyck-repro dw32-vit-t \
        dw32-shuffle-vit-t --out dw32-probe-compare
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from procedural_warmup.config import load_config
from procedural_warmup.data import build_source
from procedural_warmup.model import build_model
from procedural_warmup.utils import RunDir, set_seed

PARAM_GROUPS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2", "norm1", "norm2")


# ------------------------------------------------------------------------------------
# checkpoint loading
# ------------------------------------------------------------------------------------


def _normalize_keys(state: dict) -> dict:
    """Strip the ProceduralViT ``vit.`` prefix so keys read ``blocks.{i}...``."""
    return {(k[4:] if k.startswith("vit.") else k): v for k, v in state.items()}


def load_checkpoint_series(ckpt_dir: str | Path) -> list[tuple[int, dict]]:
    """All probe/full snapshots in ``ckpt_dir`` as ``(step, normalized_state)``, sorted."""
    ckpt_dir = Path(ckpt_dir)
    series: dict[int, dict] = {}
    for pattern in ("probe_step_*.pt", "ckpt_step_*.pt"):
        for path in sorted(ckpt_dir.glob(pattern)):
            m = re.search(r"step_(\d+)\.pt$", path.name)
            if m is None:
                continue  # skips *_stripped.pt
            step = int(m.group(1))
            if step in series:
                continue  # probe file wins; full ckpt only fills gaps
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("model_state", payload)
            series[step] = _normalize_keys(state)
    if not series:
        raise FileNotFoundError(f"no probe_step_*/ckpt_step_* snapshots in {ckpt_dir}")
    return sorted(series.items())


# ------------------------------------------------------------------------------------
# drift
# ------------------------------------------------------------------------------------


def num_blocks(state: dict) -> int:
    idx = {int(m.group(1)) for k in state if (m := re.match(r"blocks\.(\d+)\.", k))}
    return max(idx) + 1 if idx else 0


def relative_drift(series: list[tuple[int, dict]]) -> dict:
    """``{group: {block: [drift per step]}}`` with the first snapshot as W_0.

    Weights only (biases excluded). Frobenius norm; W_0 must be the step-0 probe —
    a loud error otherwise, because drift-from-mid-training is not drift-from-init.
    """
    steps = [s for s, _ in series]
    if steps[0] != 0:
        raise ValueError(
            f"first snapshot is step {steps[0]}, not 0 — enable checkpoint.probe_steps "
            "with a 0 entry (the init snapshot is the drift reference)"
        )
    w0 = series[0][1]
    n_blocks = num_blocks(w0)
    out: dict = {g: {b: [] for b in range(n_blocks)} for g in PARAM_GROUPS}
    for _, state in series:
        for g in PARAM_GROUPS:
            for b in range(n_blocks):
                key = f"blocks.{b}.{g}.weight"
                ref, cur = w0[key].float(), state[key].float()
                out[g][b].append(
                    (cur - ref).norm().item() / max(ref.norm().item(), 1e-12)
                )
    return {"steps": steps, "groups": out}


# ------------------------------------------------------------------------------------
# attention distance
# ------------------------------------------------------------------------------------


def _distance_matrix(N: int, W: int, metric: str) -> torch.Tensor:
    idx = torch.arange(N)
    if metric == "grid":
        r, c = idx // W, idx % W
        return torch.sqrt(
            (r[:, None] - r[None, :]).float() ** 2 + (c[:, None] - c[None, :]).float() ** 2
        )
    if metric == "seq":
        return (idx[:, None] - idx[None, :]).abs().float()
    raise ValueError(f"distance metric must be 'grid'|'seq', got {metric!r}")


@torch.no_grad()
def attention_distance_profile(
    model, batch_inputs: torch.Tensor, W: int, metric: str
) -> np.ndarray:
    """Mean attention-weighted token distance, shape ``(n_blocks, n_heads)``.

    Recomputes attention probabilities from hooked qkv outputs (timm's fused SDPA
    exposes none); drops the CLS row/column and renormalizes so distances are over the
    grid tokens only.
    """
    blocks = model.vit.blocks
    for blk in blocks:  # the recompute below assumes no q/k normalization (ViT-T)
        assert isinstance(blk.attn.q_norm, torch.nn.Identity), "q_norm must be Identity"
        assert isinstance(blk.attn.k_norm, torch.nn.Identity), "k_norm must be Identity"

    captured: dict[int, torch.Tensor] = {}
    hooks = [
        blk.attn.qkv.register_forward_hook(
            lambda _m, _i, out, b=b: captured.__setitem__(b, out.detach())
        )
        for b, blk in enumerate(blocks)
    ]
    try:
        model.eval()
        model.forward_tokens(batch_inputs)
    finally:
        for h in hooks:
            h.remove()

    N = batch_inputs.shape[1]
    D = _distance_matrix(N, W, metric).to(batch_inputs.device)
    profile = np.zeros((len(blocks), blocks[0].attn.num_heads))
    for b, qkv_out in captured.items():
        attn_mod = blocks[b].attn
        B, T, _ = qkv_out.shape  # T = N + 1 (CLS)
        qkv = qkv_out.reshape(B, T, 3, attn_mod.num_heads, attn_mod.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k = qkv[0].float(), qkv[1].float()
        attn = (q * attn_mod.scale) @ k.transpose(-2, -1)  # (B, heads, T, T)
        attn = attn.softmax(dim=-1)
        attn = attn[:, :, 1:, 1:]  # drop CLS row/column...
        # ...and renormalize each query row. clamp_min guards queries that park ~all
        # mass on CLS (observed on the structure-free shuffle control — an attention
        # sink; without the clamp those rows renormalize 0/0 to NaN).
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # E[distance] per (batch, head, query), then mean over batch and queries.
        profile[b] = (attn * D).sum(dim=-1).mean(dim=(0, 2)).cpu().numpy()
    return profile


def make_probe_batch(cfg, batch_size: int, seed: int, cache_path: Path) -> torch.Tensor:
    """One fixed masked probe batch from the run's own source; cached across invocations.

    The cache is self-describing and is regenerated (loudly) if the requested
    batch_size/seed no longer match — a silently stale batch would make metrics.json
    misreport its own provenance.
    """
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            isinstance(payload, dict)
            and payload.get("batch_size") == batch_size
            and payload.get("seed") == seed
        ):
            return payload["batch"]
        print(f"[probe] cache {cache_path.name} does not match batch_size={batch_size}/"
              f"seed={seed} — regenerating")
    set_seed(seed)
    dataset, masking = build_source(cfg)
    batch = torch.stack([dataset[i] for i in range(batch_size)])
    masked_input, _target, _mask = masking(batch)
    torch.save({"batch": masked_input, "batch_size": batch_size, "seed": seed}, cache_path)
    return masked_input


# ------------------------------------------------------------------------------------
# figures + CLI
# ------------------------------------------------------------------------------------


def _plot_drift(drift: dict, figures_dir: Path, run: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = drift["steps"]
    groups = drift["groups"]
    n_blocks = len(groups[PARAM_GROUPS[0]])
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, g in zip(axes.flat, PARAM_GROUPS):
        for b in range(n_blocks):
            ax.plot(steps, groups[g][b], color=cmap(b / max(n_blocks - 1, 1)), lw=1.2)
        ax.set_title(g)
        ax.set_xlabel("step")
        ax.set_ylabel("||W_t - W_0|| / ||W_0||")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_blocks - 1))
    fig.colorbar(sm, ax=axes, label="block (depth)", fraction=0.02)
    fig.suptitle(f"Warm-up weight drift by block and parameter group — {run}")
    path = figures_dir / f"{run}_probe_drift.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_attn(attn: dict[int, np.ndarray], figures_dir: Path, run: str,
               metric: str, attn_steps: list[int]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(attn)
    show = [s for s in (attn_steps or steps) if s in attn] or steps
    n_blocks = attn[steps[0]].shape[0]
    cmap = plt.get_cmap("plasma")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, s in enumerate(show):
        axes[0].plot(range(n_blocks), attn[s].mean(axis=1),
                     color=cmap(i / max(len(show) - 1, 1)), marker="o", label=f"step {s}")
    axes[0].scatter(
        np.repeat(range(n_blocks), attn[steps[-1]].shape[1]),
        attn[steps[-1]].flatten(), s=12, color="k", zorder=3, label="heads @ final",
    )
    axes[0].set_xlabel("block")
    axes[0].set_ylabel(f"mean attention distance ({metric})")
    axes[0].legend(fontsize=8)
    axes[0].set_title("distance vs depth")
    for b in range(n_blocks):
        axes[1].plot(steps, [attn[s][b].mean() for s in steps],
                     color=plt.get_cmap("viridis")(b / max(n_blocks - 1, 1)), lw=1.2)
    axes[1].set_xlabel("step")
    axes[1].set_title("distance vs training step (per block)")
    fig.suptitle(f"Attention-distance profiles — {run}")
    fig.tight_layout()
    path = figures_dir / f"{run}_probe_attn_distance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def probe_run(run: str, results_dir: str, ckpt_dir: str | None, batch_size: int,
              seed: int, device: str, metric: str | None, attn_steps: list[int]) -> None:
    cfg = load_config(str(Path(results_dir) / "reports" / run / "config.yaml"))
    ckpt_dir = ckpt_dir or str(Path(cfg.checkpoint.out_dir) / run)
    if metric is None:
        metric = "grid" if str(cfg.model.pos_embed).startswith("sincos2d") else "seq"

    series = load_checkpoint_series(ckpt_dir)
    print(f"[probe] {run}: {len(series)} snapshots (steps {[s for s, _ in series]})")
    drift = relative_drift(series)

    out_dir = RunDir.create(results_dir, f"{run}-probe")
    batch = make_probe_batch(cfg, batch_size, seed, out_dir.root / "probe_batch.pt")
    model, _head = build_model(cfg)
    model = model.to(device)
    attn: dict[int, np.ndarray] = {}
    for step, state in series:
        model.load_state_dict(
            {f"vit.{k}" if not k.startswith(("tok.", "pos.")) else k: v
             for k, v in state.items()},
            strict=False,
        )
        attn[step] = attention_distance_profile(model, batch.to(device), cfg.grid.W, metric)

    drift_fig = _plot_drift(drift, out_dir.figures_dir, run)
    attn_fig = _plot_attn(attn, out_dir.figures_dir, run, metric, attn_steps)
    final = attn[max(attn)]
    out_dir.save_metrics({
        "run": run,
        "source": cfg.data.source,
        "pos_embed": cfg.model.pos_embed,
        "dist_metric": metric,
        "probe": {"batch_size": batch_size, "seed": seed},
        "steps": drift["steps"],
        "drift": {g: {str(b): v for b, v in blocks.items()}
                  for g, blocks in drift["groups"].items()},
        "attn_distance": {str(s): p.tolist() for s, p in attn.items()},
        "final_mean_attn_distance": float(final.mean()),
        "final_attn_distance_by_block": final.mean(axis=1).tolist(),
    })
    out_dir.write_report(
        title=f"Weight-over-time probe: {run}",
        sections={
            "Setup": (
                f"- source: `{cfg.data.source}`, pos_embed: `{cfg.model.pos_embed}`\n"
                f"- snapshots: steps {drift['steps']}\n"
                f"- probe batch: {batch_size} samples from the run's own source+masking "
                f"(seed {seed}, cached as `probe_batch.pt`)\n"
                f"- distance metric: `{metric}`"
            ),
            "Result": (
                f"Final mean attention distance: **{final.mean():.2f}** "
                f"({metric} units; max possible "
                f"{13 * 2 ** 0.5:.1f} grid / 195 seq). Short-range profiles across all "
                "blocks were the CA failure signature; compare against the k-Dyck "
                "reference run before reading anything into the downstream number."
            ),
        },
        figures=[str(drift_fig), str(attn_fig)],
    )
    print(f"[probe] wrote {out_dir.root / 'report.md'}")


def compare_runs(runs: list[str], results_dir: str, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = []
    for r in runs:
        path = Path(results_dir) / "reports" / f"{r}-probe" / "metrics.json"
        if not path.exists():
            print(f"[probe-compare] skipping '{r}' (run `--run {r}` first)")
            continue
        loaded.append((r, json.loads(path.read_text(encoding="utf-8"))))
    if not loaded:
        raise FileNotFoundError("no probe metrics found for any requested run")
    metrics_used = {m["dist_metric"] for _, m in loaded}
    if len(metrics_used) > 1:
        raise ValueError(
            f"incommensurate distance metrics across runs: {sorted(metrics_used)} — "
            "re-run the probes with a common --dist (grid is well-defined for 1D runs "
            "via the fixed raster layout) before comparing"
        )

    run_dir = RunDir.create(results_dir, out)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for r, m in loaded:
        curve = m["final_attn_distance_by_block"]
        axes[0].plot(range(len(curve)), curve, marker="o",
                     label=f"{r} ({m['dist_metric']})")
        totals = [
            np.mean([blocks[b][-1] for b in blocks])
            for _g, blocks in m["drift"].items()
        ]
        label = r if len(r) <= 14 else r[:13] + "…"
        axes[1].bar(label, np.mean(totals))
    axes[0].set_xlabel("block")
    axes[0].set_ylabel("final mean attention distance")
    axes[0].set_title("attention-distance profile at 15k steps")
    axes[0].legend(fontsize=8)
    axes[1].set_title("mean final relative drift (all groups/blocks)")
    fig.suptitle("Warm-up probe comparison")
    fig.tight_layout()
    fig_path = run_dir.figures_dir / f"{out}.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    table = "| run | source | dist | final attn distance | \n|---|---|---|---|\n" + "\n".join(
        f"| `{r}` | {m['source']} | {m['dist_metric']} | {m['final_mean_attn_distance']:.2f} |"
        for r, m in loaded
    )
    run_dir.save_metrics({"runs": [r for r, _ in loaded]})
    run_dir.write_report(
        title="Warm-up weight-probe comparison",
        sections={"Profiles": table},
        figures=[str(fig_path)],
    )
    print(f"[probe-compare] wrote {run_dir.root / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Weight-over-time probe for warm-up runs.")
    ap.add_argument("--run", default=None, help="probe one warm-up run by name")
    ap.add_argument("--compare", nargs="+", default=None, help="overlay existing probes")
    ap.add_argument("--out", default="probe-compare", help="output name for --compare")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--probe-seed", type=int, default=12345)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dist", default=None, choices=["grid", "seq"],
                    help="override the pos_embed-inferred distance metric")
    ap.add_argument("--attn-steps", type=int, nargs="*", default=[],
                    help="subset of steps for the profile figure (default: all)")
    args = ap.parse_args()
    if (args.run is None) == (args.compare is None):
        ap.error("exactly one of --run / --compare is required")
    if args.run:
        probe_run(args.run, args.results_dir, args.ckpt_dir, args.batch_size,
                  args.probe_seed, args.device, args.dist, args.attn_steps)
    else:
        compare_runs(args.compare, args.results_dir, args.out)


if __name__ == "__main__":
    main()
