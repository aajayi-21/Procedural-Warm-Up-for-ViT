"""Structure statistics + pre-GPU property gate for the DW_k sampler.

Runs the design-doc §5 validation gates in executable form (CLI), writing a per-run
report to ``results/reports/<out>/`` and a figure to ``results/figures/``:

1. **Membership gate**: every sampled picture passes the independent Tier-1 checker
   *and* the Tier-2 guillotine-DW certifier (``--n`` samples, default 100k).
2. **Determinacy gate**: corner-close-only masks on a subsample audit to
   ``determinacy_rate == 1.0`` (:mod:`audit`, corner mode).
3. **Throughput gate**: measured samples/s vs. the 15k-step x batch-256 budget
   (~2.5k samples/s aggregate at GPU-bound step rates). If single-core throughput is
   below ~200/s the report recommends pre-generation (design-doc §5 gate 4).

Structure statistics contextualize the documented guillotine-only support bias and let
``p_acc`` be tuned against the 1D generator's nesting-depth profile: rectangle-span and
match-distance histograms, per-rect nesting depth (count of strictly containing boxes),
2x2 fraction (= filter-excluded fraction at the default min-match-distance), eligible-d
counts (flagging zero-eligible pictures), and accretion/split action counts. The
rectangle *count* is constant (N/4) — every cell is a corner of exactly one quadruple —
so counts are reported only as a sanity check.

    python -m procedural_warmup.data.dyck2d.stats --n 100000
    python -m procedural_warmup.data.dyck2d.stats --n 2000 --no-tier2   # quick look
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from procedural_warmup.config import load_config
from procedural_warmup.data.dyck2d.audit import audit_mask
from procedural_warmup.data.dyck2d.dataset import Dyck2DGrid
from procedural_warmup.data.dyck2d.generator import dw_picture
from procedural_warmup.data.dyck2d.validate import is_dw_guillotine, recover_rectangles
from procedural_warmup.utils import RunDir


def picture_stats(rects) -> dict:
    """Per-picture structure statistics from a rectangle list."""
    spans = np.array([[r.row_span, r.col_span] for r in rects])
    match_dist = spans.max(axis=1)
    min_span = spans.min(axis=1)
    boxes = [(r.r1, r.c1, r.r2, r.c2) for r in rects]
    depth = [
        sum(
            1
            for (s1, t1, s2, t2) in boxes
            if (s1 <= b1 and t1 <= b2 and s2 >= b3 and t2 >= b4)
            and (s1, t1, s2, t2) != (b1, b2, b3, b4)
        )
        for (b1, b2, b3, b4) in boxes
    ]
    return {
        "n_rects": len(rects),
        "row_spans": spans[:, 0].tolist(),
        "col_spans": spans[:, 1].tolist(),
        "match_dist": match_dist.tolist(),
        "min_span": min_span.tolist(),
        "depth": depth,
        "frac_2x2": float((match_dist == 1).mean()),
    }


def run_gate(
    n: int,
    k: int,
    H: int,
    W: int,
    p_acc: float,
    open_prob: float,
    min_match_distance: int,
    mask_ratio: float,
    tier2: bool = True,
    audit_every: int = 100,
    seed: int = 0,
) -> dict:
    """Sample n pictures; validate, audit a subsample, time generation. Returns metrics."""
    rng = random.Random(seed)
    mask_rng = np.random.default_rng(seed)
    trace: dict = {}
    match_hist: dict[int, int] = {}
    depth_hist: dict[int, int] = {}
    elig_counts: list[int] = []
    zero_elig = 0
    frac_2x2_sum = 0.0
    audits = audited_ok = 0
    elig_total = elig_adjacent_partner = 0  # eligible d's with min(dr,dc)==1

    t0 = time.perf_counter()
    for i in range(n):
        grid, rects = dw_picture(H, W, k, p_acc=p_acc, open_prob=open_prob, rng=rng, trace=trace)
        # Independent membership checks (Tier 1 raises on failure).
        recovered = recover_rectangles(grid, k)
        assert len(recovered) == len(rects), "Tier-1 recovery disagrees with generator"
        if tier2 and not is_dw_guillotine(grid, k):
            raise AssertionError(f"sample {i}: Tier-2 guillotine-DW certification failed")
        st = picture_stats(rects)
        frac_2x2_sum += st["frac_2x2"]
        for v in st["match_dist"]:
            match_hist[v] = match_hist.get(v, 0) + 1
        for v in st["depth"]:
            depth_hist[v] = depth_hist.get(v, 0) + 1
        elig = 0
        for r in rects:
            if max(r.row_span, r.col_span) >= min_match_distance:
                elig += 1
                elig_adjacent_partner += min(r.row_span, r.col_span) == 1
        elig_total += elig
        elig_counts.append(elig)
        zero_elig += elig == 0
        if i % audit_every == 0 and elig > 0:
            m = np.zeros((H, W), dtype=bool)
            for r in rects:
                if max(r.row_span, r.col_span) >= min_match_distance:
                    m[r.d_pos] = mask_rng.random() < mask_ratio
            report = audit_mask(grid, m, k, mode="corner")
            audits += 1
            audited_ok += report.ok and report.determinacy_rate == 1.0
    gen_seconds = time.perf_counter() - t0

    # Pure-generation throughput (no validation) on a second, smaller timing pass.
    n_time = min(n, 2000)
    t0 = time.perf_counter()
    for _ in range(n_time):
        dw_picture(H, W, k, p_acc=p_acc, open_prob=open_prob, rng=rng)
    per_sec = n_time / (time.perf_counter() - t0)

    return {
        "n_samples": n,
        "k": k,
        "grid": [H, W],
        "p_acc": p_acc,
        "open_prob": open_prob,
        "min_match_distance": min_match_distance,
        "membership_pass": True,  # gates raise on any failure
        "tier2_checked": tier2,
        "audits": audits,
        "audits_fully_determined": audited_ok,
        "match_distance_hist": {str(kk): v for kk, v in sorted(match_hist.items())},
        "depth_hist": {str(kk): v for kk, v in sorted(depth_hist.items())},
        "mean_frac_2x2": frac_2x2_sum / n,
        "mean_eligible_d": float(np.mean(elig_counts)),
        # Supervision density vs the 1D anchor: masked targets/sample at the given
        # ratio, over 196 tokens (1D close-only masks ~49/sample = 25%). Report this
        # beside any A-vs-B comparison — design principle P3 enforces input-token
        # parity, not supervised-target parity.
        "mean_masked_targets_per_sample": float(np.mean(elig_counts)) * mask_ratio,
        "masked_token_fraction": float(np.mean(elig_counts)) * mask_ratio / (H * W),
        # Eligible d's whose min(row_span, col_span) == 1: a same-index partner sits
        # 4-adjacent, so the index leaks locally (see dataset.py caveat / H7 diagnostic).
        "adjacent_partner_frac": (elig_adjacent_partner / elig_total) if elig_total else 0.0,
        "zero_eligible_pictures": zero_elig,
        "zero_eligible_frac": zero_elig / n,
        "action_counts": trace,
        "gen_only_samples_per_sec": per_sec,
        "validated_loop_seconds": gen_seconds,
        "throughput_note": (
            "on-the-fly OK" if per_sec >= 200 else "below ~200/s single-core: pre-generate"
        ),
    }


def _plot(metrics: dict, figures_dir, out: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, title, xlabel in (
        (axes[0], "match_distance_hist", "match distance max(dr, dc)", "cells (odd)"),
        (axes[1], "depth_hist", "nesting depth", "strictly-containing boxes"),
    ):
        hist = {int(kk): v for kk, v in metrics[key].items()}
        ax.bar(list(hist.keys()), list(hist.values()), color="#4878cf")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[2].axis("off")  # summary text panel
    axes[2].text(
        0.0,
        0.5,
        f"n={metrics['n_samples']}  k={metrics['k']}\n"
        f"mean eligible d/picture: {metrics['mean_eligible_d']:.1f}\n"
        f"masked targets/sample @ratio: {metrics['mean_masked_targets_per_sample']:.1f} "
        f"({metrics['masked_token_fraction']:.1%} of tokens;\n"
        f"  1D close-only anchor ~25%)\n"
        f"adjacent-partner among eligible: {metrics['adjacent_partner_frac']:.1%}\n"
        f"zero-eligible: {metrics['zero_eligible_frac']:.2%}\n"
        f"2x2 fraction: {metrics['mean_frac_2x2']:.2%}\n"
        f"gen throughput: {metrics['gen_only_samples_per_sec']:.0f}/s/core",
        fontsize=10,
        va="center",
    )
    fig.suptitle(f"DW_{metrics['k']} sampler structure statistics")
    fig.tight_layout()
    path = figures_dir / f"{out}_structure.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="DW_k sampler property gate + structure stats.")
    ap.add_argument("--n", type=int, default=100_000, help="samples for the membership gate")
    ap.add_argument("--config", default=None, help="warm-up YAML to take dyck2d/masking params from")
    ap.add_argument("--out", default="dw32-stats", help="results/reports/<out>/ run name")
    ap.add_argument("--no-tier2", action="store_true", help="skip the Tier-2 certifier (faster)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    metrics = run_gate(
        n=args.n,
        k=cfg.dyck2d.k,
        H=cfg.grid.H,
        W=cfg.grid.W,
        p_acc=cfg.dyck2d.p_acc,
        open_prob=cfg.dyck2d.open_prob,
        min_match_distance=cfg.dyck2d.min_match_distance,
        mask_ratio=cfg.masking.mask_ratio,
        tier2=not args.no_tier2,
        seed=args.seed,
    )
    # DataLoader-path throughput too (includes tensor packing).
    ds = Dyck2DGrid(cfg)
    t0 = time.perf_counter()
    for i in range(500):
        ds[i]
    metrics["dataset_samples_per_sec"] = 500 / (time.perf_counter() - t0)

    run_dir = RunDir.create(cfg.results_dir, args.out)
    fig_path = _plot(metrics, run_dir.figures_dir, args.out)
    run_dir.save_metrics(metrics)
    run_dir.write_report(
        title=f"DW_{cfg.dyck2d.k} sampler gate ({args.n} samples)",
        sections={
            "Gates": (
                f"- membership (Tier 1 + {'Tier 2' if not args.no_tier2 else 'Tier 1 only'}): "
                f"**PASS** over {args.n} samples\n"
                f"- corner-mask determinacy: {metrics['audits_fully_determined']}/{metrics['audits']} "
                f"audits fully determined\n"
                f"- throughput: {metrics['gen_only_samples_per_sec']:.0f} samples/s/core "
                f"(dataset path {metrics['dataset_samples_per_sec']:.0f}/s) — "
                f"{metrics['throughput_note']}"
            ),
            "Structure": (
                f"- mean eligible d-cells/picture: {metrics['mean_eligible_d']:.1f} "
                f"(zero-eligible: {metrics['zero_eligible_frac']:.2%})\n"
                f"- supervision density: {metrics['mean_masked_targets_per_sample']:.1f} "
                f"masked targets/sample = {metrics['masked_token_fraction']:.1%} of tokens "
                f"(1D close-only anchor ~25% — report beside any A-vs-B verdict; "
                f"the dw32-dense arm at mask_ratio 1.0 probes this axis)\n"
                f"- adjacent-partner fraction among eligible d's: "
                f"{metrics['adjacent_partner_frac']:.1%} (min-span-1 border pairs — index "
                f"leaks locally; read the H7 masked-accuracy-vs-distance diagnostic before "
                f"crediting long-range binding)\n"
                f"- 2x2-rectangle fraction: {metrics['mean_frac_2x2']:.2%}\n"
                f"- generator actions: {metrics['action_counts']}\n\n"
                "Known support bias: binary guillotine splits realize only the guillotine "
                "subset of the Simplot closure; non-guillotine tessellations are in DW_k "
                "but never sampled (see generator docstring)."
            ),
        },
        figures=[str(fig_path)],
    )
    print(f"[dw2d-stats] PASS — report at {run_dir.root / 'report.md'}")


if __name__ == "__main__":
    main()
