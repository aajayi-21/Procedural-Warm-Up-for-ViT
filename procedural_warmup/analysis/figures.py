"""Reproducible figure generation.

Every figure is produced in code and regenerable from saved results under
``results/reports/*`` (CLAUDE.md "Figures"). Figures are written to ``results/figures/``.
The non-interactive Agg backend is used so figures render headless on the training boxes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from procedural_warmup.utils import load_metrics, read_csv_log  # noqa: E402


def _ensure(out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------------------
# Warm-up training curves (called automatically at the end of a warm-up run)
# --------------------------------------------------------------------------------------


def plot_warmup_curves(results_dir: str, run_name: str, out_dir) -> Optional[Path]:
    """Loss and masked-token accuracy vs step, from a run's ``log.csv``."""
    try:
        rows = read_csv_log(results_dir, run_name)
    except FileNotFoundError:
        return None
    if not rows:
        return None
    steps = [r["step"] for r in rows]
    loss = [r["loss"] for r in rows]
    acc = [r["acc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, loss, color="tab:red", label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("masked-token loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(steps, acc, color="tab:blue", label="accuracy")
    ax2.set_ylabel("masked-token accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 1)
    plt.title(f"Warm-up curves — {run_name}")
    fig.tight_layout()

    out = _ensure(out_dir) / f"{run_name}_warmup_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_warmup_comparison(
    results_dir: str,
    runs: list[tuple[str, str]],
    out_dir,
    title: str = "Warm-up curves by source",
    note: str = "",
    filename: str = "warmup_comparison.png",
) -> Optional[Path]:
    """Overlay several warm-up runs' loss and masked-accuracy curves (from log.csv).

    ``runs`` is ``(run_name, label)`` pairs; runs without a ``log.csv`` yet are skipped
    with a warning. Masked-accuracy floors differ across sources (e.g. 1/64 for k-Dyck
    closers vs 1/32 for DW d-corners) — pass ``note`` to state them on the figure.
    """
    curves = []
    for run_name, label in runs:
        try:
            rows = read_csv_log(results_dir, run_name)
        except FileNotFoundError:
            print(f"[compare] skipping warm-up '{run_name}' (no log.csv yet)")
            continue
        if rows:
            curves.append((label, rows))
    if not curves:
        return None

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for label, rows in curves:
        steps = [r["step"] for r in rows]
        ax_loss.plot(steps, [r["loss"] for r in rows], label=label, lw=1.2)
        ax_acc.plot(steps, [r["acc"] for r in rows], label=label, lw=1.2)
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("masked-token loss")
    ax_loss.legend(fontsize=8)
    ax_acc.set_xlabel("step")
    ax_acc.set_ylabel("masked-token accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(fontsize=8)
    fig.suptitle(title + (f"\n{note}" if note else ""), fontsize=11)
    fig.tight_layout()
    out = _ensure(out_dir) / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_downstream_curve(results_dir: str, run_name: str, out_dir) -> Optional[Path]:
    """Validation top-1/top-5 and train loss vs epoch, from a run's ``log.csv``."""
    try:
        rows = read_csv_log(results_dir, run_name)
    except FileNotFoundError:
        return None
    if not rows:
        return None
    epochs = [r["epoch"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [r["val_top1"] for r in rows], color="tab:green", label="val top-1")
    ax1.plot(epochs, [r["val_top5"] for r in rows], color="tab:olive",
             linestyle="--", label="val top-5")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy (%)")
    ax1.legend(loc="lower right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [r["train_loss"] for r in rows], color="tab:red", alpha=0.5)
    ax2.set_ylabel("train loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.title(f"Downstream curves — {run_name}")
    fig.tight_layout()

    out = _ensure(out_dir) / f"{run_name}_downstream_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# CA spacetime sanity panels (generator validation)
# --------------------------------------------------------------------------------------


def plot_ca_spacetime_panel(
    rules: list[int],
    out_dir,
    width: int = 128,
    rows: int = 128,
    burn_in: int = 0,
    seed: int = 0,
    filename: str = "ca_spacetime_panel.png",
) -> Path:
    """Render spacetime diagrams for several rules side by side (visual sanity check)."""
    from procedural_warmup.analysis.complexity import rule_class
    from procedural_warmup.data.ca.eca import simulate_spacetime

    fig, axes = plt.subplots(1, len(rules), figsize=(3 * len(rules), 3.2))
    if len(rules) == 1:
        axes = [axes]
    for ax, rule in zip(axes, rules):
        rng = np.random.default_rng(seed)
        st = simulate_spacetime(rule, width=width, n_rows=rows, burn_in=burn_in, rng=rng)
        ax.imshow(st, cmap="binary", interpolation="nearest", aspect="auto")
        cls = rule_class(rule)
        ax.set_title(f"Rule {rule}" + (f" (Class {cls})" if cls else ""))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("ECA spacetime diagrams (time downward)")
    fig.tight_layout()
    out = _ensure(out_dir) / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Downstream comparison (the focused-experiment headline)
# --------------------------------------------------------------------------------------


def plot_downstream_comparison(
    results_dir: str,
    runs: list[tuple[str, str]],
    out_dir,
    metric: str = "best_top1",
    title: str = "Downstream top-1 by initialization",
    filename: str = "downstream_comparison.png",
) -> Path:
    """Bar chart of a downstream metric across runs.

    ``runs`` is a list of ``(run_name, label)`` pairs; each run's ``metrics.json`` must
    contain ``metric``. Runs without a ``metrics.json`` yet are skipped (with a warning) so a
    partially-complete experiment still charts the finished runs instead of crashing.
    """
    labels, values = [], []
    for run_name, label in runs:
        try:
            m = load_metrics(results_dir, run_name)
        except FileNotFoundError:
            print(f"[compare] skipping '{run_name}' (no metrics.json yet)")
            continue
        labels.append(label)
        values.append(m.get(metric, float("nan")))

    fig, ax = plt.subplots(figsize=(1.6 * len(runs) + 1, 4))
    bars = ax.bar(labels, values, color="tab:green")
    ax.set_ylabel(metric)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = _ensure(out_dir) / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Additive comparison (warm-up methods at full data) + substitutive (data-efficiency)
# --------------------------------------------------------------------------------------


def plot_additive_comparison(rows: list[dict], out_dir, dataset: str,
                             baseline_method: str = "random",
                             filename: str = "additive_comparison.png") -> Path:
    """Bar chart of best top-1 by warm-up method at full data, vs the random baseline.

    ``rows`` items: ``{"method": str, "best_top1": float}``. The baseline method's accuracy
    is drawn as a horizontal line and each bar is annotated with its delta over it.
    """
    rows = sorted(rows, key=lambda r: r["best_top1"])
    methods = [r["method"] for r in rows]
    vals = [r["best_top1"] for r in rows]
    base = next((r["best_top1"] for r in rows if r["method"] == baseline_method), None)

    fig, ax = plt.subplots(figsize=(1.5 * len(rows) + 1.5, 4.5))
    colors = ["tab:gray" if m == baseline_method else "tab:green" for m in methods]
    bars = ax.bar(methods, vals, color=colors)
    if base is not None:
        ax.axhline(base, color="tab:red", linestyle="--", linewidth=1,
                   label=f"{baseline_method} ({base:.2f})")
        ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        delta = f"\nΔ{v - base:+.2f}" if base is not None and v != base else ""
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}{delta}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("best top-1 (%)")
    ax.set_title(f"Additive comparison — {dataset} (full data)")
    ax.set_ylim(0, max(vals) * 1.12)
    fig.tight_layout()
    out = _ensure(out_dir) / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_substitutive_curves(rows: list[dict], out_dir, dataset: str,
                             baseline_top1: float | None, train_size: int,
                             filename: str = "substitutive_curves.png") -> Path:
    """Data-efficiency curves: best top-1 vs fraction of training images, per method.

    ``rows`` items: ``{"method": str, "fraction": float, "best_top1": float}``. The
    ``baseline_top1`` (random init at full data) is drawn as a horizontal target line; where
    a warm-up curve crosses it marks the data it can save.
    """
    by_method: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append((r["fraction"], r["best_top1"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, pts in sorted(by_method.items()):
        pts = sorted(pts)
        xs = [f * 100 for f, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, marker="o", label=method)
    if baseline_top1 is not None:
        ax.axhline(baseline_top1, color="tab:red", linestyle="--", linewidth=1,
                   label=f"random @100% ({baseline_top1:.2f})")
    ax.set_xlabel(f"% of {dataset} training images  (100% = {train_size:,})")
    ax.set_ylabel("best top-1 (%)")
    ax.set_title(f"Substitutive (data-efficiency) — {dataset}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure(out_dir) / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Rule-complexity sweep (edge-of-chaos) — scaffold, used once the sweep is run
# --------------------------------------------------------------------------------------


def plot_rule_complexity_sweep(
    points: list[dict],
    out_dir,
    filename: str = "rule_complexity_sweep.png",
) -> Path:
    """Scatter downstream accuracy vs Lempel-Ziv complexity, colored by Wolfram class.

    ``points`` is a list of dicts with keys ``lz_norm``, ``accuracy``, ``rule``, ``class``.
    """
    class_colors = {"I": "tab:gray", "II": "tab:blue", "III": "tab:orange", "IV": "tab:red"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for p in points:
        ax.scatter(p["lz_norm"], p["accuracy"],
                   color=class_colors.get(p.get("class"), "black"), s=40)
        ax.annotate(str(p["rule"]), (p["lz_norm"], p["accuracy"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Lempel-Ziv complexity (normalized)")
    ax.set_ylabel("downstream top-1")
    ax.set_title("Edge-of-chaos sweep: accuracy vs rule complexity")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=f"Class {k}")
               for k, c in class_colors.items()]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    out = _ensure(out_dir) / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
