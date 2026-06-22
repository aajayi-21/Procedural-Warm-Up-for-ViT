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
    contain ``metric``.
    """
    labels, values = [], []
    for run_name, label in runs:
        m = load_metrics(results_dir, run_name)
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
