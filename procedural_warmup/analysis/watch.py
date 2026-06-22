"""Live progress monitor for a training run (terminal-friendly, works over SSH).

Reads a run's ``log.csv`` (streamed by the trainer) and prints the latest metrics, a
compact ASCII sparkline of the key curve, and overall progress (from the run's saved
config). Use ``--live`` to refresh on an interval, and ``--figure`` to (re)generate the PNG.

    python -m procedural_warmup.analysis.watch ca-rule110 --live
    python -m procedural_warmup.analysis.watch cifar100-rule110 --live --interval 10
    python -m procedural_warmup.analysis.watch ca-rule110 --figure   # regenerate the PNG

Works for both warm-up runs (step/loss/acc) and downstream runs (epoch/val_top1).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from procedural_warmup.utils import read_csv_log

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    """Unicode sparkline; downsamples to ``width`` points."""
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, int((v - lo) / rng * (len(_BLOCKS) - 1)))]
                   for v in values)


def _load_total(results_dir: str, run_name: str) -> tuple[str, int | None]:
    """Return (unit, total) for the progress readout, from the saved config."""
    cfg_path = Path(results_dir) / "reports" / run_name / "config.yaml"
    if not cfg_path.exists():
        return "step", None
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    if "training" in cfg:  # warm-up config
        return "step", cfg.get("training", {}).get("steps")
    if "train" in cfg:  # downstream config
        return "epoch", cfg.get("train", {}).get("epochs")
    return "step", None


def render(results_dir: str, run_name: str) -> str:
    try:
        rows = read_csv_log(results_dir, run_name)
    except FileNotFoundError:
        return f"[{run_name}] no log yet — waiting for the run to produce data…"
    if not rows:
        return f"[{run_name}] log is empty — waiting…"

    keys = rows[0].keys()
    unit, total = _load_total(results_dir, run_name)
    last = rows[-1]
    lines = [f"=== {run_name} ==="]

    if "step" in keys:  # warm-up
        cur = int(last["step"])
        pct = f" ({100 * cur / total:.1f}%)" if total else ""
        lines.append(f"step {cur}/{total or '?'}{pct} | loss {last['loss']:.4f} | "
                     f"acc {last['acc']:.3f} | lr {last['lr']:.2e}")
        lines.append("loss " + sparkline([r["loss"] for r in rows]))
        lines.append("acc  " + sparkline([r["acc"] for r in rows]))
    else:  # downstream
        cur = int(last["epoch"])
        pct = f" ({100 * (cur + 1) / total:.1f}%)" if total else ""
        best = max(r["val_top1"] for r in rows)
        lines.append(f"epoch {cur}/{total or '?'}{pct} | top1 {last['val_top1']:.2f} | "
                     f"best {best:.2f} | train_loss {last['train_loss']:.4f}")
        lines.append("top1 " + sparkline([r["val_top1"] for r in rows]))
        lines.append("loss " + sparkline([r["train_loss"] for r in rows]))
    return "\n".join(lines)


def regenerate_figure(results_dir: str, run_name: str) -> None:
    from procedural_warmup.analysis.figures import (
        plot_downstream_curve,
        plot_warmup_curves,
    )

    unit, _ = _load_total(results_dir, run_name)
    figures_dir = Path(results_dir) / "figures"
    fn = plot_warmup_curves if unit == "step" else plot_downstream_curve
    out = fn(results_dir, run_name, figures_dir)
    print(f"wrote {out}" if out else "no data to plot yet")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live training-run monitor.")
    ap.add_argument("run_name")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--live", action="store_true", help="refresh continuously")
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds (--live)")
    ap.add_argument("--figure", action="store_true", help="regenerate the PNG and exit")
    args = ap.parse_args()

    if args.figure:
        regenerate_figure(args.results_dir, args.run_name)
        return
    if not args.live:
        print(render(args.results_dir, args.run_name))
        return
    try:
        while True:
            print("\033[2J\033[H", end="")  # clear screen
            print(render(args.results_dir, args.run_name))
            print(f"\n(refreshing every {args.interval:g}s — Ctrl-C to stop)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
