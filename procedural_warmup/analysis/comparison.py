"""Build the complete pretraining-method comparison (additive + substitutive).

Reads each downstream run's ``metrics.json`` and produces, under
``results/reports/<out_name>/``:
- an **additive** table + bar chart: best top-1 of every warm-up method at full data, with
  the delta over random init (reproduces the paper's Table 2/4 comparison);
- a **substitutive** table + data-efficiency curves: best top-1 vs fraction of training
  images, with the random-at-full-data target line and the "data saved" estimate
  (reproduces the paper's Figure 3 "X% fewer images" result);
- a markdown report tying it together.

Callable standalone from a manifest (so it can be regenerated from saved results), or
in-process by the experiment orchestrator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from procedural_warmup.analysis.figures import (
    plot_additive_comparison,
    plot_substitutive_curves,
)
from procedural_warmup.utils import RunDir, load_metrics

# CIFAR train-set size (both variants), for the substitutive image-count axis.
_TRAIN_SIZE = {"CIFAR10": 50_000, "CIFAR100": 50_000}


def collect(results_dir: str, runs: list[dict]) -> list[dict]:
    """Attach ``best_top1`` from each run's metrics.json; drop runs not yet finished."""
    out = []
    for r in runs:
        try:
            m = load_metrics(results_dir, r["run_name"])
        except FileNotFoundError:
            continue
        out.append({**r, "best_top1": m.get("best_top1"),
                    "final_top1": m.get("final_top1")})
    return out


def _data_saved(method_rows: list[dict], baseline: float) -> str:
    """Smallest data fraction at which a method reaches the full-data random baseline."""
    hits = sorted(r["fraction"] for r in method_rows if r["best_top1"] >= baseline)
    if not hits:
        return "did not reach baseline within the swept fractions"
    f = hits[0]
    return f"matches random@100% using {f * 100:.0f}% of images (saves {(1 - f) * 100:.0f}%)"


def build_report(results_dir: str, runs: list[dict], dataset: str,
                 out_name: str = "comparison") -> Path:
    rows = collect(results_dir, runs)
    if not rows:
        raise RuntimeError("no finished runs found — run the downstream trainings first")

    run_dir = RunDir.create(results_dir, out_name)
    baseline = next((r["best_top1"] for r in rows
                     if r["method"] == "random" and r["fraction"] == 1.0), None)

    # ---- additive: full-data rows ----
    additive = [r for r in rows if r["fraction"] == 1.0]
    add_fig = plot_additive_comparison(additive, run_dir.figures_dir, dataset,
                                       filename=f"{out_name}_additive.png")
    add_tbl = "| method | best top-1 | Δ vs random |\n|---|---|---|\n" + "\n".join(
        f"| {r['method']} | {r['best_top1']:.2f} | "
        f"{'—' if baseline is None or r['method'] == 'random' else f'{r['best_top1'] - baseline:+.2f}'} |"
        for r in sorted(additive, key=lambda r: -r["best_top1"])
    )

    # ---- substitutive: all fractions ----
    sub_fig = None
    sub_tbl = "_no substitutive runs found_"
    saved_lines = ""
    has_sub = any(r["fraction"] < 1.0 for r in rows)
    if has_sub:
        train_size = _TRAIN_SIZE.get(dataset, 50_000)
        sub_fig = plot_substitutive_curves(rows, run_dir.figures_dir, dataset,
                                           baseline, train_size,
                                           filename=f"{out_name}_substitutive.png")
        fracs = sorted({r["fraction"] for r in rows})
        header = "| method | " + " | ".join(f"{int(f*100)}%" for f in fracs) + " |"
        sep = "|---|" + "---|" * len(fracs)
        body = []
        for method in sorted({r["method"] for r in rows}):
            cells = []
            for f in fracs:
                v = next((r["best_top1"] for r in rows
                          if r["method"] == method and r["fraction"] == f), None)
                cells.append(f"{v:.2f}" if v is not None else "—")
            body.append(f"| {method} | " + " | ".join(cells) + " |")
        sub_tbl = "\n".join([header, sep, *body])
        if baseline is not None:
            for method in sorted({r["method"] for r in rows if r["method"] != "random"}):
                mr = [r for r in rows if r["method"] == method]
                saved_lines += f"- **{method}**: {_data_saved(mr, baseline)}\n"

    figures = [str(add_fig)] + ([str(sub_fig)] if sub_fig else [])
    run_dir.save_metrics({"dataset": dataset, "baseline_random_top1": baseline, "rows": rows})
    run_dir.write_report(
        title=f"Pretraining-method comparison — {dataset}",
        sections={
            "Additive (full data)": (
                "Each warm-up applied then the model trained on the **full** training set; "
                "the gain over random init is the additive benefit.\n\n" + add_tbl
            ),
            "Substitutive (data efficiency)": (
                "Warm-up methods trained on fractions of the images; how much data each "
                "warm-up makes up for relative to random-init at full data.\n\n" + sub_tbl
                + ("\n\n**Data saved (to reach random@100%):**\n" + saved_lines
                   if saved_lines else "")
            ),
        },
        figures=figures,
    )
    print(f"[comparison] wrote {run_dir.root / 'report.md'}")
    return run_dir.root / "report.md"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the pretraining-method comparison report.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--manifest", required=True,
                    help="JSON file listing run descriptors (run_name/method/fraction/dataset)")
    ap.add_argument("--out", default="comparison")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    build_report(args.results_dir, manifest["runs"], manifest["dataset"], args.out)


if __name__ == "__main__":
    main()
