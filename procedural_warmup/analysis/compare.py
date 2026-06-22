"""Build the focused-experiment comparison figure + report from finished runs.

Reads each downstream run's ``metrics.json`` (``best_top1``) and emits a bar chart plus a
markdown report summarizing the comparison. This is the deliverable for the focused
Rule-110-vs-baselines experiment.

    python -m procedural_warmup.analysis.compare --out focused_rule110 \
        --runs cifar100-random:Random cifar100-rule110:CA-Rule110 cifar100-dyck:k-Dyck
"""

from __future__ import annotations

import argparse

from procedural_warmup.analysis.figures import plot_downstream_comparison
from procedural_warmup.utils import RunDir, load_metrics


def build(results_dir: str, runs: list[tuple[str, str]], out_name: str, title: str) -> None:
    run_dir = RunDir.create(results_dir, out_name)
    fig = plot_downstream_comparison(
        results_dir, runs, run_dir.figures_dir, metric="best_top1",
        title=title, filename=f"{out_name}.png",
    )

    rows = []
    best_label, best_val = None, -1.0
    for run_name, label in runs:
        m = load_metrics(results_dir, run_name)
        v = m.get("best_top1", float("nan"))
        rows.append((label, run_name, v, m.get("dataset", "?")))
        if v > best_val:
            best_label, best_val = label, v

    table = "| init | run | best top-1 | dataset |\n|---|---|---|---|\n" + "\n".join(
        f"| {lbl} | `{rn}` | {v:.2f} | {ds} |" for lbl, rn, v, ds in rows
    )
    run_dir.save_metrics({"comparison": [
        {"label": lbl, "run": rn, "best_top1": v, "dataset": ds} for lbl, rn, v, ds in rows
    ], "winner": best_label})
    run_dir.write_report(
        title=title,
        sections={
            "Results": table,
            "Summary": (
                f"Best initialization: **{best_label}** ({best_val:.2f}% top-1).\n\n"
                "Success criterion (Stage 1): the CA (Rule 110) warm-up should match or "
                "exceed random initialization, ideally approaching or beating k-Dyck."
            ),
        },
        figures=[str(fig)] if fig else None,
    )
    print(f"[report] wrote {run_dir.root / 'report.md'}")


def _parse_run(s: str) -> tuple[str, str]:
    run_name, _, label = s.partition(":")
    return run_name, (label or run_name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Focused-experiment comparison report.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="focused_rule110", help="output run/report name")
    ap.add_argument("--title", default="CIFAR-100 top-1 by warm-up initialization")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="space-separated run_name:label pairs")
    args = ap.parse_args()
    runs = [_parse_run(s) for s in args.runs]
    build(args.results_dir, runs, args.out, args.title)


if __name__ == "__main__":
    main()
