"""Build a comparison figure + report from finished runs.

Reads each downstream run's ``metrics.json`` (``best_top1``) and emits a bar chart plus a
markdown report summarizing the comparison. Optionally overlays the corresponding
*warm-up* training curves (``--warmup-runs``) so a report contrasts both the pretraining
dynamics and the downstream outcome — the head-to-head deliverable for source-vs-source
questions like H6's k-Dyck vs DW_32.

    python -m procedural_warmup.analysis.compare --out focused_rule110 \
        --runs cifar100-random:Random cifar100-rule110:CA-Rule110 cifar100-dyck:k-Dyck

    python -m procedural_warmup.analysis.compare --out dw32_vs_kdyck_cifar100 \
        --title "CIFAR-100 top-1: 1D k-Dyck vs 2D DW_32 (H6)" \
        --runs cifar100-random-repro:Random cifar100-dyck-repro:"k-Dyck 1D" \
               cifar100-dw32:"DW32 2D" cifar100-dw32-shuffle:"DW32 shuffle" \
        --warmup-runs dyck-repro:"k-Dyck 1D" dw32-vit-t:"DW32 2D" \
        --summary "H6 bands: success A>=72.0 and A-D>=1.0; partial 70.7<=A<72.0; ..."
"""

from __future__ import annotations

import argparse

from procedural_warmup.analysis.figures import (
    plot_downstream_comparison,
    plot_warmup_comparison,
)
from procedural_warmup.utils import RunDir, load_metrics

_DEFAULT_SUMMARY = (
    "Success criterion (Stage 1): the CA (Rule 110) warm-up should match or "
    "exceed random initialization, ideally approaching or beating k-Dyck."
)


def build(
    results_dir: str,
    runs: list[tuple[str, str]],
    out_name: str,
    title: str,
    warmup_runs: list[tuple[str, str]] | None = None,
    warmup_note: str = "",
    summary: str | None = None,
) -> None:
    run_dir = RunDir.create(results_dir, out_name)
    figures = []
    fig = plot_downstream_comparison(
        results_dir, runs, run_dir.figures_dir, metric="best_top1",
        title=title, filename=f"{out_name}.png",
    )
    if fig:
        figures.append(str(fig))
    if warmup_runs:
        wfig = plot_warmup_comparison(
            results_dir, warmup_runs, run_dir.figures_dir,
            title=f"Warm-up dynamics — {title}", note=warmup_note,
            filename=f"{out_name}_warmup.png",
        )
        if wfig:
            figures.append(str(wfig))

    rows = []
    best_label, best_val = None, -1.0
    for run_name, label in runs:
        try:
            m = load_metrics(results_dir, run_name)
        except FileNotFoundError:
            print(f"[compare] skipping '{run_name}' (no metrics.json yet)")
            continue
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
    summary_text = summary if summary is not None else _DEFAULT_SUMMARY
    run_dir.write_report(
        title=title,
        sections={
            "Results": table,
            "Summary": (
                f"Best initialization: **{best_label}** ({best_val:.2f}% top-1).\n\n"
                + summary_text
            ),
        },
        figures=figures or None,
    )
    print(f"[report] wrote {run_dir.root / 'report.md'}")


def _parse_run(s: str) -> tuple[str, str]:
    run_name, _, label = s.partition(":")
    return run_name, (label or run_name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run-comparison figure + report builder.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="focused_rule110", help="output run/report name")
    ap.add_argument("--title", default="CIFAR-100 top-1 by warm-up initialization")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="space-separated downstream run_name:label pairs")
    ap.add_argument("--warmup-runs", nargs="+", default=None,
                    help="optional warm-up run_name:label pairs -> curve-overlay figure")
    ap.add_argument("--warmup-note", default="",
                    help="figure note, e.g. differing masked-accuracy chance floors")
    ap.add_argument("--summary", default=None,
                    help="override the report's Summary paragraph (e.g. H6 decision bands)")
    args = ap.parse_args()
    runs = [_parse_run(s) for s in args.runs]
    warmup_runs = [_parse_run(s) for s in args.warmup_runs] if args.warmup_runs else None
    build(args.results_dir, runs, args.out, args.title,
          warmup_runs=warmup_runs, warmup_note=args.warmup_note, summary=args.summary)


if __name__ == "__main__":
    main()
