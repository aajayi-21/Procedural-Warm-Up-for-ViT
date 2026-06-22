"""Generate the CA-generator validation report (a documented 'major step').

Produces a spacetime-diagram panel spanning the Wolfram classes and a Lempel-Ziv
complexity table, writing figures to ``results/figures/`` and a markdown report +
``complexity.json`` to ``results/reports/ca-generator/``. Run after building the generator
to visually confirm correctness and record the complexity ordering used by the rule sweep.

    python -m procedural_warmup.analysis.report
"""

from __future__ import annotations

import argparse

from procedural_warmup.analysis import complexity
from procedural_warmup.analysis.figures import plot_ca_spacetime_panel
from procedural_warmup.utils import RunDir


def generate(results_dir: str = "results") -> None:
    run = RunDir.create(results_dir, "ca-generator")

    # One representative rule per Wolfram class for the spacetime panel.
    rep_rules = [0, 108, 30, 110]  # I, II, III, IV
    panel = plot_ca_spacetime_panel(
        rep_rules, run.figures_dir, width=160, rows=160, burn_in=0,
        filename="ca_spacetime_panel.png",
    )

    scores = complexity.panel_complexity()
    scores.sort(key=lambda s: s["lz_norm"])
    run.save_metrics({"complexity": scores})

    # Markdown complexity table, ascending by normalized LZ.
    header = "| rule | class | LZ | LZ (norm) | density |\n|---|---|---|---|---|"
    body = "\n".join(
        f"| {s['rule']} | {s['class']} | {s['lz']} | {s['lz_norm']:.3f} | {s['density']:.3f} |"
        for s in scores
    )
    run.write_report(
        title="Cellular-automata generator validation",
        sections={
            "Spacetime diagrams": (
                "Representative rules per Wolfram class (time flows downward). Rule 0 is "
                "homogeneous (I), 108 periodic (II), 30 chaotic (III), 110 complex/"
                "edge-of-chaos (IV)."
            ),
            "Complexity ordering (Lempel-Ziv)": (
                "Operational complexity used as the independent variable for the rule "
                "sweep, ascending:\n\n" + header + "\n" + body
            ),
        },
        figures=[str(panel)],
    )
    print(f"[report] wrote {run.root / 'report.md'} and {panel}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CA-generator validation report.")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    generate(args.results_dir)


if __name__ == "__main__":
    main()
