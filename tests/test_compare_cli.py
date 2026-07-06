"""The compare builder's head-to-head path: downstream bars + warm-up overlay + summary."""

import csv
import json
from pathlib import Path

from procedural_warmup.analysis.compare import build


def _fake_downstream(results: Path, run: str, top1: float) -> None:
    d = results / "reports" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(
        json.dumps({"run_name": run, "dataset": "CIFAR100", "best_top1": top1})
    )


def _fake_warmup_log(results: Path, run: str) -> None:
    d = results / "reports" / run
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "acc", "lr"])
        w.writeheader()
        for step in (50, 100, 150):
            w.writerow({"step": step, "loss": 4.0 - step / 100, "acc": step / 300, "lr": 2e-3})


def test_build_head_to_head(tmp_path):
    results = tmp_path / "results"
    _fake_downstream(results, "cifar100-dyck-repro", 72.55)
    _fake_downstream(results, "cifar100-dw32", 71.90)
    _fake_warmup_log(results, "dyck-repro")
    _fake_warmup_log(results, "dw32-vit-t")

    build(
        str(results),
        [("cifar100-dyck-repro", "k-Dyck 1D"), ("cifar100-dw32", "DW32 2D"),
         ("cifar100-missing", "skipped")],
        out_name="h2h",
        title="k-Dyck vs DW32",
        warmup_runs=[("dyck-repro", "k-Dyck 1D"), ("dw32-vit-t", "DW32 2D")],
        warmup_note="chance floors 1/64 vs 1/32",
        summary="H6 bands: custom summary text.",
    )

    report = (results / "reports" / "h2h" / "report.md").read_text(encoding="utf-8")
    assert "72.55" in report and "71.90" in report
    assert "H6 bands: custom summary text." in report  # override, not the stale default
    assert "Rule 110" not in report
    assert (results / "figures" / "h2h.png").exists()
    assert (results / "figures" / "h2h_warmup.png").exists()
    meta = json.loads((results / "reports" / "h2h" / "metrics.json").read_text())
    assert meta["winner"] == "k-Dyck 1D"
    assert len(meta["comparison"]) == 2  # the missing run was skipped, not fatal
