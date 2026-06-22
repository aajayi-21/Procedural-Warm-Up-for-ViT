"""Data-fraction subset + the additive/substitutive comparison report."""

import json
from pathlib import Path

from procedural_warmup.analysis.comparison import build_report
from procedural_warmup.downstream.datasets import stratified_subset
from procedural_warmup.experiments.comparison import _build_matrix


class _Dummy:
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return i, self.targets[i]


def test_stratified_subset_size_and_balance():
    ds = _Dummy([0] * 100 + [1] * 100)
    sub = stratified_subset(ds, 0.25, seed=0)
    assert len(sub) == 50
    labels = [ds.targets[i] for i in sub.indices]
    assert labels.count(0) == 25 and labels.count(1) == 25


def test_build_matrix_dedup_and_fractions():
    spec = {
        "dataset": "CIFAR100",
        "methods": {"random": "none", "ca": "ca"},
        "substitutive": {"methods": ["random", "ca"], "fractions": [0.25, 0.5]},
    }
    matrix = _build_matrix(spec)
    keys = {(r["method"], r["fraction"]) for r in matrix}
    # additive (frac 1.0) for both + substitutive (0.25, 0.5) for both = 6 unique runs
    assert keys == {("random", 1.0), ("ca", 1.0),
                    ("random", 0.25), ("ca", 0.25),
                    ("random", 0.5), ("ca", 0.5)}


def _write_metrics(root: Path, run_name: str, top1: float):
    d = root / "reports" / run_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"best_top1": top1, "final_top1": top1}))


def test_build_report_additive_and_substitutive(tmp_path):
    results = tmp_path
    # synthetic runs: random/ca/dyck at full data + a substitutive sweep
    data = [
        ("cifar100-random-f100", "random", 1.0, 68.0),
        ("cifar100-ca-f100", "ca", 1.0, 71.5),
        ("cifar100-dyck-f100", "dyck", 1.0, 72.0),
        ("cifar100-random-f50", "random", 0.5, 60.0),
        ("cifar100-ca-f50", "ca", 0.5, 68.5),   # ca@50% beats random@100% -> data saved
        ("cifar100-random-f25", "random", 0.25, 52.0),
        ("cifar100-ca-f25", "ca", 0.25, 63.0),
    ]
    runs = []
    for run_name, method, frac, top1 in data:
        _write_metrics(results, run_name, top1)
        runs.append({"run_name": run_name, "method": method,
                     "fraction": frac, "dataset": "CIFAR100"})

    report = build_report(str(results), runs, "CIFAR100", out_name="cmp")
    assert report.exists()
    text = report.read_text()
    assert "Additive" in text and "Substitutive" in text
    assert "Δ vs random" in text
    # ca@50% (68.5) >= random@100% (68.0) -> reports saving 50%
    assert "saves 50%" in text
    figs = list((results / "figures").glob("cmp_*.png"))
    assert any("additive" in f.name for f in figs)
    assert any("substitutive" in f.name for f in figs)
