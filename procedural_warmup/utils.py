"""Shared utilities: seeding, device, metric tracking, and results/report IO.

The results helpers exist because the project requires every major step and experimental
run to produce figures *and* a written report, regenerable from saved data (see CLAUDE.md
"Figures"). A :class:`RunDir` owns a per-run folder under ``results/reports/<run>/`` that
holds the frozen config, a per-step/epoch CSV log, a final metrics JSON, and a markdown
report. Figures live under ``results/figures/``.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "cuda") -> torch.device:
    """Return the requested device, transparently falling back to CPU if CUDA is absent."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable -> falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------------------
# Metric tracking
# --------------------------------------------------------------------------------------


class AverageMeter:
    """Tracks a running average of a scalar metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def now_iso() -> str:
    """UTC timestamp (avoids importing datetime elsewhere)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------
# Results / reporting
# --------------------------------------------------------------------------------------


@dataclass
class RunDir:
    """A per-run results folder: config + CSV log + metrics JSON + markdown report.

    Layout::

        results/reports/<run_name>/
            config.yaml      # frozen config for the run
            log.csv          # per-step (warm-up) or per-epoch (downstream) metrics
            metrics.json     # final/summary metrics
            report.md        # human-readable report (links to figures)
    """

    root: Path
    figures_dir: Path
    _csv_header_written: bool = False

    @classmethod
    def create(cls, results_dir: str, run_name: str) -> "RunDir":
        root = Path(results_dir) / "reports" / run_name
        figures = Path(results_dir) / "figures"
        root.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        # A re-run (e.g. after an aborted warm-up) must not append rows to the previous
        # attempt's CSV; rotate it aside for forensics instead.
        log = root / "log.csv"
        if log.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log.rename(root / f"log.{stamp}.csv.bak")
        return cls(root=root, figures_dir=figures)

    # ---- config ----
    def save_config(self, cfg_dict: dict) -> None:
        import yaml

        with open(self.root / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)

    # ---- streaming log ----
    def log_row(self, row: dict) -> None:
        """Append a row to ``log.csv`` (header inferred from the first row)."""
        path = self.root / "log.csv"
        write_header = not self._csv_header_written and not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._csv_header_written = True

    # ---- final metrics ----
    def save_metrics(self, metrics: dict) -> None:
        with open(self.root / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # ---- report ----
    def write_report(self, title: str, sections: dict[str, str],
                     figures: Optional[list[str]] = None) -> None:
        """Write ``report.md``. ``sections`` maps headings to markdown bodies;
        ``figures`` is a list of figure paths (relative links are computed)."""
        lines = [f"# {title}", "", f"_Generated {now_iso()}_", ""]
        for heading, body in sections.items():
            lines += [f"## {heading}", "", body, ""]
        if figures:
            lines += ["## Figures", ""]
            for fig in figures:
                rel = os.path.relpath(fig, self.root)
                lines += [f"![{Path(fig).stem}]({rel})", ""]
        # UTF-8 explicitly: reports contain non-cp1252 glyphs and must write on Windows.
        (self.root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def load_metrics(results_dir: str, run_name: str) -> dict:
    """Load a run's ``metrics.json`` (used by figure regeneration)."""
    path = Path(results_dir) / "reports" / run_name / "metrics.json"
    with open(path) as f:
        return json.load(f)


def read_csv_log(results_dir: str, run_name: str) -> list[dict]:
    """Read a run's ``log.csv`` back into a list of dict rows (values as floats)."""
    path = Path(results_dir) / "reports" / run_name / "log.csv"
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows
