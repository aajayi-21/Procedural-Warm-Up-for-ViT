"""Driver for the complete pretraining-method comparison (additive + substitutive).

Given an experiment YAML, this:
  1. runs each method's warm-up (skipped if a stripped checkpoint already exists);
  2. trains on the **real** dataset for the additive runs (every method at full data) and
     the substitutive runs (selected methods at fractions of the data) — skipping any run
     whose ``metrics.json`` already exists, so the whole thing is resumable;
  3. writes a manifest and builds the comparison report (table + figures).

    python -m procedural_warmup.experiments.comparison \
        --config procedural_warmup/config/files/comparison.yaml [--dry-run] [--only-report]

Heavy: the default matrix is many 300-epoch trainings — intended for a CUDA box, run
incrementally (it resumes). Use --dry-run to see the plan and run count first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from procedural_warmup.config import load_config, load_downstream_config


def _stripped_ckpt_path(warmup_cfg_path: str) -> Path:
    """Where a warm-up config's stripped checkpoint lands (matches cli auto-strip naming)."""
    cfg = load_config(warmup_cfg_path)
    step = max(cfg.checkpoint.save_steps)
    return (Path(cfg.checkpoint.out_dir) / cfg.run_name
            / f"ckpt_step_{step:06d}_stripped.pt")


def _downstream_done(results_dir: str, run_name: str) -> bool:
    return (Path(results_dir) / "reports" / run_name / "metrics.json").exists()


def _build_matrix(spec: dict) -> list[dict]:
    """Expand the spec into a deduped list of downstream runs."""
    dataset = spec["dataset"]
    methods = spec["methods"]  # display name -> warmup key | "none"
    runs: dict[tuple[str, float], dict] = {}

    def add(method: str, fraction: float):
        key = (method, fraction)
        if key in runs:
            return
        tag = f"{dataset.lower()}-{method}-f{int(round(fraction * 100))}"
        runs[key] = {"run_name": tag, "method": method,
                     "fraction": fraction, "dataset": dataset}

    for method in methods:  # additive: every method at full data
        add(method, 1.0)
    sub = spec.get("substitutive") or {}
    for method in sub.get("methods", []):  # substitutive: fractions < 1
        for frac in sub.get("fractions", []):
            add(method, float(frac))
    return list(runs.values())


def run(spec: dict, dry_run: bool = False, only_report: bool = False) -> None:
    results_dir = spec.get("results_dir", "results")
    dataset = spec["dataset"]
    methods = spec["methods"]
    warmup_configs = spec.get("warmup_configs", {})
    matrix = _build_matrix(spec)

    # ---- resolve which warm-ups are needed and their stripped checkpoints ----
    method_ckpt: dict[str, str | None] = {}
    warmups_needed: list[tuple[str, str]] = []  # (warmup_key, config_path)
    for method, wkey in methods.items():
        if wkey in (None, "none"):
            method_ckpt[method] = None
            continue
        cfg_path = warmup_configs[wkey]
        ckpt = _stripped_ckpt_path(cfg_path)
        method_ckpt[method] = str(ckpt)
        if not ckpt.exists():
            warmups_needed.append((wkey, cfg_path))

    pending = [r for r in matrix if not _downstream_done(results_dir, r["run_name"])]

    print(f"=== comparison plan ({dataset}, {spec.get('epochs')} epochs/run) ===")
    print(f"warm-ups to run: {[w[0] for w in warmups_needed] or 'none (all cached)'}")
    print(f"downstream runs: {len(matrix)} total, {len(pending)} pending, "
          f"{len(matrix) - len(pending)} already done")
    for r in matrix:
        status = "done" if _downstream_done(results_dir, r["run_name"]) else "PENDING"
        init = "random" if method_ckpt[r["method"]] is None else "warm-up"
        print(f"  [{status:7}] {r['run_name']:28} method={r['method']:13} "
              f"frac={r['fraction']:.2f} init={init}")

    manifest_path = Path(results_dir) / "reports" / "comparison" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"dataset": dataset, "runs": matrix}, indent=2))

    if dry_run:
        print("\n(dry run — nothing executed)")
        return

    if not only_report:
        # ---- 1. warm-ups ----
        from procedural_warmup.warmup.cli import run as warmup_run
        for wkey, cfg_path in warmups_needed:
            print(f"\n=== warm-up: {wkey} ({cfg_path}) ===")
            wcfg = load_config(cfg_path)
            wcfg.results_dir = results_dir
            warmup_run(wcfg)

        # ---- 2. downstream trainings (resumable) ----
        from procedural_warmup.downstream.main import run as downstream_run
        for i, r in enumerate(matrix, 1):
            if _downstream_done(results_dir, r["run_name"]):
                print(f"\n[{i}/{len(matrix)}] skip {r['run_name']} (done)")
                continue
            print(f"\n[{i}/{len(matrix)}] train {r['run_name']} "
                  f"(method={r['method']}, frac={r['fraction']})")
            dcfg = load_downstream_config(spec["downstream_config"])
            dcfg.results_dir = results_dir
            dcfg.seed = spec.get("seed", dcfg.seed)
            dcfg.run_name = r["run_name"]
            dcfg.data.dataset = dataset
            dcfg.data.train_fraction = r["fraction"]
            dcfg.train.epochs = spec.get("epochs", dcfg.train.epochs)
            dcfg.init_checkpoint = method_ckpt[r["method"]]
            downstream_run(dcfg)

    # ---- 3. comparison report ----
    from procedural_warmup.analysis.comparison import build_report
    build_report(results_dir, matrix, dataset, out_name="comparison")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretraining-method comparison driver.")
    ap.add_argument("--config", required=True, help="experiment YAML")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--only-report", action="store_true",
                    help="skip training; just (re)build the report from finished runs")
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    run(spec, dry_run=args.dry_run, only_report=args.only_report)


if __name__ == "__main__":
    main()
