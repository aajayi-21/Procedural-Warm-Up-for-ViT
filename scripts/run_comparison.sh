#!/usr/bin/env bash
# Run the complete pretraining-method comparison (additive + substitutive).
#   scripts/run_comparison.sh [--dry-run]
# Resumable: re-running continues where it left off and rebuilds the report.
set -euo pipefail
python -m procedural_warmup.experiments.comparison \
  --config procedural_warmup/config/files/comparison.yaml "$@"
echo "See results/reports/comparison/report.md"
