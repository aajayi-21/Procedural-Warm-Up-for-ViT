#!/usr/bin/env bash
# Focused Stage-1 experiment: ECA Rule 110 vs random-init vs k-Dyck on CIFAR-100 (+ CIFAR-10).
# Chains warm-up -> strip -> downstream x3 -> comparison figure + report.
#
#   scripts/run_focused_experiment.sh [DATASET]   # default CIFAR100
#
# Heavy (two 15k-step warm-ups + three 300-epoch trainings); intended for a CUDA box.
set -euo pipefail
DATASET="${1:-CIFAR100}"
TAG="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
CFG_DIR=procedural_warmup/config/files

echo "== 0. CA generator validation report =="
python -m procedural_warmup.analysis.report

echo "== 1. Warm-up: ECA Rule 110 =="
python -m procedural_warmup.warmup.cli --config "$CFG_DIR/ca-rule110.yaml"

echo "== 2. Warm-up: k-Dyck baseline =="
python -m procedural_warmup.warmup.cli --config "$CFG_DIR/dyck-vit-t.yaml"

CA_CKPT=checkpoints/ca-rule110/ckpt_step_015000_stripped.pt
DYCK_CKPT=checkpoints/dyck-vit-t/ckpt_step_015000_stripped.pt

echo "== 3. Downstream: random init =="
scripts/run_downstream.sh "$DATASET" "${TAG}-random" none

echo "== 4. Downstream: CA Rule 110 warm-up =="
scripts/run_downstream.sh "$DATASET" "${TAG}-rule110" "$CA_CKPT"

echo "== 5. Downstream: k-Dyck warm-up =="
scripts/run_downstream.sh "$DATASET" "${TAG}-dyck" "$DYCK_CKPT"

echo "== 6. Comparison figure + report =="
python -m procedural_warmup.analysis.compare \
  --out "focused_rule110_${TAG}" \
  --title "${DATASET} top-1 by warm-up initialization" \
  --runs "${TAG}-random:Random" "${TAG}-rule110:CA-Rule110" "${TAG}-dyck:k-Dyck"

echo "Done. See results/reports/focused_rule110_${TAG}/report.md"
