#!/usr/bin/env bash
# Run one downstream image-training/transfer.
#   scripts/run_downstream.sh <CIFAR10|CIFAR100> <run_name> [stripped_ckpt|none]
set -euo pipefail
DATASET="${1:-CIFAR100}"
RUN_NAME="${2:-cifar100-run}"
INIT="${3:-none}"
python -m procedural_warmup.downstream.main \
  --config procedural_warmup/config/files/downstream-cifar.yaml \
  --dataset "$DATASET" \
  --run-name "$RUN_NAME" \
  --init "$INIT"
