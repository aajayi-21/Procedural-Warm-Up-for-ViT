#!/usr/bin/env bash
# Run one procedural warm-up and strip its checkpoint for transfer.
#   scripts/run_warmup.sh [config.yaml]
set -euo pipefail
CONFIG="${1:-procedural_warmup/config/files/ca-rule110.yaml}"
python -m procedural_warmup.warmup.cli --config "$CONFIG"
