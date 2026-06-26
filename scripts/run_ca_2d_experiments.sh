#!/usr/bin/env bash
# H3 test: retain the CA 2-D spacetime structure during warm-up, and re-examine whether the
# earlier CA rule modifications (block tokenization, forward masking) are still needed once the
# 2-D structure is exposed. Chains warm-up -> strip -> CIFAR downstream for every 2-D-retaining
# config, then builds a comparison figure vs the existing 1-D baselines.
# See results/reports/ca-2d-spacetime/report.md for the matrix and how to read it.
#
#   scripts/run_ca_2d_experiments.sh [DATASET]      # default CIFAR100
#
# Heavy (one 15k-step warm-up + one 300-epoch training per row); intended for a CUDA box.
# Resumable: skips any warm-up whose stripped checkpoint exists and any downstream with metrics.
set -euo pipefail
DATASET="${1:-CIFAR100}"
TAG="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
CFG=procedural_warmup/config/files
STEP=ckpt_step_015000_stripped.pt

# "warmup_config_basename : downstream_run_name". The warm-up run_name == config basename.
PAIRS=(
  # --- CA spacetime grid: pos {random,sincos2d} x mask {random,block2d,forward-deep}, binary ---
  "ca-rule110-2dpos:${TAG}-2dpos"                          # binary + random + sincos2d (pure H3)
  "ca-rule110-block2d:${TAG}-block2d"                      # binary + block2d  + sincos2d
  "ca-rule110-block2d-1dpos:${TAG}-block2d-1dpos"          # control: block2d + 1-D pos
  "ca-rule110-forward-2dpos:${TAG}-forward-2dpos"          # binary + forward(7) + sincos2d
  "ca-rule110-forward-deep-2dpos:${TAG}-forward-deep-2dpos" # binary + forward(12) + sincos2d (top pick)
  "ca-rule110-nextstate-2dpos:${TAG}-nextstate-2dpos"      # binary + forward(1) + sincos2d (diagnostic)
  # --- "are the earlier mods still needed once 2-D is retained?" (H2/vocab interaction) ---
  "ca-rule110-block-2dpos:${TAG}-block-2dpos"              # block tokenize + sincos2d
  "ca-rule110-hard-2dpos:${TAG}-hard-2dpos"                # block + forward + sincos2d (ceiling)
  # --- next-state operator controls (true vs shuffled; adjacency exposed) ---
  "ca-iid1step-true-1dpos:${TAG}-iid-true-1dpos"           # 1-D ECA next-state + ring pos
  "ca-iid1step-shuffled-1dpos:${TAG}-iid-shuffled-1dpos"   # shuffled control
  "gol-step-true-2dpos:${TAG}-gol-true-2dpos"              # 2-D Life next-state + sincos2d
  "gol-step-shuffled-2dpos:${TAG}-gol-shuffled-2dpos"      # shuffled control
)

for pair in "${PAIRS[@]}"; do
  cfg_name="${pair%%:*}"
  if [[ -f "checkpoints/$cfg_name/$STEP" ]]; then
    echo "== warm-up $cfg_name: stripped checkpoint exists, skipping =="
  else
    echo "== warm-up $cfg_name =="
    python -m procedural_warmup.warmup.cli --config "$CFG/$cfg_name.yaml"
  fi
done

for pair in "${PAIRS[@]}"; do
  cfg_name="${pair%%:*}"; run="${pair##*:}"
  if [[ -f "results/reports/$run/metrics.json" ]]; then
    echo "== downstream $run: metrics exist, skipping =="
  else
    echo "== downstream $run =="
    scripts/run_downstream.sh "$DATASET" "$run" "checkpoints/$cfg_name/$STEP"
  fi
done

echo "== comparison figure (new 2-D runs vs existing 1-D baselines) =="
# Baselines come from scripts/run_focused_experiment.sh + the block/hard/iid warm-ups; any run
# without a metrics.json yet (e.g. CIFAR10, where no baselines exist) is skipped with a warning
# (the compare step no longer crashes on missing runs).
python -m procedural_warmup.analysis.compare \
  --out "ca_2d_spacetime_${TAG}" \
  --title "${DATASET} top-1: retaining CA 2-D structure (H3 test)" \
  --runs \
    "${TAG}-random:Random" \
    "${TAG}-dyck:k-Dyck" \
    "${TAG}-rule110:CA binary 1D" \
    "${TAG}-2dpos:CA binary 2D" \
    "${TAG}-block2d-1dpos:CA block2d 1D" \
    "${TAG}-block2d:CA block2d 2D" \
    "${TAG}-forward-deep-2dpos:CA forward-deep 2D" \
    "${TAG}-nextstate-2dpos:CA next-state 2D" \
    "${TAG}-block:CA block 1D" \
    "${TAG}-block-2dpos:CA block 2D" \
    "${TAG}-hard:CA hard 1D" \
    "${TAG}-hard-2dpos:CA hard 2D" \
    "${TAG}-iid-true:CA iid-true 1D" \
    "${TAG}-iid-true-1dpos:CA iid-true ring" \
    "${TAG}-gol-true-2dpos:GoL-step 2D"

echo "Done. See results/reports/ca_2d_spacetime_${TAG}/report.md"
