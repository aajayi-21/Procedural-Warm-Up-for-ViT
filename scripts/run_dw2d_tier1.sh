#!/usr/bin/env bash
# Tier 1 after the H6 screen verdict (results/reports/dw32-screen/report.md):
# DW_32 with RANDOM frozen positions + its marginal-shuffle control — the decisive
# test of whether the sincos2d presentation (the screen's primary suspect) was what
# pushed the 2D arms below random init. Data, masking, budget identical to the screen;
# only model.pos_embed differs. Reuses the already-finished screen runs for context.
#
#   scripts/run_dw2d_tier1.sh [DATASET]      # default CIFAR100
#
# Pre-registered reads (1 seed; P7: differences < 0.7 are noise; in-repo anchors
# random-repro 70.34, dyck-repro 72.08, sincos2d pair 67.88/64.84):
#   dw32-randpos >= ~70.7            -> presentation was the killer; DW is viable;
#                                       next: stack the harder-masking arms on randpos.
#   dw32-randpos ~= 68 (== sincos2d) -> presentation exonerated; masking/task
#                                       difficulty promotes to primary suspect.
#   randpos treatment - control      -> the structure signal under the fixed
#                                       presentation (sincos2d pair measured +3.04).
#
# Resumable: skips any warm-up whose stripped checkpoint exists, any probe with
# metrics, and any downstream with metrics.
set -euo pipefail
DATASET="${1:-CIFAR100}"
TAG="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
CFG=procedural_warmup/config/files
STEP=ckpt_step_015000_stripped.pt

# Fail fast with a readable message if the Python environment is wrong (see
# run_dw2d_experiments.sh; same guard).
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for v in .venv/bin/activate .venv/Scripts/activate; do
    if [[ -f "$v" ]]; then echo "[env] activating $v"; source "$v"; break; fi
  done
fi
if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: 'python' cannot import torch — activate the project venv first."
  exit 1
fi
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "WARNING: CUDA unavailable — these runs are meant for a GPU box. Ctrl+C to abort."
  sleep 10
fi

warmup () { # <config_basename>
  local cfg_name="$1"
  if [[ -f "checkpoints/$cfg_name/$STEP" ]]; then
    echo "== warm-up $cfg_name: stripped checkpoint exists, skipping =="
  else
    echo "== warm-up $cfg_name =="
    python -m procedural_warmup.warmup.cli --config "$CFG/$cfg_name.yaml"
  fi
}

downstream () { # <run_name> <init>
  local run="$1" init="$2"
  if [[ -f "results/reports/$run/metrics.json" ]]; then
    echo "== downstream $run: metrics exist, skipping =="
  else
    echo "== downstream $run =="
    scripts/run_downstream.sh "$DATASET" "$run" "$init"
  fi
}

probe () { # <warmup_run_name>
  local run="$1"
  if [[ -f "results/reports/$run-probe/metrics.json" ]]; then
    echo "== probe $run: metrics exist, skipping =="
  else
    echo "== probe $run =="
    python -m procedural_warmup.analysis.weight_probe --run "$run" --dist grid
  fi
}

# ---------------------------------------------------------------------------------
# Warm-ups (the LR guard from run_dw2d_experiments.sh applies here too; with random
# positions expect the k-Dyck-like gradual climb rather than the sincos2d arms'
# step-1000 phase transition — saturation step is itself a diagnostic, note it).
# ---------------------------------------------------------------------------------
warmup dw32-randpos-vit-t
warmup dw32-randpos-shuffle-vit-t

probe dw32-randpos-vit-t
probe dw32-randpos-shuffle-vit-t
python -m procedural_warmup.analysis.weight_probe \
  --compare dyck-repro dw32-vit-t dw32-randpos-vit-t dw32-randpos-shuffle-vit-t \
  --out dw32-tier1-probe-compare

downstream "${TAG}-dw32-randpos"         "checkpoints/dw32-randpos-vit-t/$STEP"
downstream "${TAG}-dw32-randpos-shuffle" "checkpoints/dw32-randpos-shuffle-vit-t/$STEP"

# ---------------------------------------------------------------------------------
# Verdict figure + report: the position-code factorial. Rows = {structure, shuffle},
# columns = {sincos2d, random pos}, plus the anchors. The screen runs are reused
# verbatim (already on disk); analysis.compare skips any still-missing run.
# ---------------------------------------------------------------------------------
echo "== Tier-1 comparison =="
python -m procedural_warmup.analysis.compare \
  --out "dw32_tier1_${TAG}" \
  --title "${DATASET} top-1: DW_32 position-code factorial (Tier 1, post-H6)" \
  --runs \
    "${TAG}-random-repro:Random" \
    "${TAG}-dyck-repro:k-Dyck 1D (randpos)" \
    "${TAG}-dw32-randpos:DW32 randpos" \
    "${TAG}-dw32-randpos-shuffle:DW32 randpos shuffle" \
    "${TAG}-dw32:DW32 sincos2d" \
    "${TAG}-dw32-shuffle:DW32 sincos2d shuffle" \
  --warmup-runs "dyck-repro:k-Dyck 1D" "dw32-vit-t:DW32 sincos2d" \
                "dw32-randpos-vit-t:DW32 randpos" \
  --warmup-note "saturation step is diagnostic: sincos2d arms phase-transition at ~step 1000; k-Dyck climbs all 15k steps" \
  --summary "Tier-1 pre-registered reads (1 seed, P7 noise band 0.7; screen verdict: results/reports/dw32-screen/report.md): (a) DW32-randpos >= ~70.7 -> the sincos2d presentation was the primary killer; DW is viable -> next stack harder masking (min-span filter / c+d masking) on randpos and re-screen. (b) DW32-randpos ~= 68 (== the sincos2d arm) -> presentation exonerated -> masking/task-difficulty is primary; go to H11 arms. (c) randpos treatment-minus-control is the structure signal under fixed presentation (sincos2d pair: +3.04); if it collapses at randpos, the structure transfer depended on geometry being exposed -> H7's full factorial becomes the program. Anchors this branch: random 70.34, k-Dyck 72.08."

echo "Done. Read results/reports/dw32_tier1_${TAG}/report.md and"
echo "results/reports/dw32-tier1-probe-compare/report.md, then update"
echo "results/reports/dw32-screen/report.md's next-steps section with the branch taken."
