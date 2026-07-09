#!/usr/bin/env bash
# Steps 1+2 after the Tier-1 verdict (results/reports/dw32-screen/report.md §5).
#
# STEP 2 (runs first — cheapest, one downstream, no warm-up): the weight-decay-erosion
# test. The sincos2d treatment's block norms peaked at step 2000 (2.91x init) and then
# decayed to 2.30x while the task was saturated; we strip the existing step-2000 probe
# snapshot and transfer it. If DW@2k beats DW@15k (67.88), post-saturation training is
# actively harmful and "stop at saturation" becomes a rule for every future arm.
#
# STEP 1: sustained-difficulty arms on the randpos base (the Tier-1 winner):
#   strict — min(row_span,col_span)>=2 filter, ratio 1.0 (~17 hard targets/sample)
#   cd     — both closing roles masked, ratio 0.5 (~34 targets, column-forced)
#   hard   — both combined, ratio 1.0
# Pre-registered SATURATION GATE: after each warm-up, the step at which masked acc
# first reaches 0.99 is read from log.csv. An arm earns its downstream only if that
# step >= SAT_GATE (default 6000; base randpos arm: 1150, k-Dyck: 9650) — the theory
# says arms that still saturate instantly cannot transfer better, so we don't spend
# 300 GPU-epochs confirming it. Override with RUN_ALL_DOWNSTREAM=1.
#
#   scripts/run_dw2d_step12.sh [DATASET]        # default CIFAR100
#   SAT_GATE=8000 scripts/run_dw2d_step12.sh    # stricter gate
#   RUN_ALL_DOWNSTREAM=1 ...                    # ignore the gate
set -euo pipefail
DATASET="${1:-CIFAR100}"
TAG="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
CFG=procedural_warmup/config/files
STEP=ckpt_step_015000_stripped.pt
SAT_GATE="${SAT_GATE:-6000}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for v in .venv/bin/activate .venv/Scripts/activate; do
    if [[ -f "$v" ]]; then echo "[env] activating $v"; source "$v"; break; fi
  done
fi
if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: 'python' cannot import torch — activate the project venv first."
  exit 1
fi

warmup () {
  local cfg_name="$1"
  if [[ -f "checkpoints/$cfg_name/$STEP" ]]; then
    echo "== warm-up $cfg_name: stripped checkpoint exists, skipping =="
  else
    echo "== warm-up $cfg_name =="
    python -m procedural_warmup.warmup.cli --config "$CFG/$cfg_name.yaml"
  fi
}

downstream () {
  local run="$1" init="$2"
  if [[ -f "results/reports/$run/metrics.json" ]]; then
    echo "== downstream $run: metrics exist, skipping =="
  else
    echo "== downstream $run =="
    scripts/run_downstream.sh "$DATASET" "$run" "$init"
  fi
}

probe () {
  local run="$1"
  if [[ -f "results/reports/$run-probe/metrics.json" ]]; then
    echo "== probe $run: metrics exist, skipping =="
  else
    echo "== probe $run =="
    python -m procedural_warmup.analysis.weight_probe --run "$run" --dist grid
  fi
}

saturation_step () { # <run_name> -> prints the first step with acc >= 0.99, or 99999
  python - "$1" <<'PY'
import csv, sys
try:
    rows = [{k: float(v) for k, v in r.items()}
            for r in csv.DictReader(open(f"results/reports/{sys.argv[1]}/log.csv"))]
    print(next((int(r["step"]) for r in rows if r["acc"] >= 0.99), 99999))
except FileNotFoundError:
    print(0)
PY
}

# ---------------------------------------------------------------------------------
# STEP 2 — erosion test: strip the step-2000 probe snapshot (block-norm peak) of the
# sincos2d treatment and transfer it. clean_checkpoint reads any {model_state} payload.
# ---------------------------------------------------------------------------------
EARLY_SRC=checkpoints/dw32-vit-t/probe_step_002000.pt
EARLY_STRIPPED=checkpoints/dw32-vit-t/probe_step_002000_stripped.pt
if [[ -f "$EARLY_SRC" ]]; then
  if [[ ! -f "$EARLY_STRIPPED" ]]; then
    echo "== strip early snapshot (step 2000, pre-erosion norm peak) =="
    python -m procedural_warmup.warmup.process "$EARLY_SRC"
  fi
  downstream "${TAG}-dw32-early2k" "$EARLY_STRIPPED"
else
  echo "WARNING: $EARLY_SRC not found — skipping the Step-2 erosion test."
fi

# ---------------------------------------------------------------------------------
# STEP 1 — sustained-difficulty warm-ups, saturation gate, probes, gated downstreams.
# ---------------------------------------------------------------------------------
ARMS=(dw32-randpos-strict-vit-t dw32-randpos-cd-vit-t dw32-randpos-hard-vit-t)

for arm in "${ARMS[@]}"; do
  warmup "$arm"
done

echo "== saturation report (gate: first step with masked acc >= 0.99 must be >= $SAT_GATE) =="
declare -A SAT
for arm in "${ARMS[@]}"; do
  SAT[$arm]="$(saturation_step "$arm")"
  echo "  $arm: saturation step ${SAT[$arm]} (99999 = never)  [base randpos: 1150, k-Dyck: 9650]"
done

for arm in "${ARMS[@]}"; do
  probe "$arm"
done

for arm in "${ARMS[@]}"; do
  short="${arm%-vit-t}"                       # dw32-randpos-strict etc.
  if [[ "${RUN_ALL_DOWNSTREAM:-0}" == "1" || "${SAT[$arm]}" -ge "$SAT_GATE" ]]; then
    downstream "${TAG}-${short}" "checkpoints/$arm/$STEP"
  else
    echo "== downstream ${TAG}-${short}: SKIPPED by saturation gate "\
"(saturated at step ${SAT[$arm]} < $SAT_GATE — the theory predicts no transfer gain; "\
"RUN_ALL_DOWNSTREAM=1 to override) =="
  fi
done

# ---------------------------------------------------------------------------------
# Comparison: everything that has metrics gets charted (compare skips missing runs).
# ---------------------------------------------------------------------------------
echo "== Step-1/2 comparison =="
python -m procedural_warmup.analysis.compare \
  --out "dw32_step12_${TAG}" \
  --title "${DATASET} top-1: DW_32 sustained-difficulty arms + erosion test" \
  --runs \
    "${TAG}-random-repro:Random" \
    "${TAG}-dyck-repro:k-Dyck 1D" \
    "${TAG}-dw32-randpos:DW32 randpos (base)" \
    "${TAG}-dw32-randpos-strict:DW32 strict-filter" \
    "${TAG}-dw32-randpos-cd:DW32 c+d masking" \
    "${TAG}-dw32-randpos-hard:DW32 strict+cd" \
    "${TAG}-dw32:DW32 sincos2d @15k" \
    "${TAG}-dw32-early2k:DW32 sincos2d @2k (erosion test)" \
  --warmup-runs "dyck-repro:k-Dyck 1D" "dw32-randpos-vit-t:DW32 randpos base" \
                "dw32-randpos-strict-vit-t:strict" "dw32-randpos-cd-vit-t:c+d" \
                "dw32-randpos-hard-vit-t:strict+cd" \
  --warmup-note "the saturation step (acc >= 0.99) is the pre-registered difficulty readout; base randpos saturated at 1150, k-Dyck at 9650" \
  --summary "Step-1/2 pre-registered reads (1 seed; noise band 0.7; anchors: random 70.34, k-Dyck 72.08, DW32-randpos base 69.07, DW32-sincos2d@15k 67.88). STEP 2 (erosion): early2k > 67.88 + 0.7 -> post-saturation weight decay actively erodes transfer; adopt stop-at-saturation for all future warm-ups. STEP 1 (difficulty): an arm only got a downstream if its saturation step cleared the gate; if a harder arm clears random init (70.34), the sustained-pressure theory holds and that arm becomes the base for 3 seeds + the DN/DQ/DC ladder (H8). If harder arms saturate late but still transfer < random, difficulty at 14x14 is not the binding constraint either -> proceed to DN_32 (hierarchy) with the lessons: randpos + hardest determinate masking."

echo "Done. Read results/reports/dw32_step12_${TAG}/report.md; update"
echo "results/reports/dw32-screen/report.md §5 with the branch taken."
