#!/usr/bin/env bash
# Idempotent driver for the READSWAP pilot: 40 DeepMath-103K problems, OctoThinker-3B, seed 0.
# Generates, grades and reads out the pilot solutions, then checks R0 against the released LaSeR scorer
# (calculate_self_rewarding_scores.py of the LaSeR repository in $LASER_REPO) on 20 solutions. The result,
# results/pilot/octothinker_seed0_unit_compare.json, is the gate that run_readouts.sh waits for.
set -euo pipefail

ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
LASER_REPO="${LASER_REPO:-$ROOT/third_party/LaSeR}"
export HF_HUB_DISABLE_XET=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0

mkdir -p results/pilot results/analysis logs

GEN="results/pilot/octothinker_seed0_generated.jsonl"
GRADED="results/pilot/octothinker_seed0_graded.jsonl"
R04="results/pilot/octothinker_seed0_readout_r0_r4.jsonl"
READOUT="results/pilot/octothinker_seed0_readout.jsonl"
UNIT_INPUT="results/pilot/octothinker_seed0_unit_input.jsonl"
UNIT_OFFICIAL="results/pilot/octothinker_seed0_unit_official.jsonl"
UNIT_COMPARE="results/pilot/octothinker_seed0_unit_compare.json"
ANALYSIS="results/analysis/octothinker_seed0_pilot_analysis.json"
REPORT="results/pilot/pilot_report.json"
# Snapshot of OctoThinker/OctoThinker-3B-Short-Base in the Hugging Face cache (resolved by common.load_configs).
BASE="$("$PY" -c 'from common import model_config; print(model_config("OctoThinker-3B-Short-LaSeR")["base_model_local_path"])')"
START_EPOCH="$(date +%s)"

complete_jsonl() {
  local path="$1" rows="$2" responses="$3"
  [ -s "$path" ] && [ -s "$path.sha256" ] && [ -s "${path%.jsonl}_summary.json" ] && [ -s "${path%.jsonl}_summary.json.sha256" ] || return 1
  "$PY" - "$path" "$rows" "$responses" <<'PY'
import json, sys
p, expected_rows, expected_responses = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
if len(rows) != expected_rows or any(len(row.get("responses", [])) != expected_responses for row in rows):
    raise SystemExit(1)
PY
}

complete_readout() {
  local path="$1"
  complete_jsonl "$path" 40 8 || return 1
  "$PY" - "$path" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if any(len(row.get("readout_scores", [])) != 8 for row in rows):
    raise SystemExit(1)
required = {"R0_reference", "R1_marker_swap_mean", "R2_position_eos", "R3_position_answer", "R4_cref_none", "R5_cref_recomputed"}
if any(any(not required.issubset(item) for item in row["readout_scores"]) for row in rows):
    raise SystemExit(1)
PY
}

run_gpu() {
  local name="$1"
  shift
  local timing="logs/e1_${name}.seconds"
  local start end
  start="$(date +%s)"
  $GPU_RUN env \
    ${HF_HOME:+"HF_HOME=$HF_HOME"} HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
    PYTHONPATH="$PYTHONPATH" "$@"
  end="$(date +%s)"
  printf '%s\n' "$((end - start))" > "$timing"
}

# Generation and grading are guarded so rerunning after an interrupted queue
# submission never queues a duplicate writer for the same output.
if ! complete_jsonl "$GEN" 40 8; then
  run_gpu generation "$PY" -u code/generate.py \
    --model OctoThinker-3B-Short-LaSeR --dataset DeepMath-103K-pilot \
    --input data/pilot/DeepMath-103K_train_seed0_n40.jsonl \
    --output "$GEN" --seed 0 --max-num-seqs 16
fi
if ! complete_jsonl "$GRADED" 40 8; then
  run_gpu grading "$PY" -u code/grade.py --input "$GEN" --output "$GRADED"
fi

# readout.py computes R0--R4 in one policy-model load. R5 is filled in a
# fresh process below so its base-model load cannot retain policy allocations.
if ! complete_readout "$R04"; then
  run_gpu readout_r0_r4 "$PY" -u code/readout.py \
    --model OctoThinker-3B-Short-LaSeR --input "$GRADED" --output "$R04" \
    --batch-size 32 --skip-r5
fi

if [ -d "$BASE" ]; then
  if ! complete_readout "$READOUT"; then
    run_gpu readout_r5 "$PY" -u code/readout_r5.py \
      --model OctoThinker-3B-Short-LaSeR --input "$R04" --output "$READOUT" \
      --batch-size 32 --base-model-path "$BASE"
  fi
else
  # The preregistration permits R5 to be unavailable during the pilot.
  if ! complete_readout "$READOUT"; then
    cp "$R04" "$READOUT"
    cp "${R04}.sha256" "${READOUT}.sha256"
    cp "${R04%.jsonl}_summary.json" "${READOUT%.jsonl}_summary.json"
    cp "${R04%.jsonl}_summary.json.sha256" "${READOUT%.jsonl}_summary.json.sha256"
  fi
fi

if [ ! -s "$UNIT_INPUT" ]; then
  run_gpu unit_subset "$PY" code/make_unit_subset.py --input "$READOUT" --output "$UNIT_INPUT"
fi

if [ ! -s "$UNIT_OFFICIAL" ]; then
  run_gpu unit_official "$PY" "$LASER_REPO/src/calculate_self_rewarding_scores.py" \
    --input_file "$UNIT_INPUT" --model_path data/models/OctoThinker-3B-Short-LaSeR \
    --output_file "$UNIT_OFFICIAL" --max_tokens 1 --temperature 0.0 --top_p 1.0 \
    --max_num_seqs 32 --n 1 --self_rewarding_token_id 128002 \
    --self_rewarding_ratio_coef 0.1 --self_rewarding_ref_constant -25.0
fi
if [ ! -s "${UNIT_OFFICIAL%.jsonl}_summary.json" ]; then
  run_gpu unit_official_metadata "$PY" - "$UNIT_OFFICIAL" <<'PY'
from common import write_artifact_metadata
import sys
write_artifact_metadata(sys.argv[1], {"kind": "official_unit_check", "rows": 20, "model": "Keven16/OctoThinker-3B-Short-LaSeR"})
PY
fi
if [ ! -s "$UNIT_COMPARE" ]; then
  run_gpu unit_compare "$PY" code/compare_official.py \
    --ours "$UNIT_INPUT" --official "$UNIT_OFFICIAL" --pilot "$READOUT" --output "$UNIT_COMPARE"
fi

if [ ! -s "$ANALYSIS" ] || [ ! -s "$ANALYSIS.sha256" ]; then
  run_gpu analysis "$PY" -u code/analyze.py --input "$READOUT" --output "$ANALYSIS" \
    --seed 0 --bootstrap 10000
fi

if [ ! -s "$REPORT" ] || [ ! -s "$REPORT.sha256" ]; then
  run_gpu report "$PY" - "$ROOT" "$ANALYSIS" "$UNIT_COMPARE" "$REPORT" "$START_EPOCH" <<'PY'
import hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

root, analysis_path, compare_path, report_path, start = sys.argv[1:]
root = Path(root)
analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
compare = json.loads(Path(compare_path).read_text(encoding="utf-8"))
conditions = {
    "R0_reference": "R0_reference",
    "R1_marker_swap": "R1_marker_swap_mean",
    "R2_position_eos": "R2_position_eos",
    "R3_position_answer": "R3_position_answer",
    "R4_cref_none": "R4_cref_none",
    "R5_cref_recomputed": "R5_cref_recomputed",
}
metrics = analysis["metrics"]
readouts = {
    label: {key: metrics[condition].get(key) for key in ("auroc", "ece_15", "brier")}
    for label, condition in conditions.items()
}

def summary_seconds(path):
    p = Path(path)
    summary = p.with_name(p.stem + "_summary.json")
    if not summary.exists():
        return 0.0
    return float(json.loads(summary.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))

gen_seconds = summary_seconds(root / "results/pilot/octothinker_seed0_generated.jsonl")
r04_seconds = summary_seconds(root / "results/pilot/octothinker_seed0_readout_r0_r4.jsonl")
r5_seconds = summary_seconds(root / "results/pilot/octothinker_seed0_readout.jsonl")
pilot_gpu_seconds = gen_seconds + r04_seconds + r5_seconds
confirmatory_problems = 500 + 40 + 30 + 30 + 675
scale = 3 * 3 * confirmatory_problems / 40
report = {
    "schema_version": 1,
    "stage": "pilot",
    "model": "Keven16/OctoThinker-3B-Short-LaSeR",
    "dataset": "DeepMath-103K train split pilot",
    "n_problems": 40,
    "responses_per_problem": 8,
    "seed": 0,
    "readouts": readouts,
    "unit_check": {
        "n": compare.get("n"),
        "max_absolute_error": compare.get("max_absolute_error"),
        "artifact": str((root / compare_path).resolve().relative_to(root.resolve())),
    },
    "analysis": {
        "artifact": str((root / analysis_path).resolve().relative_to(root.resolve())),
        "bootstrap_resamples": analysis["bootstrap"]["resamples"],
        "bootstrap_seed": analysis["bootstrap"]["seed"],
    },
    "pilot_wall_clock_seconds": max(0, int(time.time()) - int(start)),
    "pilot_gpu_seconds_from_artifact_summaries": {
        "generation": gen_seconds,
        "readout_r0_r4": r04_seconds,
        "readout_r5": r5_seconds,
        "total": pilot_gpu_seconds,
    },
    "full_confirmatory_gpu_hours_estimate": pilot_gpu_seconds * scale / 3600.0,
    "full_confirmatory_estimate_basis": {
        "confirmatory_problems": confirmatory_problems,
        "models": 3,
        "seeds": 3,
        "scaling": "linear from this 40-problem, one-model, one-seed pilot; includes generation and R0-R5 readout artifact elapsed times",
    },
    "artifacts": {},
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
for rel in [
    "results/pilot/octothinker_seed0_generated.jsonl",
    "results/pilot/octothinker_seed0_graded.jsonl",
    "results/pilot/octothinker_seed0_readout.jsonl",
    "results/analysis/octothinker_seed0_pilot_analysis.json",
    "results/pilot/octothinker_seed0_unit_compare.json",
]:
    report["artifacts"][rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
out = Path(report_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
out.with_name(out.name + ".sha256").write_text(
    f"{hashlib.sha256(out.read_bytes()).hexdigest()}  {out.name}\n", encoding="utf-8"
)
PY
fi

if ! grep -qs 'pilot completed' logs/progress.log; then
  report_sha="$(sha256sum "$REPORT" | awk '{print $1}')"
  compare_sha="$(sha256sum "$UNIT_COMPARE" | awk '{print $1}')"
  max_error="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_absolute_error"])' "$UNIT_COMPARE")"
  printf '%s pilot completed; report=%s sha256=%s; unit_compare=%s sha256=%s max_absolute_error=%s; analysis=%s\n' \
    "$(date -u +%FT%TZ)" "$REPORT" "$report_sha" "$UNIT_COMPARE" "$compare_sha" "$max_error" "$ANALYSIS" >> logs/progress.log
fi
if [ "$(tail -n 1 logs/progress.log 2>/dev/null || true)" != "E1_DONE" ]; then
  printf 'E1_DONE\n' >> logs/progress.log
fi
