#!/usr/bin/env bash
# Idempotent vLLM cross-check of the OctoThinker seed-0 fast (transformers) readouts.
# Reads all seed-0 graded solutions of the five benchmarks again with the vLLM readout path (readout.py and
# readout_r5.py, the released scorer's engine), splits the result by benchmark and compares it with the fast
# readouts (compare_xchk.py). valid_readout takes the expected engine (vllm by default; transformers for the
# fast files). Output: results/_dev/vllm_crosscheck/.
set -euo pipefail

ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
export HF_HUB_DISABLE_XET=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0

MODEL="OctoThinker-3B-Short-LaSeR"
SEED=0
DATASETS=(
  "math500:data/confirmatory/math500/test.jsonl"
  "amc23:data/confirmatory/amc23/test.jsonl"
  "aime24:data/confirmatory/aime24/test.jsonl"
  "aime25:data/confirmatory/aime25/test.jsonl"
  "OlympiadBench:data/confirmatory/OlympiadBench/test.jsonl"
)
FAST_DIR="results/confirmatory/readout/$MODEL"
GRADED_DIR="results/confirmatory/graded/$MODEL"
OUT_DIR="results/_dev/vllm_crosscheck/$MODEL"
REPORT="results/_dev/vllm_crosscheck/xchk_report.json"
MAX_BATCH_SIZE=32
COMBINED_GRADED="results/_dev/vllm_crosscheck/graded_${MODEL}_seed${SEED}_all.jsonl"
COMBINED_R04="results/_dev/vllm_crosscheck/readout_${MODEL}_seed${SEED}_r0_r4_all.jsonl"
COMBINED_REFERENCE="results/_dev/vllm_crosscheck/readout_${MODEL}_seed${SEED}_all.jsonl"

mkdir -p "$OUT_DIR" logs

row_count() { wc -l < "$1" | tr -d ' '; }

valid_grade() {
  local path="$1" expected="$2"
  [ -s "$path" ] && [ -s "$path.sha256" ] && \
    [ -s "${path%.jsonl}_summary.json" ] && \
    [ -s "${path%.jsonl}_summary.json.sha256" ] || return 1
  "$PY" - "$path" "$expected" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if len(rows) != int(sys.argv[2]):
    raise SystemExit(1)
if any(
    len(row.get("responses", [])) != 8
    or len(row.get("correctness", [])) != 8
    or row.get("response_surface") != "released_response_v1"
    for row in rows
):
    raise SystemExit(1)
PY
}

valid_readout() {
  local path="$1" expected="$2" require_r5="$3" engine="${4:-vllm}"
  [ -s "$path" ] && [ -s "$path.sha256" ] && \
    [ -s "${path%.jsonl}_summary.json" ] && \
    [ -s "${path%.jsonl}_summary.json.sha256" ] || return 1
  "$PY" - "$path" "$expected" "$require_r5" "$engine" <<'PY'
import json, sys
required = {
    "R0_reference", "R1_marker_swap_mean", "R2_position_eos",
    "R3_position_answer", "R4_cref_none", "R5_cref_recomputed",
}
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if len(rows) != int(sys.argv[2]):
    raise SystemExit(1)
for row in rows:
    if row.get("response_surface") != "released_response_v1":
        raise SystemExit(1)
    if len(row.get("responses", [])) != 8 or len(row.get("readout_scores", [])) != 8:
        raise SystemExit(1)
    if row.get("readout_stack", {}).get("engine") != sys.argv[4]:
        raise SystemExit(1)
    for item in row["readout_scores"]:
        if not required.issubset(item):
            raise SystemExit(1)
        if sys.argv[3] == "1" and any(item[name] is None for name in required - {"R3_position_answer"}):
            raise SystemExit(1)
PY
}

run_gpu() {
  local tag="$1"
  shift
  local start end
  start="$(date +%s)"
  $GPU_RUN env \
    ${HF_HOME:+"HF_HOME=$HF_HOME"} HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
    PYTHONPATH="$PYTHONPATH" "$@"
  end="$(date +%s)"
  printf '%s\n' "$((end - start))" > "logs/xchk_${tag}.seconds"
}

config_value() {
  # Base-model snapshot of a checkpoint, resolved by common.load_configs from the Hugging Face cache.
  "$PY" - "$1" <<'PY'
import sys
from common import model_config
cfg = model_config(sys.argv[1])
print(cfg.get("base_model_local_path") or "")
PY
}

# Wait without holding a GPU.  run_readouts.sh owns the fast-readout files.
while true; do
  ready=1
  for spec in "${DATASETS[@]}"; do
    dataset="${spec%%:*}"
    input="${spec#*:}"
    expected="$(row_count "$input")"
    fast="$FAST_DIR/${dataset}_seed${SEED}.jsonl"
    grade="$GRADED_DIR/${dataset}_seed${SEED}.jsonl"
    if ! valid_grade "$grade" "$expected" || ! valid_readout "$fast" "$expected" 1 transformers; then
      ready=0
    fi
  done
  [ "$ready" -eq 1 ] && break
  sleep 600
done

BASE="$(config_value "$MODEL")"
if [ -z "$BASE" ] || [ ! -s "$BASE/config.json" ]; then
  echo "missing registered OctoThinker base snapshot: $BASE" >&2
  exit 3
fi

# Build one immutable combined input so each vLLM model is loaded once per
# surface.  The rows are unchanged and remain in benchmark order.
total_expected=0
for spec in "${DATASETS[@]}"; do
  input="${spec#*:}"
  total_expected=$((total_expected + $(row_count "$input")))
done
if ! valid_grade "$COMBINED_GRADED" "$total_expected"; then
  "$PY" - "$COMBINED_GRADED" "$GRADED_DIR" "$SEED" "${DATASETS[@]}" <<'PY'
import sys
from pathlib import Path
from common import read_jsonl, write_artifact_metadata, write_jsonl

output = Path(sys.argv[1])
graded_dir, seed = sys.argv[2], sys.argv[3]
sources = [f"{graded_dir}/{spec.split(':', 1)[0]}_seed{seed}.jsonl" for spec in sys.argv[4:]]
rows = []
for path in sources:
    rows.extend(read_jsonl(path))
write_jsonl(output, rows)
write_artifact_metadata(
    output,
    {
        "kind": "xchk_combined_graded_input",
        "rows": len(rows),
        "responses": sum(len(row.get("responses", [])) for row in rows),
        "source_order": sources,
    },
)
PY
fi
valid_grade "$COMBINED_GRADED" "$total_expected"

if ! valid_readout "$COMBINED_R04" "$total_expected" 0; then
  run_gpu "combined_r0_r4" "$PY" -u code/readout.py \
    --model "$MODEL" --input "$COMBINED_GRADED" --output "$COMBINED_R04" \
    --batch-size "$MAX_BATCH_SIZE" --skip-r5
fi
if ! valid_readout "$COMBINED_REFERENCE" "$total_expected" 1; then
  run_gpu "combined_r5" "$PY" -u code/readout_r5.py \
    --model "$MODEL" --input "$COMBINED_R04" --output "$COMBINED_REFERENCE" \
    --batch-size "$MAX_BATCH_SIZE" --base-model-path "$BASE"
fi
valid_readout "$COMBINED_REFERENCE" "$total_expected" 1

# Split the combined reference without recomputing any score.  Each split is
# a complete artifact so the CPU comparison can validate it independently.
FAST_ARGS=()
REFERENCE_ARGS=()
for spec in "${DATASETS[@]}"; do
  dataset="${spec%%:*}"
  input="${spec#*:}"
  expected="$(row_count "$input")"
  reference="$OUT_DIR/${dataset}_seed${SEED}.jsonl"
  if ! valid_readout "$reference" "$expected" 1; then
    "$PY" - "$COMBINED_REFERENCE" "$reference" "$dataset" <<'PY'
import sys
from common import read_jsonl, write_artifact_metadata, write_jsonl

source, output, dataset = sys.argv[1:]
rows = [row for row in read_jsonl(source) if row.get("dataset") == dataset]
write_jsonl(output, rows)
write_artifact_metadata(
    output,
    {
        "kind": "xchk_vllm_reference_split",
        "rows": len(rows),
        "responses": sum(len(row.get("responses", [])) for row in rows),
        "source": source,
        "dataset": dataset,
    },
)
PY
  fi
  valid_readout "$reference" "$expected" 1
  FAST_ARGS+=(--fast "$FAST_DIR/${dataset}_seed${SEED}.jsonl")
  REFERENCE_ARGS+=(--reference "$reference")
done

if [ ! -s "$REPORT" ] || [ ! -s "$REPORT.sha256" ] || \
  [ ! -s "${REPORT%.json}".md ] || [ ! -s "${REPORT%.json}".md.sha256 ]; then
  "$PY" code/compare_xchk.py "${FAST_ARGS[@]}" "${REFERENCE_ARGS[@]}" --output "$REPORT"
fi

if [ "$(tail -n 1 logs/progress.log 2>/dev/null || true)" != "XCHK_DONE" ]; then
  printf 'XCHK_DONE\n' >> logs/progress.log
fi
