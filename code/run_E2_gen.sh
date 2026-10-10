#!/usr/bin/env bash
# READSWAP confirmatory stage, phase 1: generation and grading only, idempotent.
# Generation runs up to 256 sequences at a time in vLLM. The readout phase is run_E2_readout.sh.
set -euo pipefail

ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
export HF_HUB_DISABLE_XET=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0
mkdir -p results/confirmatory/{generated,graded,readout} results/analysis logs

MODELS=("OctoThinker-3B-Short-LaSeR" "Qwen2.5-7B-LaSeR" "ORZ-7B-LaSeR")
DATASETS=("math500:data/confirmatory/math500/test.jsonl" "amc23:data/confirmatory/amc23/test.jsonl" "aime24:data/confirmatory/aime24/test.jsonl" "aime25:data/confirmatory/aime25/test.jsonl" "OlympiadBench:data/confirmatory/OlympiadBench/test.jsonl")
SEEDS=(0 1 2)

run_gpu() {
  local tag="$1"; shift
  local start end
  start="$(date +%s)"
  $GPU_RUN env ${HF_HOME:+"HF_HOME=$HF_HOME"} HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" PYTHONPATH="$PYTHONPATH" "$@"
  end="$(date +%s)"
  printf '%s\n' "$((end-start))" > "logs/e2_${tag}.seconds"
}

row_count() { wc -l < "$1" | tr -d ' '; }

valid_generated() {
  local path="$1" expected="$2"
  [ -s "$path" ] && [ -s "$path.sha256" ] && [ -s "${path%.jsonl}_summary.json" ] && [ -s "${path%.jsonl}_summary.json.sha256" ] || return 1
  "$PY" - "$path" "$expected" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding='utf-8') if x.strip()]
if len(rows) != int(sys.argv[2]): raise SystemExit(1)
for row in rows:
    n = len(row.get('responses', []))
    if n != 8 or any(len(row.get(k, [])) != n for k in ('response_token_counts','finish_reasons','stop_reasons','truncated')):
        raise SystemExit(1)
    if any(not isinstance(x, int) or x < 0 for x in row['response_token_counts']): raise SystemExit(1)
    if len(set(row['responses'])) == 1:
        row.setdefault('_identical', True)
identical = sum(len(set(row['responses'])) == 1 for row in rows)
if identical / len(rows) > 0.05:
    raise SystemExit(f'more than 5% problems have eight identical responses: {identical}/{len(rows)}')
PY
}

valid_grade() {
  local path="$1" expected="$2"
  [ -s "$path" ] && [ -s "$path.sha256" ] && [ -s "${path%.jsonl}_summary.json" ] && [ -s "${path%.jsonl}_summary.json.sha256" ] || return 1
  "$PY" - "$path" "$expected" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding='utf-8') if x.strip()]
if len(rows) != int(sys.argv[2]) or any(len(r.get('correctness', [])) != 8 or len(r.get('pred_answers', [])) != 8 for r in rows): raise SystemExit(1)
PY
}

valid_readout() {
  local path="$1" expected="$2" final="$3"
  [ -s "$path" ] && [ -s "$path.sha256" ] && [ -s "${path%.jsonl}_summary.json" ] && [ -s "${path%.jsonl}_summary.json.sha256" ] || return 1
  "$PY" - "$path" "$expected" "$final" <<'PY'
import json, sys
required = {'R0_reference','R1_marker_swap_mean','R2_position_eos','R3_position_answer','R4_cref_none','R5_cref_recomputed'}
rows = [json.loads(x) for x in open(sys.argv[1], encoding='utf-8') if x.strip()]
if len(rows) != int(sys.argv[2]): raise SystemExit(1)
for row in rows:
    if len(row.get('readout_scores', [])) != 8: raise SystemExit(1)
    for item in row['readout_scores']:
        if not required.issubset(item): raise SystemExit(1)
        if sys.argv[3] == '1' and any(item[x] is None for x in required): raise SystemExit(1)
PY
}

# All generation and grading precede the unit-comparison gate and every readout.
for model in "${MODELS[@]}"; do
  for spec in "${DATASETS[@]}"; do
    dataset="${spec%%:*}"; input="${spec#*:}"; expected="$(row_count "$input")"
    for seed in "${SEEDS[@]}"; do
      gen="results/confirmatory/generated/$model/${dataset}_seed${seed}.jsonl"
      grade="results/confirmatory/graded/$model/${dataset}_seed${seed}.jsonl"
      if ! valid_generated "$gen" "$expected"; then
        run_gpu "gen_${model}_${dataset}_${seed}" "$PY" -u code/generate.py --model "$model" --dataset "$dataset" --input "$input" --output "$gen" --seed "$seed" --max-num-seqs 256
      fi
      if ! valid_grade "$grade" "$expected"; then
        "$PY" -u code/grade.py --input "$gen" --output "$grade"
      fi
    done
  done
done

date -u +%FT%TZ > results/confirmatory/GEN_DONE
echo "$(date -u +%FT%TZ) E2 phase 1 complete: all generation and grading" >> logs/progress.log
