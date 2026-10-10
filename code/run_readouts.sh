#!/usr/bin/env bash
# Idempotent driver for the READSWAP confirmatory stage, phase 2: readouts R0-R5 and the L1-L5 analysis.
# Requires the pilot unit check of run_pilot.sh (released-scorer comparison) and the graded files of
# run_generation.sh.
set -euo pipefail

ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
export HF_HUB_DISABLE_XET=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0
mkdir -p results/confirmatory/{generated,graded,readout} results/analysis results/_dev logs

MODELS=("OctoThinker-3B-Short-LaSeR" "Qwen2.5-7B-LaSeR" "ORZ-7B-LaSeR")
DATASETS=("math500:data/confirmatory/math500/test.jsonl" "amc23:data/confirmatory/amc23/test.jsonl" "aime24:data/confirmatory/aime24/test.jsonl" "aime25:data/confirmatory/aime25/test.jsonl" "OlympiadBench:data/confirmatory/OlympiadBench/test.jsonl")
SEEDS=(0 1 2)
GATE="results/pilot/octothinker_seed0_unit_compare.json"
PILOT_GRADED="results/pilot/octothinker_seed0_graded.jsonl"
PILOT_READOUT="results/pilot/octothinker_seed0_readout.jsonl"
FAST_PILOT="results/_dev/readout_fast_pilot.jsonl"
EQUIV="results/_dev/readout_fast_equivalence.json"
MAX_BATCH_TOKENS=32768

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

valid_grade() {
  local path="$1" expected="$2"
  [ -s "$path" ] && [ -s "$path.sha256" ] && [ -s "${path%.jsonl}_summary.json" ] && [ -s "${path%.jsonl}_summary.json.sha256" ] || return 1
  "$PY" - "$path" "$expected" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding='utf-8') if x.strip()]
if len(rows) != int(sys.argv[2]):
    raise SystemExit(1)
if any(
    len(r.get('correctness', [])) != 8
    or len(r.get('pred_answers', [])) != 8
    or r.get('response_surface') != 'released_response_v1'
    for r in rows
):
    raise SystemExit(1)
PY
}

valid_readout() {
  local path="$1" expected="$2" final="$3"
  [ -s "$path" ] && [ -s "$path.sha256" ] && [ -s "${path%.jsonl}_summary.json" ] && [ -s "${path%.jsonl}_summary.json.sha256" ] || return 1
  "$PY" - "$path" "$expected" "$final" <<'PY'
import json, sys
required = {'R0_reference','R1_marker_swap_mean','R2_position_eos','R3_position_answer','R4_cref_none','R5_cref_recomputed'}
rows = [json.loads(x) for x in open(sys.argv[1], encoding='utf-8') if x.strip()]
if len(rows) != int(sys.argv[2]):
    raise SystemExit(1)
for row in rows:
    if len(row.get('readout_scores', [])) != 8:
        raise SystemExit(1)
    for item in row['readout_scores']:
        if not required.issubset(item):
            raise SystemExit(1)
        if sys.argv[3] == '1' and any(item[name] is None for name in required - {'R3_position_answer'}):
            raise SystemExit(1)
PY
}

config_value() {
  # Base-model snapshot of a checkpoint, resolved by common.load_configs from the Hugging Face cache.
  "$PY" - "$1" <<'PY'
import sys
from common import model_config
cfg = model_config(sys.argv[1])
print(cfg.get('base_model_local_path') or '')
PY
}

# Wait (without holding a GPU) for the immutable pilot official-score gate.
while [ ! -s "$GATE" ]; do
  sleep 600
done
if ! "$PY" - "$GATE" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
if x.get('token_id_match') is not True:
    raise SystemExit(1)
if x.get('max_absolute_error') is None or float(x['max_absolute_error']) > 0.01:
    raise SystemExit(1)
PY
then
  printf '%s readout stopped before fast readout: pilot official R0 comparison mismatch; see %s\n' "$(date -u +%FT%TZ)" "$GATE" >> logs/progress.log
  exit 2
fi

# The equivalence check is the first GPU operation of phase 2.
pilot_base="$(config_value OctoThinker-3B-Short-LaSeR)"
if [ ! -s "$FAST_PILOT" ] || ! valid_readout "$FAST_PILOT" 40 1; then
  run_gpu "fast_equivalence_pilot" "$PY" -u code/readout_fast.py \
    --model OctoThinker-3B-Short-LaSeR \
    --input "$PILOT_GRADED" \
    --output "$FAST_PILOT" \
    --base-model-path "$pilot_base" \
    --max-batch-tokens "$MAX_BATCH_TOKENS"
fi
# The equivalence passes, or a recorded decision to proceed exists for exactly this equivalence artifact
# (results/_dev/readout_fast_equivalence_decision.json with "proceed": true and "equivalence_sha256" equal to
# the sha256 of $EQUIV). In the original run the bound was not met and the decision was recorded; see README.
DECISION="results/_dev/readout_fast_equivalence_decision.json"
equiv_ok() {
  "$PY" - "$EQUIV" "$DECISION" <<'PY'
import hashlib, json, os, sys
eq, dec = sys.argv[1], sys.argv[2]
if json.load(open(eq, encoding='utf-8')).get('pass', False):
    raise SystemExit(0)
if os.path.exists(dec):
    d = json.load(open(dec, encoding='utf-8'))
    if d.get('proceed') is True and d.get('equivalence_sha256') == hashlib.sha256(open(eq, 'rb').read()).hexdigest():
        raise SystemExit(0)
raise SystemExit(1)
PY
}
if [ ! -s "$EQUIV" ] || [ ! -s "$EQUIV.sha256" ]; then
  "$PY" code/compare_readout_fast.py --fast "$FAST_PILOT" --reference "$PILOT_READOUT" --output "$EQUIV" || true
fi
if ! equiv_ok; then
  printf '%s readout stopped: readout_fast equivalence failed and no recorded decision; see %s\n' "$(date -u +%FT%TZ)" "$EQUIV" >> logs/progress.log
  exit 4
fi
if ! grep -qs 'readout equivalence' logs/progress.log; then
  digest="$(awk '{print $1}' "$EQUIV.sha256")"
  printf '%s readout equivalence gate: pass=%s; artifact=%s sha256=%s; decision=%s\n' "$(date -u +%FT%TZ)" \
    "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pass"])' "$EQUIV")" "$EQUIV" "$digest" "$DECISION" >> logs/progress.log
fi

# Reuse/download only the two pre-registered, public, non-gated base models (into the Hugging Face cache).
# The OctoThinker base (OctoThinker/OctoThinker-3B-Short-Base) must already be in the cache (see README).
# common.load_configs resolves every base-model path from the cache, so the registry files are not rewritten.
ensure_base() {
  local key="$1" repo="$2" revision="$3" base
  base="$(config_value "$key")"
  if [ ! -s "$base/config.json" ]; then
    "$PY" - "$repo" "$revision" <<'PY'
from huggingface_hub import HfApi, snapshot_download
import sys
repo, revision = sys.argv[1:]
info = HfApi().model_info(repo, revision=revision)
if bool(getattr(info, 'gated', False)):
    raise SystemExit(f'gated model: {repo}')
print(snapshot_download(repo_id=repo, revision=revision))
PY
  fi
  if [ ! -s "$base/config.json" ]; then
    echo "base snapshot missing after download: $base" >&2
    exit 3
  fi
}

ensure_base Qwen2.5-7B-LaSeR Qwen/Qwen2.5-7B d149729398750b98c0af14eb82c78cfe92750796
ensure_base ORZ-7B-LaSeR Open-Reasoner-Zero/Open-Reasoner-Zero-7B 0e38a598946c6fa65ec48f99b2b6a8a5107c7ee1

ANALYSIS_PATHS=()
for model in "${MODELS[@]}"; do
  model_args=()
  for seed in "${SEEDS[@]}"; do
    # All five graded datasets for one checkpoint/seed are a single GPU job.
    ready=0
    while [ "$ready" -eq 0 ]; do
      ready=1
      for spec in "${DATASETS[@]}"; do
        dataset="${spec%%:*}"
        input="${spec#*:}"
        expected="$(row_count "$input")"
        gen="results/confirmatory/generated/$model/${dataset}_seed${seed}.jsonl"
        grade="results/confirmatory/graded/$model/${dataset}_seed${seed}.jsonl"
        if ! valid_grade "$grade" "$expected"; then
          # Re-grade a graded file that lacks the released response surface, if its generation is finished.
          # Files not graded yet are left to run_generation.sh, so the two drivers never write the same file;
          # until the grade is valid, keep waiting.
          if [ -s "$grade" ] && [ -s "$gen" ] && [ -s "$gen.sha256" ] && [ -s "${gen%.jsonl}_summary.json" ]; then
            "$PY" -u code/grade.py --input "$gen" --output "$grade" || true
          fi
          valid_grade "$grade" "$expected" || ready=0
        fi
      done
      [ "$ready" -eq 1 ] || sleep 600
    done

    base="$(config_value "$model")"
    fast_ready=1
    fast_args=("$PY" -u code/readout_fast.py --model "$model" --base-model-path "$base" --max-batch-tokens "$MAX_BATCH_TOKENS")
    for spec in "${DATASETS[@]}"; do
      dataset="${spec%%:*}"
      input="${spec#*:}"
      expected="$(row_count "$input")"
      grade="results/confirmatory/graded/$model/${dataset}_seed${seed}.jsonl"
      r04="results/confirmatory/readout/$model/${dataset}_seed${seed}_r0_r4.jsonl"
      rout="results/confirmatory/readout/$model/${dataset}_seed${seed}.jsonl"
      fast_args+=(--input "$grade" --output "$rout" --r0-r4-output "$r04")
      if ! valid_readout "$r04" "$expected" 0 || ! valid_readout "$rout" "$expected" 1; then
        fast_ready=0
      fi
    done
    if [ "$fast_ready" -eq 0 ]; then
      run_gpu "fast_${model}_${seed}" "${fast_args[@]}"
    fi
    for spec in "${DATASETS[@]}"; do
      dataset="${spec%%:*}"
      rout="results/confirmatory/readout/$model/${dataset}_seed${seed}.jsonl"
      model_args+=(--input "$rout")
    done
  done

  analysis="results/analysis/${model}_confirm.json"
  if [ ! -s "$analysis" ] || [ ! -s "$analysis.sha256" ]; then
    "$PY" -u code/analyze_confirm.py "${model_args[@]}" \
      --output "$analysis" --seed 20260924 --bootstrap 10000 --metric-bootstrap 10000
  fi
  ANALYSIS_PATHS+=(--analysis "$analysis")
done

if [ ! -s results/analysis/confirm_summary.json ] || [ ! -s results/analysis/confirm_summary.json.sha256 ]; then
  "$PY" code/make_confirm_summary.py "${ANALYSIS_PATHS[@]}" --output results/analysis/confirm_summary.json
fi

printf 'E2b_DONE\n' >> logs/progress.log
