#!/usr/bin/env bash
# The three LaSeR models on AIME26 and HMMT26 (MathArena's AIME 2026 and HMMT February 2026), supplementary.
# Same generation, grading and readouts as the confirmatory runs: eight solutions per problem, seeds 0 to 2,
# temperature 1.0, top-p 1.0, at most 8,192 new tokens, readouts R0 to R5 in transformers.
# Every step is skipped when its output exists. Then: python code/newdata_analysis.py
set -u
ulimit -c 0
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
OUT=results/new_problems

for ckpt in OctoThinker-3B-Short-LaSeR Qwen2.5-7B-LaSeR ORZ-7B-LaSeR; do
  mkdir -p "$OUT/generated/$ckpt" "$OUT/graded/$ckpt" "$OUT/readout/$ckpt"
  readout_args=()
  for ds in aime_2026 hmmt_feb_2026; do
    for seed in 0 1 2; do
      name="${ds}_seed${seed}.jsonl"
      if [ ! -s "$OUT/generated/$ckpt/$name" ]; then
        $GPU_RUN "$PY" -u code/generate.py --model "$ckpt" --dataset "$ds" --input "data/new_problems/$ds/test.jsonl" \
          --output "$OUT/generated/$ckpt/$name" --seed "$seed" --max-num-seqs 256 || exit 1
      fi
      if [ ! -s "$OUT/graded/$ckpt/$name" ]; then
        "$PY" -u code/grade.py --input "$OUT/generated/$ckpt/$name" --output "$OUT/graded/$ckpt/$name" || exit 1
      fi
      [ -s "$OUT/readout/$ckpt/$name" ] || readout_args+=(--input "$OUT/graded/$ckpt/$name" --output "$OUT/readout/$ckpt/$name")
    done
  done
  if [ ${#readout_args[@]} -gt 0 ]; then
    $GPU_RUN "$PY" -u code/readout_fast.py --model "$ckpt" --max-batch-tokens 32768 "${readout_args[@]}" || exit 1
  fi
done
echo "new problems done"
