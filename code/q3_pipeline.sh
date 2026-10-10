#!/usr/bin/env bash
# Qwen3-4B replication of LaSeR's self-reward objective (exploratory).
# generate (base model) -> grade -> LoRA training on the self-reward term -> readouts (base and trained) -> analysis.
# Every stage is skipped when its output exists, so the script can be rerun after a failure.
# Run code/q3_prepare.py first: it writes the training problems to results/q3/data/train.jsonl.
# PYTHON_Q3 is the Qwen3 environment (requirements-qwen3.txt: vLLM, transformers and peft that support Qwen3);
# PYTHON is the main environment (requirements.txt), used for grading and analysis.
set -u
ulimit -c 0
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON_Q3:-${PYTHON:-python}}"
LPY="${PYTHON:-python}"
GPU="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
Q=results/q3
ENVV="env ${HF_HOME:+HF_HOME=$HF_HOME} HF_HUB_DISABLE_XET=1 PYTHONPATH=$ROOT/code VLLM_USE_V1=1"
mkdir -p $Q/gen $Q/graded $Q/readout
fail() { echo "q3 FAILED at $1 $(date -u +%FT%TZ)"; exit 1; }

if [ ! -s $Q/gen/train.jsonl ]; then
  $GPU $ENVV "$PY" -u code/q3_generate.py --input $Q/data/train.jsonl --output $Q/gen/train.jsonl --dataset deepmath \
    || fail gen-train
fi
for ds in math500 amc23 aime24 aime25 OlympiadBench; do
  if [ ! -s $Q/gen/$ds.jsonl ]; then
    $GPU $ENVV "$PY" -u code/q3_generate.py --input data/confirmatory/$ds/test.jsonl --output $Q/gen/$ds.jsonl \
      --dataset $ds || fail gen-$ds
  fi
done
for f in train math500 amc23 aime24 aime25 OlympiadBench; do
  if [ ! -s $Q/graded/$f.jsonl ]; then
    (cd code && "$LPY" -u grade.py --input ../$Q/gen/$f.jsonl --output ../$Q/graded/$f.jsonl) || fail grade-$f
    "$LPY" -c "import json;r=[json.loads(l) for l in open('$Q/graded/$f.jsonl')];c=[x for row in r for x in row['correctness']];print('$f',len(r),'problems acc',round(sum(c)/len(c),4))"
  fi
done
if [ ! -s $Q/model/train_meta.json ]; then
  $GPU $ENVV "$PY" -u code/q3_train.py || fail train
fi
if [ ! -s $Q/readout_meta.json ]; then
  $GPU $ENVV "$PY" -u code/q3_readout.py || fail readout
fi
$ENVV "$LPY" -u code/q3_analyze.py || fail analyze
echo "q3 DONE $(date -u +%FT%TZ)"
