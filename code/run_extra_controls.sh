#!/usr/bin/env bash
# Exploratory runs: exhaustive token test, vocabulary sweep at the EOS position, and two more training seeds
# (1 and 2) of the Qwen3-4B replication on the same fixed solutions (requires q3_pipeline.sh to have run).
# PYTHON is the main environment (requirements.txt); PYTHON_Q3 the Qwen3 environment (requirements-qwen3.txt).
set -u
ulimit -c 0
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
GPU="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
LPY="${PYTHON:-python}"
ENVV="env ${HF_HOME:+HF_HOME=$HF_HOME} HF_HUB_DISABLE_XET=1 PYTHONPATH=$ROOT/code"
for m in OctoThinker-3B-Short-LaSeR Qwen2.5-7B-LaSeR ORZ-7B-LaSeR; do
  [ -s "results/analysis/exhaustive_tokens_${m}.json" ] || $GPU $ENVV "$LPY" -u code/exhaustive_tokens.py --model "$m" || echo "exhaustive failed: $m"
done
for m in OctoThinker-3B-Short-LaSeR Qwen2.5-7B-LaSeR ORZ-7B-LaSeR; do
  [ -s "results/analysis/eos_vocab_${m}.json" ] || $GPU $ENVV "$LPY" -u code/eos_vocab.py --model "$m" || echo "eos vocab failed: $m"
done
PY="${PYTHON_Q3:-${PYTHON:-python}}"
for s in 1 2; do
  D=results/q3_s$s
  mkdir -p $D
  [ -e $D/graded ] || ln -s ../q3/graded $D/graded
  if [ ! -s $D/model/train_meta.json ]; then
    $GPU $ENVV Q3_DIR=$ROOT/$D "$PY" -u code/q3_train.py --seed $s --train $ROOT/results/q3/graded/train.jsonl || { echo "q3 train failed: seed $s"; continue; }
  fi
  if [ ! -s $D/readout_meta.json ]; then
    $GPU $ENVV Q3_DIR=$ROOT/$D "$PY" -u code/q3_readout.py || { echo "q3 readout failed: seed $s"; continue; }
  fi
  $ENVV Q3_DIR=$ROOT/$D Q3_OUT=$ROOT/results/analysis/q3_replication_s$s.json "$LPY" -u code/q3_analyze.py || echo "q3 analyze failed: seed $s"
done
echo "exhaustive, vocabulary and seed runs done $(date -u +%FT%TZ)"
