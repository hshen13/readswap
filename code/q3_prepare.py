#!/usr/bin/env python3
"""Training problems for the Qwen3-4B replication: 2,000 DeepMath-103K problems (LaSeR's RL data), seed 0,
excluding any problem whose text equals a confirmatory problem. Test problems: the five confirmatory files.

The pool is RL_data/DeepMath-103K/train.parquet of the Hugging Face dataset Keven16/LaSeR_training_data at the
revision below (downloaded into the Hugging Face cache; Q3_TRAIN_PARQUET points to a local copy instead).
Writes results/q3/data/train.jsonl (or $Q3_DIR/data/train.jsonl)."""
import ast
import json
import os
import random
from pathlib import Path

import pyarrow.parquet as pq

from q3_common import Q3, ROOT, write_jsonl

PARQUET = os.environ.get("Q3_TRAIN_PARQUET")
if not PARQUET:
    from huggingface_hub import hf_hub_download

    PARQUET = hf_hub_download(repo_id="Keven16/LaSeR_training_data", repo_type="dataset",
                              revision="8cc93a9d8098e7f68ffd470a258711e4051d2bcf",
                              filename="RL_data/DeepMath-103K/train.parquet")
confirm = set()
for ds in ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]:
    for line in open(ROOT / f"data/confirmatory/{ds}/test.jsonl", encoding="utf-8"):
        confirm.add(json.loads(line)["problem"].strip())
t = pq.read_table(PARQUET).to_pylist()
rows = []
for r in t:
    prompt = ast.literal_eval(r["prompt"]) if isinstance(r["prompt"], str) else r["prompt"]
    rm = ast.literal_eval(r["reward_model"]) if isinstance(r["reward_model"], str) else r["reward_model"]
    ei = ast.literal_eval(r["extra_info"]) if isinstance(r["extra_info"], str) else r["extra_info"]
    problem = prompt[0]["content"].strip()
    if problem in confirm:
        continue
    rows.append({"id": f"deepmath-{ei['index']}", "problem": problem, "answer": str(rm["ground_truth"])})
random.Random(0).shuffle(rows)
train = rows[:2000]
n = write_jsonl(Q3 / "data/train.jsonl", train)
print("train problems", n, "pool", len(rows))
