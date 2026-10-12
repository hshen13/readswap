"""Shared settings for score-only training in Qwen3.5-4B (exploratory), as in q3_common.py.

The base model writes the solutions once (non-thinking chat mode). A LoRA adapter is trained only on LaSeR's
self-reward regression term, (beta * (log p(z_c | x, y, EOS) - c) - r)^2, on the base model's own training
solutions. The untrained base model and the trained model then read the same fixed test solutions.
Sequences are built from token ids (prompt ids + generated ids + EOS + marker), with no retokenization.
BASE is the local snapshot of Qwen/Qwen3.5-4B at the revision below in the Hugging Face cache (HF_HOME);
Q3_BASE_MODEL overrides it.
"""
import json
import os
from pathlib import Path

from common import hf_snapshot_path

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
Q3 = Path(os.environ.get("Q3_DIR", str(ROOT / "results/qwen35_4b")))
BASE_REPO, BASE_REVISION = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE = os.environ.get("Q3_BASE_MODEL") or str(hf_snapshot_path(BASE_REPO, BASE_REVISION))
MARKER = 248077  # first padding row past the tokenizer's vocabulary (smallest unused id)
R1_TOKENS = [248078, 248079, 248080]  # the next three padding rows
BETA = 0.1                   # LaSeR's ratio coefficient for Qwen
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_NEW = 4096
N_SAMPLES = 8


def prompt_ids(tok, problem):
    msgs = [{"role": "user", "content": problem + SUFFIX}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False,
                                   return_dict=False)


def eos_id(tok):
    return tok.convert_tokens_to_ids("<|im_end|>")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
