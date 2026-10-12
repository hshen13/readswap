"""Shared settings for score-only training in Gemma-4-12B (exploratory), as in q3_common.py.

The base model writes the solutions once (non-thinking chat mode). A LoRA adapter is trained only on LaSeR's
self-reward regression term, (beta * (log p(z_c | x, y, EOS) - c) - r)^2, on the base model's own training
solutions. The untrained base model and the trained model then read the same fixed test solutions.
Sequences are built from token ids (prompt ids + generated ids + EOS + marker), with no retokenization.
BASE is the local snapshot of google/gemma-4-12B-it at the revision below in the Hugging Face cache (HF_HOME);
Q3_BASE_MODEL overrides it.
"""
import json
import os
from pathlib import Path

from common import hf_snapshot_path

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
Q3 = Path(os.environ.get("Q3_DIR", str(ROOT / "results/gemma4_12b")))
BASE_REPO, BASE_REVISION = "google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
BASE = os.environ.get("Q3_BASE_MODEL") or str(hf_snapshot_path(BASE_REPO, BASE_REVISION))
MARKER = 6  # <unused0>, the smallest unused id
R1_TOKENS = [7, 8, 9]  # <unused1>, <unused2>, <unused3>
BETA = 0.1                   # LaSeR's ratio coefficient for Qwen
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_NEW = 4096
N_SAMPLES = 8


def prompt_ids(tok, problem):
    msgs = [{"role": "user", "content": problem + SUFFIX}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False,
                                   return_dict=False)


def eos_id(tok):
    return tok.convert_tokens_to_ids("<turn|>")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

def head_logits(backbone, lm_head, hidden):
    """Gemma's output logits, including its final logit soft-capping."""
    import torch
    logits = lm_head(hidden)
    cap = backbone.config.get_text_config().final_logit_softcapping
    return torch.tanh(logits / cap) * cap if cap is not None else logits
