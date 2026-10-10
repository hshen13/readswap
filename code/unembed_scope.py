#!/usr/bin/env python3
"""Exploratory, CPU only: which vocabulary rows share the marker's output-embedding row?

For each released LaSeR checkpoint, cosine similarity of every output-embedding row with the marker row.
Reports how many rows exceed 0.9999, which token ids they are, and whether they are all special/added tokens.
Writes results/analysis/unembedding_scope.json.
"""
import glob
import json
import os

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

CKPTS = {
    "OctoThinker-3B-Short-LaSeR": 128002,
    "Qwen2.5-7B-LaSeR": 151652,
    "ORZ-7B-LaSeR": 151652,
}
res = {}
for name, marker in CKPTS.items():
    d = os.path.join("data/models", name)
    W = None
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, framework="pt") as h:
            if "lm_head.weight" in h.keys():
                W = h.get_tensor("lm_head.weight").float()
                break
    if W is None:
        res[name] = {"error": "no lm_head"}
        continue
    tok = AutoTokenizer.from_pretrained(d)
    m = W[marker]
    cos = torch.nn.functional.cosine_similarity(W, m.unsqueeze(0), dim=1)
    idx = torch.nonzero(cos >= 0.9999).flatten().tolist()
    vocab_size = len(tok)
    special_ids = set(tok.all_special_ids) | set(tok.added_tokens_decoder.keys())
    beyond_vocab = [i for i in idx if i >= vocab_size]
    named = [i for i in idx if i < vocab_size]
    res[name] = {
        "rows": int(W.shape[0]),
        "tokenizer_len": vocab_size,
        "n_rows_cos_ge_0.9999": len(idx),
        "n_within_tokenizer": len(named),
        "n_beyond_tokenizer": len(beyond_vocab),
        "within_tokenizer_all_special_or_added": all(i in special_ids for i in named),
        "within_tokenizer_tokens": [tok.convert_ids_to_tokens(i) for i in named][:60],
        "next_highest_cos_below_threshold": float(cos[cos < 0.9999].max()),
        "n_rows_cos_ge_0.99": int((cos >= 0.99).sum()),
    }
    print(name, json.dumps({k: v for k, v in res[name].items() if k != "within_tokenizer_tokens"}))
    print("   tokens:", res[name]["within_tokenizer_tokens"][:25])
json.dump(res, open("results/analysis/unembedding_scope.json", "w"), indent=1)
