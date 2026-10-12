#!/usr/bin/env python3
"""Output rows of unused tokens in open models first released in 2026 (exploratory, CPU only).

For each model the script reads the output matrix (the input embedding when the two are tied) and three groups of
rows: U, tokens named unused or reserved plus the padding rows beyond the tokenizer's vocabulary; S, every other
special or added token; O, 2,000 ordinary tokens drawn with a fixed seed. For each group it reports the norm, the
cosine of each row with the group mean and the pairwise cosines. It also reports the cosines among the four U
tokens with the smallest ids, the marker and R1 tokens chosen for the score-only training runs.

Weights are read from the Hugging Face cache when present. Otherwise only the index, the tokenizer files and the
shard holding the needed matrix are downloaded to a temporary folder, which is deleted afterwards.
Writes results/analysis/unembed_2026.json.
"""
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
MODELS = [  # repository, first release on Hugging Face, revision read
    ("Qwen/Qwen3.5-0.8B", "2026-02-28", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("Qwen/Qwen3.5-2B", "2026-02-28", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    ("Qwen/Qwen3.5-4B", "2026-02-27", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("Qwen/Qwen3.5-9B", "2026-02-27", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("Qwen/Qwen3.6-35B-A3B", "2026-04-15", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    ("google/gemma-4-12B-it", "2026-05-23", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16", "2026-04-20", "e5e9932441de940c9a62185c870ea5bcd4cd24e2"),
    ("zai-org/GLM-4.7-Flash", "2026-01-19", "7dd20894a642a0aa287e9827cb1a1f7f91386b67"),
]
UNUSED = re.compile(r"^(?:<unused\d+>|<\|reserved_special_token_\d+\|>|\[unused\d+\])$")
SPECIAL = re.compile(r"^<(?:\|.*|.*\||pad|eos|bos|unk|mask|start_of.*|end_of.*|image.*|audio.*|video.*|soft_token.*)>$")
META = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]
SEED = 20261011


def fetch(repo, revision, folder):
    cache = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    local = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if (local / "config.json").exists() and any(local.glob("*.safetensors")):
        return local, False
    for name in META:
        try:
            hf_hub_download(repo, name, revision=revision, local_dir=folder)
        except Exception:
            pass
    return Path(folder), True


def matrix_key(path):
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    tied = bool(config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
    index = path / "model.safetensors.index.json"
    names = list(json.loads(index.read_text())["weight_map"]) if index.exists() else []
    if not names:
        with safe_open(str(next(path.glob("*.safetensors"))), framework="np") as f:
            names = list(f.keys())
    heads = [n for n in names if n.endswith("lm_head.weight") and "mtp" not in n]
    embeds = [n for n in names if n.endswith("embed_tokens.weight") and "vision" not in n and "audio" not in n]
    key = heads[0] if heads and not tied else sorted(embeds, key=len)[0]
    return key, tied


def read_matrix(path, key, repo, revision, downloaded):
    index = path / "model.safetensors.index.json"
    shard = json.loads(index.read_text())["weight_map"][key] if index.exists() else next(path.glob("*.safetensors")).name
    if downloaded:
        hf_hub_download(repo, shard, revision=revision, local_dir=path)
    with safe_open(str(path / shard), framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def vocabulary(path):
    tok = json.loads((path / "tokenizer.json").read_text())
    vocab = {int(i): s for s, i in tok["model"]["vocab"].items()}
    added = {int(a["id"]): a["content"] for a in tok.get("added_tokens", [])}
    config = path / "tokenizer_config.json"
    if config.exists():
        for i, item in json.loads(config.read_text()).get("added_tokens_decoder", {}).items():
            added.setdefault(int(i), item["content"])
    vocab.update(added)
    return vocab, set(added)


def group_stats(rows):
    if len(rows) == 0:
        return {"n": 0}
    unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    mean = unit.mean(0)
    to_mean = unit @ (mean / max(np.linalg.norm(mean), 1e-12))
    pair = (unit @ unit.T)[np.triu_indices(len(rows), 1)] if len(rows) > 1 else np.array([1.0])
    q = lambda x, p: float(np.quantile(x, p))
    return {"n": int(len(rows)), "norm_median": q(np.linalg.norm(rows, axis=1), 0.5),
            "cos_to_mean_median": q(to_mean, 0.5), "frac_cos_to_mean_gt_0.99": float(np.mean(to_mean > 0.99)),
            "pairwise_median": q(pair, 0.5), "pairwise_p05": q(pair, 0.05)}


def one_model(repo, created, revision):
    folder = tempfile.mkdtemp(dir=ROOT / "results")
    try:
        path, downloaded = fetch(repo, revision, folder)
        key, tied = matrix_key(path)
        W = read_matrix(path, key, repo, revision, downloaded)
        vocab, added = vocabulary(path)
        n_tok = max(vocab) + 1
        special = {i for i in added} | {i for i, s in vocab.items() if SPECIAL.match(s)}
        unused = {i for i, s in vocab.items() if UNUSED.match(s)} | set(range(n_tok, W.shape[0]))
        other_special = special - unused
        ordinary = sorted(set(range(min(n_tok, W.shape[0]))) - special - unused)
        o_ids = np.random.default_rng(SEED).choice(ordinary, size=2000, replace=False)
        u_ids = sorted(unused)
        first4 = u_ids[:4]
        unit = W[first4] / np.linalg.norm(W[first4], axis=1, keepdims=True) if first4 else None
        return {"repo": repo, "created": created, "revision": revision, "matrix": key, "tied": tied,
                "shape": list(W.shape), "tokenizer_size": n_tok,
                "U": group_stats(W[u_ids]), "S": group_stats(W[sorted(other_special)]), "O": group_stats(W[o_ids]),
                "rule_tokens": [vocab.get(i, f"padding row {i}") for i in first4],
                "rule_token_ids": first4,
                "rule_cosine_to_marker": (unit @ unit[0]).tolist() if first4 else None}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main():
    out = {"exploratory": True, "ordinary_seed": SEED, "models": []}
    for repo, created, revision in MODELS:
        try:
            entry = one_model(repo, created, revision)
        except Exception as exc:  # report and continue with the next model
            entry = {"repo": repo, "error": repr(exc)}
        out["models"].append(entry)
        print(repo, {k: entry.get(k, {}).get("pairwise_median") for k in "USO"}, entry.get("error", ""), flush=True)
    path = ROOT / "results/analysis/unembed_2026.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
