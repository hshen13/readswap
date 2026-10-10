#!/usr/bin/env python3
"""Exploratory: how similar are the output-embedding rows of the LaSeR marker and the three R1 tokens?

CPU only; reads one tensor from the safetensors shards of each released checkpoint and its base model.
Writes results/analysis/unembedding_rows.json.
"""
import glob
import json
import os

import torch
from safetensors import safe_open

HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
CFG = {
    "OctoThinker-3B-Short-LaSeR": ("Keven16/OctoThinker-3B-Short-LaSeR", "OctoThinker/OctoThinker-3B-Short-Base", 128002, [128003, 128004, 128005]),
    "Qwen2.5-7B-LaSeR": ("Keven16/Qwen2.5-7B-LaSeR", "Qwen/Qwen2.5-7B", 151652, [151648, 151649, 151650]),
    "ORZ-7B-LaSeR": ("Keven16/ORZ-7B-LaSeR", "Open-Reasoner-Zero/Open-Reasoner-Zero-7B", 151652, [151648, 151649, 151650]),
}


def snapshot(repo):
    if repo.startswith("Keven16/"):
        local = os.path.join("data/models", repo.split("/", 1)[1])
        return local if os.path.isdir(local) else None
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(d + "/*"))
    return snaps[-1] if snaps else None


def rows(repo, ids):
    snap = snapshot(repo)
    if snap is None:
        return None, "not in cache"
    files = sorted(glob.glob(snap + "/*.safetensors"))
    name_order = ["lm_head.weight", "model.embed_tokens.weight", "embed_tokens.weight"]
    for name in name_order:
        for f in files:
            with safe_open(f, framework="pt") as h:
                if name in h.keys():
                    sl = h.get_slice(name)
                    out = torch.stack([sl[i:i + 1][0].float() for i in ids])
                    # a reference set of ordinary rows for scale
                    ref = torch.stack([sl[i:i + 1][0].float() for i in range(1000, 1200)])
                    return (out, ref, name), None
    return None, "no output embedding tensor"


res = {}
for label, (ck, base, marker, swaps) in CFG.items():
    entry = {}
    for role, repo in (("checkpoint", ck), ("base", base)):
        got, err = rows(repo, [marker] + swaps)
        if got is None:
            entry[role] = {"error": err}
            continue
        w, ref, tensor = got
        m = w[0]
        cos = [float(torch.nn.functional.cosine_similarity(m, w[i], dim=0)) for i in range(1, 4)]
        dist = [float((m - w[i]).norm()) for i in range(1, 4)]
        entry[role] = {
            "tensor": tensor,
            "marker_norm": float(m.norm()),
            "swap_norms": [float(w[i].norm()) for i in range(1, 4)],
            "cos_marker_swap": cos,
            "l2_marker_swap": dist,
            "ordinary_row_norm_mean": float(ref.norm(dim=1).mean()),
        }
        if role == "checkpoint":
            entry["_ck_marker"] = m
        else:
            entry["_base_marker"] = m
    if "_ck_marker" in entry and "_base_marker" in entry:
        a, b = entry.pop("_ck_marker"), entry.pop("_base_marker")
        entry["marker_change_l2"] = float((a - b).norm())
        entry["marker_change_cos"] = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    entry.pop("_ck_marker", None)
    entry.pop("_base_marker", None)
    res[label] = entry

os.makedirs("results/analysis", exist_ok=True)
with open("results/analysis/unembedding_rows.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
