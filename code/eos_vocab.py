#!/usr/bin/env python3
"""Vocabulary sweep at the end-of-sequence position (exploratory, not pre-specified).

On the seed-zero MATH500 solutions of one checkpoint, the prompt + response is encoded once with a KV cache; the cache is
then extended by one token, in chunks, for: the end-of-sequence token, 200 ordinary vocabulary tokens drawn with a fixed
seed, and every added special token. The marker's log-probability is read after each, so the token ids before that
position are identical in every condition. Reports the AUROC of every condition and the rank of the end-of-sequence
token among them. Writes results/analysis/eos_vocab_<model>.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from common import model_config, read_jsonl, released_response
from readout import sequence_for
from readout_fast import load_model

N_RANDOM = 200
CACHE_BUDGET = 8e9  # bytes of repeated KV cache per chunk


def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for layer in zip(cache.key_cache, cache.value_cache) for t in layer)


@torch.no_grad()
def read_after(model, prefix, cand, marker, device):
    out = model.model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    base_cache = out.past_key_values
    per = max(1, int(CACHE_BUDGET // max(1, cache_bytes(base_cache))))
    res = []
    for s in range(0, len(cand), per):
        chunk = cand[s:s + per]
        cache = type(base_cache)()
        for li in range(len(base_cache.key_cache)):
            cache.update(base_cache.key_cache[li].repeat(len(chunk), 1, 1, 1),
                         base_cache.value_cache[li].repeat(len(chunk), 1, 1, 1), li)
        step = model.model(input_ids=torch.tensor(chunk, device=device).view(-1, 1), past_key_values=cache,
                           use_cache=True)
        lp = torch.log_softmax(model.lm_head(step.last_hidden_state[:, -1]).float(), dim=-1)[:, marker]
        res.extend(lp.tolist())
        del cache, step
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    cfg = model_config(args.model)
    tok = AutoTokenizer.from_pretrained(cfg["local_path"])
    marker, eos = int(cfg["marker_token_id"]), tok.eos_token_id
    special = sorted(i for i in tok.added_tokens_decoder if i not in (marker, eos))
    rng = np.random.default_rng(20261009)
    pool = [i for i in range(len(tok)) if i not in tok.added_tokens_decoder]
    rand = sorted(int(i) for i in rng.choice(pool, N_RANDOM, replace=False))
    cand = [eos] + rand + special
    rows = read_jsonl(f"results/confirmatory/readout/{args.model}/{args.dataset}_seed{args.seed}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    model = load_model(cfg["local_path"], torch)
    device = torch.device("cuda")
    L, y = [], []
    for ri, row in enumerate(rows):
        for k, resp in enumerate(row["responses"]):
            r0 = sequence_for(tok, row["problem"], released_response(resp), cfg, marker, include_eos=True,
                              normalize=False)
            L.append(read_after(model, r0[:-2], cand, marker, device)); y.append(bool(row["correctness"][k]))
        if ri % 100 == 0:
            print(f"{ri}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    L = np.array(L); y = np.array(y)
    au = [float(roc_auc_score(y, L[:, j])) for j in range(len(cand))]
    kinds = ["eos"] + ["random"] * len(rand) + ["special"] * len(special)
    res = {"model": args.model, "n": int(len(y)), "eos_auroc": au[0],
           "tokens": [{"id": int(t), "token": tok.convert_ids_to_tokens(int(t)), "kind": kd, "auroc": a,
                       "median_logprob": float(np.median(L[:, j]))}
                      for j, (t, kd, a) in enumerate(zip(cand, kinds, au))],
           "rank_of_eos": int(1 + sum(a > au[0] for a in au[1:])), "n_candidates": len(cand) - 1,
           "max_other_auroc": float(max(au[1:])), "n_within_002": int(sum(a >= au[0] - 0.02 for a in au[1:])),
           "seconds": time.time() - t0}
    Path(f"results/analysis/eos_vocab_{args.model}{'_smoke' if args.limit else ''}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "tokens"}))


if __name__ == "__main__":
    main()
