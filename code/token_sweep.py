#!/usr/bin/env python3
"""Exploratory token sweep for the LaSeR readout (not pre-specified).

Picks tokens by the cosine similarity of their output-embedding row to the marker's row, in fixed bins, with a
fixed generator seed, and reads the log-probability of every picked token at the released point (after response
+ EOS) on the seed-0 MATH500 solutions, from one forward pass per solution. Reports AUROC and the rank
correlation with the marker's log-probability for every token.
Bins: >=0.9999, [0.999, 0.9999), [0.99, 0.999), [0.9, 0.99), [0.5, 0.9), <0.5; up to 20 tokens per bin
(all tokens if fewer), drawn among ids < len(tokenizer), excluding the marker.
Writes results/analysis/token_sweep_<model>.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from common import model_config, read_jsonl, released_response
from readout import sequence_for
from readout_fast import load_model, score_sequences

BINS = [(0.9999, 1.01), (0.999, 0.9999), (0.99, 0.999), (0.9, 0.99), (0.5, 0.9), (-1.01, 0.5)]
PER_BIN = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    args = ap.parse_args()
    t0 = time.time()
    cfg = model_config(args.model)
    tok = AutoTokenizer.from_pretrained(cfg["local_path"])
    tok.padding_side = "right"
    marker = int(cfg["marker_token_id"])
    model = load_model(cfg["local_path"], torch)
    W = model.get_output_embeddings().weight.detach().float()
    cos = torch.nn.functional.cosine_similarity(W, W[marker].unsqueeze(0), dim=1).cpu().numpy()
    vocab = len(tok)
    rng = np.random.default_rng(20261008)
    picked = []
    for lo, hi in BINS:
        ids = [i for i in np.nonzero((cos >= lo) & (cos < hi))[0].tolist() if i < vocab and i != marker]
        if len(ids) > PER_BIN:
            ids = sorted(rng.choice(ids, PER_BIN, replace=False).tolist())
        picked += [(i, lo, hi) for i in ids]
    target_ids = [marker] + [i for i, _, _ in picked]
    rows = read_jsonl(f"results/confirmatory/readout/{args.model}/{args.dataset}_seed{args.seed}.jsonl")
    sequences, queries, labels = [], [], []
    for ri, row in enumerate(rows):
        for k, resp in enumerate(row["responses"]):
            resp = released_response(resp)
            r0 = sequence_for(tok, row["problem"], resp, cfg, marker, include_eos=True, normalize=False)
            assert r0[-1] == marker and r0[-2] == tok.eos_token_id
            key = (ri, k)
            sequences.append((key, r0))
            queries.append((key, key, len(r0) - 2, target_ids))
            labels.append(bool(row["correctness"][k]))
    out = score_sequences(model, sequences, queries, tokenizer=tok, device=torch.device("cuda"),
                          max_batch_tokens=args.max_batch_tokens)
    keys = [(ri, k) for ri, row in enumerate(rows) for k in range(len(row["responses"]))]
    L = np.array([out[key] for key in keys])  # n x targets
    y = np.array(labels)
    marker_lp = L[:, 0]
    res = {"model": args.model, "n": len(keys), "marker_auroc": float(roc_auc_score(y, marker_lp)), "tokens": []}
    for j, (tid, lo, hi) in enumerate(picked, start=1):
        lp = L[:, j]
        res["tokens"].append({"id": int(tid), "token": tok.convert_ids_to_tokens(int(tid)), "cos": float(cos[tid]),
                              "bin": [lo, hi], "auroc": float(roc_auc_score(y, lp)),
                              "spearman_with_marker": float(spearmanr(lp, marker_lp).statistic),
                              "median_diff_lp": float(np.median(lp - marker_lp)),
                              "median_abs_diff_lp": float(np.median(np.abs(lp - marker_lp))),
                              "p99_abs_diff_lp": float(np.percentile(np.abs(lp - marker_lp), 99)),
                              "max_abs_diff_lp": float(np.max(np.abs(lp - marker_lp)))})
    res["seconds"] = time.time() - t0
    Path(f"results/analysis/token_sweep_{args.model}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for lo, hi in BINS:
        t = [x for x in res["tokens"] if x["bin"] == [lo, hi]]
        if t:
            print(f"bin [{lo}, {hi}) n={len(t)} AUROC {np.mean([x['auroc'] for x in t]):.3f} "
                  f"(min {min(x['auroc'] for x in t):.3f}) rho {np.mean([x['spearman_with_marker'] for x in t]):.3f}")


if __name__ == "__main__":
    main()
