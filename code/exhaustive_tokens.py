#!/usr/bin/env python3
"""Exhaustive token test for the LaSeR readout (exploratory, not pre-specified).

Populations, both read at the released point (after response + EOS) on the seed-zero solutions of all five benchmarks:
  S: every added special token of the tokenizer except the marker. This population is defined by the tokenizer, not by
     geometry. Each token is flagged when the chat template uses it (it appears in a formatted two-turn conversation).
  A: every vocabulary row whose cosine with the marker's output row is at least 0.9999 (ids below len(tokenizer)).
For every token: benchmark-weighted AUROC, Spearman correlation with the marker's log-probability, and the largest and
99th-percentile absolute difference from the marker's log-probability.
Equivalence bounds fixed before this run: a token reads the marker's score when its AUROC is within 0.02 of the
marker's and its score differs by at most 0.02 on every solution (beta = 0.1, so at most 0.2 nat in log-probability).
Writes results/analysis/exhaustive_tokens_<model>.json.
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

DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]


def weighted_auroc(y, s, ds):
    tot, acc = 0, 0.0
    for d in DATASETS:
        m = ds == d
        if m.sum() and 0 < y[m].sum() < m.sum():
            acc += m.sum() * roc_auc_score(y[m], s[m]); tot += m.sum()
    return float(acc / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    args = ap.parse_args()
    t0 = time.time()
    cfg = model_config(args.model)
    tok = AutoTokenizer.from_pretrained(cfg["local_path"])
    tok.padding_side = "right"
    marker = int(cfg["marker_token_id"])
    beta = float(cfg["ratio_coef"])
    model = load_model(cfg["local_path"], torch)
    W = model.get_output_embeddings().weight.detach().float()
    cos = torch.nn.functional.cosine_similarity(W, W[marker].unsqueeze(0), dim=1).cpu().numpy()
    vocab = len(tok)
    special = sorted(i for i in tok.added_tokens_decoder if i != marker and i < W.shape[0])
    conv = tok.apply_chat_template([{"role": "user", "content": "What is 1+1?"},
                                    {"role": "assistant", "content": "2"}], tokenize=True)
    used = set(conv)
    aligned = sorted(int(i) for i in np.nonzero(cos >= 0.9999)[0] if i < vocab and i != marker)
    targets = [marker] + sorted(set(special) | set(aligned))
    sequences, queries, labels, dss = [], [], [], []
    for d in DATASETS:
        rows = read_jsonl(f"results/confirmatory/readout/{args.model}/{d}_seed{args.seed}.jsonl")
        for ri, row in enumerate(rows):
            for k, resp in enumerate(row["responses"]):
                r0 = sequence_for(tok, row["problem"], released_response(resp), cfg, marker, include_eos=True,
                                  normalize=False)
                key = (d, ri, k)
                sequences.append((key, r0))
                queries.append((key, key, len(r0) - 2, targets))
                labels.append(bool(row["correctness"][k])); dss.append(d)
    out = score_sequences(model, sequences, queries, tokenizer=tok, device=torch.device("cuda"),
                          max_batch_tokens=args.max_batch_tokens)
    L = np.array([out[q[0]] for q in queries])
    y = np.array(labels); ds = np.array(dss)
    m = L[:, 0]
    res = {"model": args.model, "n": len(y), "beta": beta, "marker_auroc": weighted_auroc(y, m, ds),
           "n_special": len(special), "n_aligned": len(aligned), "tokens": []}
    for j, t in enumerate(targets[1:], start=1):
        d = np.abs(L[:, j] - m)
        res["tokens"].append({
            "id": int(t), "token": tok.convert_ids_to_tokens(int(t)), "cos": float(cos[t]),
            "special": t in special, "template_uses": t in used, "aligned": t in aligned,
            "auroc": weighted_auroc(y, L[:, j], ds), "spearman": float(spearmanr(L[:, j], m).statistic),
            "max_abs_diff_lp": float(d.max()), "p99_abs_diff_lp": float(np.percentile(d, 99))})
    res["seconds"] = time.time() - t0
    Path(f"results/analysis/exhaustive_tokens_{args.model}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for name, sel in [("special, unused by template", lambda t: t["special"] and not t["template_uses"]),
                      ("special, used by template", lambda t: t["special"] and t["template_uses"]),
                      ("aligned rows", lambda t: t["aligned"])]:
        ts = [t for t in res["tokens"] if sel(t)]
        if ts:
            print(f"{name}: n={len(ts)} AUROC {min(t['auroc'] for t in ts):.4f}-{max(t['auroc'] for t in ts):.4f} "
                  f"(marker {res['marker_auroc']:.4f}) max|dl| {max(t['max_abs_diff_lp'] for t in ts):.3f} "
                  f"within bounds {sum(abs(t['auroc'] - res['marker_auroc']) <= 0.02 and beta * t['max_abs_diff_lp'] <= 0.02 for t in ts)}",
                  flush=True)


if __name__ == "__main__":
    main()
