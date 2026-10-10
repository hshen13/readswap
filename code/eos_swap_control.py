#!/usr/bin/env python3
"""Exploratory, exact-token control for the end-of-sequence token (not pre-specified).

Takes the confirmatory R0 token ids (prompt + response + EOS + marker) of the seed-0 MATH500 solutions and replaces
only the EOS id, at the same position, by another single token id: a newline, a period, a trained special token,
an under-trained word, or nothing (the R2 prefix). The marker's log-probability is read at the same next-token
position. No retokenization, so the prefix before that position is identical in every condition.
Writes results/analysis/eos_swap_<model>.json.
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
from readout_fast import load_model, score_sequences

TRAINED_SPECIAL = {"OctoThinker-3B-Short-LaSeR": "<|start_header_id|>",
                   "Qwen2.5-7B-LaSeR": "<|im_start|>", "ORZ-7B-LaSeR": "<|im_start|>"}


def one_id(tok, text, special=False):
    if special:
        i = tok.convert_tokens_to_ids(text)
        assert isinstance(i, int) and i >= 0 and i != tok.unk_token_id, text
        return i
    ids = tok(text, add_special_tokens=False).input_ids
    assert len(ids) == 1, (text, ids)
    return ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = model_config(args.model)
    tok = AutoTokenizer.from_pretrained(cfg["local_path"])
    tok.padding_side = "right"
    marker = int(cfg["marker_token_id"])
    eos = tok.eos_token_id
    swaps = {"eos": eos, "newline": one_id(tok, "\n"), "period": one_id(tok, "."),
             "trained_special": one_id(tok, TRAINED_SPECIAL[args.model], special=True),
             "undertrained_word": one_id(tok, "PostalCodesNL")}
    rows = read_jsonl(f"results/confirmatory/readout/{args.model}/{args.dataset}_seed{args.seed}.jsonl")
    sequences, queries, labels = [], [], []
    for ri, row in enumerate(rows):
        for k, resp in enumerate(row["responses"]):
            resp = released_response(resp)
            r0 = sequence_for(tok, row["problem"], resp, cfg, marker, include_eos=True, normalize=False)
            assert r0[-1] == marker and r0[-2] == eos, (ri, k)
            for name, sid in swaps.items():
                ids = r0[:-2] + [sid, marker]
                key = (name, ri, k)
                sequences.append((key, ids))
                queries.append((key, key, len(ids) - 2, [marker]))
            ids = r0[:-2] + [marker]  # no token in place of EOS: the R2 prefix
            key = ("none", ri, k)
            sequences.append((key, ids))
            queries.append((key, key, len(ids) - 2, [marker]))
            labels.append(bool(row["correctness"][k]))
    if args.dry_run:
        print(json.dumps({"swaps": swaps, "n_sequences": len(sequences), "n_responses": len(labels),
                          "tail": {n: tok.convert_ids_to_tokens(sequences[i][1][-3:]) for i, n in enumerate(list(swaps) + ["none"])}},
                         indent=1, ensure_ascii=False))
        return
    model = load_model(cfg["local_path"], torch)
    out = score_sequences(model, sequences, queries, tokenizer=tok, device=torch.device("cuda"),
                          max_batch_tokens=args.max_batch_tokens)
    y = np.array(labels)
    keys = [(ri, k) for ri, row in enumerate(rows) for k in range(len(row["responses"]))]
    res = {"model": args.model, "dataset": args.dataset, "seed": args.seed, "n": len(keys), "swaps": swaps, "auroc": {}}
    for name in list(swaps) + ["none"]:
        lp = np.array([out[(name,) + key][0] for key in keys])
        res["auroc"][name] = {"auroc": float(roc_auc_score(y, lp)), "median_logprob": float(np.median(lp))}
    res["seconds"] = time.time() - t0
    Path(f"results/analysis/eos_swap_{args.model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["auroc"], indent=1))


if __name__ == "__main__":
    main()
