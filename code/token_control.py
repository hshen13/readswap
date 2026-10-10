#!/usr/bin/env python3
"""Exploratory token and terminator controls for the LaSeR readout (not pre-specified).

On one benchmark and seed, reads from the released readout point (after response + EOS) the log-probability of
  - the marker and the three R1 tokens,
  - three ordinary but under-trained tokens whose output rows match the marker's (cosine >= 0.9999),
  - three well-trained ordinary tokens,
and reads the marker after two ordinary terminators ("\n", ".") put where the EOS token was.
Uses the confirmatory sequence builder and scorer (readout.sequence_for, readout_fast.score_sequences).
Writes results/analysis/token_control_<model>.json.
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

UNDERTRAINED = ["PostalCodesNL", "useRalative", "\tTokenNameIdentifier"]
TRAINED = [" the", " answer", "."]
TERMINATORS = {"newline": "\n", "period": "."}


def seq_with_terminator(tokenizer, problem, response, cfg, marker_id, terminator):
    user = problem + cfg.get("prompt_template", {}).get("problem_suffix", "")
    assistant = response + terminator + tokenizer.decode([marker_id], clean_up_tokenization_spaces=False)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        add_generation_prompt=False, tokenize=False).strip()
    if full[-10:] == "<|im_end|>":
        full = full[:-len("<|im_end|>")]
    return tokenizer(full, add_special_tokens=False).input_ids


def single_id(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
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
    swaps = [int(x["id"]) for x in cfg["r1_swap_tokens"]]
    under = {t: single_id(tok, t) for t in UNDERTRAINED}
    trained = {t: single_id(tok, t) for t in TRAINED}
    targets = [("marker", marker)] + [(f"r1_{s}", s) for s in swaps] + \
              [(f"under:{t}", i) for t, i in under.items()] + [(f"trained:{t}", i) for t, i in trained.items()]
    target_ids = [i for _, i in targets]

    rows = read_jsonl(f"results/confirmatory/readout/{args.model}/{args.dataset}_seed{args.seed}.jsonl")
    sequences, queries, labels, r0_ref = [], [], [], []
    for ri, row in enumerate(rows):
        for k, resp in enumerate(row["responses"]):
            resp = released_response(resp)
            r0 = sequence_for(tok, row["problem"], resp, cfg, marker, include_eos=True, normalize=False)
            assert r0[-1] == marker and r0[-2] == tok.eos_token_id, (ri, k)
            key = (ri, k)
            sequences.append((("eos",) + key, r0))
            queries.append((("eos",) + key, ("eos",) + key, len(r0) - 2, target_ids))
            for name, term in TERMINATORS.items():
                ids = seq_with_terminator(tok, row["problem"], resp, cfg, marker, term)
                assert ids[-1] == marker, (ri, k, name)
                sequences.append(((name,) + key, ids))
                queries.append(((name,) + key, (name,) + key, len(ids) - 2, [marker]))
            labels.append(bool(row["correctness"][k]))
            r0_ref.append(row["readout_logprobs"][k]["R0_marker"])
    if args.dry_run:
        lens = [len(ids) for _, ids in sequences]
        print(json.dumps({"targets": targets, "n_sequences": len(sequences), "n_responses": len(labels),
                          "max_len": max(lens), "mean_len": float(np.mean(lens)),
                          "sample_tail_eos": tok.convert_ids_to_tokens(sequences[0][1][-4:]),
                          "sample_tail_newline": tok.convert_ids_to_tokens(sequences[1][1][-4:]),
                          "sample_tail_period": tok.convert_ids_to_tokens(sequences[2][1][-4:])}, indent=1))
        return
    model = load_model(cfg["local_path"], torch)
    out = score_sequences(model, sequences, queries, tokenizer=tok, device=torch.device("cuda"),
                          max_batch_tokens=args.max_batch_tokens)
    keys = [(ri, k) for ri, row in enumerate(rows) for k in range(len(row["responses"]))]
    y = np.array(labels)
    eos = np.array([out[("eos",) + key] for key in keys])  # n x len(targets)
    res = {"model": args.model, "dataset": args.dataset, "seed": args.seed, "n": len(keys),
           "n_correct": int(y.sum()), "targets": {}, "terminators": {}}
    marker_lp = eos[:, 0]
    res["max_abs_diff_vs_confirmatory_R0_marker"] = float(np.max(np.abs(marker_lp - np.array(r0_ref))))
    for j, (name, tid) in enumerate(targets):
        lp = eos[:, j]
        res["targets"][name] = {"token_id": int(tid), "auroc": float(roc_auc_score(y, lp)),
                                "spearman_with_marker": float(np.corrcoef(np.argsort(np.argsort(lp)),
                                                                           np.argsort(np.argsort(marker_lp)))[0, 1]),
                                "median_logprob": float(np.median(lp))}
    for name in TERMINATORS:
        lp = np.array([out[(name,) + key][0] for key in keys])
        res["terminators"][name] = {"auroc": float(roc_auc_score(y, lp)), "median_logprob": float(np.median(lp))}
    res["auroc_eos_marker"] = res["targets"]["marker"]["auroc"]
    res["seconds"] = time.time() - t0
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    Path(f"results/analysis/token_control_{args.model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
