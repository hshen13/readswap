#!/usr/bin/env python3
"""Exploratory mechanism control for the LaSeR readout (not pre-specified).

On the seed-0 solutions of all five benchmarks, for one checkpoint:
  1. one forward pass of prompt + response with a KV cache; the marker's log-probability at the last response token
     is the "none" condition (the R2 prefix);
  2. the cache is repeated and five single tokens are appended in one batch: the end-of-sequence token, a newline,
     a period, a special token of the chat template, and the rare word PostalCodesNL; the marker's log-probability
     is read after each (same token ids before that position in every condition);
  3. after the end-of-sequence token, the log-probability of the marker is split into its two terms,
     logit = h^T w_marker and log Z(h) = logsumexp over the vocabulary;
  4. the base model (the R5 reference) gives the same two terms after the end-of-sequence token.
Writes results/analysis/mech/<model>.jsonl (one row per solution) and prints a short summary.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from common import model_config, read_jsonl, released_response
from readout import sequence_for
from readout_fast import load_model

TRAINED_SPECIAL = {"OctoThinker-3B-Short-LaSeR": "<|start_header_id|>",
                   "Qwen2.5-7B-LaSeR": "<|im_start|>", "ORZ-7B-LaSeR": "<|im_start|>"}
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
NAMES = ["eos", "newline", "period", "chat_special", "rare_word"]


def one_id(tok, text, special=False):
    if special:
        i = tok.convert_tokens_to_ids(text)
        assert isinstance(i, int) and i >= 0 and i != tok.unk_token_id, text
        return i
    ids = tok(text, add_special_tokens=False).input_ids
    assert len(ids) == 1, (text, ids)
    return ids[0]


@torch.no_grad()
def trained_pass(model, prefix, swap_ids, marker, r1_first, device):
    inp = torch.tensor([prefix], device=device)
    out = model.model(input_ids=inp, use_cache=True)
    last = torch.log_softmax(model.lm_head(out.last_hidden_state[:, -1]).float()[0], dim=-1)
    rec = {"none": float(last[marker])}
    cache = out.past_key_values
    cache.batch_repeat_interleave(len(swap_ids))
    step = model.model(input_ids=torch.tensor(swap_ids, device=device).view(-1, 1), past_key_values=cache,
                       use_cache=True)
    logits = model.lm_head(step.last_hidden_state[:, -1]).float()
    logz = torch.logsumexp(logits, dim=-1)
    for j, name in enumerate(NAMES):
        rec[name] = float(logits[j, marker] - logz[j])
    rec["eos_logit_marker"] = float(logits[0, marker])
    rec["eos_logz"] = float(logz[0])
    rec["eos_logit_r1"] = float(logits[0, r1_first])
    return rec


@torch.no_grad()
def base_pass(model, seq, marker, device):
    out = model.model(input_ids=torch.tensor([seq], device=device), use_cache=False)
    logits = model.lm_head(out.last_hidden_state[:, -1]).float()[0]
    logz = torch.logsumexp(logits, dim=-1)
    return {"base_eos_logit_marker": float(logits[marker]), "base_eos_logz": float(logz),
            "base_eos": float(logits[marker] - logz)}


def load_cpu(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: solutions per dataset")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    cfg = model_config(args.model)
    tok = AutoTokenizer.from_pretrained(cfg["local_path"])
    marker, eos = int(cfg["marker_token_id"]), tok.eos_token_id
    r1_first = int(cfg["r1_swap_tokens"][0]["id"])
    swap_ids = [eos, one_id(tok, "\n"), one_id(tok, "."), one_id(tok, TRAINED_SPECIAL[args.model], special=True),
                one_id(tok, "PostalCodesNL")]
    items = []
    for ds in DATASETS:
        rows = read_jsonl(f"results/confirmatory/readout/{args.model}/{ds}_seed{args.seed}.jsonl")
        n = 0
        for ri, row in enumerate(rows):
            for k, resp in enumerate(row["responses"]):
                r0 = sequence_for(tok, row["problem"], released_response(resp), cfg, marker, include_eos=True,
                                  normalize=False)
                assert r0[-1] == marker and r0[-2] == eos, (ds, ri, k)
                items.append({"ds": ds, "problem_id": str(row["problem_id"]), "k": k,
                              "correct": bool(row["correctness"][k]),
                              "length": int(row["response_token_counts"][k]), "seq": r0[:-1]})
                n += 1
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
    device = torch.device(args.device)
    model = load_model(cfg["local_path"], torch) if args.device == "cuda" else load_cpu(cfg["local_path"])
    for i, it in enumerate(items):
        it.update(trained_pass(model, it["seq"][:-1], swap_ids, marker, r1_first, device))
        if args.limit:  # smoke: the cached EOS read must equal a full forward pass on prompt + response + EOS
            full = base_pass(model, it["seq"], marker, device)["base_eos"]
            print("cache check", it["ds"], round(it["eos"], 4), round(full, 4), flush=True)
        if i % 1000 == 0:
            print(f"trained {i}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    base = load_model(cfg["base_model_local_path"], torch) if args.device == "cuda" else load_cpu(cfg["base_model_local_path"])
    for i, it in enumerate(items):
        it.update(base_pass(base, it["seq"], marker, device))
        if i % 1000 == 0:
            print(f"base {i}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    out = Path(f"results/analysis/mech/{args.model}{'_smoke' if args.limit else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for it in items:
            it.pop("seq")
            fh.write(json.dumps(it) + "\n")
    print(json.dumps({"model": args.model, "n": len(items), "swap_ids": dict(zip(NAMES, swap_ids)),
                      "seconds": time.time() - t0}))


if __name__ == "__main__":
    main()
