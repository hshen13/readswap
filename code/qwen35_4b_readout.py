#!/usr/bin/env python3
"""Readouts of score-only training in Qwen3.5-4B on fixed test solutions (exploratory).

For the base model and for the trained model, on token-id prefixes built without retokenization:
  P0 = prompt + response + EOS   -> marker, R1 tokens (R0, R1); on MATH500 also under-trained and trained words
  P2 = prompt + response         -> marker (R2)
  P3 = prompt + response up to the final boxed answer (retokenized text) -> marker (R3)
  on MATH500 only, trained model: prompt + response + X -> marker, X in {newline, period, <|im_start|>,
  under-trained word} (exact-token EOS swap).
Scores: R0 = BETA*(lp0 - c), R1 the mean over the R1 tokens, R2/R3 likewise, R4 = BETA*lp0,
R5 = BETA*(lp0_trained - lp0_base). Writes results/q3/readout/<dataset>.jsonl and results/q3/readout_meta.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_4b_common import BASE, BETA, MARKER, Q3, R1_TOKENS, eos_id, prompt_ids, read_jsonl, write_jsonl  # noqa: E402

DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench", "aime_2026", "hmmt_feb_2026"]


def truncate_at_last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed")
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[: j + 1]
        j += 1
    return None


@torch.no_grad()
def score(model, prefixes, targets_of, device, max_tokens):
    """prefixes: list of (key, ids); next-token log-probs at the last position for targets_of[key]."""
    backbone, lm_head = model.model, model.lm_head
    items = sorted(prefixes, key=lambda kv: len(kv[1]))
    out, b, mx = {}, [], 0

    def run(batch):
        L = max(len(ids) for _, ids in batch)
        inp = torch.zeros((len(batch), L), dtype=torch.long, device=device)
        att = torch.zeros_like(inp)
        for i, (_, ids) in enumerate(batch):
            inp[i, : len(ids)] = torch.tensor(ids, device=device)
            att[i, : len(ids)] = 1
        h = backbone(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state
        pos = torch.tensor([len(ids) - 1 for _, ids in batch], device=device)
        lp = torch.log_softmax(lm_head(h[torch.arange(len(batch), device=device), pos]).float(), dim=-1)
        for i, (key, _) in enumerate(batch):
            t = torch.tensor(targets_of[key], device=device)
            out[key] = lp[i].gather(0, t).tolist()

    for kv in items:
        mx2 = max(mx, len(kv[1]))
        if b and mx2 * (len(b) + 1) > max_tokens:
            run(b)
            b, mx2 = [], len(kv[1])
        b.append(kv); mx = mx2
    if b:
        run(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    eos = eos_id(tok)
    one = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    words_under = {w: one(w) for w in ["PostalCodesNL", "useRalative", "\tTokenNameIdentifier"]}
    words_trained = {w: one(w) for w in [" the", " answer", "."]}
    for w, ids in {**words_under, **words_trained}.items():
        assert len(ids) == 1, (w, ids)
    swaps = {"newline": one("\n")[0], "period": one(".")[0], "trained_special": tok.convert_tokens_to_ids("<|im_start|>"),
             "undertrained_word": words_under["PostalCodesNL"][0]}
    meta = json.loads((Q3 / "model/train_meta.json").read_text())
    c = meta["reference_constant"]
    data = {ds: read_jsonl(Q3 / f"graded/{ds}.jsonl") for ds in DATASETS}
    if args.limit:
        data = {ds: rows[: args.limit] for ds, rows in data.items()}
    pre, tgt = [], {}
    sweep_t = [MARKER] + R1_TOKENS + [v[0] for v in words_under.values()] + [v[0] for v in words_trained.values()]
    for ds, rows in data.items():
        for ri, r in enumerate(rows):
            p = prompt_ids(tok, r["problem"])
            for k, ids in enumerate(r["response_token_ids"]):
                ids = list(ids)
                key = (ds, ri, k)
                pre.append((key + ("P0",), p + ids + [eos]))
                tgt[key + ("P0",)] = sweep_t if ds == "math500" else [MARKER] + R1_TOKENS
                pre.append((key + ("P2",), p + ids)); tgt[key + ("P2",)] = [MARKER]
                tr = truncate_at_last_boxed(r["responses"][k])
                if tr is not None:
                    pre.append((key + ("P3",), p + tok(tr, add_special_tokens=False).input_ids)); tgt[key + ("P3",)] = [MARKER]
    swap_pre = []
    for ri, r in enumerate(data["math500"]):
        p = prompt_ids(tok, r["problem"])
        for k, ids in enumerate(r["response_token_ids"]):
            for name, sid in swaps.items():
                key = ("math500", ri, k, "S_" + name)
                swap_pre.append((key, p + list(ids) + [sid])); tgt[key] = [MARKER]
    device = torch.device("cuda")
    res = {}
    t0 = time.time()
    for name, path in [("base", BASE), ("trained", str(Q3 / "model/merged"))]:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        res[name] = score(model, pre + (swap_pre if name == "trained" else []), tgt, device, args.max_tokens)
        print(name, "done", f"{time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    for ds, rows in data.items():
        out_rows = []
        for ri, r in enumerate(rows):
            scores, lps = [], []
            for k in range(len(r["response_token_ids"])):
                key = (ds, ri, k)
                t0v, b0v = res["trained"][key + ("P0",)], res["base"][key + ("P0",)]
                lp0, lp0b = t0v[0], b0v[0]
                r1 = {str(t): BETA * (v - c) for t, v in zip(R1_TOKENS, t0v[1:4])}
                lp2 = res["trained"][key + ("P2",)][0]
                lp3 = res["trained"].get(key + ("P3",), [None])[0]
                sc = {"R0_reference": BETA * (lp0 - c), "R1_marker_swap": r1,
                      "R1_marker_swap_mean": sum(r1.values()) / 3, "R2_position_eos": BETA * (lp2 - c),
                      "R3_position_answer": None if lp3 is None else BETA * (lp3 - c),
                      "R4_cref_none": BETA * lp0, "R5_cref_recomputed": BETA * (lp0 - lp0b),
                      "BASE_R0": BETA * (lp0b - c)}
                lp = {"R0_marker": lp0, "R2_marker": lp2, "R3_marker": lp3, "R5_base_marker": lp0b}
                if ds == "math500":
                    lp["P0_targets"] = dict(zip([str(x) for x in sweep_t], t0v))
                    lp["P0_targets_base"] = dict(zip([str(x) for x in sweep_t], b0v))
                    lp["eos_swap"] = {n: res["trained"][(ds, ri, k, "S_" + n)][0] for n in swaps}
                scores.append(sc); lps.append(lp)
            o = {kk: v for kk, v in r.items() if kk != "response_token_ids"}
            o["readout_scores"], o["readout_logprobs"], o["reference_constant"] = scores, lps, c
            out_rows.append(o)
        write_jsonl(Q3 / f"readout/{ds}.jsonl", out_rows)
    (Q3 / "readout_meta.json").write_text(json.dumps({"swaps": swaps, "sweep_targets": sweep_t,
                                                       "words_under": words_under, "words_trained": words_trained,
                                                       "reference_constant": c, "seconds": time.time() - t0}, indent=1))
    print("readout written")


if __name__ == "__main__":
    main()
