#!/usr/bin/env python3
"""Train LaSeR's self-reward term on Qwen3-4B with LoRA (exploratory replication).

Loss per solution: (BETA * (log p(z_c | prompt, response, EOS) - c) - r)^2 with r = 1 for a correct solution.
c is the mean base-model log-probability of z_c after EOS on 256 training solutions (LaSeR replaces the reference
log-probability by such a constant). Only the hidden state at the EOS position reaches the LM head.
10% of the training problems are held out; the script reports their AUROC before and after training.
Saves the LoRA adapter and a merged bfloat16 model under results/q3/.
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
from scipy.stats import rankdata

from q3_common import BASE, BETA, MARKER, Q3, eos_id, prompt_ids, read_jsonl


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build(tok, rows):
    eos = eos_id(tok)
    items = []
    for r in rows:
        p = prompt_ids(tok, r["problem"])
        for ids, ok in zip(r["response_token_ids"], r["correctness"]):
            items.append({"ids": p + list(ids) + [eos], "y": float(bool(ok)), "pid": r["problem_id"]})
    return items


def marker_logprob(backbone, lm_head, batch, device):
    L = max(len(x["ids"]) for x in batch)
    inp = torch.zeros((len(batch), L), dtype=torch.long, device=device)
    att = torch.zeros((len(batch), L), dtype=torch.long, device=device)
    for i, x in enumerate(batch):
        inp[i, : len(x["ids"])] = torch.tensor(x["ids"], device=device)
        att[i, : len(x["ids"])] = 1
    h = backbone(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state
    pos = torch.tensor([len(x["ids"]) - 1 for x in batch], device=device)
    sel = h[torch.arange(len(batch), device=device), pos]
    return torch.log_softmax(lm_head(sel).float(), dim=-1)[:, MARKER]


def batches(items, max_tokens):
    items = sorted(items, key=lambda x: len(x["ids"]))
    b, mx = [], 0
    for x in items:
        mx2 = max(mx, len(x["ids"]))
        if b and mx2 * (len(b) + 1) > max_tokens:
            yield b
            b, mx2 = [], len(x["ids"])
        b.append(x); mx = mx2
    if b:
        yield b


@torch.no_grad()
def eval_lp(backbone, lm_head, items, device, max_tokens):
    out = {}
    for b in batches(items, max_tokens):
        lp = marker_logprob(backbone, lm_head, b, device)
        for x, v in zip(b, lp.tolist()):
            out[id(x)] = v
    return [out[id(x)] for x in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(Q3 / "graded/train.jsonl"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--micro-tokens", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: number of problems")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="training seed: LoRA initialization and batch order")
    args = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    rows = read_jsonl(args.train)
    if args.limit:
        rows = rows[: args.limit]
    pids = sorted({r["problem_id"] for r in rows})
    random.Random(0).shuffle(pids)
    val_p = set(pids[: max(1, len(pids) // 10)])
    train = build(tok, [r for r in rows if r["problem_id"] not in val_p])
    val = build(tok, [r for r in rows if r["problem_id"] in val_p])
    print(f"train solutions {len(train)} (correct {np.mean([x['y'] for x in train]):.3f}), val {len(val)}")
    if args.dry_run:
        print("max len", max(len(x["ids"]) for x in train), "example tail", tok.convert_ids_to_tokens(train[0]["ids"][-4:]))
        return
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    backbone, lm_head = model.model, model.lm_head
    model.eval()
    calib = random.Random(1).sample(train, min(256, len(train)))
    c = float(np.mean(eval_lp(backbone, lm_head, calib, device, args.micro_tokens)))
    val_before = eval_lp(backbone, lm_head, val, device, args.micro_tokens)
    print(f"reference constant c = {c:.3f}; val AUROC before training {auroc([x['y'] for x in val], val_before):.4f}")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(model, lcfg)
    pm.print_trainable_parameters()
    backbone, lm_head = pm.base_model.model.model, pm.base_model.model.lm_head
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    bl = list(batches(train, args.micro_tokens))
    total = math.ceil(len(bl) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(1, total)))
    step, t0, log = 0, time.time(), []
    pm.train()
    for ep in range(args.epochs):
        random.Random(1000 * args.seed + ep).shuffle(bl)
        for i, b in enumerate(bl):
            lp = marker_logprob(backbone, lm_head, b, device)
            y = torch.tensor([x["y"] for x in b], device=device)
            loss = ((BETA * (lp - c) - y) ** 2).mean() / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0 or i == len(bl) - 1:
                torch.nn.utils.clip_grad_norm_(pm.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    log.append({"step": step, "loss": float(loss.item() * args.accum), "s": time.time() - t0})
                    print(f"step {step}/{total} loss {loss.item() * args.accum:.4f} {time.time() - t0:.0f}s", flush=True)
    pm.eval()
    val_after = eval_lp(backbone, lm_head, val, device, args.micro_tokens)
    a_after = auroc([x["y"] for x in val], val_after)
    print(f"val AUROC after training {a_after:.4f}")
    out = Q3 / "model"
    pm.save_pretrained(str(out / "adapter"))
    merged = pm.merge_and_unload()
    merged.save_pretrained(str(out / "merged"), safe_serialization=True)
    tok.save_pretrained(str(out / "merged"))
    meta = {"base": BASE, "marker": MARKER, "beta": BETA, "reference_constant": c, "lr": args.lr, "rank": args.rank,
            "epochs": args.epochs, "micro_tokens": args.micro_tokens, "accum": args.accum, "steps": step,
            "train_solutions": len(train), "val_solutions": len(val),
            "val_auroc_before": auroc([x["y"] for x in val], val_before), "val_auroc_after": a_after, "log": log}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
