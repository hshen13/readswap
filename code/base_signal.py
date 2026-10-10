#!/usr/bin/env python3
"""Exploratory: does the base model (before LaSeR training) already carry the correctness signal?

Uses the confirmatory readout files: R5 stored the base model's log-probability of the marker after
response + EOS (readout_logprobs[k]["R5_base_marker"]), on exactly the R0 token ids. Its AUROC against correctness
is computed per benchmark and averaged with weights proportional to the number of solutions (the primary AUROC
of Amendment A1); also within problems. Writes results/analysis/base_signal.json.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
CKPTS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
out = {}
for m in CKPTS:
    per = {}
    tot_n = 0
    wsum = {"base": 0.0, "ckpt": 0.0}
    within = {"base": [], "ckpt": []}
    for ds in DATASETS:
        y, b, c = [], [], []
        byp = defaultdict(lambda: ([], [], []))
        for seed in (0, 1, 2):
            for line in open(ROOT / f"results/confirmatory/readout/{m}/{ds}_seed{seed}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                for k, lp in enumerate(r["readout_logprobs"]):
                    if lp.get("R5_base_marker") is None:
                        continue
                    yy = bool(r["correctness"][k])
                    y.append(yy); b.append(lp["R5_base_marker"]); c.append(lp["R0_marker"])
                    t = byp[r["problem_id"]]
                    t[0].append(yy); t[1].append(lp["R5_base_marker"]); t[2].append(lp["R0_marker"])
        y = np.array(y)
        if y.all() or (~y).all():
            continue
        ab, ac = roc_auc_score(y, b), roc_auc_score(y, c)
        per[ds] = {"n": int(len(y)), "auroc_base": float(ab), "auroc_checkpoint": float(ac)}
        tot_n += len(y)
        wsum["base"] += len(y) * ab
        wsum["ckpt"] += len(y) * ac
        for yy, bb, cc in byp.values():
            yy = np.array(yy)
            if yy.any() and (~yy).any():
                within["base"].append(roc_auc_score(yy, bb)); within["ckpt"].append(roc_auc_score(yy, cc))
    out[m] = {"primary_auroc_base": wsum["base"] / tot_n, "primary_auroc_checkpoint": wsum["ckpt"] / tot_n,
              "within_problem_auroc_base": float(np.mean(within["base"])),
              "within_problem_auroc_checkpoint": float(np.mean(within["ckpt"])), "by_benchmark": per}
    print(m, {k: round(v, 4) for k, v in out[m].items() if k != "by_benchmark"})
    print("   ", {d: (round(v["auroc_base"], 3), round(v["auroc_checkpoint"], 3)) for d, v in per.items()})
(ROOT / "results/analysis/base_signal.json").write_text(json.dumps(out, indent=1))
