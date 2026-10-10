#!/usr/bin/env python3
"""Exploratory paired test of calibration: ECE and Brier of R5 minus R0, cluster bootstrap over problems.

Scores clipped to [0, 1]; ECE with 15 equal-width bins (Amendment A1). 2,000 stratified problem resamples,
generator seed 20260924. Writes results/analysis/calibration_paired.json.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
CKPTS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
B = 2000
edges = np.linspace(0, 1, 16)


def ece(p, y):
    idx = np.clip(np.digitize(p, edges) - 1, 0, 14)
    tot = 0.0
    for b in range(15):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return tot / len(p)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


out = {}
for model in CKPTS:
    cl = defaultdict(lambda: {"y": [], "R0": [], "R5": []})
    bench = {}
    for ds in DATASETS:
        for seed in (0, 1, 2):
            for line in open(ROOT / f"results/confirmatory/readout/{model}/{ds}_seed{seed}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                pid = f"{ds}:{r['problem_id']}"
                bench[pid] = ds
                for k, item in enumerate(r["readout_scores"]):
                    cl[pid]["y"].append(float(r["correctness"][k]))
                    cl[pid]["R0"].append(min(1.0, max(0.0, item["R0_reference"])))
                    cl[pid]["R5"].append(min(1.0, max(0.0, item["R5_cref_recomputed"])))
    pids = sorted(cl)
    Y = [np.array(cl[p]["y"]) for p in pids]
    P0 = [np.array(cl[p]["R0"]) for p in pids]
    P5 = [np.array(cl[p]["R5"]) for p in pids]
    groups = defaultdict(list)
    for i, p in enumerate(pids):
        groups[bench[p]].append(i)

    def stats(idx):
        y = np.concatenate([Y[i] for i in idx]); a = np.concatenate([P0[i] for i in idx]); b = np.concatenate([P5[i] for i in idx])
        return ece(b, y) - ece(a, y), brier(b, y) - brier(a, y), ece(a, y), ece(b, y), brier(a, y), brier(b, y)

    pt = stats(range(len(pids)))
    rng = np.random.default_rng(20260924)
    dE, dB = [], []
    for _ in range(B):
        idx = []
        for g in sorted(groups):
            gi = groups[g]
            idx.extend(gi[j] for j in rng.integers(0, len(gi), len(gi)))
        s = stats(idx)
        dE.append(s[0]); dB.append(s[1])
    out[model] = {"ece_R0": pt[2], "ece_R5": pt[3], "brier_R0": pt[4], "brier_R5": pt[5],
                  "ece_diff": pt[0], "ece_diff_ci95": [float(np.quantile(dE, .025)), float(np.quantile(dE, .975))],
                  "brier_diff": pt[1], "brier_diff_ci95": [float(np.quantile(dB, .025)), float(np.quantile(dB, .975))]}
    print(model, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in out[model].items()})
(ROOT / "results/analysis/calibration_paired.json").write_text(json.dumps(out, indent=1))
