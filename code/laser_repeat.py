#!/usr/bin/env python3
"""Exploratory: is the score-length association among correct solutions explained by repetitive (looping) text?

Repetitiveness = gzip compressed size / raw size of the response (lower = more repetitive).
Writes results/analysis/laser_repeat.json.
"""
import gzip
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("READSWAP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]


def ratio(text):
    raw = text.encode("utf-8")
    return len(gzip.compress(raw, 6)) / max(len(raw), 1)


def resid_groups(x, groups):
    x = np.asarray(x, float).copy()
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    for ii in idx.values():
        x[ii] -= x[ii].mean()
    return x


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def partial(y, x, covs):
    """Correlation of y and x after linear regression of both on covs (columns), plus intercept."""
    Z = np.column_stack([np.ones(len(y))] + covs)
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return corr(ry, rx)


res = {}
for model in MODELS:
    rows = []
    ok = True
    for ds in DATASETS:
        for seed in (0, 1, 2):
            path = f"{ROOT}/results/confirmatory/readout/{model}/{ds}_seed{seed}.jsonl"
            if not os.path.exists(path):
                ok = False
                break
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                for k, text in enumerate(r["responses"]):
                    rows.append((f"{ds}:{r['problem_id']}", bool(r["correctness"][k]), int(r["response_token_counts"][k]),
                                 bool(r["truncated"][k]), float(r["readout_scores"][k]["R0_reference"]), ratio(text)))
        if not ok:
            break
    if not ok:
        res[model] = {"status": "incomplete"}
        continue
    pid = [r[0] for r in rows]
    corr_ = np.array([r[1] for r in rows])
    ln = np.array([r[2] for r in rows], float)
    tr = np.array([r[3] for r in rows])
    sc = np.array([r[4] for r in rows])
    rr = np.array([r[5] for r in rows])
    keep = corr_ & ~tr
    g = [p for p, k in zip(pid, keep) if k]
    y = resid_groups(sc[keep], g)
    x = resid_groups(np.log(ln[keep]), g)
    c = resid_groups(rr[keep], g)
    out = {
        "n_correct_untruncated": int(keep.sum()),
        "within_problem_corr_score_loglen": corr(y, x),
        "within_problem_partial_score_loglen_given_ratio": partial(y, x, [c]),
        "within_problem_corr_score_ratio": corr(y, c),
        "within_problem_corr_loglen_ratio": corr(x, c),
    }
    # Least repetitive half (ratio above the median of correct, untruncated solutions).
    med = np.median(rr[keep])
    k2 = keep & (rr > med)
    g2 = [p for p, k in zip(pid, k2) if k]
    out["least_repetitive_half"] = {
        "n": int(k2.sum()),
        "within_problem_corr_score_loglen": corr(resid_groups(sc[k2], g2), resid_groups(np.log(ln[k2]), g2)),
    }
    res[model] = out
    print(model, json.dumps(out))

os.makedirs(f"{ROOT}/results/analysis", exist_ok=True)
json.dump(res, open(f"{ROOT}/results/analysis/laser_repeat.json", "w"), indent=1)
