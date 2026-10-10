#!/usr/bin/env python3
"""Exploratory, CPU only (not pre-registered): length and truncation checks for the LaSeR readouts.

Writes results/analysis/laser_extra.json. Reads the confirmatory readout files; never modifies them.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("READSWAP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]


def rankdata(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    # average ties
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def auroc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return None
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos - 1) / 2 - npos * 0) / (npos * nneg) - 0.0) if False else \
        float((r[labels].sum() + npos - npos * (npos + 1) / 2) / (npos * nneg))


def residualize(values, groups):
    values = np.asarray(values, float)
    out = values.copy()
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    for g, ii in idx.items():
        out[ii] = values[ii] - values[ii].mean()
    return out


def load(model):
    rows = []
    for ds in DATASETS:
        for seed in (0, 1, 2):
            path = f"{ROOT}/results/confirmatory/readout/{model}/{ds}_seed{seed}.jsonl"
            if not os.path.exists(path):
                return None
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = f"{ds}:{r['problem_id']}"
                for k in range(len(r["responses"])):
                    rows.append({
                        "pid": pid, "ds": ds, "seed": seed, "k": k,
                        "correct": bool(r["correctness"][k]),
                        "len": int(r["response_token_counts"][k]),
                        "trunc": bool(r["truncated"][k]),
                        "R0": float(r["readout_scores"][k]["R0_reference"]),
                        "R5": float(r["readout_scores"][k]["R5_cref_recomputed"]),
                        "tail": r["responses"][k][-240:],
                    })
    return rows


res = {}
for model in MODELS:
    rows = load(model)
    if rows is None:
        res[model] = {"status": "incomplete"}
        continue
    corr = np.array([r["correct"] for r in rows])
    length = np.array([r["len"] for r in rows], float)
    score = np.array([r["R0"] for r in rows])
    trunc = np.array([r["trunc"] for r in rows])
    pids = [r["pid"] for r in rows]

    out = {"n_responses": len(rows)}
    # 1. Global partial correlation given correctness (should match L3), all and without truncated responses.
    out["partial_all"] = pearson(residualize(score, corr), residualize(length, corr))
    keep = ~trunc
    out["partial_no_trunc"] = pearson(residualize(score[keep], corr[keep]), residualize(length[keep], corr[keep]))
    out["truncated"] = {"total": int(trunc.sum()), "correct": int((trunc & corr).sum()),
                        "incorrect": int((trunc & ~corr).sum())}
    # 2. Within problem and correctness: remove problem-by-correctness means (controls for difficulty).
    groups = [f"{p}|{c}" for p, c in zip(pids, corr)]
    out["partial_within_problem"] = pearson(residualize(score, groups), residualize(length, groups))
    out["partial_within_problem_no_trunc"] = pearson(
        residualize(score[keep], [g for g, k in zip(groups, keep) if k]),
        residualize(length[keep], [g for g, k in zip(groups, keep) if k]))
    # 3. Spearman among correct responses within each problem (>= 4 correct responses with distinct lengths).
    byp = defaultdict(list)
    for i, r in enumerate(rows):
        byp[r["pid"]].append(i)
    sp_c, sp_i = [], []
    for p, ii in byp.items():
        for want, acc in ((True, sp_c), (False, sp_i)):
            jj = [i for i in ii if corr[i] == want and not trunc[i]]
            if len(jj) >= 4 and len(set(length[jj])) > 1 and len(set(score[jj])) > 1:
                acc.append(pearson(rankdata(length[jj]), rankdata(score[jj])))
    out["spearman_within_problem_correct"] = {"n_problems": len(sp_c), "mean": float(np.mean(sp_c)),
                                              "share_negative": float(np.mean(np.array(sp_c) < 0))}
    out["spearman_within_problem_incorrect"] = {"n_problems": len(sp_i), "mean": float(np.mean(sp_i)),
                                                "share_negative": float(np.mean(np.array(sp_i) < 0))}
    # 4. Within-problem AUROC (problems with both correct and incorrect responses), mean over problems.
    aucs = []
    for p, ii in byp.items():
        a = auroc(score[ii], corr[ii])
        if a is not None:
            aucs.append(a)
    out["within_problem_auroc"] = {"n_problems": len(aucs), "mean": float(np.mean(aucs))}
    # 5. Pairs inside a problem: an incorrect response that is shorter than a correct one and outscores it.
    n_pairs = n_inv = n_pairs_short = n_inv_short = 0
    for p, ii in byp.items():
        cs = [i for i in ii if corr[i]]
        ws = [i for i in ii if not corr[i]]
        for c in cs:
            for w in ws:
                n_pairs += 1
                inv = score[w] > score[c]
                n_inv += inv
                if length[w] < length[c]:
                    n_pairs_short += 1
                    n_inv_short += inv
    out["pairs"] = {"pairs": n_pairs, "incorrect_scores_higher": n_inv / max(n_pairs, 1),
                    "pairs_incorrect_shorter": n_pairs_short,
                    "incorrect_shorter_scores_higher": n_inv_short / max(n_pairs_short, 1),
                    "incorrect_longer_scores_higher": (n_inv - n_inv_short) / max(n_pairs - n_pairs_short, 1)}
    # 6. Example: two correct, non-truncated responses to one problem, same seed, lengths differ >= 3x.
    best = None
    for p, ii in byp.items():
        cs = [i for i in ii if corr[i] and not trunc[i]]
        for a in cs:
            for b in cs:
                if rows[a]["seed"] == rows[b]["seed"] and length[b] >= 3 * length[a] and length[a] >= 150:
                    gap = score[a] - score[b]
                    if best is None or gap > best[0]:
                        best = (gap, a, b)
    if best:
        gap, a, b = best
        out["example"] = {"pid": rows[a]["pid"], "seed": rows[a]["seed"],
                          "short": {"k": rows[a]["k"], "len": int(length[a]), "R0": float(score[a]), "tail": rows[a]["tail"]},
                          "long": {"k": rows[b]["k"], "len": int(length[b]), "R0": float(score[b]), "tail": rows[b]["tail"]}}
    res[model] = out

secs = {}
for f in glob.glob(f"{ROOT}/logs/e2_*.seconds"):
    name = os.path.basename(f)[:-8]
    try:
        secs[name] = int(open(f).read().strip())
    except ValueError:
        pass
gpu = defaultdict(float)
for name, s in secs.items():
    for model in MODELS:
        if model in name:
            kind = "generation" if name.startswith("e2_gen_") else "readout"
            gpu[f"{model}|{kind}"] += s / 3600
res["gpu_hours_wallclock_incl_queue"] = dict(gpu)

os.makedirs(f"{ROOT}/results/analysis", exist_ok=True)
with open(f"{ROOT}/results/analysis/laser_extra.json", "w") as fh:
    json.dump(res, fh, indent=1)
for m, v in res.items():
    print("==", m)
    if isinstance(v, dict):
        for k, x in v.items():
            if k != "example":
                print("  ", k, x)
        if "example" in v:
            e = v["example"]
            print("   example", e["pid"], e["seed"], e["short"]["len"], round(e["short"]["R0"], 3), e["long"]["len"], round(e["long"]["R0"], 3))
