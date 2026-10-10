#!/usr/bin/env python3
"""Exploratory, CPU only: does R0 discriminate correctness beyond length?

For each checkpoint, on all confirmatory solutions (three seeds, clusters = problems):
  - length-only baseline: AUROC of -length, benchmark-weighted and within problem;
  - R0 within problem, with the number of eligible problems (both correct and wrong solutions);
  - length-matched pairs: within a problem, every (correct, wrong) pair whose token lengths differ by at most 10%;
    the share of pairs in which the correct solution scores higher (ties count one half);
  - R0 after removing log length: a linear fit of R0 on log length in one half of the problems is applied to the
    other half (two folds by problem), and the AUROC of the residual is benchmark-weighted;
  - voting gain of R0 over majority voting (linear and exponential weights), per problem.
Intervals: 2,000 cluster-bootstrap resamples of problems within benchmark (seed 20261008).
Writes results/analysis/length_control.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_confirm_a1 as A  # noqa: E402

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
MODELS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
N_BOOT = 2000


def auc(y, s):
    return A.auroc(np.asarray(y, int), np.asarray(s, float))


def flat(cluster, cond="R0_reference"):
    s, y, n = [], [], []
    for row in cluster["rows"]:
        for i, item in enumerate(row["readout_scores"]):
            s.append(float(item[cond])); y.append(int(bool(row["correctness"][i])))
            n.append(int(row["response_token_counts"][i]))
    return np.array(s), np.array(y), np.array(n, float)


def per_cluster(c):
    s, y, n = flat(c)
    out = {"ds": c["dataset"]}
    if 0 < y.sum() < len(y):
        out["wp_r0"] = auc(y, s)
        out["wp_len"] = auc(y, -n)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    conc, tot = 0.0, 0
    for i in pos:
        for j in neg:
            if abs(np.log(n[i] / n[j])) <= np.log(1.1):
                tot += 1
                conc += 1.0 if s[i] > s[j] else (0.5 if s[i] == s[j] else 0.0)
    out["pairs"], out["conc"] = tot, conc
    out["vote_lin"] = A.linear_weighted_majority(c, "R0_reference") - A.unweighted_majority(c, "R0_reference")
    out["vote_exp"] = A.exp_weighted_majority(c, "R0_reference") - A.unweighted_majority(c, "R0_reference")
    return out


def bench_weighted(records):
    """records: list of (ds, y, score); AUROC within benchmark, weighted by the number of solutions."""
    tot, acc = 0, 0.0
    for ds in DATASETS:
        yy = np.concatenate([r[1] for r in records if r[0] == ds] or [np.array([])])
        ss = np.concatenate([r[2] for r in records if r[0] == ds] or [np.array([])])
        a = auc(yy, ss) if len(yy) else None
        if a is not None:
            acc += len(yy) * a; tot += len(yy)
    return acc / tot if tot else None


def residual_auroc(clusters):
    rng = np.random.default_rng(7)
    fold = {id(c): int(rng.integers(0, 2)) for c in clusters}
    recs = []
    for f in (0, 1):
        tr = [flat(c) for c in clusters if fold[id(c)] != f]
        x = np.log(np.concatenate([t[2] for t in tr])); s = np.concatenate([t[0] for t in tr])
        b, a = np.polyfit(x, s, 1)
        for c in clusters:
            if fold[id(c)] == f:
                s1, y1, n1 = flat(c)
                recs.append((c["dataset"], y1, s1 - (a + b * np.log(n1))))
    return bench_weighted(recs)


def summarize(pcs):
    wp = [p for p in pcs if "wp_r0" in p]
    pairs = sum(p["pairs"] for p in pcs)
    return {"within_problem_auroc_r0": float(np.mean([p["wp_r0"] for p in wp])),
            "within_problem_auroc_len": float(np.mean([p["wp_len"] for p in wp])),
            "n_eligible_problems": len(wp),
            "matched_pair_auroc": float(sum(p["conc"] for p in pcs) / pairs) if pairs else None,
            "n_matched_pairs": int(pairs),
            "vote_gain_linear": float(np.nanmean([p["vote_lin"] for p in pcs])),
            "vote_gain_exp": float(np.nanmean([p["vote_exp"] for p in pcs]))}


def main():
    out = {}
    for m in MODELS:
        paths = [str(ROOT / f"results/confirmatory/readout/{m}/{ds}_seed{s}.jsonl") for ds in DATASETS for s in (0, 1, 2)]
        clusters, _ = A.load_clusters(paths)
        pcs = [per_cluster(c) for c in clusters]
        obs = summarize(pcs)
        recs_len = [(c["dataset"], flat(c)[1], -flat(c)[2]) for c in clusters]
        recs_r0 = [(c["dataset"], flat(c)[1], flat(c)[0]) for c in clusters]
        obs["benchmark_auroc_len"] = bench_weighted(recs_len)
        obs["benchmark_auroc_r0"] = bench_weighted(recs_r0)
        obs["benchmark_auroc_r0_minus_loglen"] = residual_auroc(clusters)
        by_ds = {}
        for i, c in enumerate(clusters):
            by_ds.setdefault(c["dataset"], []).append(i)
        rng = np.random.default_rng(20261008)
        draws = []
        for _ in range(N_BOOT):
            idx = np.concatenate([rng.choice(v, len(v), replace=True) for v in by_ds.values()])
            draws.append(summarize([pcs[i] for i in idx]))
        ci = {k: [float(np.percentile([d[k] for d in draws], 2.5)), float(np.percentile([d[k] for d in draws], 97.5))]
              for k in ["within_problem_auroc_r0", "within_problem_auroc_len", "matched_pair_auroc",
                        "vote_gain_linear", "vote_gain_exp"]}
        out[m] = {"observed": obs, "ci95": ci, "n_problems": len(clusters)}
        print(m, json.dumps(obs), json.dumps(ci), flush=True)
    (ROOT / "results/analysis/length_control.json").write_text(json.dumps(out, indent=1))
    print("wrote length_control.json")


if __name__ == "__main__":
    main()
