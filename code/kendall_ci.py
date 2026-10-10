#!/usr/bin/env python3
"""Recompute Kendall tau (each readout vs R0) and its interval with scipy on stratified problem resamples.

The vectorized bootstrap in analyze_confirm_a1.py produced Kendall draws that do not bracket the scipy point
estimate. This script recomputes the interval directly: 2,000 resamples of problems within benchmark
(generator seed 20260924), Kendall tau-b over all solutions of the resampled problems.
It replaces observed["kendall_tau_vs_R0_ci95"] in each *_confirm.json (backup kept), sets the unreported
per-benchmark Kendall intervals to null, records the method, and rewrites the sha256 sidecar.
Run it after analyze_confirm.py; the released *_confirm.json files already contain its output.
"""
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
CKPTS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
CONDS = ["R0_reference", "R1_marker_swap_mean", "R2_position_eos", "R3_position_answer", "R4_cref_none", "R5_cref_recomputed"]
B = 2000
summary = {}
for model in CKPTS:
    clusters = defaultdict(lambda: {c: [] for c in CONDS})
    bench_of = {}
    for ds in DATASETS:
        for seed in (0, 1, 2):
            for line in open(ROOT / f"results/confirmatory/readout/{model}/{ds}_seed{seed}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                pid = f"{ds}:{r['problem_id']}"
                bench_of[pid] = ds
                for item in r["readout_scores"]:
                    for c in CONDS:
                        v = item.get(c)
                        clusters[pid][c].append(np.nan if v is None else float(v))
    pids = sorted(clusters)
    arr = {c: [np.array(clusters[p][c]) for p in pids] for c in CONDS}
    by_bench = defaultdict(list)
    for i, p in enumerate(pids):
        by_bench[bench_of[p]].append(i)

    def tau(idx, cond):
        x = np.concatenate([arr["R0_reference"][i] for i in idx])
        y = np.concatenate([arr[cond][i] for i in idx])
        m = ~(np.isnan(x) | np.isnan(y))
        return float(kendalltau(x[m], y[m]).statistic)

    allidx = list(range(len(pids)))
    point = {c: tau(allidx, c) for c in CONDS}
    rng = np.random.default_rng(20260924)
    draws = {c: [] for c in CONDS}
    for _ in range(B):
        idx = []
        for b in sorted(by_bench):
            g = by_bench[b]
            idx.extend(g[j] for j in rng.integers(0, len(g), len(g)))
        for c in CONDS:
            draws[c].append(tau(idx, c))
    ci = {c: [float(np.quantile(draws[c], 0.025)), float(np.quantile(draws[c], 0.975))] for c in CONDS}
    summary[model] = {"point": point, "ci95": ci}
    p = ROOT / f"results/analysis/{model}_confirm.json"
    d = json.loads(p.read_text())
    shutil.copy(p, str(p) + ".bak_kendall")
    old_point = d["metrics"]["kendall_tau_vs_R0"]
    d["metrics"]["kendall_tau_vs_R0_ci95"] = ci
    d["metrics"]["kendall_ci_method"] = (f"scipy kendalltau over all solutions of {B} stratified problem resamples, "
                                         "generator seed 20260924 (code/kendall_ci.py, 2026-10-08); replaces the "
                                         "vectorized draws, whose intervals did not bracket the point estimate")
    for bname, bd in d.get("by_benchmark", {}).items():
        if "kendall_tau_vs_R0_ci95" in bd:
            bd["kendall_tau_vs_R0_ci95"] = None
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    Path(str(p) + ".sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    for c in CONDS:
        lo, hi = ci[c]
        flag = "" if lo <= point[c] <= hi else "  OUTSIDE"
        print(model[:10], c[:14], "json", round(old_point[c], 4), "scipy", round(point[c], 4), "ci", [round(lo, 4), round(hi, 4)], flag)
(ROOT / "results/analysis/kendall_ci.json").write_text(json.dumps(summary, indent=1))
