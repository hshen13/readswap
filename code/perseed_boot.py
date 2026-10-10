#!/usr/bin/env python3
"""Per-seed problem-bootstrap intervals for every metric of the plan (exploratory reporting).

For one checkpoint, each seed and each readout: primary AUROC, pooled AUROC, ECE, Brier, linear and exponential
weighted voting and partial correlation with length, with 95% intervals from 200 problem resamples within benchmark
(analyze_confirm_a1.metric on one-seed clusters). Writes results/analysis/perseed_boot_<model>.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_confirm_a1 as A  # noqa: E402

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
READ = ["R0_reference", "R1_marker_swap_mean", "R2_position_eos", "R3_position_answer", "R4_cref_none", "R5_cref_recomputed"]
KEYS = ["primary_auroc", "auroc_pooled", "ece_15", "brier", "weighted_majority_accuracy_k8",
        "exp_weighted_majority_accuracy_k8", "partial_correlation_score_length_given_correctness"]
N_BOOT = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    out = {"model": args.model, "bootstrap": N_BOOT, "seeds": {}}
    for seed in (0, 1, 2):
        clusters = []
        for d in DATASETS:
            for line in open(ROOT / f"results/confirmatory/readout/{args.model}/{d}_seed{seed}.jsonl"):
                r = json.loads(line)
                clusters.append({"dataset": d, "problem_id": str(r["problem_id"]), "rows": [r]})
        rng = np.random.default_rng(20261009 + seed)
        draws = {r: {k: [] for k in KEYS} for r in READ}
        for _ in range(N_BOOT):
            s = A.stratified_resample(clusters, rng)
            for r in READ:
                m = A.metric(s, r)
                for k in KEYS:
                    draws[r][k].append(np.nan if m.get(k) is None else m[k])
        res = {}
        for r in READ:
            res[r] = {k: [float(np.nanpercentile(draws[r][k], 2.5)), float(np.nanpercentile(draws[r][k], 97.5))] for k in KEYS}
        out["seeds"][str(seed)] = res
        print(args.model, "seed", seed, "done", flush=True)
    (ROOT / f"results/analysis/perseed_boot_{args.model}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
