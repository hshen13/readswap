#!/usr/bin/env python3
"""CPU analysis of results/analysis/mech/<model>.jsonl (exploratory).

Seed-0 solutions of all five benchmarks. Benchmark-weighted AUROC (as the primary metric) with 95% cluster-bootstrap
intervals over problems within benchmark (1,000 resamples), for:
  - the marker read after the end-of-sequence token and after each replacement token, and with no token (R2 prefix);
  - paired differences EOS minus each replacement;
  - the two terms of the log-probability after EOS: the marker logit h^T w and the normalizer -log Z, in the released
    checkpoint and in the base model, and the training change of each term (checkpoint minus base).
Writes results/analysis/mech_summary.json.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
MODELS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
N_BOOT = 1000


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    order = s.argsort(kind="mergesort")
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    vals, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals)); np.add.at(sums, inv, r); r = (sums / cnt)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def weighted(rows, key):
    tot, acc = 0, 0.0
    for ds in DATASETS:
        rr = [r for r in rows if r["ds"] == ds]
        a = auroc([r["correct"] for r in rr], [key(r) for r in rr]) if rr else None
        if a is not None:
            acc += len(rr) * a; tot += len(rr)
    return acc / tot if tot else None


def main():
    keys = {
        "eos": lambda r: r["eos"], "newline": lambda r: r["newline"], "period": lambda r: r["period"],
        "chat_special": lambda r: r["chat_special"], "rare_word": lambda r: r["rare_word"], "none": lambda r: r["none"],
        "logit_marker": lambda r: r["eos_logit_marker"], "neg_logz": lambda r: -r["eos_logz"],
        "base_eos": lambda r: r["base_eos"], "base_logit_marker": lambda r: r["base_eos_logit_marker"],
        "base_neg_logz": lambda r: -r["base_eos_logz"],
        "change_logit_marker": lambda r: r["eos_logit_marker"] - r["base_eos_logit_marker"],
        "change_neg_logz": lambda r: -(r["eos_logz"] - r["base_eos_logz"]),
        "neg_length": lambda r: -r["length"],
    }
    out = {}
    names = list(keys)
    for m in MODELS:
        rows = [json.loads(l) for l in open(ROOT / f"results/analysis/mech/{m}.jsonl")]
        y = np.array([r["correct"] for r in rows], bool)
        S = np.array([[f(r) for f in keys.values()] for r in rows], float)
        dsi = np.array([DATASETS.index(r["ds"]) for r in rows])
        clusters = defaultdict(list)
        for i, r in enumerate(rows):
            clusters[(r["ds"], r["problem_id"])].append(i)
        by_ds = defaultdict(list)
        for k, v in clusters.items():
            by_ds[k[0]].append(np.array(v))

        def wauc(idx):
            res = np.zeros(len(names)); tot = 0
            for d in range(len(DATASETS)):
                ii = idx[dsi[idx] == d]
                if len(ii) == 0 or y[ii].all() or (~y[ii]).all():
                    continue
                res += len(ii) * np.array([auroc(y[ii], S[ii, j]) for j in range(len(names))]); tot += len(ii)
            return res / tot

        obs = dict(zip(names, wauc(np.arange(len(rows)))))
        rng = np.random.default_rng(20261009)
        D = []
        for _ in range(N_BOOT):
            idx = np.concatenate([v[i] for ds, vs in by_ds.items() for v in [vs] for i in rng.integers(0, len(vs), len(vs))])
            D.append(wauc(idx))
        D = np.array(D)
        draws = {k: D[:, j] for j, k in enumerate(names)}
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        res = {k: {"auroc": float(obs[k]), "ci95": ci(draws[k])} for k in names}
        for k in ["newline", "period", "chat_special", "rare_word", "none"]:
            res["eos_minus_" + k] = {"diff": float(obs["eos"] - obs[k]), "ci95": ci(draws["eos"] - draws[k])}
        lm = np.array([r["eos_logit_marker"] for r in rows]); lz = np.array([r["eos_logz"] for r in rows])
        blm = np.array([r["base_eos_logit_marker"] for r in rows]); blz = np.array([r["base_eos_logz"] for r in rows])
        res["terms"] = {
            "sd_logit_marker": float(lm.std()), "sd_logz": float(lz.std()),
            "corr_logit_logz": float(np.corrcoef(lm, lz)[0, 1]),
            "mean_change_logit_marker": float((lm - blm).mean()), "mean_change_logz": float((lz - blz).mean()),
            "sd_change_logit_marker": float((lm - blm).std()), "sd_change_logz": float((lz - blz).std()),
            "n": len(rows), "n_problems": len(clusters),
        }
        out[m] = res
        print(m, json.dumps({k: (round(v["auroc"], 3) if "auroc" in v else round(v["diff"], 3)) for k, v in res.items()
                             if k != "terms"}), json.dumps({k: round(v, 3) for k, v in res["terms"].items()}), flush=True)
    (ROOT / "results/analysis/mech_summary.json").write_text(json.dumps(out, indent=1))
    print("wrote mech_summary.json")


if __name__ == "__main__":
    main()
