#!/usr/bin/env python3
"""Analysis of score-only training in Gemma-4-12B (exploratory).

Reuses the metric code of analyze_confirm_a1.py with one sample set (seed 0) per problem. Reports, for the trained
model and for the untrained base model, the benchmark-weighted AUROC of every readout with a 95% cluster-bootstrap
interval (problems resampled within benchmark), voting accuracy at k = 8, the contrasts R0 - R1, R0 - R2 and
trained - base, the exact-token EOS swaps on MATH500, the token sweep on MATH500, and the cosine geometry of the
output-embedding rows involved. Writes results/analysis/gemma4_12b_replication.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_confirm_a1 as A  # noqa: E402
from gemma4_12b_common import BASE, BETA, MARKER, Q3, R1_TOKENS, ROOT, read_jsonl  # noqa: E402

DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench", "aime_2026", "hmmt_feb_2026"]
CONDS = ["R0_reference", "R1_marker_swap_mean", "R2_position_eos", "R3_position_answer", "R4_cref_none",
         "R5_cref_recomputed", "BASE_R0"]
N_BOOT = 2000


def load():
    clusters = []
    for ds in DATASETS:
        for r in read_jsonl(Q3 / f"readout/{ds}.jsonl"):
            clusters.append({"dataset": ds, "problem_id": str(r["problem_id"]), "rows": [r]})
    return clusters


def primary(clusters, cond):
    return A.primary_auroc_details(clusters, cond)[0]


def summarize(clusters, rng_seed=0):
    obs = {c: A.metric(clusters, c) for c in CONDS}
    contrasts = {
        "R0_minus_R1": lambda cl: primary(cl, "R0_reference") - primary(cl, "R1_marker_swap_mean"),
        "R0_minus_R2": lambda cl: primary(cl, "R0_reference") - primary(cl, "R2_position_eos"),
        "trained_minus_base": lambda cl: primary(cl, "R0_reference") - primary(cl, "BASE_R0"),
    }
    con_obs = {k: f(clusters) for k, f in contrasts.items()}
    rng = np.random.default_rng(rng_seed)
    draws = {c: [] for c in CONDS}
    vote = {c: [] for c in ["R0_reference", "BASE_R0"]}
    cdraw = {k: [] for k in contrasts}
    for _ in range(N_BOOT):
        s = A.stratified_resample(clusters, rng)
        p = {c: primary(s, c) for c in CONDS}
        for c in CONDS:
            draws[c].append(p[c])
        cdraw["R0_minus_R1"].append(p["R0_reference"] - p["R1_marker_swap_mean"])
        cdraw["R0_minus_R2"].append(p["R0_reference"] - p["R2_position_eos"])
        cdraw["trained_minus_base"].append(p["R0_reference"] - p["BASE_R0"])
        for c in vote:
            vote[c].append(np.nanmean([A.exp_weighted_majority(x, c) for x in s]))
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    out = {}
    for c in CONDS:
        m = obs[c]
        out[c] = {"primary_auroc": m["primary_auroc"], "primary_auroc_ci95": ci(draws[c]),
                  "by_benchmark": {k: v["auroc"] for k, v in m["primary_auroc_by_benchmark"].items()},
                  "exp_weighted_majority_k8": m["exp_weighted_majority_accuracy_k8"],
                  "unweighted_majority_k8": m["unweighted_majority_accuracy_k8"],
                  "ece_15": m["ece_15"], "brier": m["brier"],
                  "partial_corr_length": m["partial_correlation_score_length_given_correctness"],
                  "n_responses": m["n_responses"]}
        if c in vote:
            out[c]["exp_weighted_majority_k8_ci95"] = ci(vote[c])
    return out, {k: {"value": con_obs[k], "ci95": ci(cdraw[k])} for k in contrasts}


def pooled_auroc(y, s):
    return A.auroc(np.asarray(y, int), np.asarray(s, float))


def eos_and_sweep(clusters):
    meta = json.loads((Q3 / "readout_meta.json").read_text())
    rows = [x["rows"][0] for x in clusters if x["dataset"] == "math500"]
    y, swap, tgt, tgt_b = [], {}, {}, {}
    for r in rows:
        for i, lp in enumerate(r["readout_logprobs"]):
            y.append(int(bool(r["correctness"][i])))
            swap.setdefault("eos", []).append(lp["R0_marker"])
            swap.setdefault("none", []).append(lp["R2_marker"])
            for n, v in lp["eos_swap"].items():
                swap.setdefault(n, []).append(v)
            for t, v in lp["P0_targets"].items():
                tgt.setdefault(t, []).append(v)
            for t, v in lp["P0_targets_base"].items():
                tgt_b.setdefault(t, []).append(v)
    eos = {n: {"auroc": pooled_auroc(y, v), "median_logprob": float(np.median(v))} for n, v in swap.items()}
    names = {str(MARKER): "marker"}
    names.update({str(t): f"R1 token {t}" for t in R1_TOKENS})
    names.update({str(v[0]): f"low-norm ordinary token {w!r}" for w, v in meta["words_under"].items()})
    names.update({str(v[0]): f"trained word {w!r}" for w, v in meta["words_trained"].items()})
    sweep = {t: {"name": names.get(t, t), "auroc_trained": pooled_auroc(y, tgt[t]),
                 "auroc_base": pooled_auroc(y, tgt_b[t]),
                 "median_logprob_trained": float(np.median(tgt[t])),
                 "median_shift_trained_minus_base": float(np.median(np.asarray(tgt[t]) - np.asarray(tgt_b[t]))),
                 "corr_with_marker_trained": float(np.corrcoef(tgt[t], tgt[str(MARKER)])[0, 1])}
             for t in tgt}
    return {"n": len(y), "swaps": meta["swaps"], "eos_swap": eos, "token_sweep": sweep}


def unembedding_geometry(token_ids):
    import torch
    from safetensors import safe_open
    idx = json.loads((Path(BASE) / "model.safetensors.index.json").read_text())["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in idx else "model.language_model.embed_tokens.weight"
    with safe_open(str(Path(BASE) / idx[key]), framework="pt") as f:
        w = f.get_slice(key)
        rows = torch.stack([w[t:t + 1][0].float() for t in token_ids])
    rows = torch.nn.functional.normalize(rows, dim=-1)
    cos = (rows @ rows.T).tolist()
    return {"weight": key, "token_ids": token_ids, "cosine": cos}


def main():
    clusters = load()
    res, con = summarize(clusters)
    train_meta = json.loads((Q3 / "model/train_meta.json").read_text())
    rd = json.loads((Q3 / "readout_meta.json").read_text())
    sweep_ids = [int(t) for t in rd["sweep_targets"]]
    out = {"model": "Gemma-4-12B + LoRA on LaSeR's self-reward term", "beta": BETA, "marker": MARKER,
           "reference_constant": train_meta["reference_constant"],
           "train": {k: train_meta[k] for k in ["train_solutions", "val_solutions", "val_auroc_before",
                                                 "val_auroc_after", "steps", "lr", "rank"]},
           "n_problems": len(clusters), "readouts": res, "contrasts": con, "math500": eos_and_sweep(clusters),
           "unembedding": unembedding_geometry(sweep_ids), "bootstrap": N_BOOT,
           "accuracy": {ds: float(np.mean([np.mean(c["rows"][0]["correctness"]) for c in clusters
                                           if c["dataset"] == ds])) for ds in DATASETS}}
    path = Path(os.environ.get("Q3_OUT", str(ROOT / "results/analysis/gemma4_12b_replication.json")))
    path.write_text(json.dumps(out, indent=1))
    for c in CONDS:
        r = res[c]
        print(f"{c:22s} AUROC {r['primary_auroc']:.3f} [{r['primary_auroc_ci95'][0]:.3f}, {r['primary_auroc_ci95'][1]:.3f}]"
              f"  vote {r['exp_weighted_majority_k8']:.3f}")
    print(json.dumps(con, indent=1))
    print(json.dumps(out["math500"]["eos_swap"], indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
