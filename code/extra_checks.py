#!/usr/bin/env python3
"""Additional CPU analyses for the LaSeR readouts (exploratory).

1. Paired problem-bootstrap intervals (1,000 resamples within benchmark, three seeds pooled per problem):
   AUROC of R0 minus length only, R0 minus R0 without log length (cross-fitted), R0 minus R5, and R0 minus R3 on the
   solutions with a box.
2. Truncation sensitivity: the same AUROCs and L2 on the solutions that reached the end-of-sequence token, and the
   end-of-sequence control and the two log-probability terms on the seed-zero solutions without truncation.
3. Duplicate audit of the Qwen3-4B training problems against the five benchmarks: exact match after normalization and
   near matches by word 8-gram Jaccard similarity of at least 0.5.
Writes results/analysis/extra_checks.json.
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
A = ROOT / "results/analysis"
MODELS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
N_BOOT = 1000


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = s.argsort(kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    vals, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals)); np.add.at(sums, inv, r); r = (sums / cnt)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wauc(y, s, ds, idx):
    tot, acc = 0, 0.0
    for d in range(len(DATASETS)):
        ii = idx[ds[idx] == d]
        ii = ii[~np.isnan(s[ii])]
        if len(ii) and 0 < y[ii].sum() < len(ii):
            acc += len(ii) * auroc(y[ii], s[ii]); tot += len(ii)
    return acc / tot if tot else np.nan


def load(model):
    recs = []
    for d in DATASETS:
        for seed in (0, 1, 2):
            for line in open(ROOT / f"results/confirmatory/readout/{model}/{d}_seed{seed}.jsonl"):
                r = json.loads(line)
                for k, sc in enumerate(r["readout_scores"]):
                    recs.append({"ds": DATASETS.index(d), "pid": f"{d}:{r['problem_id']}", "seed": seed, "k": k,
                                 "y": bool(r["correctness"][k]), "len": float(r["response_token_counts"][k]),
                                 "trunc": bool(r["truncated"][k]), "R0": sc["R0_reference"], "R2": sc["R2_position_eos"],
                                 "R3": sc.get("R3_position_answer"), "R5": sc["R5_cref_recomputed"]})
    return recs


def cross_fit_resid(s, length, pid):
    rng = np.random.default_rng(7)
    ups = sorted(set(pid)); fold = dict(zip(ups, rng.integers(0, 2, len(ups))))
    f = np.array([fold[p] for p in pid]); out = np.empty(len(s)); x = np.log(length)
    for k in (0, 1):
        tr, te = f != k, f == k
        b, a = np.polyfit(x[tr], s[tr], 1)
        out[te] = s[te] - (a + b * x[te])
    return out


def boot(y, ds, pid, scores, pairs, mask=None):
    """scores: name -> array; pairs: list of (name_a, name_b). Returns point and CI of AUROCs and paired differences."""
    idx_all = np.arange(len(y)) if mask is None else np.nonzero(mask)[0]
    by = defaultdict(list)
    for i in idx_all:
        by[(ds[i], pid[i])].append(i)
    clusters = defaultdict(list)
    for (d, p), ii in by.items():
        clusters[d].append(np.array(ii))
    pt = {n: wauc(y, s, ds, idx_all) for n, s in scores.items()}
    rng = np.random.default_rng(20261009)
    draws = defaultdict(list)
    for _ in range(N_BOOT):
        idx = np.concatenate([cl[i] for d, cl in clusters.items() for i in rng.integers(0, len(cl), len(cl))])
        v = {n: wauc(y, s, ds, idx) for n, s in scores.items()}
        for n in scores:
            draws[n].append(v[n])
        for a, b in pairs:
            draws[a + "-" + b].append(v[a] - v[b])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    out = {n: {"auroc": pt[n], "ci95": ci(draws[n])} for n in scores}
    for a, b in pairs:
        out[a + "-" + b] = {"diff": pt[a] - pt[b], "ci95": ci(draws[a + "-" + b])}
    return out


def main():
    res = {"bootstrap": N_BOOT}
    for m in MODELS:
        R = load(m)
        y = np.array([r["y"] for r in R]); ds = np.array([r["ds"] for r in R]); pid = [r["pid"] for r in R]
        length = np.array([r["len"] for r in R]); trunc = np.array([r["trunc"] for r in R])
        S = {k: np.array([np.nan if r[k] is None else r[k] for r in R], float) for k in ["R0", "R2", "R3", "R5"]}
        S["len_only"] = -length
        S["R0_wo_len"] = cross_fit_resid(S["R0"], length, pid)
        boxed = ~np.isnan(S["R3"])
        S["R0_boxed"] = np.where(boxed, S["R0"], np.nan)
        main_pairs = [("R0", "len_only"), ("R0", "R0_wo_len"), ("R0", "R5"), ("R0", "R2"), ("R0_boxed", "R3")]
        res[m] = {"all": boot(y, ds, pid, S, main_pairs),
                  "untruncated": boot(y, ds, pid, {k: S[k] for k in ["R0", "R2", "R5", "len_only"]},
                                      [("R0", "R2"), ("R0", "len_only")], mask=~trunc),
                  "n_truncated": int(trunc.sum()), "n": int(len(y))}
        # seed-zero mechanism control without truncated solutions
        tr0 = {(r["pid"], r["k"]): r["trunc"] for r in R if r["seed"] == 0}
        mech = [json.loads(l) for l in open(A / f"mech/{m}.jsonl")]
        keep = [x for x in mech if not tr0[(f"{x['ds']}:{x['problem_id']}", x["k"])]]
        ym = np.array([x["correct"] for x in keep]); dm = np.array([DATASETS.index(x["ds"]) for x in keep])
        ii = np.arange(len(keep))
        res[m]["mech_untruncated"] = {
            "n": len(keep),
            **{k: wauc(ym, np.array([x[k] for x in keep]), dm, ii)
               for k in ["eos", "newline", "period", "chat_special", "rare_word", "none"]},
            "logit_marker": wauc(ym, np.array([x["eos_logit_marker"] for x in keep]), dm, ii),
            "neg_logz": wauc(ym, -np.array([x["eos_logz"] for x in keep]), dm, ii),
            "base_logit_marker": wauc(ym, np.array([x["base_eos_logit_marker"] for x in keep]), dm, ii)}
        print(m, json.dumps({k: (round(v["auroc"], 3) if "auroc" in v else round(v["diff"], 3)) for k, v in res[m]["all"].items()}),
              json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[m]["mech_untruncated"].items()}), flush=True)

    # duplicate audit
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()  # noqa: E731
    grams = lambda t: {tuple(w[i:i + 8]) for w in [norm(t).split()] for i in range(max(1, len(w) - 7))}  # noqa: E731
    train = [json.loads(l)["problem"] for l in open(ROOT / "results/q3/data/train.jsonl")]
    evals = [(d, json.loads(l)["problem"]) for d in DATASETS for l in open(ROOT / f"data/confirmatory/{d}/test.jsonl")]
    tn = {norm(t) for t in train}
    exact = [(d, p[:80]) for d, p in evals if norm(p) in tn]
    index = defaultdict(set)
    tg = [grams(t) for t in train]
    for i, g in enumerate(tg):
        for x in g:
            index[x].add(i)
    near = []
    for d, p in evals:
        g = grams(p)
        cand = set().union(*(index.get(x, set()) for x in g)) if g else set()
        for i in cand:
            j = len(g & tg[i]) / max(1, len(g | tg[i]))
            if j >= 0.5:
                near.append({"dataset": d, "eval": p[:80], "train": train[i][:80], "jaccard": round(j, 3)})
    res["duplicates"] = {"n_train": len(train), "n_eval": len(evals), "exact": exact, "near_jaccard_ge_0.5": near}
    print("duplicates: exact", len(exact), "near", len(near))
    (A / "extra_checks.json").write_text(json.dumps(res, indent=1))
    print("wrote extra_checks.json")


if __name__ == "__main__":
    main()
