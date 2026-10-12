#!/usr/bin/env python3
"""Exploratory analyses beyond the pre-specified runs, recomputed from the per-solution readouts.

Part 1, new problems: the three LaSeR checkpoints on AIME 2026 and HMMT February 2026 (eight solutions, seeds 0-2).
Part 2, new models: the self-score term trained alone into Qwen3-4B (three seeds), Qwen3.5-4B and Gemma-4-12B
(seed 0), each read on the five benchmarks and on the two 2026 sets.

Every metric comes from analyze_confirm_a1.py unchanged. Length-only is the AUROC of the negative number of
generated tokens. Intervals are 95% percentile intervals over problems resampled within each benchmark; paired
differences use the same resamples for both sides. Writes results/analysis/newdata.json.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

import analyze_confirm_a1 as A
from common import read_jsonl

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
NEW_SETS = ["aime_2026", "hmmt_feb_2026"]
BENCHMARKS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
CHECKPOINTS = ["OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR"]
TRAINED = {
    "Qwen3-4B seed 0": "results/q3/readout",
    "Qwen3-4B seed 1": "results/q3_s1/readout",
    "Qwen3-4B seed 2": "results/q3_s2/readout",
    "Qwen3.5-4B": "results/qwen35_4b/readout",
    "Gemma-4-12B": "results/gemma4_12b/readout",
}
READOUTS = ["R0_reference", "R1_marker_swap_mean", "R2_position_eos", "R3_position_answer", "R4_cref_none",
            "R5_cref_recomputed"]
PAIRS = [("R0_reference", "R1_marker_swap_mean"), ("R0_reference", "R2_position_eos"),
         ("R0_reference", "length_only")]
SEED = 20260924


def add_length_score(clusters):
    """Copy the clusters and store the negative token count of each solution as the readout 'length_only'."""
    out = copy.deepcopy(clusters)
    for cluster in out:
        for row in cluster["rows"]:
            for item, n in zip(row["readout_scores"], row["response_token_counts"]):
                item["length_only"] = -float(n)
    return out


def within_problem(clusters, condition):
    """Mean AUROC among the solutions of each problem that has both correct and wrong ones."""
    values = []
    for cluster in clusters:
        score, correct, _ = A.arrays([cluster], condition)
        value = A.auroc(correct, score)
        if value is not None:
            values.append(value)
    return {"auroc": float(np.mean(values)) if values else None, "eligible_problems": len(values)}


def summarize(clusters, conditions, n_boot):
    """Point metrics with intervals for each condition, paired differences and within-problem AUROC."""
    clusters = add_length_score(clusters)
    rng = np.random.default_rng(SEED)
    samples = A.stratified_bootstrap_samples(clusters, rng, n_boot)
    out, draws = {"readouts": {}, "paired": {}, "within_problem": {}}, {}
    for cond in conditions + ["length_only"]:
        point = A.metric(clusters, cond)
        draws[cond] = A._vectorized_metric_draws(clusters, cond, samples)
        A._add_metric_intervals(point, draws[cond])
        out["readouts"][cond] = point
    for a, b in PAIRS:
        diff = [x.get("primary_auroc") - y.get("primary_auroc") for x, y in zip(draws[a], draws[b])
                if x.get("primary_auroc") is not None and y.get("primary_auroc") is not None]
        out["paired"][f"{a} - {b}"] = {
            "value": out["readouts"][a]["primary_auroc"] - out["readouts"][b]["primary_auroc"],
            "ci95": A._ci95(diff), "valid_draws": len(diff)}
    for cond in ["R0_reference", "length_only"]:
        out["within_problem"][cond] = within_problem(clusters, cond)
    out["n_problems"] = len(clusters)
    out["n_solutions"] = sum(len(r["correctness"]) for c in clusters for r in c["rows"])
    out["accuracy"] = float(np.mean([x for c in clusters for r in c["rows"] for x in r["correctness"]]))
    return out


def laser_new_problems(n_boot):
    result = {}
    for ckpt in CHECKPOINTS:
        base = ROOT / "results/new_problems/readout" / ckpt
        paths = [str(base / f"{ds}_seed{s}.jsonl") for ds in NEW_SETS for s in (0, 1, 2)]
        clusters, meta = A.load_clusters(paths)
        assert not meta["excluded_problems"], meta["excluded_problems"]
        entry = {"pooled": summarize(clusters, READOUTS, n_boot), "by_dataset": {}}
        for ds in NEW_SETS:
            entry["by_dataset"][ds] = summarize([c for c in clusters if c["dataset"] == ds], READOUTS, n_boot)
        result[ckpt] = entry
        print(ckpt, "R0", round(entry["pooled"]["readouts"]["R0_reference"]["primary_auroc"], 3), flush=True)
    return result


def load_single_seed(directory, datasets):
    clusters = []
    for ds in datasets:
        path = ROOT / directory / f"{ds}.jsonl"
        if not path.exists():
            return None
        for row in read_jsonl(path):
            clusters.append({"dataset": ds, "problem_id": str(row["problem_id"]), "rows": [row]})
    return clusters


def trained_models(n_boot, names=None):
    result = {}
    for name, directory in TRAINED.items():
        if names and name not in names:
            continue
        entry = {}
        for label, datasets in [("benchmarks", BENCHMARKS), ("new_problems", NEW_SETS)]:
            clusters = load_single_seed(directory, datasets)
            if clusters is not None:
                entry[label] = summarize(clusters, ["BASE_R0"] + READOUTS, n_boot)
        result[name] = entry
        print(name, {k: round(v["readouts"]["R0_reference"]["primary_auroc"], 3) for k, v in entry.items()}, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--output", default=str(ROOT / "results/analysis/newdata.json"))
    ap.add_argument("--part", choices=["all", "laser", "trained", "merge"], default="all",
                    help="run one part, or one trained model with --model, to spread the work over processes")
    ap.add_argument("--model", action="append", help="trained model name, as in TRAINED")
    ap.add_argument("--parts", nargs="*", default=[], help="partial outputs to merge with --part merge")
    args = ap.parse_args()
    out = {"exploratory": True, "bootstrap": args.bootstrap, "bootstrap_seed": SEED,
           "laser_new_problems": {}, "trained_models": {}}
    if args.part == "merge":
        for path in args.parts:
            part = json.loads(Path(path).read_text())
            out["laser_new_problems"].update(part["laser_new_problems"])
            out["trained_models"].update(part["trained_models"])
    if args.part in ("all", "laser"):
        out["laser_new_problems"] = laser_new_problems(args.bootstrap)
    if args.part in ("all", "trained"):
        out["trained_models"] = trained_models(args.bootstrap, args.model)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
