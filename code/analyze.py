#!/usr/bin/env python3
"""Preregistered metrics and clustered-bootstrap contrasts for READSWAP (pilot)."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import read_jsonl, sigmoid_free_normalized, write_artifact_metadata, write_json


CONDITIONS = [
    "R0_reference",
    "R1_marker_swap_mean",
    "R2_position_eos",
    "R3_position_answer",
    "R4_cref_none",
    "R5_cref_recomputed",
]


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    positives = y == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration(y: np.ndarray, score: np.ndarray) -> tuple[float | None, float | None]:
    if not len(y):
        return None, None
    p = np.asarray(sigmoid_free_normalized(score.tolist()))
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    for lo in np.linspace(0, 1, 16)[:-1]:
        hi = lo + 1 / 15
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            ece += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece), brier


def kendall(x: np.ndarray, y: np.ndarray) -> float | None:
    try:
        from scipy.stats import kendalltau

        value = kendalltau(x, y, nan_policy="omit").statistic
        return None if np.isnan(value) else float(value)
    except Exception:
        return None


def partial_corr(score: np.ndarray, length: np.ndarray, correctness: np.ndarray) -> float | None:
    if len(score) < 4 or np.std(score) == 0 or np.std(length) == 0:
        return None
    x = np.column_stack([np.ones(len(correctness)), correctness])
    rx = score - x @ np.linalg.lstsq(x, score, rcond=None)[0]
    ry = length - x @ np.linalg.lstsq(x, length, rcond=None)[0]
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return None if denom == 0 else float(rx.dot(ry) / denom)


def weighted_majority(rows: list[dict[str, Any]], condition: str) -> float | None:
    correct, total = 0, 0
    for row in rows:
        scores = row["readout_scores"]
        answers = row.get("pred_answers", [None] * len(scores))
        by_answer: dict[str, float] = {}
        for ans, item in zip(answers, scores):
            if ans is None or item.get(condition) is None:
                continue
            key = str(ans)
            by_answer[key] = by_answer.get(key, 0.0) + max(float(item[condition]), 0.0)
        if not by_answer:
            continue
        chosen = max(by_answer, key=by_answer.get)
        expected = str(row.get("answer"))
        correct += int(chosen == expected or any(str(a) == chosen and c for a, c in zip(answers, row.get("correctness", []))))
        total += 1
    return None if total == 0 else correct / total


def flattened(rows: list[dict[str, Any]], condition: str):
    values = []
    for problem_idx, row in enumerate(rows):
        for response_idx, item in enumerate(row["readout_scores"]):
            value = item.get(condition)
            if value is None:
                continue
            values.append((problem_idx, response_idx, float(value), int(bool(row["correctness"][response_idx])), len(row["responses"][response_idx])))
    if not values:
        return (np.array([], dtype=int),) * 2 + (np.array([]),) * 3
    return tuple(np.asarray(x) for x in zip(*values))


def metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    problem, response, score, correct, length = flattened(rows, condition)
    del response
    ece, brier = calibration(correct, score)
    return {
        "n_responses": int(len(score)),
        "n_problems": int(len(np.unique(problem))) if len(problem) else 0,
        "auroc": auroc(correct, score),
        "ece_15": ece,
        "brier": brier,
        "weighted_majority_accuracy_k8": weighted_majority(rows, condition),
        "partial_correlation_score_length_given_correctness": partial_corr(score, length, correct),
    }


def kendall_vs_reference(rows: list[dict[str, Any]], condition: str) -> float | None:
    """Rank agreement with R0 on the same response instances."""
    reference = []
    candidate = []
    for row in rows:
        for item in row["readout_scores"]:
            x = item.get("R0_reference")
            y = item.get(condition)
            if x is not None and y is not None:
                reference.append(float(x))
                candidate.append(float(y))
    if not reference:
        return None
    return kendall(np.asarray(reference), np.asarray(candidate))


def contrast_values(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    m = {c: metrics(rows, c) for c in CONDITIONS}
    r0, r1, r2, r3, r4 = (m[c]["auroc"] for c in CONDITIONS[:5])
    out["L1_auroc_R0_minus_R1"] = None if r0 is None or r1 is None else r0 - r1
    out["L2_auroc_R0_minus_R2"] = None if r0 is None or r2 is None else r0 - r2
    out["L3_partial_corr_R0"] = m["R0_reference"]["partial_correlation_score_length_given_correctness"]
    w0 = m["R0_reference"]["weighted_majority_accuracy_k8"]
    w4 = m["R4_cref_none"]["weighted_majority_accuracy_k8"]
    out["L4_weighted_majority_R0_minus_R4"] = None if w0 is None or w4 is None else w0 - w4
    short0, long0, short3, long3 = [], [], [], []
    for row in rows:
        lengths = np.asarray([len(x) for x in row["responses"]])
        if not len(lengths):
            continue
        median = float(np.median(lengths))
        for idx, item in enumerate(row["readout_scores"]):
            label = int(bool(row["correctness"][idx]))
            if item.get("R0_reference") is not None:
                (short0 if lengths[idx] <= median else long0).append((label, item["R0_reference"]))
            if item.get("R3_position_answer") is not None:
                (short3 if lengths[idx] <= median else long3).append((label, item["R3_position_answer"]))
    def subgroup_gap(short, long):
        if not short or not long:
            return None
        a = auroc(np.asarray([x[0] for x in short]), np.asarray([x[1] for x in short]))
        b = auroc(np.asarray([x[0] for x in long]), np.asarray([x[1] for x in long]))
        return None if a is None or b is None else a - b
    gap0 = subgroup_gap(short0, long0)
    gap3 = subgroup_gap(short3, long3)
    out["L5_auroc_gap_short_long_R0_minus_R3"] = None if gap0 is None or gap3 is None else gap0 - gap3
    return out


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    usable = sorted(((k, p) for k, p in p_values.items() if p is not None), key=lambda x: x[1])
    result = {k: None for k in p_values}
    running = 0.0
    for rank, (key, p) in enumerate(usable):
        adjusted = min(1.0, (len(usable) - rank) * p)
        running = max(running, adjusted)
        result[key] = running
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    rows = read_jsonl(args.input)
    observed = {c: metrics(rows, c) for c in CONDITIONS}
    observed["kendall_tau_vs_R0"] = {
        c: kendall_vs_reference(rows, c) for c in CONDITIONS
    }
    observed_contrasts = contrast_values(rows)
    problem_ids = np.arange(len(rows))
    rng = np.random.default_rng(args.seed)
    draws = {key: [] for key, value in observed_contrasts.items() if value is not None}
    for _ in range(args.bootstrap):
        sample = rng.choice(problem_ids, size=len(problem_ids), replace=True)
        boot_rows = [rows[int(i)] for i in sample]
        for key, value in contrast_values(boot_rows).items():
            if key in draws and value is not None:
                draws[key].append(value)
    contrasts = {}
    p_values = {}
    for key, value in observed_contrasts.items():
        if value is None:
            contrasts[key] = {"estimate": None, "ci95": None, "p_two_sided": None}
            p_values[key] = None
            continue
        arr = np.asarray(draws[key])
        ci = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if len(arr) else None
        p = float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))) if len(arr) else None
        contrasts[key] = {"estimate": value, "ci95": ci, "p_two_sided": p}
        p_values[key] = p
    adjusted = holm(p_values)
    for key in contrasts:
        contrasts[key]["p_holm"] = adjusted.get(key)
        contrasts[key]["holds"] = bool(
            contrasts[key]["p_holm"] is not None
            and contrasts[key]["p_holm"] < 0.05
            and contrasts[key]["ci95"] is not None
            and contrasts[key]["ci95"][0] > 0
            or contrasts[key]["p_holm"] is not None
            and contrasts[key]["p_holm"] < 0.05
            and contrasts[key]["ci95"] is not None
            and contrasts[key]["ci95"][1] < 0
        )
    output = Path(args.output)
    result = {
        "input": str(Path(args.input)),
        "unit": "problem; responses clustered within problem",
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "ci": "95%"},
        "metrics": observed,
        "contrasts": contrasts,
    }
    write_json(output, result)
    output.with_name(output.name + ".sha256").write_text(
        f"{__import__('hashlib').sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
