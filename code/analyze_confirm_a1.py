#!/usr/bin/env python3
"""A1 analysis for the confirmatory LaSeR readouts.

The unit of resampling is a problem, stratified by benchmark.  R3 is a
defined readout only for responses containing a final boxed answer; the other
readouts determine complete-case exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import read_jsonl, write_json


CONDITIONS = [
    "R0_reference", "R1_marker_swap_mean", "R2_position_eos",
    "R3_position_answer", "R4_cref_none", "R5_cref_recomputed",
]
BASE_CONDITIONS = [c for c in CONDITIONS if c != "R3_position_answer"]
CONTRAST_NAMES = [
    "L1_auroc_R0_minus_R1", "L2_auroc_R0_minus_R2",
    "L3_partial_corr_R0", "L4_weighted_majority_R0_minus_R4",
    "L5_auroc_gap_short_long_R0_minus_R3",
]
EQUIVALENCE_BOUNDS = {
    "L1_auroc_R0_minus_R1": 0.02,
    "L2_auroc_R0_minus_R2": 0.02,
    "L3_partial_corr_R0": 0.05,
    "L4_weighted_majority_R0_minus_R4": 0.02,
    "L5_auroc_gap_short_long_R0_minus_R3": 0.02,
}


def avg_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        out[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return out


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    pos = y.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    ranks = avg_rank(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration(y: np.ndarray, score: np.ndarray) -> tuple[float | None, float | None]:
    """A1 calibration: score itself clipped to [0, 1], with 15 bins."""
    if not len(y):
        return None, None
    p = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for i in range(15):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < 14 else (p <= edges[i + 1]))
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece), brier


def partial_corr(score: np.ndarray, length: np.ndarray, correct: np.ndarray) -> float | None:
    if len(score) < 4 or np.std(score) == 0 or np.std(length) == 0:
        return None
    z = np.column_stack([np.ones(len(correct)), correct])
    rs = score - z @ np.linalg.lstsq(z, score, rcond=None)[0]
    rl = length - z @ np.linalg.lstsq(z, length, rcond=None)[0]
    denom = np.linalg.norm(rs) * np.linalg.norm(rl)
    return None if denom == 0 else float(rs.dot(rl) / denom)


def kendall(x: np.ndarray, y: np.ndarray) -> float | None:
    try:
        from scipy.stats import kendalltau
        value = kendalltau(x, y, nan_policy="omit").statistic
        return None if np.isnan(value) else float(value)
    except Exception:
        return None


def _r1_token_conditions(clusters: list[dict[str, Any]]) -> list[str]:
    tokens: set[str] = set()
    for cluster in clusters:
        for row in cluster["rows"]:
            for item in row.get("readout_scores", []):
                tokens.update(str(k) for k in item.get("R1_marker_swap", {}))
    return [f"R1_marker_swap_token_{token}" for token in sorted(tokens, key=lambda x: (int(x) if x.isdigit() else x))]


def score_value(item: dict[str, Any], condition: str) -> float | None:
    if condition.startswith("R1_marker_swap_token_"):
        token = condition.removeprefix("R1_marker_swap_token_")
        value = item.get("R1_marker_swap", {}).get(token)
    else:
        value = item.get(condition)
    return None if value is None else float(value)


def arrays(clusters: list[dict[str, Any]], condition: str) -> tuple[np.ndarray, ...]:
    scores, correct, lengths = [], [], []
    for cluster in clusters:
        for row in cluster["rows"]:
            for i, item in enumerate(row["readout_scores"]):
                value = score_value(item, condition)
                if value is None:
                    continue
                scores.append(value)
                correct.append(int(bool(row["correctness"][i])))
                lengths.append(int(row["response_token_counts"][i]))
    return np.asarray(scores, dtype=float), np.asarray(correct, dtype=int), np.asarray(lengths, dtype=float)


def char_arrays(clusters: list[dict[str, Any]], condition: str) -> tuple[np.ndarray, np.ndarray]:
    scores, lengths = [], []
    for cluster in clusters:
        for row in cluster["rows"]:
            for i, item in enumerate(row["readout_scores"]):
                value = score_value(item, condition)
                if value is not None:
                    scores.append(value)
                    lengths.append(len(row["responses"][i]))
    return np.asarray(scores, dtype=float), np.asarray(lengths, dtype=float)


def _records_primary(records: Iterable[tuple[str, int, float]]) -> tuple[float | None, dict[str, dict[str, Any]]]:
    by_benchmark: dict[str, list[tuple[int, float]]] = {}
    for benchmark, correct, score in records:
        by_benchmark.setdefault(str(benchmark), []).append((int(correct), float(score)))
    details: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0
    total = 0
    for benchmark in sorted(by_benchmark):
        values = by_benchmark[benchmark]
        y = np.asarray([x[0] for x in values], dtype=int)
        s = np.asarray([x[1] for x in values], dtype=float)
        value = auroc(y, s)
        details[benchmark] = {"n_responses": len(values), "auroc": value}
        if value is not None:
            weighted_sum += len(values) * value
            total += len(values)
    return (weighted_sum / total if total else None), details


def primary_auroc_details(clusters: list[dict[str, Any]], condition: str) -> tuple[float | None, dict[str, dict[str, Any]]]:
    records = []
    for cluster in clusters:
        for row in cluster["rows"]:
            for i, item in enumerate(row["readout_scores"]):
                value = score_value(item, condition)
                if value is not None:
                    records.append((cluster["dataset"], int(bool(row["correctness"][i])), value))
    return _records_primary(records)


def _parse_answer(answer: Any) -> str | None:
    return None if answer is None else str(answer)


@lru_cache(maxsize=200000)
def answers_equivalent(left: str, right: str) -> bool:
    """Use the same math_verify equivalence relation as the grader."""
    if left == right:
        return True
    try:
        from math_verify import parse, verify
        gold = parse(r"\boxed{" + left + "}")
        target = parse(r"\boxed{" + right + "}")
        return bool(verify(gold, target))
    except Exception:
        return False


def equivalence_classes(answers: list[Any]) -> list[int | None]:
    """Return first-occurrence class ids; None is not a vote class."""
    labels: list[int | None] = [None] * len(answers)
    representatives: list[str] = []
    for i, raw in enumerate(answers):
        answer = _parse_answer(raw)
        if answer is None:
            continue
        for class_id, representative in enumerate(representatives):
            if answers_equivalent(answer, representative):
                labels[i] = class_id
                break
        else:
            labels[i] = len(representatives)
            representatives.append(answer)
    return labels


@lru_cache(maxsize=20000)
def _cached_equivalence_classes(answers: tuple[Any, ...]) -> tuple[int | None, ...]:
    return tuple(equivalence_classes(list(answers)))


def _vote_row(row: dict[str, Any], condition: str, weighting: str) -> float | None:
    answers = row.get("pred_answers", [None] * len(row.get("readout_scores", [])))
    scores = row.get("readout_scores", [])
    labels = _cached_equivalence_classes(tuple(answers))
    eligible = [i for i, item in enumerate(scores) if i < len(labels) and labels[i] is not None and score_value(item, condition) is not None]
    if not eligible:
        return None
    if weighting == "exp":
        maximum = max(score_value(scores[i], condition) for i in eligible)
        weights = {i: math.exp(score_value(scores[i], condition) - maximum) for i in eligible}
    elif weighting == "linear":
        weights = {i: score_value(scores[i], condition) for i in eligible}
    elif weighting == "count":
        weights = {i: 1.0 for i in eligible}
    else:
        raise ValueError(f"unknown vote weighting: {weighting}")
    totals: dict[int, float] = {}
    first: dict[int, int] = {}
    counts: dict[int, int] = {}
    for i in eligible:
        label = int(labels[i])
        totals[label] = totals.get(label, 0.0) + float(weights[i])
        counts[label] = counts.get(label, 0) + 1
        first.setdefault(label, i)
    # Python's tuple comparison implements the A1 tie order.  The final
    # component is negative because an earlier response wins.
    chosen = max(totals, key=lambda label: (totals[label], counts[label], -first[label]))
    correctness = row.get("correctness", [])
    return float(any(i < len(correctness) and bool(correctness[i]) for i in eligible if labels[i] == chosen))


def vote_accuracy(cluster: dict[str, Any], condition: str, weighting: str) -> float | None:
    values = [_vote_row(row, condition, weighting) for row in cluster["rows"]]
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def exp_weighted_majority(cluster: dict[str, Any], condition: str) -> float:
    value = vote_accuracy(cluster, condition, "exp")
    return float("nan") if value is None else value


def linear_weighted_majority(cluster: dict[str, Any], condition: str) -> float:
    value = vote_accuracy(cluster, condition, "linear")
    return float("nan") if value is None else value


def unweighted_majority(cluster: dict[str, Any], condition: str) -> float:
    value = vote_accuracy(cluster, condition, "count")
    return float("nan") if value is None else value


def metric(clusters: list[dict[str, Any]], condition: str, ranges: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    score, correct, length = arrays(clusters, condition)
    ece, brier = calibration(correct, score)
    pooled = auroc(correct, score)
    primary, per_benchmark = primary_auroc_details(clusters, condition)
    linear_values = [linear_weighted_majority(c, condition) for c in clusters]
    exp_values = [exp_weighted_majority(c, condition) for c in clusters]
    unweighted_values = [unweighted_majority(c, condition) for c in clusters]
    mean_or_none = lambda values: float(np.mean([x for x in values if not math.isnan(x)])) if any(not math.isnan(x) for x in values) else None
    char_score, char_length = char_arrays(clusters, condition)
    return {
        "n_responses": int(len(score)), "n_problems": int(len(clusters)),
        "auroc": pooled, "auroc_pooled": pooled, "primary_auroc": primary,
        "primary_auroc_by_benchmark": per_benchmark,
        "ece_15": ece, "brier": brier,
        "weighted_majority_accuracy_k8": mean_or_none(linear_values),
        "exp_weighted_majority_accuracy_k8": mean_or_none(exp_values),
        "unweighted_majority_accuracy_k8": mean_or_none(unweighted_values),
        "partial_correlation_score_length_given_correctness": partial_corr(score, length, correct),
        "partial_correlation_score_characters_given_correctness": partial_corr(char_score, char_length, correct) if len(char_score) == len(correct) else None,
    }


def metric_arrays(score: np.ndarray, correct: np.ndarray, length: np.ndarray, ranges: dict[str, tuple[float, float]] | None, condition: str) -> dict[str, Any]:
    ece, brier = calibration(correct, score)
    value = auroc(correct, score)
    return {"n_responses": int(len(score)), "auroc": value, "auroc_pooled": value,
            "primary_auroc": value, "ece_15": ece, "brier": brier,
            "partial_correlation_score_length_given_correctness": partial_corr(score, length, correct)}


def length_half_metrics(clusters: list[dict[str, Any]], condition: str, ranges: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    values = {"short": [[], [], []], "long": [[], [], []]}
    for cluster in clusters:
        lengths = np.asarray([n for row in cluster["rows"] for n in row["response_token_counts"]])
        median = float(np.median(lengths))
        for row in cluster["rows"]:
            for i, item in enumerate(row["readout_scores"]):
                value = score_value(item, condition)
                if value is None:
                    continue
                side = "short" if row["response_token_counts"][i] <= median else "long"
                values[side][0].append(value)
                values[side][1].append(int(bool(row["correctness"][i])))
                values[side][2].append(int(row["response_token_counts"][i]))
    return {side: metric_arrays(np.asarray(score), np.asarray(correct, dtype=int), np.asarray(length, dtype=float), ranges, condition)
            for side, (score, correct, length) in values.items()}


def length_score_series(clusters: list[dict[str, Any]], condition: str) -> dict[str, list[float | None]]:
    score, correct, length = arrays(clusters, condition)
    if not len(score):
        return {"length_decile": list(range(1, 11)), "correct": [None] * 10, "incorrect": [None] * 10}
    z = (score - score.mean()) / (score.std() if score.std() else 1.0)
    edges = np.quantile(length, np.linspace(0, 1, 11))
    bins = np.searchsorted(edges[1:-1], length, side="right")
    return {
        "length_decile": list(range(1, 11)),
        "correct": [float(z[(bins == i) & (correct == 1)].mean()) if np.any((bins == i) & (correct == 1)) else None for i in range(10)],
        "incorrect": [float(z[(bins == i) & (correct == 0)].mean()) if np.any((bins == i) & (correct == 0)) else None for i in range(10)],
    }


def _length_half_records(clusters: list[dict[str, Any]], condition: str, side: str,
                         population: str | None = None) -> list[tuple[str, int, float]]:
    """Responses of one length half; with `population`, only responses where that readout is defined.

    The median split is within problem and is taken over the same population (Amendment A1 §3 for L5).
    """
    records = []
    for cluster in clusters:
        keep = [[population is None or score_value(item, population) is not None for item in row["readout_scores"]]
                for row in cluster["rows"]]
        lengths = [x for row, k in zip(cluster["rows"], keep)
                   for x, ok in zip(row["response_token_counts"], k) if ok]
        if not lengths:
            continue
        median = float(np.median(np.asarray(lengths)))
        for row, k in zip(cluster["rows"], keep):
            for i, item in enumerate(row["readout_scores"]):
                if not k[i]:
                    continue
                response_side = "short" if row["response_token_counts"][i] <= median else "long"
                value = score_value(item, condition)
                if response_side == side and value is not None:
                    records.append((cluster["dataset"], int(bool(row["correctness"][i])), value))
    return records


def l5_value(clusters: list[dict[str, Any]]) -> float | None:
    gaps = []
    for condition in ("R0_reference", "R3_position_answer"):
        short, _ = _records_primary(_length_half_records(clusters, condition, "short", "R3_position_answer"))
        long, _ = _records_primary(_length_half_records(clusters, condition, "long", "R3_position_answer"))
        if short is None or long is None:
            return None
        gaps.append(short - long)
    return gaps[0] - gaps[1]


def contrast_values(clusters: list[dict[str, Any]], ranges: dict[str, tuple[float, float]] | None = None) -> dict[str, float | None]:
    primary = {condition: primary_auroc_details(clusters, condition)[0] for condition in CONDITIONS}
    l4_linear = [linear_weighted_majority(c, "R0_reference") - linear_weighted_majority(c, "R4_cref_none") for c in clusters]
    l4_linear = [x for x in l4_linear if not math.isnan(x)]
    return {
        "L1_auroc_R0_minus_R1": _minus(primary["R0_reference"], primary["R1_marker_swap_mean"]),
        "L2_auroc_R0_minus_R2": _minus(primary["R0_reference"], primary["R2_position_eos"]),
        "L3_partial_corr_R0": metric(clusters, "R0_reference").get("partial_correlation_score_length_given_correctness"),
        "L4_weighted_majority_R0_minus_R4": float(np.mean(l4_linear)) if l4_linear else None,
        "L5_auroc_gap_short_long_R0_minus_R3": l5_value(clusters),
    }


def _minus(a: Any, b: Any) -> float | None:
    return None if a is None or b is None else float(a - b)


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    usable = sorted(((k, p) for k, p in p_values.items() if p is not None), key=lambda x: x[1])
    result = {k: None for k in p_values}
    running = 0.0
    for rank, (key, p) in enumerate(usable):
        running = max(running, min(1.0, (len(usable) - rank) * p))
        result[key] = running
    return result


def cluster_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset", "unknown")), str(row.get("problem_id", row.get("unique_id")))


def load_clusters(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    model_ids, revisions = set(), set()
    all_rows = []
    for path in paths:
        rows = read_jsonl(path)
        all_rows.extend(rows)
        for row in rows:
            model_ids.add(row.get("model")); revisions.add(row.get("model_revision"))
            grouped.setdefault(cluster_key(row), {})[int(row["seed"])] = row
    if len(model_ids) != 1 or len(revisions) != 1:
        raise RuntimeError(f"inconsistent checkpoint identity: models={model_ids}, revisions={revisions}")
    clusters, excluded = [], []
    truncated = 0
    for key, seed_rows in sorted(grouped.items()):
        reason = []
        if set(seed_rows) != {0, 1, 2}:
            reason.append("missing_seed")
        for row in seed_rows.values():
            truncated += sum(bool(x) for x in row.get("truncated", []))
            if len(row.get("responses", [])) != 8:
                reason.append("response_count")
            if len(row.get("response_token_counts", [])) != 8:
                reason.append("missing_token_counts")
            if len(row.get("correctness", [])) != 8 or len(row.get("readout_scores", [])) != 8:
                reason.append("missing_grade_or_readout")
            # A1: only R0/R1/R2/R4/R5 drive complete-case exclusion.  R3 may
            # be absent and is counted in the missingness report instead.
            for item in row.get("readout_scores", []):
                if any(score_value(item, c) is None for c in BASE_CONDITIONS):
                    reason.append("missing_condition")
        if reason:
            excluded.append({"dataset": key[0], "problem_id": key[1], "reasons": sorted(set(reason))})
        else:
            clusters.append({"dataset": key[0], "problem_id": key[1], "rows": [seed_rows[s] for s in (0, 1, 2)]})
    metadata = {
        "model": next(iter(model_ids)) if model_ids else None,
        "model_revision": next(iter(revisions)) if revisions else None,
        "input_files": paths, "excluded_problems": excluded,
        "flag_counts": {"truncated_responses": int(truncated), "excluded_problems": len(excluded)},
        "raw_rows": len(all_rows),
    }
    return clusters, metadata


def r3_missingness(clusters: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for cluster in clusters:
        bench = out.setdefault(cluster["dataset"], {"correct": {"total": 0, "missing": 0}, "incorrect": {"total": 0, "missing": 0}})
        for row in cluster["rows"]:
            for i, item in enumerate(row["readout_scores"]):
                key = "correct" if bool(row["correctness"][i]) else "incorrect"
                bench[key]["total"] += 1
                bench[key]["missing"] += int(score_value(item, "R3_position_answer") is None)
    for bench in out.values():
        for counts in bench.values():
            counts["rate"] = counts["missing"] / counts["total"] if counts["total"] else None
    return out


def kendall_by_reference(clusters: list[dict[str, Any]], condition: str) -> float | None:
    x, y = [], []
    for cluster in clusters:
        for row in cluster["rows"]:
            for item in row["readout_scores"]:
                left, right = score_value(item, "R0_reference"), score_value(item, condition)
                if left is not None and right is not None:
                    x.append(left); y.append(right)
    return kendall(np.asarray(x), np.asarray(y)) if x else None


def stratified_resample(clusters: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for cluster in clusters:
        by_benchmark.setdefault(cluster["dataset"], []).append(cluster)
    sample = []
    for benchmark in sorted(by_benchmark):
        group = by_benchmark[benchmark]
        indices = rng.integers(0, len(group), len(group))
        sample.extend(group[int(i)] for i in indices)
    return sample


def bootstrap_auroc_ci(clusters: list[dict[str, Any]], condition: str, rng: np.random.Generator, n: int, primary: bool) -> list[float] | None:
    if not clusters or n <= 0:
        return None
    values = []
    for _ in range(n):
        sample = stratified_resample(clusters, rng)
        if primary:
            value = primary_auroc_details(sample, condition)[0]
        else:
            score, correct, _ = arrays(sample, condition)
            value = auroc(correct, score)
        if value is not None:
            values.append(value)
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None


def stratified_bootstrap_samples(
    clusters: list[dict[str, Any]], rng: np.random.Generator, n: int
) -> list[list[dict[str, Any]]]:
    """Create one shared, benchmark-stratified resample plan.

    The plan is generated once and then reused for every condition, metric,
    benchmark, and R3 sensitivity calculation.  This is both reproducible and
    materially faster than drawing a fresh Python-level bootstrap for each
    reported quantity.
    """

    return [stratified_resample(clusters, rng) for _ in range(max(0, n))]


def _ci95(values: list[Any]) -> list[float] | None:
    usable = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not usable:
        return None
    return [float(np.quantile(usable, 0.025)), float(np.quantile(usable, 0.975))]


def _add_metric_intervals(observed: dict[str, Any], draws: list[dict[str, Any]]) -> None:
    """Attach percentile intervals for every scalar returned by ``metric``."""

    skip = {"n_responses", "n_problems", "primary_auroc_by_benchmark"}
    for key, value in list(observed.items()):
        if key in skip or not isinstance(value, (int, float)):
            continue
        observed[f"{key}_ci95"] = _ci95([item.get(key) for item in draws])

    point_by_benchmark = observed.get("primary_auroc_by_benchmark", {})
    for benchmark, point in point_by_benchmark.items():
        point["auroc_ci95"] = _ci95(
            [
                draw.get("primary_auroc_by_benchmark", {}).get(benchmark, {}).get("auroc")
                for draw in draws
            ]
        )


def _attach_length_half_intervals(
    observed: dict[str, Any], draws: list[dict[str, Any]]
) -> None:
    for side, point in observed.items():
        if not isinstance(point, dict):
            continue
        for key, value in list(point.items()):
            if key in {"n_responses", "n_problems"} or not isinstance(value, (int, float)):
                continue
            point[f"{key}_ci95"] = _ci95(
                [draw.get(side, {}).get(key) for draw in draws]
            )


def _sample_counts(
    clusters: list[dict[str, Any]], samples: list[list[dict[str, Any]]]
) -> np.ndarray:
    """Encode a bootstrap plan as a B x K cluster multiplicity matrix."""

    positions = {id(cluster): i for i, cluster in enumerate(clusters)}
    counts = np.zeros((len(samples), len(clusters)), dtype=np.int32)
    for draw_i, sample in enumerate(samples):
        indices = [positions[id(cluster)] for cluster in sample]
        if indices:
            counts[draw_i] = np.bincount(indices, minlength=len(clusters))
    return counts


def _padded_condition_arrays(
    clusters: list[dict[str, Any]], condition: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return padded response arrays and a validity mask, one row per cluster."""

    data = [arrays([cluster], condition) for cluster in clusters]
    width = max((len(scores) for scores, _correct, _length in data), default=0)
    scores = np.zeros((len(data), width), dtype=float)
    correct = np.zeros((len(data), width), dtype=float)
    lengths = np.zeros((len(data), width), dtype=float)
    chars = np.zeros((len(data), width), dtype=float)
    valid = np.zeros((len(data), width), dtype=bool)
    for i, (score, label, length) in enumerate(data):
        size = len(score)
        scores[i, :size] = score
        correct[i, :size] = label
        lengths[i, :size] = length
        char_score, char_length = char_arrays([clusters[i]], condition)
        # char_score has the same validity population as score.
        chars[i, :size] = char_length
        valid[i, :size] = True
    return scores, correct, lengths, chars, valid


def _pair_auc_matrix(
    scores: np.ndarray, correct: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute positive-vs-negative pair contributions by cluster."""

    n_clusters = scores.shape[0]
    pair = np.zeros((n_clusters, n_clusters), dtype=float)
    cluster_ids = np.broadcast_to(np.arange(n_clusters)[:, None], scores.shape)[valid]
    flat_scores = scores[valid]
    flat_correct = correct[valid].astype(int)
    order = np.argsort(flat_scores, kind="mergesort")
    sorted_scores = flat_scores[order]
    sorted_clusters = cluster_ids[order]
    sorted_correct = flat_correct[order]
    positives = np.bincount(
        sorted_clusters[sorted_correct == 1], minlength=n_clusters
    ).astype(float)
    negatives = np.bincount(
        sorted_clusters[sorted_correct == 0], minlength=n_clusters
    ).astype(float)
    cumulative_negative = np.zeros(n_clusters, dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_clusters = sorted_clusters[start:end]
        group_correct = sorted_correct[start:end]
        group_negative = np.bincount(
            group_clusters[group_correct == 0], minlength=n_clusters
        ).astype(float)
        positive_group = np.bincount(
            group_clusters[group_correct == 1], minlength=n_clusters
        ).astype(float)
        for cluster_i in np.flatnonzero(positive_group):
            pair[cluster_i] += positive_group[cluster_i] * (
                cumulative_negative + 0.5 * group_negative
            )
        cumulative_negative += group_negative
        start = end
    return pair, positives, negatives


def _weighted_auc_draws(
    counts: np.ndarray, scores: np.ndarray, correct: np.ndarray, valid: np.ndarray
) -> list[float | None]:
    if not len(counts):
        return []
    pair, positives, negatives = _pair_auc_matrix(scores, correct, valid)
    n_pos = counts @ positives
    n_neg = counts @ negatives
    numerator = np.einsum("bi,ij,bj->b", counts, pair, counts)
    denominator = n_pos * n_neg
    values = np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)
    return [None if not np.isfinite(value) else float(value) for value in values]


def _weighted_calibration_draws(
    counts: np.ndarray, scores: np.ndarray, correct: np.ndarray, valid: np.ndarray
) -> tuple[list[float | None], list[float | None]]:
    bins = np.clip(scores, 0.0, 1.0)
    bin_ids = np.minimum((bins * 15).astype(int), 14)
    bin_counts = np.zeros((len(scores), 15), dtype=float)
    bin_score_sums = np.zeros((len(scores), 15), dtype=float)
    bin_label_sums = np.zeros((len(scores), 15), dtype=float)
    for i in range(len(scores)):
        for b in range(15):
            mask = valid[i] & (bin_ids[i] == b)
            bin_counts[i, b] = np.sum(mask)
            bin_score_sums[i, b] = np.sum(bins[i, mask])
            bin_label_sums[i, b] = np.sum(correct[i, mask])
    counts_b = counts @ bin_counts
    score_b = counts @ bin_score_sums
    label_b = counts @ bin_label_sums
    total = counts_b.sum(axis=1)
    ece = np.zeros(len(counts), dtype=float)
    for b in range(15):
        nonzero = counts_b[:, b] > 0
        ece[nonzero] += (
            counts_b[nonzero, b] / total[nonzero]
            * np.abs(score_b[nonzero, b] / counts_b[nonzero, b] - label_b[nonzero, b] / counts_b[nonzero, b])
        )
    p = np.clip(scores, 0.0, 1.0)
    brier_cluster = np.where(valid, (p - correct) ** 2, 0.0).sum(axis=1)
    brier = np.divide(counts @ brier_cluster, total, out=np.full(len(counts), np.nan), where=total > 0)
    return (
        [None if not np.isfinite(value) else float(value) for value in ece],
        [None if not np.isfinite(value) else float(value) for value in brier],
    )


def _weighted_partial_corr_draws(
    counts: np.ndarray, score: np.ndarray, length: np.ndarray,
    correct: np.ndarray, valid: np.ndarray,
) -> list[float | None]:
    numerator = np.zeros(len(counts), dtype=float)
    denominator_x = np.zeros(len(counts), dtype=float)
    denominator_z = np.zeros(len(counts), dtype=float)
    for group in (0, 1):
        base = valid & (correct == group)
        n_cluster = base.sum(axis=1).astype(float)
        sx_cluster = np.where(base, score, 0.0).sum(axis=1)
        sz_cluster = np.where(base, length, 0.0).sum(axis=1)
        sxx_cluster = np.where(base, score * score, 0.0).sum(axis=1)
        szz_cluster = np.where(base, length * length, 0.0).sum(axis=1)
        sxz_cluster = np.where(base, score * length, 0.0).sum(axis=1)
        n = counts @ n_cluster
        sx = counts @ sx_cluster; sz = counts @ sz_cluster
        sxx = counts @ sxx_cluster; szz = counts @ szz_cluster; sxz = counts @ sxz_cluster
        usable = n > 0
        numerator[usable] += sxz[usable] - sx[usable] * sz[usable] / n[usable]
        denominator_x[usable] += sxx[usable] - sx[usable] * sx[usable] / n[usable]
        denominator_z[usable] += szz[usable] - sz[usable] * sz[usable] / n[usable]
    denom = np.sqrt(np.maximum(0.0, denominator_x * denominator_z))
    values = np.divide(numerator, denom, out=np.full(len(counts), np.nan), where=denom > 0)
    return [None if not np.isfinite(value) else float(value) for value in values]


def _vectorized_metric_draws(
    clusters: list[dict[str, Any]], condition: str, samples: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Vectorized draws for one condition; counts are the only draw loop."""

    counts = _sample_counts(clusters, samples)
    scores, correct, lengths, chars, valid = _padded_condition_arrays(clusters, condition)
    pooled_auc = _weighted_auc_draws(counts, scores, correct, valid)
    ece, brier = _weighted_calibration_draws(counts, scores, correct, valid)
    partial = _weighted_partial_corr_draws(counts, scores, lengths, correct, valid)
    char_partial = _weighted_partial_corr_draws(counts, scores, chars, correct, valid)
    vote_values = {
        "weighted_majority_accuracy_k8": np.asarray([linear_weighted_majority(c, condition) for c in clusters], dtype=float),
        "exp_weighted_majority_accuracy_k8": np.asarray([exp_weighted_majority(c, condition) for c in clusters], dtype=float),
        "unweighted_majority_accuracy_k8": np.asarray([unweighted_majority(c, condition) for c in clusters], dtype=float),
    }
    output = []
    valid_problem_count = counts.sum(axis=1)
    for draw_i in range(len(samples)):
        item = {
            "n_responses": int(round(float(counts[draw_i] @ valid.sum(axis=1)))),
            "n_problems": int(round(float(valid_problem_count[draw_i]))),
            "auroc": pooled_auc[draw_i],
            "auroc_pooled": pooled_auc[draw_i],
            "primary_auroc": pooled_auc[draw_i],
            "ece_15": ece[draw_i],
            "brier": brier[draw_i],
            "partial_correlation_score_length_given_correctness": partial[draw_i],
            "partial_correlation_score_characters_given_correctness": char_partial[draw_i],
            "primary_auroc_by_benchmark": {},
        }
        for key, values in vote_values.items():
            denominator = counts[draw_i] @ np.isfinite(values)
            numerator = counts[draw_i] @ np.nan_to_num(values, nan=0.0)
            item[key] = None if denominator == 0 else float(numerator / denominator)
        output.append(item)
    for benchmark in sorted({cluster["dataset"] for cluster in clusters}):
        indices = [i for i, cluster in enumerate(clusters) if cluster["dataset"] == benchmark]
        sub_counts = counts[:, indices]
        sub_scores, sub_correct, _sub_lengths, _sub_chars, sub_valid = (
            scores[indices], correct[indices], lengths[indices], chars[indices], valid[indices]
        )
        sub_auc = _weighted_auc_draws(sub_counts, sub_scores, sub_correct, sub_valid)
        for draw_i, value in enumerate(sub_auc):
            output[draw_i]["primary_auroc_by_benchmark"][benchmark] = {
                "n_responses": int(round(float(sub_counts[draw_i] @ sub_valid.sum(axis=1)))),
                "auroc": value,
            }
    for draw_i, item in enumerate(output):
        weighted, total = 0.0, 0
        for details in item["primary_auroc_by_benchmark"].values():
            if details["auroc"] is not None:
                weighted += details["n_responses"] * details["auroc"]
                total += details["n_responses"]
        item["primary_auroc"] = None if not total else float(weighted / total)
    return output


def _length_half_clusters(
    clusters: list[dict[str, Any]], side: str, population: str | None = "R3_position_answer"
) -> list[dict[str, Any]]:
    """Materialize the A1 box-compliant, within-problem length halves."""

    output = []
    for cluster in clusters:
        lengths = [
            length
            for row in cluster["rows"]
            for length, item in zip(row["response_token_counts"], row["readout_scores"])
            if population is None or score_value(item, population) is not None
        ]
        median = float(np.median(lengths)) if lengths else None
        rows = []
        for row in cluster["rows"]:
            indices = [
                i for i, (length, item) in enumerate(
                    zip(row["response_token_counts"], row["readout_scores"])
                )
                if (population is None or score_value(item, population) is not None)
                and median is not None
                and (length <= median if side == "short" else length > median)
            ]
            row_copy = dict(row)
            for key in (
                "responses", "pred_answers", "correctness", "readout_scores",
                "response_token_counts",
            ):
                if key in row:
                    row_copy[key] = [row[key][i] for i in indices]
            rows.append(row_copy)
        output.append({**cluster, "rows": rows})
    return output


def _kendall_pair_stats(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_clusters = len(x)
    concord = np.zeros((n_clusters, n_clusters), dtype=float)
    discord = np.zeros_like(concord)
    tie_x = np.zeros_like(concord)
    tie_y = np.zeros_like(concord)
    joint = np.zeros_like(concord)
    within_concord = np.zeros(n_clusters, dtype=float)
    within_discord = np.zeros(n_clusters, dtype=float)
    within_tie_x = np.zeros(n_clusters, dtype=float)
    within_tie_y = np.zeros(n_clusters, dtype=float)
    within_joint = np.zeros(n_clusters, dtype=float)
    n_observations = valid.sum(axis=1).astype(float)
    # Aggregate all response-position pairs with K x K NumPy comparisons.
    # The diagonal is rebuilt separately because cross-cluster aggregation
    # includes self-pairs there, which are not part of Kendall's statistic.
    for position in range(x.shape[1]):
        pair_valid = valid[:, position, None] & valid[None, :, :]
        dx = x[:, position, None] - x[None, :, :]
        dy = y[:, position, None] - y[None, :, :]
        concord += np.sum((dx * dy > 0) & pair_valid, axis=2)
        discord += np.sum((dx * dy < 0) & pair_valid, axis=2)
        tie_x += np.sum((dx == 0) & (dy != 0) & pair_valid, axis=2)
        tie_y += np.sum((dx != 0) & (dy == 0) & pair_valid, axis=2)
        joint += np.sum((dx == 0) & (dy == 0) & pair_valid, axis=2)
    np.fill_diagonal(concord, 0.0)
    np.fill_diagonal(discord, 0.0)
    np.fill_diagonal(tie_x, 0.0)
    np.fill_diagonal(tie_y, 0.0)
    np.fill_diagonal(joint, 0.0)
    for i in range(n_clusters):
        xi, yi = x[i, valid[i]], y[i, valid[i]]
        if len(xi) < 2:
            continue
        left, right = np.triu_indices(len(xi), 1)
        dx, dy = xi[left] - xi[right], yi[left] - yi[right]
        within_concord[i] = np.sum(dx * dy > 0)
        within_discord[i] = np.sum(dx * dy < 0)
        within_tie_x[i] = np.sum((dx == 0) & (dy != 0))
        within_tie_y[i] = np.sum((dx != 0) & (dy == 0))
        within_joint[i] = np.sum((dx == 0) & (dy == 0))
    return (
        concord,
        discord,
        tie_x,
        tie_y,
        joint,
        np.asarray([within_concord, within_discord, within_tie_x, within_tie_y, within_joint, n_observations]),
    )


def _kendall_draws_vectorized(
    clusters: list[dict[str, Any]], condition: str, samples: list[list[dict[str, Any]]]
) -> list[float | None]:
    counts = _sample_counts(clusters, samples)
    def aligned(target: str) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(clusters), 24), dtype=float)
        valid = np.zeros((len(clusters), 24), dtype=bool)
        for cluster_i, cluster in enumerate(clusters):
            slot = 0
            for row in cluster["rows"]:
                for item in row["readout_scores"]:
                    value = score_value(item, target)
                    if value is not None:
                        values[cluster_i, slot] = value
                        valid[cluster_i, slot] = True
                    slot += 1
        return values, valid

    x, xvalid = aligned("R0_reference")
    y, yvalid = aligned(condition)
    valid = xvalid & yvalid
    concord, discord, tie_x, tie_y, joint, within = _kendall_pair_stats(x, y, valid)
    within_c, within_d, within_tx, within_ty, within_joint, n_obs = within
    values = []
    for count in counts:
        cross = np.outer(count, count)
        cross_pairs = np.triu(cross, 1)
        c = float(np.sum(cross_pairs * concord) + count @ within_c)
        d = float(np.sum(cross_pairs * discord) + count @ within_d)
        tx = float(np.sum(cross_pairs * tie_x) + count @ within_tx)
        ty = float(np.sum(cross_pairs * tie_y) + count @ within_ty)
        both = float(np.sum(cross_pairs * joint) + count @ within_joint)
        duplicate_pairs = count * (count - 1.0) / 2.0
        c += float(duplicate_pairs @ (2.0 * within_c))
        d += float(duplicate_pairs @ (2.0 * within_d))
        tx += float(duplicate_pairs @ (2.0 * within_tx))
        ty += float(duplicate_pairs @ (2.0 * within_ty))
        both += float(duplicate_pairs @ (2.0 * within_joint + n_obs))
        total_obs = float(count @ n_obs)
        total = total_obs * (total_obs - 1.0) / 2.0
        # Pairs between duplicate copies preserve the within-cluster ordering;
        # the same underlying response is the only additional joint tie.
        denom = math.sqrt(max(0.0, (total - ty - both) * (total - tx - both)))
        values.append(None if denom == 0 else float((c - d) / denom))
    return values


def bootstrap_report_intervals(
    clusters: list[dict[str, Any]],
    conditions: list[str],
    samples: list[list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[float | None]], dict[str, dict[str, list[float | None]]]]:
    """Compute shared-resample draws for all registered report metrics.

    Metric calculations operate on NumPy arrays at the response level; only
    the outer cluster-resample plan is Python-level.  Keeping the plan shared
    is important because every interval in the report must use the same
    generator seed and the same resampled problems.
    """

    metric_draws = {condition: [] for condition in conditions}
    benchmark_names = sorted({cluster["dataset"] for cluster in clusters})
    benchmark_draws = {
        benchmark: {condition: [] for condition in conditions}
        for benchmark in benchmark_names
    }
    length_draws = {
        condition: [] for condition in ("R0_reference", "R3_position_answer")
    }
    kendall_draws = {condition: [] for condition in CONDITIONS}
    benchmark_kendall_draws = {
        benchmark: {condition: [] for condition in CONDITIONS}
        for benchmark in benchmark_names
    }
    for condition in conditions:
        metric_draws[condition] = _vectorized_metric_draws(clusters, condition, samples)
    for benchmark in benchmark_names:
        subset = [cluster for cluster in clusters if cluster["dataset"] == benchmark]
        subset_samples = [[cluster for cluster in sample if cluster["dataset"] == benchmark] for sample in samples]
        for condition in conditions:
            benchmark_draws[benchmark][condition] = _vectorized_metric_draws(subset, condition, subset_samples)
        for condition in CONDITIONS:
            benchmark_kendall_draws[benchmark][condition] = _kendall_draws_vectorized(
                subset, condition, subset_samples
            )
    for condition in length_draws:
        length_draws[condition] = []
        for side in ("short", "long"):
            half_clusters = _length_half_clusters(clusters, side)
            by_id = {id(cluster): half for cluster, half in zip(clusters, half_clusters)}
            half_samples = [[by_id[id(cluster)] for cluster in sample] for sample in samples]
            half_draws = _vectorized_metric_draws(half_clusters, condition, half_samples)
            for draw_i, draw in enumerate(half_draws):
                if len(length_draws[condition]) <= draw_i:
                    length_draws[condition].append({})
                length_draws[condition][draw_i][side] = draw
    for condition in CONDITIONS:
        kendall_draws[condition] = _kendall_draws_vectorized(clusters, condition, samples)
    return metric_draws, benchmark_draws, length_draws, kendall_draws, benchmark_kendall_draws


def _contrast_verdict(key: str, item: dict[str, Any]) -> None:
    bound = EQUIVALENCE_BOUNDS[key]
    ci90 = item.get("ci90")
    item["equivalence_bound"] = bound
    item["negligible"] = bool(ci90 is not None and ci90[0] >= -bound and ci90[1] <= bound)
    item["verdict"] = "holds" if item.get("holds") else ("negligible" if item["negligible"] else "inconclusive")


def _with_fallback_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    if out.get("R3_position_answer") is None:
        out["R3_position_answer"] = out.get("R2_position_eos")
    return out


def metric_with_r3_fallback(clusters: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    copied = []
    for cluster in clusters:
        rows = []
        for row in cluster["rows"]:
            row_copy = dict(row)
            row_copy["readout_scores"] = [_with_fallback_item(item) for item in row["readout_scores"]]
            rows.append(row_copy)
        copied.append({**cluster, "rows": rows})
    return metric(copied, condition)


def l5_value_with_r3_fallback(clusters: list[dict[str, Any]]) -> float | None:
    copied = []
    for cluster in clusters:
        rows = []
        for row in cluster["rows"]:
            row_copy = dict(row)
            row_copy["readout_scores"] = [_with_fallback_item(item) for item in row["readout_scores"]]
            rows.append(row_copy)
        copied.append({**cluster, "rows": rows})
    return l5_value(copied)


def _l5_fallback_draws(
    clusters: list[dict[str, Any]], samples: list[list[dict[str, Any]]]
) -> list[float | None]:
    """Vectorized L5 draws after replacing missing R3 with R2."""

    fallback_base = []
    for cluster in clusters:
        rows = []
        for row in cluster["rows"]:
            row_copy = dict(row)
            row_copy["readout_scores"] = [
                _with_fallback_item(item) for item in row["readout_scores"]
            ]
            rows.append(row_copy)
        fallback_base.append({**cluster, "rows": rows})
    by_id = {id(cluster): item for cluster, item in zip(clusters, fallback_base)}
    fallback_samples = [[by_id[id(cluster)] for cluster in sample] for sample in samples]
    half_draws = {}
    for side in ("short", "long"):
        half_base = _length_half_clusters(fallback_base, side, population=None)
        half_by_id = {id(cluster): item for cluster, item in zip(fallback_base, half_base)}
        half_samples = [[half_by_id[id(cluster)] for cluster in sample] for sample in fallback_samples]
        for condition in ("R0_reference", "R3_position_answer"):
            half_draws[(side, condition)] = _vectorized_metric_draws(
                half_base, condition, half_samples
            )
    values: list[float | None] = []
    for draw_i in range(len(samples)):
        short0 = half_draws[("short", "R0_reference")][draw_i]["primary_auroc"]
        long0 = half_draws[("long", "R0_reference")][draw_i]["primary_auroc"]
        short3 = half_draws[("short", "R3_position_answer")][draw_i]["primary_auroc"]
        long3 = half_draws[("long", "R3_position_answer")][draw_i]["primary_auroc"]
        values.append(
            None if None in (short0, long0, short3, long3)
            else float((short0 - long0) - (short3 - long3))
        )
    return values


def analyze(paths: list[str], seed: int, bootstrap: int, metric_bootstrap: int, label: str | None = None) -> dict[str, Any]:
    clusters, meta = load_clusters(paths)
    token_conditions = _r1_token_conditions(clusters)
    all_metric_conditions = CONDITIONS + token_conditions
    rng = np.random.default_rng(seed)
    shared_samples = stratified_bootstrap_samples(
        clusters, rng, max(int(bootstrap), int(metric_bootstrap))
    )
    metric_samples = shared_samples[: int(metric_bootstrap)]
    metric_draws, benchmark_draws, length_draws, kendall_draws, benchmark_kendall_draws = bootstrap_report_intervals(
        clusters, all_metric_conditions, metric_samples
    )
    observed: dict[str, Any] = {}
    for condition in all_metric_conditions:
        item = metric(clusters, condition)
        _add_metric_intervals(item, metric_draws[condition])
        # auroc_ci95 is the interval of the pooled `auroc`; primary_auroc_ci95 belongs to primary_auroc.
        observed[condition] = item
    observed["kendall_tau_vs_R0"] = {c: kendall_by_reference(clusters, c) for c in CONDITIONS}
    observed["kendall_tau_vs_R0_ci95"] = {
        c: _ci95(kendall_draws[c]) for c in CONDITIONS
    }
    observed["length_halves"] = {}
    for condition in ("R0_reference", "R3_position_answer"):
        point = length_half_metrics(clusters, condition)
        _attach_length_half_intervals(point, length_draws[condition])
        observed["length_halves"][condition] = point
    benchmarks = sorted({c["dataset"] for c in clusters})
    by_benchmark = {}
    for benchmark in benchmarks:
        subset = [c for c in clusters if c["dataset"] == benchmark]
        by_benchmark[benchmark] = {}
        for condition in all_metric_conditions:
            point = metric(subset, condition)
            _add_metric_intervals(point, benchmark_draws[benchmark][condition])
            by_benchmark[benchmark][condition] = point
        by_benchmark[benchmark]["kendall_tau_vs_R0"] = {c: kendall_by_reference(subset, c) for c in CONDITIONS}
        by_benchmark[benchmark]["kendall_tau_vs_R0_ci95"] = {
            c: _ci95(benchmark_kendall_draws[benchmark][c])
            for c in CONDITIONS
        }
        by_benchmark[benchmark]["length_halves"] = {}
        for condition in ("R0_reference", "R3_position_answer"):
            point = length_half_metrics(subset, condition)
            _attach_length_half_intervals(
                point,
                [
                    length_half_metrics(
                        [cluster for cluster in sample if cluster["dataset"] == benchmark],
                        condition,
                    )
                    for sample in metric_samples
                ],
            )
            by_benchmark[benchmark]["length_halves"][condition] = point
    per_seed = {}
    per_seed_metrics = {}
    per_seed_contrasts = {str(seed_id): None for seed_id in (0, 1, 2)}
    for seed_id in (0, 1, 2):
        seed_clusters = [{**c, "rows": [c["rows"][seed_id]]} for c in clusters]
        per_seed_contrasts[str(seed_id)] = contrast_values(seed_clusters)
        per_seed_metrics[str(seed_id)] = {
            condition: metric(seed_clusters, condition)
            for condition in all_metric_conditions
        }
        per_seed_metrics[str(seed_id)]["kendall_tau_vs_R0"] = {
            condition: kendall_by_reference(seed_clusters, condition)
            for condition in CONDITIONS
        }
    for benchmark in benchmarks:
        per_seed[benchmark] = {}
        for seed_id in (0, 1, 2):
            subset = [{**c, "rows": [c["rows"][seed_id]]} for c in clusters if c["dataset"] == benchmark]
            per_seed[benchmark][str(seed_id)] = {c: metric(subset, c) for c in all_metric_conditions}
            per_seed[benchmark][str(seed_id)]["kendall_tau_vs_R0"] = {c: kendall_by_reference(subset, c) for c in CONDITIONS}
            per_seed[benchmark][str(seed_id)]["contrasts"] = contrast_values(subset)
    observed_contrasts = contrast_values(clusters)
    length_score = {c: length_score_series(clusters, c) for c in ("R0_reference", "R3_position_answer")}
    draws = {k: [] for k, value in observed_contrasts.items() if value is not None}
    for sample in shared_samples[: int(bootstrap)]:
        for key, value in contrast_values(sample).items():
            if key in draws and value is not None:
                draws[key].append(float(value))
    contrasts, p_values = {}, {}
    for key, estimate in observed_contrasts.items():
        arr = np.asarray(draws.get(key, []), dtype=float)
        ci95 = [float(np.quantile(arr, .025)), float(np.quantile(arr, .975))] if len(arr) else None
        ci90 = [float(np.quantile(arr, .05)), float(np.quantile(arr, .95))] if len(arr) else None
        p = float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))) if len(arr) else None
        contrasts[key] = {"estimate": estimate, "ci95": ci95, "ci90": ci90, "p_bootstrap_two_sided": p, "p_two_sided": p}
        p_values[key] = p
    adjusted = holm(p_values)
    for key, item in contrasts.items():
        item["p_holm"] = adjusted.get(key)
        item["holds"] = bool(item["p_holm"] is not None and item["p_holm"] < .05 and item["ci95"] and (item["ci95"][0] > 0 or item["ci95"][1] < 0))
        _contrast_verdict(key, item)
    sensitivity = {}
    fallback_base = []
    for cluster in clusters:
        rows = []
        for row in cluster["rows"]:
            row_copy = dict(row)
            row_copy["readout_scores"] = [
                _with_fallback_item(item) for item in row["readout_scores"]
            ]
            rows.append(row_copy)
        fallback_base.append({**cluster, "rows": rows})
    fallback_by_id = {id(cluster): fallback for cluster, fallback in zip(clusters, fallback_base)}
    fallback_samples = [
        [fallback_by_id[id(cluster)] for cluster in sample] for sample in metric_samples
    ]
    for condition in CONDITIONS:
        if condition == "R3_position_answer":
            point = metric_with_r3_fallback(clusters, condition)
            fallback_draws = _vectorized_metric_draws(
                fallback_base, condition, fallback_samples
            )
            _add_metric_intervals(point, fallback_draws)
            sensitivity[condition] = point
        else:
            sensitivity[condition] = observed[condition]
    sensitivity_l5_draws = _l5_fallback_draws(clusters, metric_samples)
    return {
        "schema_version": 3, "stage": "E2c", "label": label, "model": meta["model"],
        "model_revision": meta["model_revision"], "input_files": paths,
        "unit": "problem cluster; all 8 responses for each of seeds 0, 1, 2",
        "decisions": {
            "score": "beta_v * (readout log probability - reference constant), as emitted by readout_fast.py",
            "calibration": "clip(score, 0, 1); 15 equal-width bins; no per-condition normalization",
            "weighted_majority": "RM@8 uses linear score sums over math_verify equivalence classes; ties by class count then first response; exp(score) is secondary",
            "primary_auroc": "response-weighted mean of per-benchmark AUROCs; pooled and per-benchmark values are secondary",
            "r3": "R3 metrics and L5 use box-compliant responses; complete-case exclusion covers R0/R1/R2/R4/R5 only",
            "response_length": "generated token count; characters are a secondary check",
            "bootstrap": "stratified resample of problems within benchmark; 10000 resamples for confirmatory runs",
            "equivalence": "90% interval inside A1 bound is negligible; otherwise PREREG holds or inconclusive",
        },
        "bootstrap": {
            "resamples": bootstrap,
            "metric_resamples": metric_bootstrap,
            "seed": seed,
            "ci": "95% percentile; 90% equivalence interval",
            "shared_resample_plan": True,
            "p_label": "bootstrap p",
        },
        "r1_token_conditions": token_conditions,
        "metrics": observed, "by_benchmark": by_benchmark, "by_benchmark_seed": per_seed,
        "per_seed_metrics": per_seed_metrics,
        "per_seed_contrasts": per_seed_contrasts,
        "length_score": length_score, "contrasts": contrasts,
        "r3_missingness": r3_missingness(clusters),
        "sensitivity_r3_as_r2": {
            "metrics": sensitivity,
            "l5": l5_value_with_r3_fallback(clusters),
            "l5_ci95": _ci95(sensitivity_l5_draws),
        },
        "exclusions": meta["excluded_problems"], "flag_counts": meta["flag_counts"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--metric-bootstrap", type=int, default=10000)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    result = analyze(args.input, args.seed, args.bootstrap, args.metric_bootstrap, args.label)
    output = Path(args.output)
    write_json(output, result)
    output.with_name(output.name + ".sha256").write_text(f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
