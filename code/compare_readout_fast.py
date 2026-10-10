#!/usr/bin/env python3
"""Compare fast and reference readouts in raw log-probability space."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import read_jsonl, sha256_file, write_json


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 1.0 if all(x == y for x, y in zip(xs, ys)) else 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def auroc(scores: list[float], labels: list[bool]) -> float:
    positives = sum(bool(x) for x in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        mean_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += sum(mean_rank for k in order[i:j] if labels[k])
        rank += j - i
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def flatten(rows: list[dict[str, Any]], fast: bool) -> dict[str, list[float | None] | list[bool]]:
    values: dict[str, list[float | None]] = {
        "R0": [],
        "R1_0": [],
        "R1_1": [],
        "R1_2": [],
        "R2": [],
        "R3": [],
        "R4": [],
        "R5": [],
    }
    labels: list[bool] = []
    for row in rows:
        for response_i, correct in enumerate(row["correctness"]):
            logs = row["readout_logprobs"][response_i]
            r0 = logs["R0_marker"]
            swaps = logs["R1_markers"]
            raw_values = [r0, swaps[sorted(swaps)[0]], swaps[sorted(swaps)[1]], swaps[sorted(swaps)[2]]]
            raw_values += [logs["R2_marker"], logs["R3_marker"], r0]
            base = logs.get("R5_base_marker")
            raw_values.append(r0 - base if r0 is not None and base is not None else None)
            if any(value is None for value in raw_values[:5] + raw_values[6:]):
                raise ValueError("missing non-R3 readout value in equivalence input")
            for key, value in zip(values, raw_values):
                values[key].append(None if value is None else float(value))
            labels.append(bool(correct))
    values["correctness"] = labels
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    fast_rows = read_jsonl(args.fast)
    reference_rows = read_jsonl(args.reference)
    if len(fast_rows) != len(reference_rows):
        raise ValueError("fast/reference row count differs")
    for fast, reference in zip(fast_rows, reference_rows):
        if fast.get("problem_id") != reference.get("problem_id"):
            raise ValueError("fast/reference response order differs")
        if len(fast["responses"]) != len(reference["responses"]):
            raise ValueError("fast/reference response count differs")
        if fast["responses"] != reference["responses"]:
            raise ValueError("fast/reference are not on the same raw response surface")
    fast_values = flatten(fast_rows, True)
    reference_values = flatten(reference_rows, False)
    report: dict[str, Any] = {
        "kind": "readout_fast_equivalence",
        "fast": str(args.fast),
        "reference": str(args.reference),
        "surface": "raw pilot text (normalization disabled by --raw-surface)",
        "criterion": {"pearson_r_min": 0.999, "max_absolute_difference": 0.05},
        "responses": len(fast_values["correctness"]),
        "readouts": {},
    }
    passed = True
    for key in (
        "R0",
        "R1_0",
        "R1_1",
        "R1_2",
        "R2",
        "R3",
        "R4",
        "R5",
    ):
        ref = reference_values[key]
        fast = fast_values[key]
        pairs = [(a, b) for a, b in zip(fast, ref) if a is not None and b is not None]
        one_sided_nulls = sum((a is None) != (b is None) for a, b in zip(fast, ref))
        both_nulls = sum(a is None and b is None for a, b in zip(fast, ref))
        if key != "R3" and (one_sided_nulls or both_nulls):
            raise ValueError(f"missing non-R3 readout value for {key}")
        paired_fast = [a for a, _ in pairs]
        paired_ref = [b for _, b in pairs]
        differences = [abs(a - b) for a, b in pairs]
        r = pearson(paired_fast, paired_ref)
        item = {
            "pearson_r": r,
            "max_absolute_difference": max(differences) if differences else None,
            "mean_absolute_difference": (
                sum(differences) / len(differences) if differences else None
            ),
            "nulls_fast": sum(value is None for value in fast),
            "nulls_reference": sum(value is None for value in ref),
            "both_nulls": both_nulls,
            "one_sided_nulls": one_sided_nulls,
            "auroc_reference": auroc(paired_ref, [label for value, label in zip(ref, reference_values["correctness"]) if value is not None]),
            "auroc_fast": auroc(paired_fast, [label for value, label in zip(fast, fast_values["correctness"]) if value is not None]),
        }
        report["readouts"][key] = item
        if one_sided_nulls or not math.isfinite(r) or r < 0.999 or (
            item["max_absolute_difference"] is not None
            and item["max_absolute_difference"] > 0.05
        ):
            passed = False
    report["r3_nulls"] = report["readouts"]["R3"] | {
        "agreement": report["readouts"]["R3"]["one_sided_nulls"] == 0,
    }
    report["pass"] = passed
    output = Path(args.output)
    write_json(output, report)
    output.with_name(output.name + ".sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
