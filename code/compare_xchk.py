#!/usr/bin/env python3
"""Compare seed-0 fast and vLLM readouts on the score scale.

The point estimates deliberately call the registered A1 analysis functions;
this file only assembles one seed of already graded/read-out rows and does not
alter the confirmatory analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from analyze_confirm_a1 import (
    CONDITIONS,
    EQUIVALENCE_BOUNDS,
    contrast_values,
    metric,
    score_value,
)
from common import model_config, read_jsonl, sha256_file, write_json


CONDITION_LABELS = list(CONDITIONS)
CONTRASTS = list(EQUIVALENCE_BOUNDS)
REQUIRED_CONDITIONS = set(CONDITIONS)


def _token_condition(item: dict[str, Any], token: str) -> str:
    return f"R1_marker_swap_token_{token}"


def _token_conditions(rows: list[dict[str, Any]]) -> list[str]:
    tokens: set[str] = set()
    for row in rows:
        for item in row.get("readout_scores", []):
            tokens.update(str(token) for token in item.get("R1_marker_swap", {}))
    return [
        _token_condition({}, token)
        for token in sorted(tokens, key=lambda value: (int(value) if value.isdigit() else value))
    ]


def _json_fingerprint(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assert_internal_consistency(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} has no rows")
    models = {row.get("model") for row in rows}
    revisions = {row.get("model_revision") for row in rows}
    seeds = {row.get("seed") for row in rows}
    surfaces = {row.get("response_surface") for row in rows}
    if len(models) != 1 or len(revisions) != 1 or seeds != {0}:
        raise ValueError(
            f"{label} has inconsistent model/revision/seed: "
            f"models={models}, revisions={revisions}, seeds={seeds}"
        )
    if surfaces != {"released_response_v1"}:
        raise ValueError(f"{label} is not entirely on released_response_v1: {surfaces}")
    stacks = [row.get("readout_stack") for row in rows]
    if any(stack is None for stack in stacks):
        raise ValueError(f"{label} has a row without readout_stack")
    fingerprints = {_json_fingerprint(stack) for stack in stacks}
    if len(fingerprints) != 1:
        raise ValueError(f"{label} has inconsistent readout inference stacks")
    for row in rows:
        if len(row.get("responses", [])) != 8:
            raise ValueError(f"{label} has a row without eight responses: {row.get('problem_id')}")
        if len(row.get("correctness", [])) != 8:
            raise ValueError(f"{label} has a row without eight grades: {row.get('problem_id')}")
        if len(row.get("response_token_counts", [])) != 8:
            raise ValueError(f"{label} has missing response_token_counts: {row.get('problem_id')}")
        scores = row.get("readout_scores", [])
        if len(scores) != 8:
            raise ValueError(f"{label} has incomplete readout_scores: {row.get('problem_id')}")
        for item in scores:
            if not REQUIRED_CONDITIONS.issubset(item):
                raise ValueError(f"{label} has missing readout condition")


def _read_paths(paths: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        part = read_jsonl(path)
        if not part:
            raise ValueError(f"{label} input is empty: {path}")
        rows.extend(part)
    _fill_missing_token_counts(rows)
    _assert_internal_consistency(rows, label)
    return rows


def _fill_missing_token_counts(rows: list[dict[str, Any]]) -> None:
    """Backfill old pilot artifacts that predate the registered length field.

    Confirmatory graded/readout rows already carry the exact generated-token
    counts.  The immutable pilot readouts used by the required smoke test do
    not, so derive those counts with the checkpoint tokenizer in memory only.
    """
    if all(len(row.get("response_token_counts", [])) == 8 for row in rows):
        return
    from transformers import AutoTokenizer

    model = rows[0].get("model")
    config = model_config(str(model))
    tokenizer = AutoTokenizer.from_pretrained(config["local_path"])
    for row in rows:
        counts = row.get("response_token_counts")
        if len(counts or []) == 8:
            continue
        row["response_token_counts"] = [
            len(tokenizer(response, add_special_tokens=False).input_ids)
            for response in row.get("responses", [])
        ]


def _assert_pair(fast: list[dict[str, Any]], reference: list[dict[str, Any]]) -> None:
    if len(fast) != len(reference):
        raise ValueError(f"fast/reference row count differs: {len(fast)} != {len(reference)}")
    for fast_row, reference_row in zip(fast, reference):
        for field in ("dataset", "problem_id", "responses", "correctness", "response_token_counts"):
            if fast_row.get(field) != reference_row.get(field):
                raise ValueError(
                    f"fast/reference mismatch in {field} for {fast_row.get('problem_id')}"
                )


def _clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": str(row.get("dataset", "unknown")),
            "problem_id": str(row.get("problem_id")),
            "rows": [row],
        }
        for row in rows
    ]


def _subtract(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if not xs or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x == 0 or var_y == 0:
        return 1.0 if all(x == y for x, y in zip(xs, ys)) else 0.0
    return float(
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        / math.sqrt(var_x * var_y)
    )


def _score_difference(
    fast_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], condition: str
) -> dict[str, Any]:
    fast_values: list[float] = []
    reference_values: list[float] = []
    nulls_fast = 0
    nulls_reference = 0
    for fast_row, reference_row in zip(fast_rows, reference_rows):
        for fast_item, reference_item in zip(
            fast_row["readout_scores"], reference_row["readout_scores"]
        ):
            fast_value = score_value(fast_item, condition)
            reference_value = score_value(reference_item, condition)
            nulls_fast += int(fast_value is None)
            nulls_reference += int(reference_value is None)
            if (fast_value is None) != (reference_value is None):
                raise ValueError(f"one-sided null for {condition}")
            if fast_value is not None:
                fast_values.append(float(fast_value))
                reference_values.append(float(reference_value))
    differences = [abs(x - y) for x, y in zip(fast_values, reference_values)]
    return {
        "n_pairs": len(fast_values),
        "nulls_fast": nulls_fast,
        "nulls_vllm": nulls_reference,
        "mean_absolute_difference": (
            float(sum(differences) / len(differences)) if differences else None
        ),
        "max_absolute_difference": float(max(differences)) if differences else None,
        "pearson_r": _pearson(fast_values, reference_values),
    }


def _auroc_report(
    fast_clusters: list[dict[str, Any]],
    reference_clusters: list[dict[str, Any]],
    conditions: list[str],
) -> dict[str, Any]:
    fast_by_benchmark = {}
    reference_by_benchmark = {}
    benchmarks = sorted({cluster["dataset"] for cluster in fast_clusters})
    for benchmark in benchmarks:
        fast_by_benchmark[benchmark] = [
            cluster for cluster in fast_clusters if cluster["dataset"] == benchmark
        ]
        reference_by_benchmark[benchmark] = [
            cluster for cluster in reference_clusters if cluster["dataset"] == benchmark
        ]

    def one_scope(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
        result = {}
        for condition in conditions:
            left_metric = metric(left, condition)
            right_metric = metric(right, condition)
            result[condition] = {
                "fast": left_metric.get("primary_auroc"),
                "vllm": right_metric.get("primary_auroc"),
                "difference_fast_minus_vllm": _subtract(
                    left_metric.get("primary_auroc"), right_metric.get("primary_auroc")
                ),
                "pooled_secondary": {
                    "fast": left_metric.get("auroc_pooled"),
                    "vllm": right_metric.get("auroc_pooled"),
                    "difference_fast_minus_vllm": _subtract(
                        left_metric.get("auroc_pooled"), right_metric.get("auroc_pooled")
                    ),
                },
            }
        return result

    return {
        "primary_all_benchmarks": one_scope(fast_clusters, reference_clusters),
        "by_benchmark": {
            benchmark: one_scope(fast_by_benchmark[benchmark], reference_by_benchmark[benchmark])
            for benchmark in benchmarks
        },
    }


def _contrast_report(
    fast_clusters: list[dict[str, Any]],
    reference_clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    benchmarks = sorted({cluster["dataset"] for cluster in fast_clusters})
    scopes = {
        "primary_all_benchmarks": (fast_clusters, reference_clusters),
    }
    for benchmark in benchmarks:
        scopes[f"benchmark:{benchmark}"] = (
            [cluster for cluster in fast_clusters if cluster["dataset"] == benchmark],
            [cluster for cluster in reference_clusters if cluster["dataset"] == benchmark],
        )
    report = {}
    for scope, (left, right) in scopes.items():
        fast_values = contrast_values(left)
        reference_values = contrast_values(right)
        report[scope] = {
            contrast: {
                "fast": fast_values.get(contrast),
                "vllm": reference_values.get(contrast),
                "difference_fast_minus_vllm": _subtract(
                    fast_values.get(contrast), reference_values.get(contrast)
                ),
                "a1_equivalence_bound": EQUIVALENCE_BOUNDS[contrast],
            }
            for contrast in CONTRASTS
        }
    return report


def _verify_pilot_equivalence(report: dict[str, Any], path: str) -> dict[str, Any]:
    expected = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = {
        "R0": "R0_reference",
        "R1_0": "R1_marker_swap_token_128003",
        "R1_1": "R1_marker_swap_token_128004",
        "R1_2": "R1_marker_swap_token_128005",
        "R2": "R2_position_eos",
        "R3": "R3_position_answer",
        "R4": "R4_cref_none",
        "R5": "R5_cref_recomputed",
    }
    actual = report["auroc"]["primary_all_benchmarks"]
    checks = {}
    passed = True
    for key, condition in mapping.items():
        item = actual[condition]
        expected_item = expected["readouts"][key]
        checks[key] = {
            "expected_fast": expected_item["auroc_fast"],
            "actual_fast": item["fast"],
            "expected_vllm": expected_item["auroc_reference"],
            "actual_vllm": item["vllm"],
            "fast_match": math.isclose(item["fast"], expected_item["auroc_fast"], abs_tol=1e-12),
            "vllm_match": math.isclose(item["vllm"], expected_item["auroc_reference"], abs_tol=1e-12),
        }
        passed = passed and checks[key]["fast_match"] and checks[key]["vllm_match"]
    result = {"path": path, "pass": passed, "readouts": checks}
    report["pilot_equivalence_check"] = result
    if not passed:
        raise SystemExit("pilot AUROC reproduction failed")
    return result


def _markdown(report: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.6g}"

    lines = [
        "# Readout cross-check: transformers vs. vLLM",
        "",
        "OctoThinker-3B-Short-LaSeR, seed 0; score-scale comparison of the fast readout and the vLLM reference.",
        "",
        "## Score agreement",
        "",
        "| Readout | mean abs diff | max abs diff | Pearson r | pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, item in report["score_scale_agreement"].items():
        lines.append(
            f"| {condition} | {fmt(item['mean_absolute_difference'])} | "
            f"{fmt(item['max_absolute_difference'])} | {fmt(item['pearson_r'])} | {item['n_pairs']} |"
        )
    lines.extend(["", "## Primary AUROC", ""])
    lines.append("| Scope / readout | Fast | vLLM | Fast − vLLM |")
    lines.append("|---|---:|---:|---:|")
    primary = report["auroc"]["primary_all_benchmarks"]
    for condition, item in primary.items():
        lines.append(
            f"| primary / {condition} | {fmt(item['fast'])} | {fmt(item['vllm'])} | "
            f"{fmt(item['difference_fast_minus_vllm'])} |"
        )
    lines.extend(["", "## A1 contrasts", ""])
    lines.append("| Scope / contrast | Fast | vLLM | Fast − vLLM | A1 bound |")
    lines.append("|---|---:|---:|---:|---:|")
    for scope, contrasts in report["contrasts"].items():
        for contrast, item in contrasts.items():
            fast = "n/a" if item["fast"] is None else f"{item['fast']:.6g}"
            vllm = "n/a" if item["vllm"] is None else f"{item['vllm']:.6g}"
            diff = "n/a" if item["difference_fast_minus_vllm"] is None else f"{item['difference_fast_minus_vllm']:.6g}"
            lines.append(
                f"| {scope} / {contrast} | {fast} | {vllm} | {diff} | ±{item['a1_equivalence_bound']:.6g} |"
            )
    if report.get("pilot_equivalence_check"):
        lines.extend(["", f"Pilot AUROC reproduction: **{report['pilot_equivalence_check']['pass']}**.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="append", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-equivalence")
    args = parser.parse_args()
    if len(args.fast) != len(args.reference):
        raise ValueError("pass the same number of --fast and --reference files")

    fast_rows = _read_paths(args.fast, "fast")
    reference_rows = _read_paths(args.reference, "vLLM")
    _assert_pair(fast_rows, reference_rows)
    fast_clusters = _clusters(fast_rows)
    reference_clusters = _clusters(reference_rows)
    conditions = CONDITION_LABELS + _token_conditions(fast_rows)
    reference_conditions = CONDITION_LABELS + _token_conditions(reference_rows)
    if conditions != reference_conditions:
        raise ValueError(f"fast/vLLM R1 token sets differ: {conditions} != {reference_conditions}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vllm_crosscheck",
        "model": next(iter({row["model"] for row in fast_rows})),
        "model_revision": next(iter({row["model_revision"] for row in fast_rows})),
        "seed": 0,
        "benchmarks": sorted({cluster["dataset"] for cluster in fast_clusters}),
        "fast_inputs": args.fast,
        "vllm_inputs": args.reference,
        "readout_conditions": conditions,
        "score_scale": "readout_scores emitted by the registered scorer (beta_v=0.1)",
        "auroc": _auroc_report(fast_clusters, reference_clusters, conditions),
        "score_scale_agreement": {
            condition: _score_difference(fast_rows, reference_rows, condition)
            for condition in conditions
        },
        "contrasts": _contrast_report(fast_clusters, reference_clusters),
        "inference_stacks": {
            "fast": fast_rows[0]["readout_stack"],
            "vllm": reference_rows[0]["readout_stack"],
        },
    }
    if args.expected_equivalence:
        _verify_pilot_equivalence(report, args.expected_equivalence)

    output = Path(args.output)
    write_json(output, report)
    output.with_name(output.name + ".sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(report), encoding="utf-8")
    markdown.with_name(markdown.name + ".sha256").write_text(
        f"{sha256_file(markdown)}  {markdown.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
