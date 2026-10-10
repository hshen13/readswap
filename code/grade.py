#!/usr/bin/env python3
"""Apply the LaSeR math_verify boxed-answer grader to generated JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    read_jsonl,
    released_last_boxed_only_string,
    released_remove_boxed,
    released_response,
    write_artifact_metadata,
    write_jsonl,
)


def boxed_answer(text: str) -> str | None:
    return released_remove_boxed(released_last_boxed_only_string(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from math_verify import parse, verify

    rows = []
    for row in read_jsonl(args.input):
        expected = row.get("answer")
        normalized_responses = []
        response_answers = []
        correctness = []
        for raw_response in row["responses"]:
            response = released_response(raw_response)
            normalized_responses.append(response)
            pred = boxed_answer(response)
            response_answers.append(pred)
            if pred is None or expected is None:
                correctness.append(False)
                continue
            try:
                correctness.append(
                    bool(verify(parse(r"\boxed{" + str(expected) + "}"), parse(r"\boxed{" + pred + "}")))
                )
            except Exception:
                correctness.append(False)
        out = dict(row)
        out["responses"] = normalized_responses
        out["pred_answers"] = response_answers
        out["correctness"] = correctness
        out["grader"] = "math_verify (LaSeR eval_math.py)"
        out["response_surface"] = "released_response_v1"
        rows.append(out)
    n = write_jsonl(args.output, rows)
    write_artifact_metadata(
        args.output,
        {
            "kind": "graded",
            "rows": n,
            "grader": "math_verify",
            "response_surface": "released_response_v1",
            "answer_extractor": "released_last_boxed_only_string/remove_boxed",
        },
    )


if __name__ == "__main__":
    main()
