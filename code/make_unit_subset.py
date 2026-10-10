#!/usr/bin/env python3
"""Make the first 20 valid pilot solutions a one-response-per-problem scorer input."""

import argparse
from pathlib import Path

from common import read_jsonl, released_response, write_jsonl
from grade import boxed_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = read_jsonl(args.input)
    selected = []
    for row in rows:
        for i, (raw_response, _answer, correct) in enumerate(
            zip(row["responses"], row.get("pred_answers", []), row.get("correctness", []))
        ):
            response = released_response(raw_response)
            answer = boxed_answer(response)
            # A unit row is valid only when the released final-box extractor
            # produced an answer; correctness is intentionally not the filter.
            if answer is None:
                continue
            item = dict(row)
            item["responses"] = [response]
            item["pred_answers"] = [answer]
            item["correctness"] = [correct]
            if "readout_scores" in row:
                item["readout_scores"] = [row["readout_scores"][i]]
            if "readout_logprobs" in row:
                item["readout_logprobs"] = [row["readout_logprobs"][i]]
            item["unit_response_index"] = i
            selected.append(item)
            if len(selected) == 20:
                write_jsonl(args.output, selected)
                return
    raise RuntimeError(f"only found {len(selected)} responses")


if __name__ == "__main__":
    main()
