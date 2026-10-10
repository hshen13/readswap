#!/usr/bin/env python3
"""Compare our R0 scores with the released calculate_self_rewarding_scores output."""

import argparse
import json
import statistics
from pathlib import Path

from common import model_config, read_jsonl, released_response, sha256_file, write_json
from readout import sequence_for


def released_sequence_ids(tokenizer, problem: str, response: str, config: dict, marker: int) -> list[int]:
    """Build the exact full sequence used by released scorer's helper."""

    user = problem + config.get("prompt_template", {}).get("problem_suffix", "")
    solution = released_response(response) + tokenizer.eos_token
    solution += tokenizer.decode([marker], clean_up_tokenization_spaces=False)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": solution}],
        add_generation_prompt=False,
        tokenize=False,
    ).strip()
    if text[-10:] == "<|im_end|>":
        text = text[:-10]
    return tokenizer(text, add_special_tokens=False).input_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pilot", default=None, help="Full pilot readout used for R0 spread.")
    args = ap.parse_args()
    ours, official = read_jsonl(args.ours), read_jsonl(args.official)
    if len(ours) != len(official):
        raise RuntimeError(f"row count differs: {len(ours)} vs {len(official)}")
    if not ours:
        raise RuntimeError("empty official comparison input")
    cfg = model_config(ours[0].get("model", ""))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["local_path"])
    marker = int(cfg["marker_token_id"])
    comparisons = []
    token_id_mismatches = []
    for ours_row, official_row in zip(ours, official):
        # make_unit_subset.py keeps the selected response index on the copied
        # row. It is not always zero when earlier responses lack a boxed
        # answer, so compare the same response that the official scorer saw.
        unit_index = int(ours_row.get("unit_response_index", 0))
        response_index = 0 if len(ours_row["responses"]) == 1 else unit_index
        score_index = 0 if len(ours_row["readout_scores"]) == 1 else unit_index
        ours_response = released_response(ours_row["responses"][response_index])
        official_response = released_response(official_row["responses"][0])
        if ours_response != official_response:
            raise RuntimeError("official scorer input response differs after normalization")
        ours_ids = sequence_for(
            tokenizer, ours_row["problem"], ours_response, cfg, marker, include_eos=True
        )
        released_ids = released_sequence_ids(tokenizer, official_row["problem"], official_response, cfg, marker)
        if ours_ids != released_ids:
            token_id_mismatches.append(ours_row.get("problem_id"))
        ours_score = ours_row["readout_scores"][score_index]["R0_reference"]
        official_score = official_row["self_rewarding_scores"][0]
        comparisons.append(
            {
                "problem_id": ours_row.get("problem_id"),
                "ours_r0": ours_score,
                "official_r0": official_score,
                "absolute_error": abs(float(ours_score) - float(official_score)),
            }
        )
    max_error = max(x["absolute_error"] for x in comparisons) if comparisons else None
    if args.pilot:
        pilot_rows = read_jsonl(args.pilot)
        pilot_scores = [
            float(item["R0_reference"])
            for row in pilot_rows
            for item in row.get("readout_scores", [])
            if item.get("R0_reference") is not None
        ]
    else:
        pilot_scores = [float(x["ours_r0"]) for x in comparisons]
    r0_stddev = statistics.pstdev(pilot_scores) if len(pilot_scores) >= 2 else None
    result = {
        "n": len(comparisons),
        "max_absolute_error": max_error,
        "score_tolerance": 0.01,
        "token_id_match": not token_id_mismatches,
        "token_id_mismatches": token_id_mismatches,
        "normalized_surface": True,
        "pilot_r0_standard_deviation": r0_stddev,
        "max_error_to_pilot_r0_standard_deviation": (
            max_error / r0_stddev if max_error is not None and r0_stddev else None
        ),
        "comparisons": comparisons,
    }
    write_json(args.output, result)
    Path(args.output + ".sha256").write_text(
        f"{sha256_file(args.output)}  {Path(args.output).name}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "n": len(comparisons),
        "max_absolute_error": max_error,
        "token_id_match": not token_id_mismatches,
        "max_error_to_pilot_r0_standard_deviation": result["max_error_to_pilot_r0_standard_deviation"],
    }))


if __name__ == "__main__":
    main()
