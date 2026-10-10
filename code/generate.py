#!/usr/bin/env python3
"""Generate shared 8-response samples for one checkpoint/dataset/seed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (
    ROOT,
    inference_stack,
    model_config,
    parse_runtime_version,
    prompt_ids,
    read_jsonl,
    write_artifact_metadata,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="config key or Hugging Face id")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    args = ap.parse_args()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cfg = model_config(args.model)
    input_rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(cfg["local_path"])
    prompts = [prompt_ids(tokenizer, row["problem"], cfg) for row in input_rows]
    started = time.perf_counter()
    llm = LLM(
        model=cfg["local_path"],
        tokenizer=cfg["local_path"],
        dtype="bfloat16",
        gpu_memory_utilization=0.65,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=8192, n=8, seed=args.seed
    )
    generations = llm.generate(prompt_token_ids=prompts, sampling_params=sampling)
    elapsed = time.perf_counter() - started
    stack = inference_stack(cfg)
    stack["engine_version"] = parse_runtime_version("vllm")
    stack["batching"]["max_num_seqs"] = args.max_num_seqs
    rows = []
    for row, result in zip(input_rows, generations):
        outputs = [item.text for item in result.outputs]
        token_counts = []
        finish_reasons = []
        stop_reasons = []
        for item in result.outputs:
            token_ids = getattr(item, "token_ids", None)
            if token_ids is None:
                raise RuntimeError("vLLM output did not expose token_ids")
            token_counts.append(len(token_ids))
            finish_reasons.append(getattr(item, "finish_reason", None))
            stop_reasons.append(getattr(item, "stop_reason", None))
        rows.append(
            {
                "dataset": args.dataset,
                "problem_id": row.get("id", row.get("unique_id")),
                "problem": row["problem"],
                "answer": row.get("answer"),
                "responses": outputs,
                "response_token_counts": token_counts,
                "finish_reasons": finish_reasons,
                "stop_reasons": stop_reasons,
                "truncated": [reason == "length" or count >= 8192 for reason, count in zip(finish_reasons, token_counts)],
                "model": cfg["model_id"],
                "model_revision": cfg["revision"],
                "seed": args.seed,
                "inference_stack": stack,
            }
        )
    output = Path(args.output).resolve()
    n = write_jsonl(output, rows)
    write_artifact_metadata(
        output,
        {
            "artifact": str(output.relative_to(ROOT)),
            "kind": "generation",
            "rows": n,
            "responses": n * 8,
            "elapsed_seconds": elapsed,
            "model": cfg["model_id"],
            "dataset": args.dataset,
            "seed": args.seed,
            "inference_stack": stack,
        },
    )


if __name__ == "__main__":
    main()
