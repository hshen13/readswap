#!/usr/bin/env python3
"""Fill R5 in an existing R0 to R4 readout artifact in a fresh GPU process."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

from common import (
    model_config,
    parse_runtime_version,
    read_jsonl,
    released_response,
    write_artifact_metadata,
    write_jsonl,
)
from readout import _reap_children_and_exit, score_batch, sequence_for, unload


def special_token_ids(tokenizer):
    """Return every special token and its id in a stable, comparable form."""
    tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    for value in (getattr(tokenizer, "special_tokens_map", {}) or {}).values():
        if isinstance(value, (list, tuple)):
            tokens.update(str(item) for item in value)
        elif value is not None:
            tokens.add(str(value))
    return {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in sorted(tokens)
    }


def assert_tokenizer_compatibility(policy_tok, base_tok):
    """R5 scores policy-tokenizer ids with the base model only if ids agree."""
    policy_vocab = policy_tok.get_vocab()
    base_vocab = base_tok.get_vocab()
    policy_special = special_token_ids(policy_tok)
    base_special = special_token_ids(base_tok)
    vocabularies_identical = policy_vocab == base_vocab
    special_ids_identical = policy_special == base_special
    check = {
        "vocabularies_identical": vocabularies_identical,
        "special_token_ids_identical": special_ids_identical,
        "policy_vocab_size": len(policy_vocab),
        "base_vocab_size": len(base_vocab),
        "policy_special_token_ids": policy_special,
        "base_special_token_ids": base_special,
    }
    if not vocabularies_identical or not special_ids_identical:
        raise RuntimeError(
            "R5 requires identical checkpoint/base vocabularies and special-token ids: "
            + repr(check)
        )
    return check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--base-model-path", default=None)
    args = ap.parse_args()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM

    cfg = model_config(args.model)
    base_path = args.base_model_path or cfg.get("base_model_local_path")
    if not base_path or not Path(base_path).exists():
        raise RuntimeError("R5 base model snapshot is unavailable; no substitute is allowed")
    rows = read_jsonl(args.input)
    policy_tok = AutoTokenizer.from_pretrained(cfg["local_path"])
    base_tok = AutoTokenizer.from_pretrained(base_path)
    tokenizer_check = assert_tokenizer_compatibility(policy_tok, base_tok)
    started = time.perf_counter()
    base_llm = LLM(
        model=base_path,
        tokenizer=base_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.65,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
    )
    sequences, targets, keys = [], [], []
    marker = int(cfg["marker_token_id"])
    for row_i, row in enumerate(rows):
        for response_i, response in enumerate(row["responses"]):
            # The base checkpoint has no chat template.  Construct the exact
            # LaSeR/R0 sequence with the checkpoint tokenizer, then pass those
            # token ids directly to the base model.
            sequences.append(sequence_for(policy_tok, row["problem"], response, cfg, marker, True))
            targets.append(marker)
            keys.append((row_i, response_i))
    values = score_batch(base_llm, sequences, targets, args.batch_size)
    unload(base_llm)
    r5_map = dict(zip(keys, values))
    ratio = float(cfg["ratio_coef"])
    out = []
    for row_i, row in enumerate(rows):
        item = dict(row)
        item["responses"] = [released_response(response) for response in row["responses"]]
        scores = []
        logs = []
        for response_i, existing in enumerate(row["readout_scores"]):
            key = (row_i, response_i)
            updated = dict(existing)
            policy_logprob = row["readout_logprobs"][response_i].get("R0_marker")
            base_logprob = r5_map.get(key)
            updated["R5_cref_recomputed"] = (
                (policy_logprob - base_logprob) * ratio
                if policy_logprob is not None and base_logprob is not None
                else None
            )
            scores.append(updated)
            log_item = dict(row["readout_logprobs"][response_i])
            log_item["R5_base_marker"] = base_logprob
            logs.append(log_item)
        item["readout_scores"] = scores
        item["readout_logprobs"] = logs
        stack = dict(item.get("readout_stack", {}))
        stack["r5_base_model"] = {
            "model_id": cfg["base_model_repo_id"],
            "revision": cfg["base_model_revision"],
            "local_path": str(base_path),
            "engine_version": parse_runtime_version("vllm"),
        }
        stack["r5_tokenizer_compatibility"] = tokenizer_check
        item["readout_stack"] = stack
        out.append(item)
    output = Path(args.output)
    n = write_jsonl(output, out)
    write_artifact_metadata(
        output,
        {
            "kind": "readout_with_r5",
            "rows": n,
            "responses": len(keys),
            "elapsed_seconds": time.perf_counter() - started,
            "model": cfg["model_id"],
            "base_model": cfg["base_model_repo_id"],
        },
    )


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        _reap_children_and_exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    _reap_children_and_exit(0)
