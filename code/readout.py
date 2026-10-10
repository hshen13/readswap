#!/usr/bin/env python3
"""Compute the six preregistered readout surfaces on existing responses."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

from common import (
    append_marker_ids,
    last_boxed_span,
    model_config,
    parse_runtime_version,
    prompt_text,
    read_jsonl,
    released_response,
    write_artifact_metadata,
    write_jsonl,
)


def sequence_for(
    tokenizer: Any,
    problem: str,
    response: str,
    config: dict[str, Any],
    marker_id: int,
    include_eos: bool,
    normalize: bool = True,
) -> list[int]:
    """Build the exact full chat-template sequence used by the released scorer."""
    user = problem + config.get("prompt_template", {}).get("problem_suffix", "")
    assistant = released_response(response) if normalize else response
    if include_eos and tokenizer.eos_token:
        assistant += tokenizer.eos_token
    assistant += tokenizer.decode([marker_id], clean_up_tokenization_spaces=False)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        add_generation_prompt=False,
        tokenize=False,
    ).strip()
    if full[-10:] == "<|im_end|>":
        full = full[:-len("<|im_end|>")]
    return tokenizer(full, add_special_tokens=False).input_ids


def score_batch(llm: Any, sequences: list[list[int]], targets: list[int], batch_size: int) -> list[float]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        n=1,
        seed=42,
        prompt_logprobs=1,
        logprobs=1,
        ignore_eos=True,
    )
    values: list[float] = []
    for start in range(0, len(sequences), batch_size):
        batch_seq = sequences[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        outputs = llm.generate(prompt_token_ids=batch_seq, sampling_params=sampling)
        for output, target in zip(outputs, batch_targets):
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if not prompt_logprobs:
                raise RuntimeError("vLLM returned no prompt_logprobs")
            entry = prompt_logprobs[-1] or {}
            item = entry.get(target)
            if item is None:
                item = entry.get(str(target))
            if item is None:
                raise RuntimeError(f"target token {target} absent from final prompt_logprobs")
            values.append(float(item.logprob if hasattr(item, "logprob") else item["logprob"]))
    return values


def unload(llm: Any) -> None:
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    # vLLM V0 can release its CUDA blocks asynchronously after the Python
    # object is deleted.  Give the driver time to reclaim them before R5
    # loads the base checkpoint in the same locked GPU slot.
    time.sleep(15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="graded or generated JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--base-model-path", default=None)
    ap.add_argument("--skip-r5", action="store_true")
    args = ap.parse_args()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM

    cfg = model_config(args.model)
    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(cfg["local_path"])
    marker = int(cfg["marker_token_id"])
    swaps = [int(x["id"]) for x in cfg["r1_swap_tokens"]]
    flat: list[tuple[int, int, str]] = []
    for row_i, row in enumerate(rows):
        for response_i, response in enumerate(row["responses"]):
            flat.append((row_i, response_i, released_response(response)))

    def build(include_eos: bool, marker_ids: list[int], truncate: bool = False):
        seqs, targets = [], []
        valid = []
        for row_i, response_i, response in flat:
            text = response
            if truncate:
                text = response[: response.rfind(last_boxed_span(response)) + len(last_boxed_span(response))] if last_boxed_span(response) else None
            if text is None:
                continue
            m = marker_ids[len(seqs) % len(marker_ids)] if len(marker_ids) > 1 else marker_ids[0]
            seqs.append(sequence_for(tokenizer, rows[row_i]["problem"], text, cfg, m, include_eos))
            targets.append(m)
            valid.append((row_i, response_i))
        return seqs, targets, valid

    started = time.perf_counter()
    llm = LLM(
        model=cfg["local_path"],
        tokenizer=cfg["local_path"],
        dtype="bfloat16",
        gpu_memory_utilization=0.65,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
    )
    r0_seq, r0_targets, r0_valid = build(True, [marker])
    r0_raw = score_batch(llm, r0_seq, r0_targets, args.batch_size)
    r1_values: dict[int, list[float]] = {}
    for swap in swaps:
        seq, targets, valid = build(True, [swap])
        vals = score_batch(llm, seq, targets, args.batch_size)
        r1_values[swap] = vals
    r2_seq, r2_targets, r2_valid = build(False, [marker])
    r2_values = score_batch(llm, r2_seq, r2_targets, args.batch_size)
    r3_seq, r3_targets, r3_valid = build(False, [marker], truncate=True)
    r3_values = score_batch(llm, r3_seq, r3_targets, args.batch_size)
    unload(llm)

    base_path = args.base_model_path or cfg.get("base_model_local_path")
    r5_map: dict[tuple[int, int], float] = {}
    if not args.skip_r5 and base_path and Path(base_path).exists():
        base_tok = AutoTokenizer.from_pretrained(base_path)
        base_llm = LLM(
            model=base_path,
            tokenizer=base_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.65,
            tensor_parallel_size=max(1, torch.cuda.device_count()),
        )
        seq, targets, valid = [], [], []
        for row_i, response_i, response in flat:
            seq.append(sequence_for(base_tok, rows[row_i]["problem"], response, cfg, marker, True))
            targets.append(marker)
            valid.append((row_i, response_i))
        for key, value in zip(valid, score_batch(base_llm, seq, targets, args.batch_size)):
            r5_map[key] = value
        unload(base_llm)

    def map_values(keys, values):
        return {key: value for key, value in zip(keys, values)}

    r0_map = map_values(r0_valid, r0_raw)
    r2_map = map_values(r2_valid, r2_values)
    r3_map = map_values(r3_valid, r3_values)
    r1_maps = {swap: map_values(r0_valid, vals) for swap, vals in r1_values.items()}
    out = []
    for row_i, row in enumerate(rows):
        item = dict(row)
        item["responses"] = [released_response(response) for response in row["responses"]]
        scores = []
        for response_i, _response in enumerate(row["responses"]):
            key = (row_i, response_i)
            r1_raw = {str(swap): r1_maps[swap].get(key) for swap in swaps}
            r0_raw_value = r0_map.get(key)
            r2_raw_value = r2_map.get(key)
            r3_raw_value = r3_map.get(key)
            ratio = float(cfg["ratio_coef"])
            cref = float(cfg["reference_constant"])
            r1 = {k: (v - cref) * ratio if v is not None else None for k, v in r1_raw.items()}
            scores.append(
                {
                    "R0_reference": (r0_raw_value - cref) * ratio if r0_raw_value is not None else None,
                    "R1_marker_swap": r1,
                    "R1_marker_swap_mean": sum(x for x in r1.values() if x is not None) / len([x for x in r1.values() if x is not None]) if any(x is not None for x in r1.values()) else None,
                    "R2_position_eos": (r2_raw_value - cref) * ratio if r2_raw_value is not None else None,
                    "R3_position_answer": (r3_raw_value - cref) * ratio if r3_raw_value is not None else None,
                    "R4_cref_none": r0_raw_value * ratio if r0_raw_value is not None else None,
                    "R5_cref_recomputed": (r0_raw_value - r5_map[key]) * ratio if key in r5_map and r0_raw_value is not None else None,
                }
            )
        item["readout_scores"] = scores
        item["readout_logprobs"] = [
            {
                "R0_marker": r0_map.get((row_i, i)),
                "R1_markers": {str(swap): r1_maps[swap].get((row_i, i)) for swap in swaps},
                "R2_marker": r2_map.get((row_i, i)),
                "R3_marker": r3_map.get((row_i, i)),
                "R5_base_marker": r5_map.get((row_i, i)),
            }
            for i in range(len(row["responses"]))
        ]
        item["readout_stack"] = {
            "engine": "vllm",
            "engine_version": parse_runtime_version("vllm"),
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.65,
            "marker_token_id": marker,
            "reference_constant": cfg["reference_constant"],
        }
        item["marker_token_id"] = marker
        item["marker_token"] = cfg.get("marker_token")
        item["reference_constant"] = cfg["reference_constant"]
        out.append(item)
    output = Path(args.output)
    n = write_jsonl(output, out)
    write_artifact_metadata(
        output,
        {"kind": "readout", "rows": n, "responses": len(flat), "elapsed_seconds": time.perf_counter() - started, "model": cfg["model_id"]},
    )


def _reap_children_and_exit(code: int) -> None:
    """Reap vLLM's EngineCore before hard-exiting the process.

    vLLM can leave its engine child alive during normal interpreter teardown;
    that orphan keeps holding GPU memory.  Hard exit is intentional here:
    callers may run under an exclusive GPU wrapper (GPU_RUN), so the child
    must be gone before the wrapper releases the GPU.
    """
    import os as _os_r

    try:
        import psutil as _ps

        children = _ps.Process().children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        _ps.wait_procs(children, timeout=30)
    except Exception:
        import multiprocessing as _mp

        for child in _mp.active_children():
            try:
                child.kill()
            except Exception:
                pass
    _os_r._exit(code)


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
