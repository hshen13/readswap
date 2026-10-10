#!/usr/bin/env python3
"""Single-pass HF readout for the preregistered LaSeR surfaces.

The final marker (or swap token) is included in each token sequence only as
the scored target.  The model is run on the full R0 sequence once; logits are
formed only for the hidden states needed by R0/R1/R2/R3.  Non-prefix cases are
scored in small fallback batches.  R5 is one separate pass through the base
model on the exact R0 token ids of the checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import gc
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import (
    model_config,
    parse_runtime_version,
    read_jsonl,
    released_response,
    truncate_at_last_boxed,
    write_artifact_metadata,
    write_json,
    write_jsonl,
)
from readout import sequence_for
from readout_r5 import assert_tokenizer_compatibility


def is_token_prefix_with_target(candidate: list[int], full: list[int]) -> bool:
    """Whether candidate's context is a prefix of full's context.

    Both sequences end in their scored target.  The final target is therefore
    excluded from the prefix comparison, while the candidate target position
    remains available as a query into the full forward pass.
    """

    if len(candidate) < 2 or len(candidate) > len(full):
        return False
    return candidate[:-1] == full[: len(candidate) - 1]


def iter_token_batches(
    items: Iterable[tuple[Any, list[int]]], max_batch_tokens: int
) -> Iterable[list[tuple[Any, list[int]]]]:
    """Yield right-padding batches bounded by padded token count."""

    ordered = sorted(items, key=lambda item: (len(item[1]), repr(item[0])))
    batch: list[tuple[Any, list[int]]] = []
    max_len = 0
    for item in ordered:
        length = len(item[1])
        proposed_max = max(max_len, length)
        proposed_size = proposed_max * (len(batch) + 1)
        if batch and proposed_size > max_batch_tokens:
            yield batch
            batch = []
            max_len = 0
        batch.append(item)
        max_len = max(max_len, length)
        if len(batch) == 1 and max_len > max_batch_tokens:
            yield batch
            batch = []
            max_len = 0
    if batch:
        yield batch


def score_sequences(
    model: Any,
    sequences: list[tuple[Any, list[int]]],
    queries: list[tuple[Any, Any, int, list[int]]],
    *,
    tokenizer: Any,
    device: Any,
    max_batch_tokens: int,
) -> dict[Any, list[float]]:
    """Score selected next-token distributions from batched final states.

    ``queries`` contains (query_key, sequence_key, hidden_position,
    target_ids).  The hidden position is the token immediately before the
    target in the unpadded sequence.  Only those positions reach the LM head
    and log-softmax.
    """

    import torch

    sequence_map = dict(sequences)
    by_sequence: dict[Any, list[tuple[Any, int, list[int]]]] = defaultdict(list)
    for query_key, sequence_key, position, target_ids in queries:
        if sequence_key not in sequence_map:
            raise KeyError(f"query refers to missing sequence {sequence_key!r}")
        if position < 0 or position >= len(sequence_map[sequence_key]) - 1:
            raise ValueError(
                f"invalid next-token position {position} for sequence "
                f"{sequence_key!r} of length {len(sequence_map[sequence_key])}"
            )
        by_sequence[sequence_key].append((query_key, position, target_ids))

    results: dict[Any, list[float]] = {}
    backbone = getattr(model, "model", None)
    if backbone is None:
        backbone = getattr(model, "base_model", None)
    if backbone is None:
        raise RuntimeError("cannot locate the causal-LM backbone")
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("causal LM has no output embedding / lm_head")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    with torch.inference_mode():
        for batch in iter_token_batches(sequences, max_batch_tokens):
            max_len = max(len(ids) for _, ids in batch)
            input_ids = torch.full(
                (len(batch), max_len), int(pad_id), dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(batch), max_len), dtype=torch.long, device=device
            )
            local_keys = []
            for row_i, (sequence_key, ids) in enumerate(batch):
                values = torch.tensor(ids, dtype=torch.long, device=device)
                input_ids[row_i, : len(ids)] = values
                attention_mask[row_i, : len(ids)] = 1
                local_keys.append(sequence_key)

            base_output = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = base_output.last_hidden_state
            selected: list[Any] = []
            metadata: list[tuple[Any, list[int]]] = []
            local_index = {key: i for i, key in enumerate(local_keys)}
            for sequence_key in local_keys:
                for query_key, position, target_ids in by_sequence[sequence_key]:
                    selected.append(hidden[local_index[sequence_key], position])
                    metadata.append((query_key, target_ids))
            if selected:
                selected_hidden = torch.stack(selected, dim=0)
                logits = lm_head(selected_hidden).float()
                log_probs = torch.log_softmax(logits, dim=-1)
                for row_i, (query_key, target_ids) in enumerate(metadata):
                    ids = torch.tensor(target_ids, dtype=torch.long, device=device)
                    values = log_probs[row_i].gather(0, ids)
                    results[query_key] = [float(x) for x in values.cpu().tolist()]
                del selected_hidden, logits, log_probs
            del hidden, base_output, input_ids, attention_mask

    return results


def unload_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    except Exception:
        pass
    # Let the CUDA allocator and model workers settle before loading the base.
    time.sleep(5)


def load_model(path: str, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to("cuda")
    model.eval()
    return model


def validate_pairs(inputs: list[str], outputs: list[str], r04_outputs: list[str]) -> None:
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("--input and --output must be supplied in equal nonzero counts")
    if r04_outputs and len(r04_outputs) != len(inputs):
        raise ValueError("--r0-r4-output must have one path per --input")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--output", action="append", required=True)
    ap.add_argument("--r0-r4-output", action="append", default=[])
    ap.add_argument("--base-model-path", default=None)
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--skip-r5", action="store_true")
    ap.add_argument(
        "--raw-surface",
        action="store_true",
        help="Do not apply released_response normalization (pilot equivalence only).",
    )
    args = ap.parse_args()
    validate_pairs(args.input, args.output, args.r0_r4_output)
    if args.max_batch_tokens <= 0:
        raise ValueError("--max-batch-tokens must be positive")

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("readout_fast.py requires a CUDA GPU")
    device = torch.device("cuda")
    cfg = model_config(args.model)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(cfg["local_path"])
    tokenizer.padding_side = "right"
    marker = int(cfg["marker_token_id"])
    swaps = [int(x["id"]) for x in cfg["r1_swap_tokens"]]
    input_rows = [read_jsonl(path) for path in args.input]

    records: list[dict[str, Any]] = []
    for dataset_i, rows in enumerate(input_rows):
        for row_i, row in enumerate(rows):
            for response_i, response in enumerate(row["responses"]):
                response = response if args.raw_surface else released_response(response)
                r0 = sequence_for(
                    tokenizer, row["problem"], response, cfg, marker,
                    include_eos=True, normalize=False,
                )
                r1 = {
                    swap: sequence_for(
                        tokenizer, row["problem"], response, cfg, swap,
                        include_eos=True, normalize=False,
                    )
                    for swap in swaps
                }
                r2 = sequence_for(
                    tokenizer, row["problem"], response, cfg, marker,
                    include_eos=False, normalize=False,
                )
                truncated = truncate_at_last_boxed(response)
                r3 = (
                    sequence_for(
                        tokenizer,
                        row["problem"],
                        truncated,
                        cfg,
                        marker,
                        include_eos=False,
                        normalize=False,
                    )
                    if truncated is not None
                    else None
                )
                records.append(
                    {
                        "dataset_i": dataset_i,
                        "row_i": row_i,
                        "response_i": response_i,
                        "r0": r0,
                        "r1": r1,
                        "r2": r2,
                        "r3": r3,
                        "r1_prefix": all(
                            is_token_prefix_with_target(seq, r0) for seq in r1.values()
                        ),
                        "r2_prefix": is_token_prefix_with_target(r2, r0),
                        "r3_prefix": r3 is not None
                        and is_token_prefix_with_target(r3, r0),
                        "raw": {"r1": {}},
                    }
                )

    record_index = {
        (record["dataset_i"], record["row_i"], record["response_i"]): i
        for i, record in enumerate(records)
    }

    fallback_counts = {
        "r1_response_fallbacks": sum(not x["r1_prefix"] for x in records),
        "r2_response_fallbacks": sum(not x["r2_prefix"] for x in records),
        "r3_tokenization_fallbacks": sum(
            x["r3"] is not None and not x["r3_prefix"] for x in records
        ),
        "r3_missing_boxed": sum(x["r3"] is None for x in records),
    }

    # One LaSeR model pass per batch group: R0, all R1 swaps, and any prefix
    # R2/R3 queries are read from the same hidden states.
    model = load_model(cfg["local_path"], torch)
    r0_sequences = [(i, record["r0"]) for i, record in enumerate(records)]
    core_queries: list[tuple[Any, Any, int, list[int]]] = []
    for i, record in enumerate(records):
        core_queries.append(
            (("r0", i), i, len(record["r0"]) - 2, [marker, *swaps])
        )
        if record["r2_prefix"]:
            core_queries.append((("r2", i), i, len(record["r2"]) - 2, [marker]))
        if record["r3_prefix"]:
            core_queries.append((("r3", i), i, len(record["r3"]) - 2, [marker]))
    core_values = score_sequences(
        model,
        r0_sequences,
        core_queries,
        tokenizer=tokenizer,
        device=device,
        max_batch_tokens=args.max_batch_tokens,
    )

    r1_fallback_sequences: list[tuple[Any, list[int]]] = []
    r1_fallback_queries: list[tuple[Any, Any, int, list[int]]] = []
    r2_fallback_sequences: list[tuple[Any, list[int]]] = []
    r2_fallback_queries: list[tuple[Any, Any, int, list[int]]] = []
    r3_fallback_sequences: list[tuple[Any, list[int]]] = []
    r3_fallback_queries: list[tuple[Any, Any, int, list[int]]] = []
    for i, record in enumerate(records):
        if not record["r1_prefix"]:
            for swap in swaps:
                key = (i, swap)
                seq = record["r1"][swap]
                r1_fallback_sequences.append((key, seq))
                r1_fallback_queries.append(
                    (key, key, len(seq) - 2, [swap])
                )
        if not record["r2_prefix"]:
            key = ("r2_fallback", i)
            seq = record["r2"]
            r2_fallback_sequences.append((key, seq))
            r2_fallback_queries.append((key, key, len(seq) - 2, [marker]))
        if record["r3"] is not None and not record["r3_prefix"]:
            key = ("r3_fallback", i)
            seq = record["r3"]
            r3_fallback_sequences.append((key, seq))
            r3_fallback_queries.append((key, key, len(seq) - 2, [marker]))

    for sequences, queries, kind in (
        (r1_fallback_sequences, r1_fallback_queries, "r1"),
        (r2_fallback_sequences, r2_fallback_queries, "r2"),
        (r3_fallback_sequences, r3_fallback_queries, "r3"),
    ):
        if not sequences:
            continue
        values = score_sequences(
            model,
            sequences,
            queries,
            tokenizer=tokenizer,
            device=device,
            max_batch_tokens=args.max_batch_tokens,
        )
        for query_key, value in values.items():
            if kind == "r1":
                response_i, swap = query_key
                records[response_i]["raw"]["r1"][swap] = value[0]
            elif kind == "r2":
                records[query_key[1]]["raw"]["r2"] = value[0]
            else:
                records[query_key[1]]["raw"]["r3"] = value[0]

    for i, record in enumerate(records):
        core = core_values[("r0", i)]
        record["raw"]["r0"] = core[0]
        record["raw"]["r1"].update(
            {swap: core[j + 1] for j, swap in enumerate(swaps)}
            if record["r1_prefix"]
            else {}
        )
        if record["r2_prefix"]:
            record["raw"]["r2"] = core_values[("r2", i)][0]
        if record["r3_prefix"]:
            record["raw"]["r3"] = core_values[("r3", i)][0]

    unload_model(model)

    base_path = args.base_model_path or cfg.get("base_model_local_path")
    r5_map: dict[int, float] = {}
    tokenizer_check: dict[str, Any] | None = None
    base_runtime = None
    if not args.skip_r5:
        if not base_path or not Path(base_path).exists():
            raise RuntimeError("R5 base model snapshot is unavailable; no substitute is allowed")
        base_tokenizer = AutoTokenizer.from_pretrained(base_path)
        base_tokenizer.padding_side = "right"
        tokenizer_check = assert_tokenizer_compatibility(tokenizer, base_tokenizer)
        base_sequences = []
        for i, record in enumerate(records):
            row = input_rows[record["dataset_i"]][record["row_i"]]
            response = row["responses"][record["response_i"]]
            # R5 scores the exact R0 token ids (checkpoint tokenizer and chat
            # template) with the base model, as the released training feeds its
            # reference model and as readout_r5.py does.  The base checkpoints
            # need not ship a chat template (OctoThinker-3B-Short-Base has none).
            # The vocabulary/special-token assertion above keeps the ids valid.
            seq = sequence_for(
                tokenizer, row["problem"], response, cfg, marker,
                include_eos=True, normalize=not args.raw_surface,
            )
            base_sequences.append((i, seq))
        base_model = load_model(base_path, torch)
        base_queries = [
            (("r5", i), i, len(seq) - 2, [marker]) for i, seq in base_sequences
        ]
        base_values = score_sequences(
            base_model,
            base_sequences,
            base_queries,
            tokenizer=base_tokenizer,
            device=device,
            max_batch_tokens=args.max_batch_tokens,
        )
        for i in range(len(records)):
            r5_map[i] = base_values[("r5", i)][0]
        base_runtime = parse_runtime_version("transformers")
        unload_model(base_model)

    ratio = float(cfg["ratio_coef"])
    cref = float(cfg["reference_constant"])
    for i, record in enumerate(records):
        raw = record["raw"]
        raw_r0 = raw.get("r0")
        raw_r1 = raw.get("r1", {})
        raw_r2 = raw.get("r2")
        raw_r3 = raw.get("r3")
        raw_r5 = r5_map.get(i)
        record["scores"] = {
            "R0_reference": (raw_r0 - cref) * ratio if raw_r0 is not None else None,
            "R1_marker_swap": {
                str(swap): (raw_r1.get(swap) - cref) * ratio
                if raw_r1.get(swap) is not None
                else None
                for swap in swaps
            },
            "R1_marker_swap_mean": (
                sum((raw_r1[swap] - cref) * ratio for swap in swaps) / len(swaps)
                if all(raw_r1.get(swap) is not None for swap in swaps)
                else None
            ),
            "R2_position_eos": (raw_r2 - cref) * ratio if raw_r2 is not None else None,
            "R3_position_answer": (raw_r3 - cref) * ratio if raw_r3 is not None else None,
            "R4_cref_none": raw_r0 * ratio if raw_r0 is not None else None,
            "R5_cref_recomputed": (
                (raw_r0 - raw_r5) * ratio
                if raw_r0 is not None and raw_r5 is not None
                else None
            ),
        }

    readout_stack = {
        "engine": "transformers",
        "engine_version": parse_runtime_version("transformers"),
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "device": "cuda",
        "model_id": cfg["model_id"],
        "model_revision": cfg["revision"],
        "max_batch_tokens": args.max_batch_tokens,
        "padding_side": "right",
        "sort_by_length": True,
        "marker_token_id": marker,
        "reference_constant": cfg["reference_constant"],
    }
    if base_path and not args.skip_r5:
        readout_stack["r5_base_model"] = {
            "model_id": cfg["base_model_repo_id"],
            "revision": cfg["base_model_revision"],
            "local_path": str(base_path),
            "engine_version": base_runtime,
        }
        readout_stack["r5_tokenizer_compatibility"] = tokenizer_check

    output_rows: list[list[dict[str, Any]]] = [[] for _ in input_rows]
    for dataset_i, rows in enumerate(input_rows):
        for row_i, row in enumerate(rows):
            item = dict(row)
            item["responses"] = [
                response if args.raw_surface else released_response(response)
                for response in row["responses"]
            ]
            scores = []
            logs = []
            for response_i, _response in enumerate(row["responses"]):
                record_i = record_index[(dataset_i, row_i, response_i)]
                record = records[record_i]
                raw = record["raw"]
                scores.append(record["scores"])
                logs.append(
                    {
                        "R0_marker": raw.get("r0"),
                        "R1_markers": {
                            str(swap): raw.get("r1", {}).get(swap) for swap in swaps
                        },
                        "R2_marker": raw.get("r2"),
                        "R3_marker": raw.get("r3"),
                        "R5_base_marker": r5_map.get(record_i),
                    }
                )
            item["readout_scores"] = scores
            item["readout_logprobs"] = logs
            item["readout_stack"] = dict(readout_stack)
            item["marker_token_id"] = marker
            item["marker_token"] = cfg.get("marker_token")
            item["reference_constant"] = cfg["reference_constant"]
            output_rows[dataset_i].append(item)

    for dataset_i, output_path in enumerate(args.output):
        output = Path(output_path)
        n = write_jsonl(output, output_rows[dataset_i])
        summary = {
            "kind": "readout_with_r5" if not args.skip_r5 else "readout",
            "rows": n,
            "responses": sum(len(row["responses"]) for row in output_rows[dataset_i]),
            "elapsed_seconds": time.perf_counter() - started,
            "model": cfg["model_id"],
            "base_model": cfg.get("base_model_repo_id") if not args.skip_r5 else None,
            "batching": {
                "max_batch_tokens": args.max_batch_tokens,
                "padding_side": "right",
                "sort_by_length": True,
            },
            "fallback_counts": {
                "r1_response_fallbacks": sum(
                    x["dataset_i"] == dataset_i and not x["r1_prefix"] for x in records
                ),
                "r2_response_fallbacks": sum(
                    x["dataset_i"] == dataset_i and not x["r2_prefix"] for x in records
                ),
                "r3_tokenization_fallbacks": sum(
                    x["dataset_i"] == dataset_i
                    and x["r3"] is not None
                    and not x["r3_prefix"]
                    for x in records
                ),
                "r3_missing_boxed": sum(
                    x["dataset_i"] == dataset_i and x["r3"] is None for x in records
                ),
            },
        }
        write_artifact_metadata(output, summary)

    if args.r0_r4_output:
        r04_stack = dict(readout_stack)
        r04_stack.pop("r5_base_model", None)
        r04_stack.pop("r5_tokenizer_compatibility", None)
        for dataset_i, output_path in enumerate(args.r0_r4_output):
            rows = copy.deepcopy(output_rows[dataset_i])
            for row in rows:
                row["readout_stack"] = dict(r04_stack)
                for score in row["readout_scores"]:
                    score["R5_cref_recomputed"] = None
                for log_item in row["readout_logprobs"]:
                    log_item["R5_base_marker"] = None
            output = Path(output_path)
            n = write_jsonl(output, rows)
            write_artifact_metadata(
                output,
                {
                    "kind": "readout",
                    "rows": n,
                    "responses": sum(len(row["responses"]) for row in rows),
                    "elapsed_seconds": time.perf_counter() - started,
                    "model": cfg["model_id"],
                    "batching": {
                        "max_batch_tokens": args.max_batch_tokens,
                        "padding_side": "right",
                        "sort_by_length": True,
                    },
                    "fallback_counts": {
                        "r1_response_fallbacks": sum(
                            x["dataset_i"] == dataset_i and not x["r1_prefix"]
                            for x in records
                        ),
                        "r2_response_fallbacks": sum(
                            x["dataset_i"] == dataset_i and not x["r2_prefix"]
                            for x in records
                        ),
                        "r3_tokenization_fallbacks": sum(
                            x["dataset_i"] == dataset_i
                            and x["r3"] is not None
                            and not x["r3_prefix"]
                            for x in records
                        ),
                        "r3_missing_boxed": sum(
                            x["dataset_i"] == dataset_i and x["r3"] is None
                            for x in records
                        ),
                    },
                },
            )


if __name__ == "__main__":
    main()
