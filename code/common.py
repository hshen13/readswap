"""Shared helpers for the pre-registered READSWAP experiment.

The functions ``released_response``, ``released_last_boxed_only_string`` and
``released_remove_boxed`` reproduce the answer handling of the LaSeR
repository (https://github.com/RUCBM/LaSeR, src/eval_math.py),
Copyright (c) 2025 RUCBM, released under the MIT License.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
CONFIG_PATH = ROOT / "data" / "LASER_CONFIGS.json"


def hf_hub_cache() -> Path:
    """Hugging Face hub cache directory, following HF_HUB_CACHE / HF_HOME."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def hf_snapshot_path(repo_id: str, revision: str) -> Path:
    """Local snapshot directory of ``repo_id`` at ``revision`` in the hub cache."""
    return hf_hub_cache() / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision


def load_configs() -> dict[str, Any]:
    """Read data/LASER_CONFIGS.json and resolve local model paths.

    Relative ``local_path`` entries are taken relative to the repository root.
    An empty ``base_model_local_path`` is resolved to the snapshot of
    ``base_model_repo_id`` at ``base_model_revision`` in the hub cache.
    """
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        configs = json.load(fh)
    for cfg in configs.get("models", {}).values():
        local = cfg.get("local_path")
        if local and not Path(local).is_absolute():
            cfg["local_path"] = str(ROOT / local)
        if not cfg.get("base_model_local_path") and cfg.get("base_model_repo_id") and cfg.get("base_model_revision"):
            cfg["base_model_local_path"] = str(hf_snapshot_path(cfg["base_model_repo_id"], cfg["base_model_revision"]))
    return configs


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def write_artifact_metadata(jsonl_path: str | Path, summary: dict[str, Any]) -> None:
    jsonl_path = Path(jsonl_path)
    summary_path = jsonl_path.with_name(jsonl_path.stem + "_summary.json")
    write_json(summary_path, summary)
    digest = sha256_file(jsonl_path)
    jsonl_path.with_name(jsonl_path.name + ".sha256").write_text(
        f"{digest}  {jsonl_path.name}\n", encoding="utf-8"
    )
    summary_path.with_name(summary_path.name + ".sha256").write_text(
        f"{sha256_file(summary_path)}  {summary_path.name}\n", encoding="utf-8"
    )


def model_config(model_key: str) -> dict[str, Any]:
    configs = load_configs()
    if model_key in configs["models"]:
        return configs["models"][model_key]
    for key, value in configs["models"].items():
        if value.get("model_id") == model_key:
            return value
    raise KeyError(f"unknown model: {model_key}")


def prompt_text(tokenizer: Any, problem: str, config: dict[str, Any]) -> str:
    suffix = config.get("prompt_template", {}).get("problem_suffix", "")
    user_content = problem + suffix
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ).strip()


def prompt_ids(tokenizer: Any, problem: str, config: dict[str, Any]) -> list[int]:
    return tokenizer(
        prompt_text(tokenizer, problem, config), add_special_tokens=False
    ).input_ids


def released_response(text: str) -> str:
    """Normalize a response exactly as released ``eval_math.py`` does.

    Source: LaSeR repository, ``src/eval_math.py:141-144`` (MIT License).
    This is deliberately applied only on the consuming side; raw generation
    artifacts remain byte-for-byte unchanged.
    """

    response = text.strip()
    if "</answer>" in response:
        response = response.split("</answer>")[0] + "</answer>"
    return response


def append_marker_ids(
    tokenizer: Any,
    prompt: str,
    response: str,
    marker_id: int,
    include_eos: bool,
) -> list[int]:
    text = prompt + released_response(response)
    if include_eos and tokenizer.eos_token:
        text += tokenizer.eos_token
    text += tokenizer.decode([marker_id], clean_up_tokenization_spaces=False)
    return tokenizer(text, add_special_tokens=False).input_ids


def released_last_boxed_only_string(text: str) -> str | None:
    """Copy of the released grader's final-box extraction.

    Source: LaSeR repository, ``src/eval_math.py:10-32`` (MIT License).
    In particular, the final marker wins and an incomplete final marker is
    invalid even when an earlier marker is complete.
    """

    idx = text.rfind(r"\boxed")
    if idx < 0:
        idx = text.rfind(r"\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def released_remove_boxed(value: str | None) -> str | None:
    """Copy of released ``remove_boxed``; source eval_math.py:35-42."""

    left = r"\boxed{"
    try:
        assert value is not None
        assert value[: len(left)] == left
        assert value[-1] == "}"
        return value[len(left) : -1]
    except Exception:
        return None


def last_boxed_span(text: str) -> str | None:
    """Compatibility alias for the released final-box extraction."""

    return released_last_boxed_only_string(text)


def truncate_at_last_boxed(text: str) -> str | None:
    span = last_boxed_span(text)
    if span is None:
        return None
    return text[: text.rfind(span) + len(span)]


def sigmoid_free_normalized(scores: list[float]) -> list[float]:
    """The affine [0,1] normalization used for calibration, without sigmoid."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if math.isclose(lo, hi):
        return [0.5] * len(scores)
    return [(x - lo) / (hi - lo) for x in scores]


def inference_stack(config: dict[str, Any], *, dtype: str = "bfloat16") -> dict[str, Any]:
    return {
        "engine": "vllm",
        "engine_version": "0.7.3 (required by LaSeR README; runtime recorded by caller)",
        "dtype": dtype,
        "model_id": config["model_id"],
        "model_revision": config["revision"],
        "decoding": config["decoding"],
        "batching": {"prompt_token_ids": True, "max_num_seqs": 32},
    }


def parse_runtime_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"
