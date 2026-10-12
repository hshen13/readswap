#!/usr/bin/env python3
"""Generate eight solutions per problem with base Qwen3.5-4B (non-thinking chat mode), LaSeR decoding settings
(temperature 1.0, top-p 1.0), at most 4,096 new tokens. Output rows follow the schema of code/generate.py, plus
the generated token ids, so that code/grade.py can grade them and readouts need no retokenization."""
import argparse
import os
import sys
import time

from qwen35_4b_common import BASE, MAX_NEW, N_SAMPLES, eos_id, prompt_ids, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(BASE)
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids(tok, r["problem"])) for r in rows]
    t0 = time.time()
    llm = LLM(model=BASE, tokenizer=BASE, dtype="bfloat16", gpu_memory_utilization=0.65, max_model_len=6144,
              max_num_seqs=256, seed=args.seed)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_NEW, n=N_SAMPLES, seed=args.seed)
    outs = llm.generate(prompts, sampling_params=sp)
    eos = eos_id(tok)
    out_rows = []
    for r, res in zip(rows, outs):
        ids = [list(o.token_ids) for o in res.outputs]
        out_rows.append({
            "dataset": args.dataset,
            "problem_id": r.get("id", r.get("unique_id")),
            "problem": r["problem"],
            "answer": r.get("answer"),
            "responses": [o.text for o in res.outputs],
            "response_token_ids": [x[:-1] if x and x[-1] == eos else x for x in ids],
            "response_token_counts": [len(x) for x in ids],
            "finish_reasons": [o.finish_reason for o in res.outputs],
            "stop_reasons": [o.stop_reason for o in res.outputs],
            "truncated": [o.finish_reason == "length" for o in res.outputs],
            "model": "Qwen/Qwen3.5-4B",
            "seed": args.seed,
        })
    write_jsonl(args.output, out_rows)
    print(f"wrote {len(out_rows)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # vLLM V1 can leave its engine process alive after an error
        print("FAILED:", repr(exc), file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
