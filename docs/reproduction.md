# Reproducing the paper step by step

This page gives every command of the pipeline with its inputs and outputs. The short version is in the [README](../README.md).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `READSWAP_ROOT` | the repository root (from the script location) | where `data/`, `results/` and `outputs/` live |
| `PYTHON` | `python` | interpreter of the main environment (use a command on `PATH` or an absolute path) |
| `PYTHON_Q3` | `$PYTHON` | interpreter of the Qwen3 environment (replication only) |
| `HF_HOME` | Hugging Face default (`~/.cache/huggingface`) | Hugging Face cache; the base models are read from it |
| `GPU_RUN` | empty | optional command prefix for GPU jobs, e.g. a scheduler or lock wrapper; empty runs them directly |
| `LASER_REPO` | `$READSWAP_ROOT/third_party/LaSeR` | clone of the LaSeR repository (pilot scorer check) |
| `Q3_DIR`, `Q3_OUT`, `Q3_BASE_MODEL`, `Q3_TRAIN_PARQUET` | see `code/q3_common.py`, `code/q3_prepare.py` | replication paths |

All commands below are run from the repository root.

## Pre-registration

`prereg/PREREG_v1.json` (the plan) is byte-identical to the version that was hashed (SHA-256 `c0c2bc230dac9e46...`,
as printed in Appendix A of the paper). `prereg/AMENDMENT_A1.md` (Amendment A1, locked before any confirmatory
solution existed) differs from the locked file only in line 3, from which an internal attribution was removed
before release. The locked file has SHA-256 `a04f5c089ba93688537966e2efc6e374aa4f4b58eb5819a79e4d60a3a95eee00`
(the hash printed in the paper); `prereg/SHA256SUMS_A1` lists the released copy. Check both files with

```bash
cd prereg && sha256sum -c SHA256SUMS SHA256SUMS_A1
```

Paths inside these frozen files (for example the location of the LaSeR checkout) refer to the original working
layout; in this repository the LaSeR clone is `third_party/LaSeR`.

Pre-specified: the six readouts, the contrasts L1 to L5, the metrics (primary AUROC, ECE, Brier, weighted voting,
Kendall tau, partial correlation with length), the cluster bootstrap over problems with Holm correction, and the
R3 sensitivity analysis of Amendment A1. These come from `code/analyze_confirm.py` and are stored in
`results/analysis/<checkpoint>_confirm.json` and `results/analysis/confirm_summary.json`. Everything else in this
repository is exploratory and labelled as such in the paper.

## Pipeline

### 0. Pilot and check against LaSeR's scorer

```bash
bash code/run_pilot.sh
```

Generates, grades and reads out eight solutions for the 40 pilot problems with OctoThinker-3B (seed 0), then scores
20 of them with the released `calculate_self_rewarding_scores.py` and compares R0 with it
(`results/pilot/octothinker_seed0_unit_compare.json`). The readout stage waits for this check to pass.

### 1. Generation and grading

```bash
bash code/run_generation.sh
```

For each checkpoint, benchmark and seed (0, 1, 2), `code/generate.py` samples eight solutions with vLLM (temperature
1.0, top-p 1.0, at most 8,192 new tokens) and `code/grade.py` grades the final boxed answer with Math-Verify, using
the answer extraction of the released LaSeR evaluator. Outputs: `results/confirmatory/generated/` and
`results/confirmatory/graded/`. Single run:

```bash
PYTHONPATH=code python code/generate.py --model OctoThinker-3B-Short-LaSeR --dataset math500 \
  --input data/confirmatory/math500/test.jsonl --output results/confirmatory/generated/OctoThinker-3B-Short-LaSeR/math500_seed0.jsonl \
  --seed 0 --max-num-seqs 256
PYTHONPATH=code python code/grade.py --input results/confirmatory/generated/OctoThinker-3B-Short-LaSeR/math500_seed0.jsonl \
  --output results/confirmatory/graded/OctoThinker-3B-Short-LaSeR/math500_seed0.jsonl
```

### 2. Readouts R0 to R5 and the confirmatory analysis

```bash
bash code/run_readouts.sh
python code/kendall_ci.py
```

`run_readouts.sh` first checks the single-pass transformers readout (`code/readout_fast.py`) against the vLLM
readout path (`code/readout.py`, `code/readout_r5.py`) on the pilot solutions (`code/compare_readout_fast.py`). The
bound set for this check (0.05 nat on the raw log-probabilities) was not met in our run (largest difference 0.14 nat,
on R0; scores differ by at most 0.014), and the run continued with the transformers readouts on a recorded decision,
backed by the full vLLM cross-check below. The driver stops unless the check passes or
`results/_dev/readout_fast_equivalence_decision.json` records `"proceed": true` together with the SHA-256 of
`results/_dev/readout_fast_equivalence.json`. After inspecting that file, the decision can be recorded with

```bash
python - <<'EOF'
import hashlib, json
eq = "results/_dev/readout_fast_equivalence.json"
json.dump({"proceed": True, "equivalence_sha256": hashlib.sha256(open(eq, "rb").read()).hexdigest()},
          open("results/_dev/readout_fast_equivalence_decision.json", "w"), indent=2)
EOF
```

It then downloads the two Qwen-family base models if needed, computes R0 to R5 for every solution (one forward
pass of the checkpoint per solution, extra passes where a readout prefix is not a prefix of the R0 sequence, and one
pass of the base model for R5), and runs `code/analyze_confirm.py` per checkpoint (seed 20260924, 10,000 cluster
bootstrap resamples) and `code/make_confirm_summary.py`. `code/kendall_ci.py` afterwards recomputes the Kendall tau
intervals in the three `*_confirm.json` files with scipy (backups are kept); the released JSONs contain its output.

vLLM cross-check of all seed-0 OctoThinker-3B solutions (`results/_dev/vllm_crosscheck/xchk_report.json`):

```bash
bash code/run_vllm_crosscheck.sh
```

### 3. Exploratory controls

GPU runs (each driver skips outputs that already exist):

```bash
bash code/run_token_control.sh          # token_control_<ckpt>.json (OctoThinker-3B, Qwen2.5-7B)
bash code/run_token_control_orz.sh      # token_control_ORZ-7B-LaSeR.json
bash code/run_token_sweep.sh            # token_sweep_<ckpt>.json
bash code/run_eos_swap.sh               # eos_swap_<ckpt>.json
bash code/run_mech.sh                   # mech/<ckpt>.jsonl, then the OctoThinker sweep with raw differences
bash code/run_extra_controls.sh        # exhaustive_tokens_<ckpt>.json, eos_vocab_<ckpt>.json, Qwen3 seeds 1 and 2
PYTHONPATH=code python code/unembed_check.py   # unembedding_rows.json (reads the weight files, CPU)
PYTHONPATH=code python code/unembed_scope.py   # unembedding_scope.json (CPU)
```

The second half of `run_extra_controls.sh` trains and reads out Qwen3-4B seeds 1 and 2 and needs step 4.

CPU analyses (all write to `results/analysis/`):

```bash
python code/mech_analyze.py      # mech_summary.json (from results/analysis/mech/*.jsonl, included)
python code/base_signal.py       # base_signal.json
python code/laser_extra.py       # laser_extra.json
python code/laser_lengthctl.py   # length_control.json
python code/laser_repeat.py      # laser_repeat.json
python code/calib_paired.py      # calibration_paired.json
python code/extra_checks.py      # extra_checks.json (needs results/q3/data/train.jsonl from step 4)
python code/perseed_boot.py --model OctoThinker-3B-Short-LaSeR   # perseed_boot_<ckpt>.json, one per checkpoint
python code/laser_gpu_hours.py   # gpu_hours.json (parses the GPU wrapper logs, not included)
```

`base_signal.py`, `laser_extra.py`, `laser_lengthctl.py`, `laser_repeat.py`, `calib_paired.py`, `extra_checks.py`,
`perseed_boot.py` and `kendall_ci.py` read the per-solution readouts in `results/confirmatory/readout/`.

### 4. Qwen3-4B replication

```bash
python code/q3_prepare.py      # results/q3/data/train.jsonl: 2,000 DeepMath-103K problems, seed 0
bash code/q3_pipeline.sh       # generate, grade, LoRA training, readouts, q3_replication.json
```

`q3_prepare.py` downloads LaSeR's DeepMath-103K training parquet (`Keven16/LaSeR_training_data`, fixed revision),
removes problems whose text equals a benchmark problem, shuffles with seed 0 and keeps the first 2,000. The pipeline
uses `PYTHON_Q3` for generation, training and readout and `PYTHON` for grading and analysis. Training seeds 1 and 2
(`q3_replication_s1.json`, `q3_replication_s2.json`) come from `run_extra_controls.sh`.

### 5. Problems and models from 2026 (exploratory)

These runs came after the confirmatory analysis. A separate plan fixed their models, data, marker rule and protocol
before they started; the paper's Appendix A gives its hash.

```bash
bash code/run_new_problems.sh          # the three checkpoints on AIME 2026 and HMMT February 2026
bash code/run_new_models.sh            # score-only training in Qwen3.5-4B and Gemma-4-12B (needs step 4's train.jsonl)
python code/newdata_analysis.py        # newdata.json: both, plus the Qwen3-4B seeds of step 4
python code/unembed_2026.py            # unembed_2026.json: unused-token rows in eight models from 2026 (CPU)
```

`run_new_problems.sh` uses the confirmatory generation, grading and readout scripts unchanged on
`data/new_problems/` (eight solutions, seeds 0 to 2). `run_new_models.sh` repeats `q3_pipeline.sh` with seed 0 for
each model; the scripts `qwen35_4b_*.py` and `gemma4_12b_*.py` differ from `q3_*.py` only in the model, the chat
template call, the end-of-turn token, the marker and R1 ids (the four unused tokens with the smallest ids) and, for
Gemma, its logit soft-capping. They run in a third environment, `PYTHON_NEW` (`requirements-2026.txt`).
`newdata_analysis.py` takes about an hour in one process with 10,000 bootstrap resamples per model; `--part laser`,
`--part trained --model NAME` and `--part merge --parts ...` split the work over processes.
`unembed_2026.py` reads cached weights in place and otherwise downloads only the shard holding the output matrix,
at most 12 GB per model, and deletes it after use.

### 6. Tables and figures

```bash
bash code/make_tables_and_figures.sh
cd figures/demo && pdflatex demo.tex && cd ../..
```

The table and figure generators read only `results/analysis/`, except `make_calibration_fig.py`, which reads the
per-solution readouts and is skipped when they are absent. Tables are written to `outputs/tables/`, figures to
`outputs/figs/`.
