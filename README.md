# READSWAP

Code for our paper **Almost Any Unused Token Gives the Same Self-Score: Auditing LaSeR's Trained Confidence Token**.

[LaSeR](https://github.com/RUCBM/LaSeR) trains a model to score its own solutions. After the solution and the EOS
token, it reads the log-probability of a marker token and subtracts a constant. We take the three released LaSeR
checkpoints (OctoThinker-3B-Short, Qwen2.5-7B, ORZ-7B), sample solutions on MATH500, AMC23, AIME24, AIME25 and
OlympiadBench, keep those solutions fixed, and change only how the score is read:

- R0: the released readout
- R1: another unused special token as the marker
- R2: the marker read without the EOS token before it
- R3: the marker read right after the final `\boxed{}` answer
- R4: no constant
- R5: the base model's log-probability in place of the constant

The comparisons L1 to L5 between these readouts were registered before the main runs (`prereg/`). The token
sweeps, EOS controls, length checks and the Qwen3-4B replication came later and are exploratory.

## Repository layout

```
README.md, LICENSE, requirements.txt, requirements-qwen3.txt
code/                 all Python scripts and shell drivers (run from the repository root)
  eaclplot.py         shared matplotlib style of the data figures
tests/                unit tests of the analysis code (CPU only)
prereg/               frozen pre-registration (PREREG_v1.json), Amendment A1 and their SHA-256 files
data/                 benchmark files, pilot problems, checkpoint registries (see data/README.md)
results/analysis/     analysis outputs used by the paper (summary JSONs, per-solution mechanism control)
figures/fig1/         Figure 1 as an editable SVG (overview.svg) and the PDF used in the paper
figures/demo/         TikZ source of Figure 4 (demo.tex) and its style file
outputs/              tables and figures written by the generators (created on demand)
```

Directories that the pipeline creates but that are not part of this repository: `data/models/` (checkpoints),
`results/confirmatory/` (per-solution generations, grades and readouts, about 3 GB), `results/pilot/`,
`results/_dev/` (readout equivalence check and vLLM cross-check), `results/q3*/` (Qwen3-4B solutions and
adapters), `logs/`, `third_party/LaSeR/`.

## Setup

Hardware: every run used one NVIDIA A800 GPU with 80 GB of memory. Generation and readouts took 16.1, 13.3 and
18.8 GPU hours for OctoThinker-3B, Qwen2.5-7B and ORZ-7B (`results/analysis/gpu_hours.json`). The analyses,
tables and figures run on a CPU.

Main environment (Python 3.12.14):

```bash
pip install -r requirements.txt
```

The Qwen3-4B replication generates, trains and reads out with a newer stack (Python 3.10.19):

```bash
pip install -r requirements-qwen3.txt     # in a separate environment
```

The data figures use matplotlib's pgf backend and need `pdflatex` with the `newtx` fonts; the TikZ figure (Figure 4)
needs `pdflatex` with TikZ and `fontawesome5`.

Models, from Hugging Face (ids and revisions as in `data/MODELS.json`). The checkpoints go to `data/models/`, the
base models stay in the Hugging Face cache (`HF_HOME`), where the code looks for them:

```bash
huggingface-cli download Keven16/OctoThinker-3B-Short-LaSeR --revision d964b7c68d5f55eecf586f4eaeffb5786690ad89 --local-dir data/models/OctoThinker-3B-Short-LaSeR
huggingface-cli download Keven16/Qwen2.5-7B-LaSeR --revision e77250921afdf935ebc00d3f8c40f9864f063004 --local-dir data/models/Qwen2.5-7B-LaSeR
huggingface-cli download Keven16/ORZ-7B-LaSeR --revision d0439aa43bd6922b6c26289a387b43ed34700559 --local-dir data/models/ORZ-7B-LaSeR
huggingface-cli download OctoThinker/OctoThinker-3B-Short-Base --revision 01b58beede3c9de3cef10cd11dadbd89fcf96e48
huggingface-cli download Qwen/Qwen2.5-7B --revision d149729398750b98c0af14eb82c78cfe92750796
huggingface-cli download Open-Reasoner-Zero/Open-Reasoner-Zero-7B --revision 0e38a598946c6fa65ec48f99b2b6a8a5107c7ee1
huggingface-cli download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c     # replication only
```

The pilot compares R0 with the released LaSeR scorer, so it needs the LaSeR repository:

```bash
git clone https://github.com/RUCBM/LaSeR.git third_party/LaSeR
git -C third_party/LaSeR checkout 48996f78e76986982f8678a892c21483998823fe
```

### Environment variables

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

## Reproducing the paper

### 0. Pilot and released-scorer check

```bash
bash code/run_e1_rest.sh
```

Generates, grades and reads out eight solutions for the 40 pilot problems with OctoThinker-3B (seed 0), then scores
20 of them with the released `calculate_self_rewarding_scores.py` and compares R0 with it
(`results/pilot/octothinker_seed0_unit_compare.json`). The readout stage waits for this check to pass.

### 1-2. Generation and grading

```bash
bash code/run_E2_gen.sh
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

### 3-4. Readouts R0-R5 and the confirmatory analysis

```bash
bash code/run_E2_readout.sh
python code/kendall_ci.py
```

`run_E2_readout.sh` first checks the single-pass transformers readout (`code/readout_fast.py`) against the vLLM
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
bash code/run_xchk_v2.sh
```

### 5. Exploratory controls

GPU runs (each driver skips outputs that already exist):

```bash
bash code/run_token_control.sh          # token_control_<ckpt>.json (OctoThinker-3B, Qwen2.5-7B)
bash code/run_token_control_orz.sh      # token_control_ORZ-7B-LaSeR.json
bash code/run_token_sweep.sh            # token_sweep_<ckpt>.json
bash code/run_eos_swap.sh               # eos_swap_<ckpt>.json
bash code/run_mech.sh                   # mech/<ckpt>.jsonl, then the OctoThinker sweep with raw differences
bash code/run_exhaustive_vocab_seeds.sh # exhaustive_tokens_<ckpt>.json, eos_vocab_<ckpt>.json, Qwen3 seeds 1-2
PYTHONPATH=code python code/unembed_check.py   # unembedding_rows.json (reads the weight files, CPU)
PYTHONPATH=code python code/unembed_scope.py   # unembedding_scope.json (CPU)
```

The second half of `run_exhaustive_vocab_seeds.sh` trains and reads out Qwen3-4B seeds 1 and 2 and needs step 6.

CPU analyses (all write to `results/analysis/`):

```bash
python code/mech_analyze.py      # mech_summary.json (from results/analysis/mech/*.jsonl, included)
python code/base_signal.py       # base_signal.json
python code/laser_extra.py       # laser_extra.json
python code/laser_lengthctl.py   # length_control.json
python code/laser_repeat.py      # laser_repeat.json
python code/calib_paired.py      # calibration_paired.json
python code/extra_checks.py      # extra_checks.json (needs results/q3/data/train.jsonl from step 6)
python code/perseed_boot.py --model OctoThinker-3B-Short-LaSeR   # perseed_boot_<ckpt>.json, one per checkpoint
python code/laser_gpu_hours.py   # gpu_hours.json (parses the GPU wrapper logs, not included)
```

`base_signal.py`, `laser_extra.py`, `laser_lengthctl.py`, `laser_repeat.py`, `calib_paired.py`, `extra_checks.py`,
`perseed_boot.py` and `kendall_ci.py` read the per-solution readouts in `results/confirmatory/readout/`.

### 6. Qwen3-4B replication

```bash
python code/q3_prepare.py      # results/q3/data/train.jsonl: 2,000 DeepMath-103K problems, seed 0
bash code/q3_pipeline.sh       # generate, grade, LoRA training, readouts, q3_replication.json
```

`q3_prepare.py` downloads LaSeR's DeepMath-103K training parquet (`Keven16/LaSeR_training_data`, fixed revision),
removes problems whose text equals a benchmark problem, shuffles with seed 0 and keeps the first 2,000. The pipeline
uses `PYTHON_Q3` for generation, training and readout and `PYTHON` for grading and analysis. Training seeds 1 and 2
(`q3_replication_s1.json`, `q3_replication_s2.json`) come from `run_exhaustive_vocab_seeds.sh`.

### 7. Tables and figures

```bash
bash code/finalize_laser_assets.sh
cd figures/demo && pdflatex demo.tex && cd ../..
```

The table and figure generators read only `results/analysis/`, except `make_calibration_fig.py`, which reads the
per-solution readouts and is skipped when they are absent. Tables are written to `outputs/tables/`, figures to
`outputs/figs/`.

## Scripts and the paper

Numbers refer to the compiled submission. "Pre-specified" marks results under the pre-registration and Amendment A1.

| Paper element | Generator | Inputs |
|---|---|---|
| Figure 1 (LaSeR's self-score in the model; unused tokens share one output direction; the three readout choices) | `figures/fig1/overview.svg` (vector drawing; `overview.pdf` is the paper version) | none |
| Table 1 (AUROC, ECE and weighted-voting accuracy of R0-R5; pre-specified) | `code/laser_paper_tables.py` -> `readouts.tex` | `*_confirm.json` |
| Table 2 (contrasts L1-L5 with intervals and verdicts; pre-specified) | `code/laser_paper_tables.py` -> `contrasts.tex` | `*_confirm.json` |
| Figure 2 (token sweep by cosine with the marker's output row; marker read after tokens in the EOS position) | `code/make_token_figure_v3.py` -> `token_control.pdf` | `token_sweep_*.json`, `mech_summary.json` |
| Table 3 (length checks of R0) | `code/laser_new_tables.py` -> `length_checks.tex` | `laser_extra.json`, `length_control.json` |
| Figure 3 (mean R0 and R3 score by length decile) | `code/make_figures.py` -> `length_score.pdf` | `*_confirm.json` |
| Figure 4 (two MATH500 examples) | `figures/demo/demo.tex` | none (content from the seed-1 OctoThinker-3B solutions) |
| Tables 4-6 (readout configuration, models, datasets; Appendices A-D) | written by hand | `data/LASER_CONFIGS.json`, `data/MANIFEST.json`, `data/MODELS.json` |
| Appendix D compute hours | `code/laser_gpu_hours.py` | GPU wrapper logs -> `gpu_hours.json` |
| Figure 5 (AUROC of four readouts per benchmark) | `code/make_laser_more_figs.py` -> `benchmarks.pdf` | `*_confirm.json` |
| Figure 6 (AUROC of every readout on each seed) | `code/make_mechanism_figs.py` -> `perseed.pdf` | `*_confirm.json` |
| Table 7 (output-embedding rows of marker and R1 tokens) | `code/laser_paper_tables.py` -> `app_unembed.tex` | `unembedding_rows.json` (`code/unembed_check.py`) |
| Appendix F.1, shared output direction | `code/unembed_scope.py` | `unembedding_scope.json` |
| Figure 7 (exhaustive token test: score difference against cosine with the marker's row) | `code/make_mechanism_figs.py` -> `exhaustive.pdf` | `exhaustive_tokens_*.json` (`code/exhaustive_tokens.py`) |
| Figure 8 (marker read after candidate tokens in the EOS position) | `code/make_mechanism_figs.py` -> `eos_vocab.pdf` | `eos_vocab_*.json` (`code/eos_vocab.py`) |
| Figure 9 (marker logit after EOS, base model and LaSeR checkpoint) | `code/make_mechanism_figs.py` -> `terms.pdf` | `mech/*.jsonl` (`code/mech_control.py`) |
| Appendix F.2-F.3, EOS control and the two terms | `code/mech_control.py`, `code/mech_analyze.py`, `code/eos_swap_control.py` | `mech_summary.json`, `eos_swap_*.json` |
| Appendix F.3, base-model control | `code/base_signal.py` | `base_signal.json` |
| Appendix F.4, length controls, truncation, repetition, paired intervals | `code/laser_lengthctl.py`, `code/laser_extra.py`, `code/laser_repeat.py`, `code/extra_checks.py` | `length_control.json`, `laser_extra.json`, `laser_repeat.json`, `extra_checks.json` |
| Figure 10 (reliability of R0 and R5) | `code/make_calibration_fig.py` -> `calibration.pdf` | per-solution readouts; binned values in `calibration_bins.json` |
| Appendix F.5, paired calibration test | `code/calib_paired.py` | `calibration_paired.json` |
| Figure 11 (weighted minus majority voting) | `code/make_laser_more_figs.py` -> `voting.pdf` | `*_confirm.json` |
| Figure 12 (Qwen3-4B tokens before and after training) | `code/make_mechanism_figs.py` -> `q3_tokens.pdf` | `q3_replication.json` |
| Table 8 and Section 5.5 (Qwen3-4B replication) | `code/laser_new_tables.py` -> `app_q3.tex` | `q3_replication.json` (`code/q3_*.py`) |
| Per-benchmark and per-seed tables of every metric (repository only) | `code/laser_paper_tables.py` -> `app_full.tex`, `app_perseed.tex`, `app_perseed_auroc.tex`, `app_secondary.tex`; `code/laser_perseed_tables.py` -> `app_perseed_metrics.tex`; `code/perseed_boot.py` | `*_confirm.json`, `perseed_boot_*.json`, `kendall_ci.json` |
| Token sweep, exhaustive-test and EOS-control tables (repository only) | `code/laser_new_tables.py` -> `app_sweep.tex`, `app_mech.tex`; `code/laser_exh_table.py` -> `app_exhaustive.tex` | `token_sweep_*.json`, `exhaustive_tokens_*.json`, `mech_summary.json` |
| Appendix A, released-scorer and vLLM checks | `code/compare_official.py`, `code/compare_readout_fast.py`, `code/compare_xchk.py` | `results/pilot/`, `results/_dev/` (created by the drivers) |

`laser_paper_tables.py` also writes `app_benchmarks.tex` (results by benchmark), which the paper does not include,
and a first version of `length_checks.tex`, which `laser_new_tables.py` replaces. `make_figures.py` also writes
`readout_auroc.pdf`, which the paper does not include. `token_control_*.json` (`code/token_control.py`) holds the
first token and terminator control on MATH500 and enters no table or figure.

## Tests

```bash
PYTHONPATH=code python -m unittest discover -s tests -v
```

## Third-party code and data

`code/common.py` reproduces three small answer-handling functions of the LaSeR evaluator
(https://github.com/RUCBM/LaSeR, `src/eval_math.py`), Copyright (c) 2025 RUCBM, MIT License. The LaSeR repository
is otherwise used only as an external dependency (`third_party/LaSeR`). The benchmark files in `data/confirmatory/`
are copied from the LaSeR repository; their sources and licenses are listed in `data/README.md`.

## License

The code in this repository is released under the MIT License (see `LICENSE`). Data files keep the licenses of their
original sources (`data/README.md`).
