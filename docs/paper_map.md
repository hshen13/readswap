# Which script produces which table and figure

Numbers refer to the compiled paper. "Pre-specified" marks results under the pre-registration and Amendment A1;
everything else is exploratory.

| Paper element | Generator | Inputs |
|---|---|---|
| Figure 1 (LaSeR's self-score in the model; unused tokens share one output direction; the three readout choices) | `figures/fig1/overview.svg` (vector drawing; `overview.pdf` is the paper version) | none |
| Table 1 (AUROC, ECE and weighted-voting accuracy of R0 to R5; pre-specified) | `code/laser_paper_tables.py` -> `readouts.tex` | `*_confirm.json` |
| Figure 2 (two MATH500 examples) | `figures/demo/demo.tex` | none (content from the seed-1 OctoThinker-3B solutions) |
| Table 2 (contrasts L1 to L5 with intervals and verdicts; pre-specified) | `code/laser_paper_tables.py` -> `contrasts.tex` | `*_confirm.json` |
| Figure 3 (token sweep by cosine with the marker's output row; marker read after tokens in the EOS position) | `code/make_token_figure.py` -> `token_control.pdf` | `token_sweep_*.json`, `mech_summary.json` |
| Table 3 (length checks of R0) | `code/laser_new_tables.py` -> `length_checks.tex` | `laser_extra.json`, `length_control.json` |
| Table 4 and Section 5.5 (2026 problems; score-only training in Qwen3-4B, Qwen3.5-4B and Gemma-4-12B) | `code/newdata_tables.py` -> `newdata.tex` | `newdata.json` (`code/newdata_analysis.py`) |
| Tables 5 to 7 (readout configuration, models, datasets; Appendices A to D) | written by hand | `data/LASER_CONFIGS.json`, `data/MANIFEST.json`, `data/MODELS.json` |
| Appendix D compute hours | `code/laser_gpu_hours.py` | GPU wrapper logs -> `gpu_hours.json` |
| Figure 4 (AUROC of four readouts per benchmark and on the 2026 sets) | `code/make_laser_more_figs.py` -> `benchmarks.pdf` | `*_confirm.json`, `newdata.json` |
| Figure 5 (AUROC of every readout on each seed) | `code/make_mechanism_figs.py` -> `perseed.pdf` | `*_confirm.json` |
| Figure 6 (exhaustive token test: score difference against cosine with the marker's row) | `code/make_mechanism_figs.py` -> `exhaustive.pdf` | `exhaustive_tokens_*.json` (`code/exhaustive_tokens.py`) |
| Table 8 (2026 problems and score-only models with intervals) | `code/newdata_tables.py` -> `app_newdata.tex` | `newdata.json` |
| Figure 7 (marker read after candidate tokens in the EOS position) | `code/make_mechanism_figs.py` -> `eos_vocab.pdf` | `eos_vocab_*.json` (`code/eos_vocab.py`) |
| Table 9 (output-embedding rows of marker and R1 tokens) | `code/laser_paper_tables.py` -> `app_unembed.tex` | `unembedding_rows.json` (`code/unembed_check.py`) |
| Appendix F.1, shared output direction | `code/unembed_scope.py` | `unembedding_scope.json` |
| Figure 8 (marker logit after EOS, base model and LaSeR checkpoint) | `code/make_mechanism_figs.py` -> `terms.pdf` | `mech/*.jsonl` (`code/mech_control.py`) |
| Table 10 and Figure 9 (unused-token rows in eight models from 2026) | `code/newdata_tables.py` -> `app_unembed2026.tex`; `code/newdata_figs.py` -> `unembed_2026.pdf` | `unembed_2026.json` (`code/unembed_2026.py`) |
| Appendix F.2 and F.3, EOS control and the two terms | `code/mech_control.py`, `code/mech_analyze.py`, `code/eos_swap_control.py` | `mech_summary.json`, `eos_swap_*.json` |
| Appendix F.3, base-model control | `code/base_signal.py` | `base_signal.json` |
| Figure 10 (mean R0 and R3 score by length decile) | `code/make_figures.py` -> `length_score.pdf` | `*_confirm.json` |
| Appendix F.4, length controls, truncation, repetition, paired intervals | `code/laser_lengthctl.py`, `code/laser_extra.py`, `code/laser_repeat.py`, `code/extra_checks.py` | `length_control.json`, `laser_extra.json`, `laser_repeat.json`, `extra_checks.json` |
| Figure 11 (reliability of R0 and R5) | `code/make_calibration_fig.py` -> `calibration.pdf` | per-solution readouts; binned values in `calibration_bins.json` |
| Appendix F.5, paired calibration test | `code/calib_paired.py` | `calibration_paired.json` |
| Figure 12 (weighted minus majority voting) | `code/make_laser_more_figs.py` -> `voting.pdf` | `*_confirm.json` |
| Figure 13 (tokens before and after score-only training in three models) | `code/newdata_figs.py` -> `trained_tokens.pdf` | `q3_replication.json`, `qwen35_4b_replication.json`, `gemma4_12b_replication.json` |
| Table 11 (every readout of the score-only models) | `code/newdata_tables.py` -> `app_trained.tex` | `newdata.json` |
| Appendix G, Qwen3-4B seeds and training details | `code/q3_*.py`, `code/qwen35_4b_*.py`, `code/gemma4_12b_*.py` | `q3_replication*.json`, `qwen35_4b_replication.json`, `gemma4_12b_replication.json` |
| Per-benchmark and per-seed tables of every metric (repository only) | `code/laser_paper_tables.py` -> `app_full.tex`, `app_perseed.tex`, `app_perseed_auroc.tex`, `app_secondary.tex`; `code/laser_perseed_tables.py` -> `app_perseed_metrics.tex`; `code/perseed_boot.py` | `*_confirm.json`, `perseed_boot_*.json`, `kendall_ci.json` |
| Token sweep, exhaustive-test and EOS-control tables (repository only) | `code/laser_new_tables.py` -> `app_sweep.tex`, `app_mech.tex`; `code/laser_exh_table.py` -> `app_exhaustive.tex` | `token_sweep_*.json`, `exhaustive_tokens_*.json`, `mech_summary.json` |
| Appendix A, released-scorer and vLLM checks | `code/compare_official.py`, `code/compare_readout_fast.py`, `code/compare_xchk.py` | `results/pilot/`, `results/_dev/` (created by the drivers) |

`laser_paper_tables.py` also writes `app_benchmarks.tex` (results by benchmark), which the paper does not include,
and a first version of `length_checks.tex`, which `laser_new_tables.py` replaces. `laser_new_tables.py` also writes
`app_q3.tex` (readouts of the first Qwen3-4B seed), which `app_trained.tex` replaces in the paper. `make_figures.py`
also writes `readout_auroc.pdf`, which the paper does not include. `token_control_*.json` (`code/token_control.py`)
holds the first token and terminator control on MATH500 and enters no table or figure.
