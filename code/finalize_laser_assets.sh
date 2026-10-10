#!/usr/bin/env bash
# Regenerate every table and data figure of the paper from the analysis JSONs in results/analysis/.
# Tables go to outputs/tables/, figures to outputs/figs/. The figures need pdflatex (matplotlib pgf backend).
# make_calibration_fig.py reads the per-solution readouts in results/confirmatory/readout/ and is skipped
# when they are absent (the released results/analysis/calibration_bins.json holds its binned output).
set -u
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
test -s results/analysis/ORZ-7B-LaSeR_confirm.json || { echo "ORZ confirm JSON missing"; exit 1; }
"$PY" code/laser_paper_tables.py
"$PY" code/make_figures.py --analysis-root results/analysis --output-dir outputs
"$PY" code/make_laser_more_figs.py
"$PY" code/make_token_figure_v3.py
"$PY" code/make_mechanism_figs.py
if [ -d results/confirmatory/readout/ORZ-7B-LaSeR ]; then
  "$PY" code/make_calibration_fig.py
else
  echo "skipping make_calibration_fig.py: results/confirmatory/readout/ not present"
fi
"$PY" - <<'EOF'
import json
d = json.load(open("results/analysis/ORZ-7B-LaSeR_confirm.json"))
for k, v in d["contrasts"].items():
    print(k, round(v["estimate"], 4), [round(x, 4) for x in v["ci95"]], [round(x, 4) for x in v["ci90"]], v["verdict"], v["p_holm"])
s = d["sensitivity_r3_as_r2"]
print("sens", round(s["l5"], 4), [round(x, 4) for x in s["l5_ci95"]],
      round(s["metrics"]["R3_position_answer"]["primary_auroc"], 3),
      [round(x, 3) for x in s["metrics"]["R3_position_answer"]["primary_auroc_ci95"]])
EOF
echo "finalized $(date -u +%FT%TZ)"
"$PY" code/laser_new_tables.py
"$PY" code/laser_perseed_tables.py
"$PY" code/laser_exh_table.py
