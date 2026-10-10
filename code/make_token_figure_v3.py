#!/usr/bin/env python3
"""Figure 2 (token control): token sweep and end-of-sequence control.

Top: AUROC of tokens read after EOS, sampled at random within bins of cosine similarity between their output row and
the marker's row (results/analysis/token_sweep_*.json, seed-zero MATH500). Bottom: the marker read after the token
at the end-of-sequence position, benchmark-weighted AUROC over the seed-zero solutions of all five benchmarks with
95% cluster-bootstrap intervals (results/analysis/mech_summary.json). Writes outputs/figs/token_control.pdf.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eaclplot  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
A = ROOT / "results/analysis"
CKPTS = [("OctoThinker-3B-Short-LaSeR", "OctoThinker-3B"), ("Qwen2.5-7B-LaSeR", "Qwen2.5-7B"), ("ORZ-7B-LaSeR", "ORZ-7B")]
BINS = [(0.9999, 1.01, r"$\geq$.9999"), (0.999, 0.9999, r"$\geq$.999"), (0.99, 0.999, r"$\geq$.99"),
        (0.9, 0.99, r"$\geq$.9"), (0.5, 0.9, r"$\geq$.5"), (-1.01, 0.5, r"$<$.5")]
EOS = [("eos", "EOS (released)"), ("newline", "newline"), ("period", "period"), ("chat_special", "chat token"),
       ("rare_word", "rare word"), ("none", "nothing ($R_2$)")]

eaclplot.setup()
colours = [eaclplot.BLUE, eaclplot.ORANGE, eaclplot.GREEN]
markers = ["o", "s", "D"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(eaclplot.COL, 3.55), gridspec_kw={"height_ratios": [1.0, 1.05]})
rng = np.random.default_rng(0)
mech = json.loads((A / "mech_summary.json").read_text())
for j, (key, label) in enumerate(CKPTS):
    sw = json.loads((A / f"token_sweep_{key}.json").read_text())
    for b, (lo, hi, _) in enumerate(BINS):
        vals = [t["auroc"] for t in sw["tokens"] if t["bin"] == [lo, hi]]
        x = b + (j - 1) * 0.22 + rng.uniform(-0.05, 0.05, len(vals))
        ax1.plot(x, vals, linestyle="none", marker=markers[j], markersize=2.6, alpha=0.75, color=colours[j],
                 label=label if b == 0 else None)
    ax1.axhline(sw["marker_auroc"], color=colours[j], linewidth=0.6, linestyle=":")
    m = mech[key]
    for i, (k, _) in enumerate(EOS):
        v = m[k]["auroc"]; lo, hi = m[k]["ci95"]
        y = i + (j - 1) * 0.24
        ax2.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt=markers[j], markersize=3.2, color=colours[j],
                     elinewidth=0.8, capsize=0)
ax1.set_xticks(range(len(BINS)), [b[2] for b in BINS])
ax1.set_xlabel("cosine with the marker's output row")
ax1.set_ylabel("AUROC")
ax1.set_ylim(0.0, 1.0)
ax1.axvspan(-0.5, 0.5, color=eaclplot.BLUE_BG, zorder=0, linewidth=0)
ax2.set_yticks(range(len(EOS)), [e[1] for e in EOS])
ax2.invert_yaxis()
ax2.set_xlim(0.2, 1.0)
ax2.axvline(0.5, color=eaclplot.GREY, linestyle="--", linewidth=0.8)
ax2.axhspan(-0.5, 0.5, color=eaclplot.BLUE_BG, zorder=0, linewidth=0)
ax2.set_xlabel("AUROC of the marker read after this token")
eaclplot.finish(ax1, "y")
eaclplot.finish(ax2, "x")
fig.legend(*ax1.get_legend_handles_labels(), loc="outside upper center", ncol=3, handlelength=1.0,
           handletextpad=0.3, columnspacing=0.8)
out = ROOT / "outputs/figs/token_control.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
eaclplot.save(fig, out)
print("written", out)
