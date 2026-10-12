#!/usr/bin/env python3
"""Two data figures for READSWAP from the confirmatory JSONs.

voting.pdf (one column): weighted-voting accuracy of each readout minus unweighted voting, per checkpoint.
benchmarks.pdf (two columns): primary AUROC of R0, R2, R3 and R5 on each benchmark, one panel per checkpoint,
               followed by AIME and HMMT 2026 when results/analysis/newdata.json exists (exploratory).
LASER_ANALYSIS_DIR overrides the analysis directory. Writes outputs/figs/.
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
A = Path(os.environ.get("LASER_ANALYSIS_DIR", str(ROOT / "results/analysis")))
OUT = ROOT / "outputs/figs"
OUT.mkdir(parents=True, exist_ok=True)
CKPTS = [("OctoThinker-3B-Short-LaSeR", "OctoThinker-3B"), ("Qwen2.5-7B-LaSeR", "Qwen2.5-7B"), ("ORZ-7B-LaSeR", "ORZ-7B")]
data = [(label, json.loads((A / f"{k}_confirm.json").read_text())) for k, label in CKPTS if (A / f"{k}_confirm.json").exists()]
colours = [eaclplot.BLUE, eaclplot.ORANGE, eaclplot.GREEN]
markers = ["o", "s", "D"]
eaclplot.setup()

# ---- voting ----
rows = [("$R_0$ released", "R0_reference", "weighted_majority_accuracy_k8"),
        ("$R_2$ before EOS", "R2_position_eos", "weighted_majority_accuracy_k8"),
        ("$R_3$ at the answer", "R3_position_answer", "weighted_majority_accuracy_k8"),
        ("$R_4$ no constant", "R4_cref_none", "weighted_majority_accuracy_k8"),
        ("$R_4$, weights $\\exp(s)$", "R4_cref_none", "exp_weighted_majority_accuracy_k8"),
        ("$R_5$ base reference", "R5_cref_recomputed", "weighted_majority_accuracy_k8")]
fig, ax = plt.subplots(figsize=(eaclplot.COL, 2.2))
y = np.arange(len(rows), dtype=float)
for j, (label, d) in enumerate(data):
    base = d["metrics"]["R0_reference"]["unweighted_majority_accuracy_k8"]
    xs = [100 * (d["metrics"][cond][key] - base) for _, cond, key in rows]
    off = (j - (len(data) - 1) / 2) * 0.22
    ax.plot(xs, y + off, linestyle="none", marker=markers[j], markersize=3.5, color=colours[j], label=label)
ax.axvline(0, color=eaclplot.GREY, linestyle="--", linewidth=0.8)
ax.set_yticks(y, [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlabel("Gain over unweighted vote")
ax.set_xlim(-23, 4)
eaclplot.finish(ax, "x")
fig.legend(loc="outside upper center", ncol=len(data), handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
eaclplot.save(fig, OUT / "voting.pdf")

# ---- per-benchmark AUROC ----
bench = [("math500", "MATH500"), ("amc23", "AMC23"), ("aime24", "AIME24"), ("aime25", "AIME25"), ("OlympiadBench", "OlympiadBench")]
new_sets = [("aime_2026", "AIME 2026"), ("hmmt_feb_2026", "HMMT 2026")]
newdata = json.loads((A / "newdata.json").read_text())["laser_new_problems"] if (A / "newdata.json").exists() else {}
keys = {label: key for key, label in CKPTS}
reads = [("R0_reference", "$R_0$", eaclplot.BLUE, "o"), ("R5_cref_recomputed", "$R_5$", eaclplot.GREEN, "^"),
         ("R2_position_eos", "$R_2$", eaclplot.ORANGE, "s"), ("R3_position_answer", "$R_3$", eaclplot.BROWN, "D")]
shown = bench + (new_sets if newdata else [])
fig, axes = plt.subplots(1, len(data), figsize=(eaclplot.FULL, 2.2 if newdata else 1.75), sharey=True, squeeze=False)
for i, (label, d) in enumerate(data):
    ax = axes[0][i]
    yb = np.arange(len(shown), dtype=float)
    for r_i, (cond, rl, col, mk) in enumerate(reads):
        xs = [d["by_benchmark"][b][cond]["primary_auroc"] for b, _ in bench]
        if newdata:
            xs += [newdata[keys[label]]["by_dataset"][b]["readouts"][cond]["primary_auroc"] for b, _ in new_sets]
        off = (r_i - (len(reads) - 1) / 2) * 0.16
        ax.plot(xs, yb + off, linestyle="none", marker=mk, markersize=3.2, color=col, label=rl if i == 0 else None)
    ax.axvline(0.5, color=eaclplot.GREY, linestyle="--", linewidth=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    if newdata:
        ax.axhline(len(bench) - 0.5, color=eaclplot.GREY, linestyle=":", linewidth=0.6)
    ax.set_yticks(yb, [b[1] for b in shown])
    ax.invert_yaxis()
    ax.set_xlabel("AUROC")
    eaclplot.panel_title(ax, label)
    eaclplot.finish(ax, "x")
fig.legend(loc="outside upper center", ncol=len(reads), handlelength=1.0, handletextpad=0.3, columnspacing=0.9)
eaclplot.save(fig, OUT / "benchmarks.pdf")
print("figures:", [l for l, _ in data])
