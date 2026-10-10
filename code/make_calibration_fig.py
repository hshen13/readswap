#!/usr/bin/env python3
"""Reliability diagram of the released score (R0) and the base-reference score (R5), clipped to [0, 1].

Reads the confirmatory readout files directly (results/confirmatory/readout/, not part of the code release);
15 equal-width bins as in Amendment A1 (ECE).
Writes outputs/figs/calibration.pdf and results/analysis/calibration_bins.json.
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
CKPTS = [("OctoThinker-3B-Short-LaSeR", "OctoThinker-3B"), ("Qwen2.5-7B-LaSeR", "Qwen2.5-7B"), ("ORZ-7B-LaSeR", "ORZ-7B")]
DATASETS = ["math500", "amc23", "aime24", "aime25", "OlympiadBench"]
READS = [("R0_reference", "$R_0$ released", eaclplot.BLUE, "o", "-"), ("R5_cref_recomputed", "$R_5$ base reference", eaclplot.GREEN, "^", "--")]
edges = np.linspace(0, 1, 16)
out = {}
eaclplot.setup()
fig, axes = plt.subplots(1, 3, figsize=(eaclplot.FULL, 1.95), sharey=True, squeeze=False)
for i, (key, label) in enumerate(CKPTS):
    scores = {r[0]: [] for r in READS}
    correct = []
    files = [ROOT / f"results/confirmatory/readout/{key}/{d}_seed{s}.jsonl" for d in DATASETS for s in (0, 1, 2)]
    if not all(f.exists() for f in files):
        continue
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for k, sc in enumerate(r["readout_scores"]):
                correct.append(bool(r["correctness"][k]))
                for cond, *_ in READS:
                    scores[cond].append(sc[cond])
    y = np.array(correct, float)
    ax = axes[0][i]
    ax.plot([0, 1], [0, 1], color=eaclplot.GREY, linewidth=0.8, linestyle=":")
    out[key] = {}
    for cond, name, col, mk, ls in READS:
        s = np.clip(np.array(scores[cond]), 0, 1)
        idx = np.clip(np.digitize(s, edges) - 1, 0, 14)
        xs, ys, ns = [], [], []
        for b in range(15):
            m = idx == b
            if m.sum() >= 50:
                xs.append(float(s[m].mean())); ys.append(float(y[m].mean())); ns.append(int(m.sum()))
        out[key][cond] = {"mean_score": xs, "accuracy": ys, "n": ns}
        ax.plot(xs, ys, color=col, marker=mk, markersize=3, linestyle=ls, linewidth=1.0, label=name if i == 0 else None)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel("Score, clipped to [0, 1]")
    if i == 0:
        ax.set_ylabel("Share correct")
    eaclplot.panel_title(ax, label)
    eaclplot.finish(ax, "y")
fig.legend(loc="outside upper center", ncol=2, handlelength=1.8, handletextpad=0.4, columnspacing=1.0)
(ROOT / "outputs/figs").mkdir(parents=True, exist_ok=True)
eaclplot.save(fig, ROOT / "outputs/figs/calibration.pdf")
(ROOT / "results/analysis/calibration_bins.json").write_text(json.dumps(out, indent=1))
print("calibration figure:", list(out))
