#!/usr/bin/env python3
"""Figures for the exploratory runs on new models.

trained_tokens.pdf (two columns): tokens read at the released point on MATH500 before and after training only the
                   score term, in Qwen3-4B, Qwen3.5-4B and Gemma-4-12B, with the median gain in log-probability.
unembed_2026.pdf   (one column): pairwise cosine of the output rows of unused tokens in eight models released in
                   2026, next to ordinary tokens and the four lowest-id unused tokens.
Reads results/analysis/{q3,qwen35_4b,gemma4_12b}_replication.json and unembed_2026.json. Writes outputs/figs/.
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
OUT = ROOT / "outputs/figs"
OUT.mkdir(parents=True, exist_ok=True)
eaclplot.setup()

# ---- tokens before and after score-only training ----
MODELS = [("q3_replication.json", "Qwen3-4B"), ("qwen35_4b_replication.json", "Qwen3.5-4B"),
          ("gemma4_12b_replication.json", "Gemma-4-12B")]


def token_rows(sweep):
    """Order the swept tokens as marker, R1 tokens, rare tokens, then three frequent ones."""
    marker, r1, rare, words = [], [], [], {}
    for v in sweep.values():
        name = v["name"]
        if name == "marker":
            marker.append(("marker", v))
        elif name.startswith("R1 token"):
            r1.append(v)
        elif name.startswith("under-trained word") or name.startswith("low-norm ordinary token"):
            rare.append(v)
        elif name.startswith("trained word"):
            words[name.split("'")[1]] = v
    rows = marker + [(f"$R_1$ token {i + 1}", v) for i, v in enumerate(r1)]
    rows += [(f"rare token {i + 1}", v) for i, v in enumerate(rare)]
    rows += [(label, words[w]) for w, label in [(" the", "``the''"), (" answer", "``answer''"), (".", "period")] if w in words]
    return rows


data = [(label, json.loads((A / f).read_text())["math500"]["token_sweep"]) for f, label in MODELS if (A / f).exists()]
fig, axes = plt.subplots(1, len(data), figsize=(eaclplot.FULL, 2.25), sharey=True, squeeze=False)
for j, (label, sweep) in enumerate(data):
    ax = axes[0][j]
    rows = token_rows(sweep)
    for i, (_, v) in enumerate(rows):
        ax.plot([v["auroc_base"], v["auroc_trained"]], [i, i], color=eaclplot.GRID, linewidth=2.0, zorder=1)
        ax.scatter([v["auroc_base"]], [i], s=12, color=eaclplot.GREY, zorder=2,
                   label="untrained" if (i == 0 and j == 0) else None)
        ax.scatter([v["auroc_trained"]], [i], s=12, color=eaclplot.BLUE, zorder=2,
                   label="trained" if (i == 0 and j == 0) else None)
        ax.text(1.03, i, f"+{v['median_shift_trained_minus_base']:.0f}", va="center", fontsize=6.5,
                color=eaclplot.INK, transform=ax.get_yaxis_transform())
    ax.text(1.03, -1.1, "nat", va="center", fontsize=6.5, color=eaclplot.INK, transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=eaclplot.GREY, linestyle="--", linewidth=0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -1.6)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel("AUROC on MATH500")
    eaclplot.panel_title(ax, label)
    eaclplot.finish(ax, "x")
fig.subplots_adjust(wspace=0.32)
fig.legend(loc="outside upper center", ncol=2, handletextpad=0.3, columnspacing=1.0)
eaclplot.save(fig, OUT / "trained_tokens.pdf")

# ---- unused rows in 2026 models ----
geo = [m for m in json.loads((A / "unembed_2026.json").read_text())["models"] if "error" not in m]
fig, ax = plt.subplots(figsize=(eaclplot.COL, 2.3))
for i, m in enumerate(geo):
    u, o = m["U"], m["O"]
    ax.scatter([o["pairwise_median"]], [i], s=14, facecolor="white", edgecolor=eaclplot.GREY, zorder=3,
               label="ordinary tokens" if i == 0 else None)
    if u["n"]:
        ax.plot([u["pairwise_p05"], u["pairwise_median"]], [i, i], color=eaclplot.BLUE_L, linewidth=2.0, zorder=1)
        ax.scatter([u["pairwise_median"]], [i], s=14, color=eaclplot.BLUE, zorder=3,
                   label="unused tokens" if i == 0 else None)
    else:
        ax.text(0.30, i, "no unused tokens", va="center", fontsize=6.5, color=eaclplot.INK)
    rule = m.get("rule_cosine_to_marker")
    if rule and len(rule) > 1:
        ax.scatter([min(rule[1:])], [i], s=16, marker="x", color=eaclplot.ORANGE, zorder=4,
                   label="lowest four unused ids" if i == 0 else None)
short = {"gemma-4-12B-it": "Gemma-4-12B", "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": "Nemotron-3-Nano-30B"}
names = [short.get(m["repo"].split("/")[1], m["repo"].split("/")[1]) for m in geo]
ax.set_yticks(range(len(geo)), names)
ax.invert_yaxis()
ax.set_xlim(-0.05, 1.05)
ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
ax.set_xlabel("cosine of output rows")
eaclplot.finish(ax, "x")
fig.legend(loc="outside upper center", ncol=2, handletextpad=0.3, columnspacing=0.8, fontsize=7)
eaclplot.save(fig, OUT / "unembed_2026.pdf")
print("figures written")
