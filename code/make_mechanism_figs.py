#!/usr/bin/env python3
"""Appendix figures that show the mechanisms behind the READSWAP results.

exhaustive.pdf  (two columns): score difference to the marker against the cosine of each token's output row
                 with the marker's, for every aligned and every special token (exhaustive test).
eos_vocab.pdf   (two columns): AUROC of the marker read after each candidate token in the EOS position.
terms.pdf       (two columns): the marker's logit after EOS for correct and wrong solutions, base model and
                 LaSeR checkpoint.
perseed.pdf     (two columns): AUROC of every readout on each generation seed.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eaclplot  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = ROOT / "results/analysis"
OUT = ROOT / "outputs/figs"
OUT.mkdir(parents=True, exist_ok=True)
CK = [("OctoThinker-3B-Short-LaSeR", "OctoThinker-3B"), ("Qwen2.5-7B-LaSeR", "Qwen2.5-7B"), ("ORZ-7B-LaSeR", "ORZ-7B")]
BETA = 0.1
eaclplot.setup()


def auroc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    order = s.argsort(kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    vals, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---- exhaustive test: same direction, same score ----
fig, axes = plt.subplots(1, 3, figsize=(eaclplot.FULL, 1.95), sharey=True)
for ax, (k, label) in zip(axes, CK):
    e = json.loads((A / f"exhaustive_tokens_{k}.json").read_text())
    m_auc = e["marker_auroc"]
    groups = [("aligned", eaclplot.BLUE, "o", "on the marker's direction"),
              ("special", eaclplot.ORANGE, "s", "other special tokens")]
    for name, col, mk, lab in groups:
        ts = [t for t in e["tokens"] if (t["aligned"] if name == "aligned" else (t["special"] and not t["aligned"]))]
        x = np.clip([1 - t["cos"] for t in ts], 1e-7, None)
        y = np.clip([BETA * t["max_abs_diff_lp"] for t in ts], 1e-4, None)
        ok = np.array([abs(t["auroc"] - m_auc) <= 0.02 and BETA * t["max_abs_diff_lp"] <= 0.02 for t in ts])
        ax.scatter(x[ok], y[ok], s=6, marker=mk, color=col, alpha=0.6, linewidths=0, label=lab)
        ax.scatter(x[~ok], y[~ok], s=12, marker=mk, facecolors="none", edgecolors=col, linewidths=0.7)
    ax.axhline(0.02, color=eaclplot.GREY, linestyle="--", linewidth=0.8)
    ax.axvline(1e-4, color=eaclplot.GREY, linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-8, 2)
    n_al = sum(t["aligned"] for t in e["tokens"])
    n_sp = sum(t["special"] and not t["aligned"] for t in e["tokens"])
    ax.text(0.03, 0.95, f"{n_al:,} aligned, {n_sp} other special", transform=ax.transAxes, va="top", fontsize=6.5)
    eaclplot.panel_title(ax, label)
    eaclplot.finish(ax, "y")
axes[0].set_ylabel("largest score difference")
axes[1].set_xlabel("1 $-$ cosine with the marker's output row")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=2, handletextpad=0.3, columnspacing=1.2, markerscale=1.6)
eaclplot.save(fig, OUT / "exhaustive.pdf")

# ---- EOS vocabulary sweep: nothing else in that position works as well ----
fig, axes = plt.subplots(1, 3, figsize=(eaclplot.FULL, 1.45), sharey=True)
rng = np.random.default_rng(0)
for ax, (k, label) in zip(axes, CK):
    v = json.loads((A / f"eos_vocab_{k}.json").read_text())
    rows = [("random", "200 random tokens", eaclplot.GREY), ("special", "special tokens", eaclplot.ORANGE)]
    for i, (kind, lab, col) in enumerate(rows):
        xs = np.array([t["auroc"] for t in v["tokens"] if t["kind"] == kind])
        ax.scatter(xs, i + rng.uniform(-0.18, 0.18, len(xs)), s=5, color=col, alpha=0.55, linewidths=0)
    ax.axvline(v["eos_auroc"], color=eaclplot.BLUE, linewidth=1.4)
    ax.text(v["eos_auroc"] - 0.02, 1.45, "EOS", color=eaclplot.BLUE, ha="right", va="center", fontsize=7)
    ax.axvline(0.5, color=eaclplot.GREY, linestyle="--", linewidth=0.6)
    ax.set_yticks([0, 1], [r[1] for r in rows])
    ax.set_ylim(-0.5, 1.7)
    ax.set_xlim(0, 1)
    eaclplot.panel_title(ax, label)
    eaclplot.finish(ax, "x")
axes[1].set_xlabel("AUROC of the marker read after the token")
eaclplot.save(fig, OUT / "eos_vocab.pdf")

# ---- the two terms: training turns the projection around ----
fig, axes = plt.subplots(2, 3, figsize=(eaclplot.FULL, 2.6))
for j, (k, label) in enumerate(CK):
    recs = [json.loads(x) for x in open(A / f"mech/{k}.jsonl")]
    y = np.array([r["correct"] for r in recs], bool)
    for i, (key, row) in enumerate([("base_eos_logit_marker", "base model"), ("eos_logit_marker", "LaSeR")]):
        ax = axes[i][j]
        s = np.array([r[key] for r in recs], float)
        lo, hi = np.percentile(s, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(s[y], bins=bins, density=True, histtype="step", color=eaclplot.BLUE, linewidth=1.0, label="correct")
        ax.hist(s[~y], bins=bins, density=True, histtype="step", color=eaclplot.ORANGE, linewidth=1.0, label="wrong")
        ax.text(0.5, 0.97, f"{row}: AUROC {auroc(y, s):.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.set_yticks([])
        if i == 0:
            eaclplot.panel_title(ax, label)
        elif j == 1:
            ax.set_xlabel("logit of the marker after EOS")
        eaclplot.finish(ax, "x")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=2, handlelength=1.4, columnspacing=1.2)
eaclplot.save(fig, OUT / "terms.pdf")

# ---- per seed ----
READ = [("R0_reference", "$R_0$"), ("R1_marker_swap_mean", "$R_1$"), ("R2_position_eos", "$R_2$"),
        ("R3_position_answer", "$R_3$"), ("R4_cref_none", "$R_4$"), ("R5_cref_recomputed", "$R_5$")]
fig, axes = plt.subplots(1, 3, figsize=(eaclplot.FULL, 1.6), sharey=True)
cols = [eaclplot.BLUE, eaclplot.ORANGE, eaclplot.GREEN]
for ax, (k, label) in zip(axes, CK):
    d = json.loads((A / f"{k}_confirm.json").read_text())
    for si, seed in enumerate(["0", "1", "2"]):
        xs = [d["per_seed_metrics"][seed][r]["auroc"] for r, _ in READ]
        ax.scatter(xs, np.arange(len(READ)) + (si - 1) * 0.18, s=9, color=cols[si], label=f"seed {seed}", zorder=2)
    ax.axvline(0.5, color=eaclplot.GREY, linestyle="--", linewidth=0.6)
    ax.set_yticks(range(len(READ)), [r[1] for r in READ])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC")
    eaclplot.panel_title(ax, label)
    eaclplot.finish(ax, "x")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3, handletextpad=0.2, columnspacing=1.0)
eaclplot.save(fig, OUT / "perseed.pdf")
print("mechanism figures written to", OUT)
