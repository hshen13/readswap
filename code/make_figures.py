#!/usr/bin/env python3
"""Generate two vector figures from the confirmatory analysis JSONs.

Run ``python code/make_figures.py --analysis-root results/analysis
--output-dir outputs``.  Writes ``<output-dir>/figs/length_score.pdf`` (the
length figure of the paper) and ``<output-dir>/figs/readout_auroc.pdf``.
Without ``--dev``, output under ``outputs/`` requires confirm_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(root: Path) -> list[dict]:
    paths = sorted(root.glob("*_confirm.json")) or sorted(root.glob("*.json"))
    paths = [p for p in paths if p.name != "confirm_summary.json"]
    if not paths:
        raise RuntimeError(f"no analysis JSON in {root}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    if not args.dev and "outputs" in out.parts:
        summary = Path(args.analysis_root) / "confirm_summary.json"
        if not summary.exists():
            raise RuntimeError("output under outputs/ requires the complete confirmatory summary")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eaclplot
    import matplotlib.pyplot as plt
    import numpy as np

    eaclplot.setup()
    # Fixed checkpoint order and labels keyed by the model id, never by file order.
    order = ["Keven16/OctoThinker-3B-Short-LaSeR", "Keven16/Qwen2.5-7B-LaSeR", "Keven16/ORZ-7B-LaSeR"]
    short = {"Keven16/OctoThinker-3B-Short-LaSeR": "OctoThinker-3B", "Keven16/Qwen2.5-7B-LaSeR": "Qwen2.5-7B",
             "Keven16/ORZ-7B-LaSeR": "ORZ-7B"}
    analyses = sorted(load(Path(args.analysis_root)),
                      key=lambda a: order.index(a.get("model")) if a.get("model") in order else len(order))
    out.mkdir(parents=True, exist_ok=True)
    colours = [eaclplot.BLUE, eaclplot.ORANGE, eaclplot.GREEN]
    markers = ["o", "s", "D"]
    names = {"$R_0$": "$R_0$ released", "$R_1$": "$R_1$ other marker", "$R_2$": "$R_2$ before EOS",
             "$R_3$": "$R_3$ at the answer", "$R_4$": "$R_4$ no constant", "$R_5$": "$R_5$ base reference"}
    conditions = [("$R_0$", "R0_reference"), ("$R_1$", "R1_marker_swap_mean"), ("$R_2$", "R2_position_eos"),
                  ("$R_3$", "R3_position_answer"), ("$R_4$", "R4_cref_none"), ("$R_5$", "R5_cref_recomputed")]

    fig, ax = plt.subplots(figsize=(eaclplot.COL, 2.55))
    y = np.arange(len(conditions), dtype=float)
    for j, a in enumerate(analyses):
        xs, lo, hi = [], [], []
        for _, condition in conditions:
            m = a["metrics"][condition]
            xs.append(m.get("primary_auroc", m.get("auroc")))
            ci = m.get("primary_auroc_ci95") or m.get("auroc_ci95") or [xs[-1], xs[-1]]
            lo.append(ci[0]); hi.append(ci[1])
        offset = (j - (len(analyses) - 1) / 2) * 0.2
        ax.errorbar(xs, y + offset, xerr=[np.asarray(xs) - np.asarray(lo), np.asarray(hi) - np.asarray(xs)],
                    fmt=markers[j % len(markers)], linestyle="none", color=colours[j % len(colours)], label=short.get(a.get("model"), str(a.get("model"))),
                    capsize=0, elinewidth=.7, markersize=3.5)
    ax.axvline(.5, color=eaclplot.GREY, linestyle="--", linewidth=.8)
    ax.set_yticks(y, [names[x[0]] for x in conditions]); ax.invert_yaxis(); ax.set_xlabel("AUROC")
    eaclplot.finish(ax, "x")
    fig.legend(loc="outside upper center", ncol=3, handlelength=1.0, handletextpad=0.3, columnspacing=0.7)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    eaclplot.save(fig, out / "figs" / "readout_auroc.pdf")

    fig, ax = plt.subplots(figsize=(eaclplot.COL, 2.35))
    for condition, color, label, mk in (("R0_reference", eaclplot.BLUE, "$R_0$", "o"), ("R3_position_answer", eaclplot.ORANGE, "$R_3$", "s")):
        for correct, style, suffix in (("correct", "-", " correct"), ("incorrect", "--", " incorrect")):
            curves = []
            for a in analyses:
                curve = a.get("length_score", {}).get(condition, {}).get(correct, [])
                curves.append([np.nan if x is None else x for x in curve])
            mean = np.nanmean(np.asarray(curves, dtype=float), axis=0)
            ax.plot(range(1, 11), mean, color=color, linestyle=style, marker=mk, label=label + suffix)
    ax.set_xlabel("Response-length decile"); ax.set_ylabel("Within-model z-scored score")
    eaclplot.finish(ax, "y"); fig.legend(loc="outside upper center", ncol=2)
    eaclplot.save(fig, out / "figs" / "length_score.pdf")


if __name__ == "__main__":
    main()
