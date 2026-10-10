"""Shared matplotlib style for the data figures of the paper.

Serif text matching the body (Times via newtx, typeset by pdflatex through the pgf backend, so every font is an
embedded Type 1 font), text >= 9 pt, Okabe-Ito colours (<= 5 per figure) with line style or marker as a second
cue, only left/bottom spines, light grid. Requires a LaTeX installation with pdflatex and the newtx fonts.

usage:
    import eaclplot
    eaclplot.setup()                      # before creating figures
    fig, ax = plt.subplots(figsize=(eaclplot.COL, 2.1))
    ...
    eaclplot.finish(ax)
    eaclplot.save(fig, "figs/x.pdf")

Include the PDF with \\includegraphics{...} at its natural size (no width=), so 9 pt stays 9 pt.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

# Widths in inches: ACL column is 7.7 cm, text width 16.0 cm.
COL = 3.03
FULL = 6.30

# Okabe-Ito palette (colour-blind safe). Names are kept stable so figure scripts need no change.
BLUE = "#0072B2"      # blue: the method's main condition
ORANGE = "#E69F00"    # orange: its swap / control
BROWN = "#D55E00"     # vermillion: a third condition
GREEN = "#009E73"     # bluish green: a fourth condition
GREY = "#7F7F7F"      # neutral reference (baselines, unweighted, none)
BLUE_L = "#56B4E9"    # sky blue
RED = "#CC79A7"       # reddish purple
BLUE_BG = "#E8F1F8"   # light tint of BLUE for bands and shaded rows
ORANGE_BG = "#FBF0D9"  # light tint of ORANGE
INK = "#222222"
GRID = "#DDDDDD"

# Series order: the main condition first, then its swap/control, then references.
SERIES = [BLUE, ORANGE, GREEN, GREY, BROWN, BLUE_L, RED]

PREAMBLE = "\n".join([
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{newtxtext}",
    r"\usepackage{newtxmath}",
])


def setup() -> None:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": PREAMBLE,
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.2,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        # Constrained layout keeps the saved PDF exactly at figsize (COL or FULL), so LaTeX includes it
        # at 100% and the text stays 9 pt.  Put legends outside with loc="outside upper center".
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "savefig.bbox": "standard",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def finish(ax, grid: str = "y") -> None:
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_title(ax, text: str, color: str = BLUE) -> None:
    """Bold left-aligned panel title in the Figure 1 title colour."""
    ax.set_title(text, loc="left", fontweight="bold", color=color, pad=4)


def save(fig, path) -> None:
    fig.savefig(str(path), backend="pgf")
    plt.close(fig)
