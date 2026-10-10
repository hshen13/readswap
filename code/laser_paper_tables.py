#!/usr/bin/env python3
"""Main-text and appendix tables for the READSWAP paper, written from the analysis JSONs.

Inputs : results/analysis/<checkpoint>_confirm.json (pre-specified), results/analysis/laser_extra.json and
         results/analysis/unembedding_rows.json (exploratory).
Outputs: outputs/tables/readouts.tex, contrasts.tex, length_checks.tex, app_benchmarks.tex, app_perseed.tex,
         app_unembed.tex, app_full.tex, app_secondary.tex and app_perseed_auroc.tex, each with a .sha256.
Checkpoints without a confirm JSON are left out.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
A = Path(os.environ.get("LASER_ANALYSIS_DIR", str(ROOT / "results/analysis")))
OUT = ROOT / "outputs/tables"
CKPTS = [("OctoThinker-3B-Short-LaSeR", "OctoThinker-3B"), ("Qwen2.5-7B-LaSeR", "Qwen2.5-7B"),
         ("ORZ-7B-LaSeR", "ORZ-7B")]
SHORT = {"OctoThinker-3B": "OctoThinker", "Qwen2.5-7B": "Qwen2.5", "ORZ-7B": "ORZ"}
READOUTS = [("R0_reference", "R0 released"), ("R1_marker_swap_mean", "R1 other marker"),
            ("R2_position_eos", "R2 before EOS"), ("R3_position_answer", "R3 at the answer"),
            ("R4_cref_none", "R4 no constant"), ("R5_cref_recomputed", "R5 base reference")]
CONTRASTS = [("L1_auroc_R0_minus_R1", "L1", "AUROC R0$-$R1"), ("L2_auroc_R0_minus_R2", "L2", "AUROC R0$-$R2"),
             ("L3_partial_corr_R0", "L3", "partial $r$ (R0)"),
             ("L4_weighted_majority_R0_minus_R4", "L4", "RM@8 R0$-$R4"),
             ("L5_auroc_gap_short_long_R0_minus_R3", "L5", "gap R0$-$R3")]


def num(x, d=3):
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    if float(s.replace("$-$", "-")) == 0:
        s = f"{0:.{d}f}"
    return s

import re as _re


def mathify(text):
    text = _re.sub(r"\bR(\d)\s*\$-\$\s*R(\d)\b", r"$R_\1-R_\2$", text)
    text = _re.sub(r"(?<![\w$\\{])R([0-5])\b", r"$R_\1$", text)
    text = _re.sub(r"(?<![\w$\\{])L([1-5])\b", r"$L_\1$", text)
    text = text.replace("RM@8", "Vote").replace("Maj@8", "Majority")
    return text


def write(path, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    text = mathify("\n".join(lines) + "\n")
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")


confirm = {}
for key, label in CKPTS:
    p = A / f"{key}_confirm.json"
    if p.exists():
        confirm[key] = json.loads(p.read_text(encoding="utf-8"))
avail = [(k, l) for k, l in CKPTS if k in confirm]
n = len(avail)

# Table 1: readouts x checkpoints.
cols = "l" + "rrr" * n
lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{l}}}" for _, l in avail) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n)),
         "Readout & " + " & ".join(["AUROC & ECE & RM@8"] * n) + r" \\", r"\midrule"]
for cond, label in READOUTS:
    cells = []
    for k, _ in avail:
        m = confirm[k]["metrics"][cond]
        cells += [num(m["primary_auroc"]), num(m["ece_15"]), num(m["weighted_majority_accuracy_k8"])]
    lines.append(label + " & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
cells = []
for k, _ in avail:
    m = confirm[k]["metrics"]["R0_reference"]
    cells += ["", "", num(m["unweighted_majority_accuracy_k8"])]
lines.append(r"Unweighted vote & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Correctness signal of each readout on the same solutions: AUROC averaged over benchmarks,"
          r" calibration error and accuracy of score-weighted voting over eight samples; R3 covers the 95 to 97\% of"
          r" solutions with a box. Changing the marker or constant keeps the AUROC, and shortening the prefix"
          r" removes most of it.}",
          r"\label{tab:main}", r"\end{table*}"]
write(OUT / "readouts.tex", lines)

# Table 2: registered contrasts, one row per checkpoint with the interval on a second line.
def bold(s):
    return r"\textbf{" + s.replace("$-$", r"{\boldmath$-$}") + "}"


SCALE = {"L1": 1000.0}
lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r" & $L_1$: marker & $L_2$: EOS token & $L_3$: length & $L_4$: constant & $L_5$: answer position \\",
         r" & $\Delta$AUROC ($\times 10^{3}$) & $\Delta$AUROC & partial $r$ & $\Delta$vote accuracy"
         r" & $\Delta$gap, short vs.\ long \\",
         r"\midrule"]
for i, (k, label) in enumerate(avail):
    est, ivs = [], []
    for key, short, _ in CONTRASTS:
        c = confirm[k]["contrasts"][key]
        sc = SCALE.get(short, 1.0)
        d = 1 if short == "L1" else 3
        s = num(c["estimate"] * sc, d)
        if c["verdict"] == "holds":
            s = bold(s)
        elif c["verdict"] == "negligible":
            s += r"$^{\dagger}$"
        est.append(s)
        ivs.append(r"{\scriptsize[" + num(c["ci95"][0] * sc, d) + ", " + num(c["ci95"][1] * sc, d) + "]}")
    lines.append(label + " & " + " & ".join(est) + r" \\")
    lines.append(" & " + " & ".join(ivs) + r" \\" + ("[3pt]" if i < len(avail) - 1 else ""))
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Pre-specified contrasts for each checkpoint, with the 95\% cluster-bootstrap interval over"
          r" problems below each estimate. Bold marks a Holm-corrected $p<0.05$ with an interval that excludes"
          r" zero, and $^{\dagger}$ marks a 90\% interval inside the equivalence bound. $L_1$ is scaled by"
          r" $10^{3}$.}",
          r"\label{tab:contrasts}", r"\end{table*}"]
write(OUT / "contrasts.tex", lines)

# Table 3: exploratory length checks.
extra = json.loads((A / "laser_extra.json").read_text(encoding="utf-8"))
ext = [(k, l) for k, l in avail if isinstance(extra.get(k), dict) and "partial_all" in extra[k]]
rows = [
    ("Partial $r$, all solutions", lambda e, c: num(e["partial_all"], 2)),
    ("\\quad within problem", lambda e, c: num(e["partial_within_problem"], 2)),
    ("\\quad same, untruncated", lambda e, c: num(e["partial_within_problem_no_trunc"], 2)),
    ("Spearman, correct only", lambda e, c: num(e["spearman_within_problem_correct"]["mean"], 2)),
    ("AUROC, benchmark", lambda e, c: num(c["metrics"]["R0_reference"]["primary_auroc"], 2)),
    ("AUROC, within problem", lambda e, c: num(e["within_problem_auroc"]["mean"], 2)),
    ("Wrong beats right, shorter", lambda e, c: f"{100 * e['pairs']['incorrect_shorter_scores_higher']:.0f}\\%"),
    ("\\quad longer", lambda e, c: f"{100 * e['pairs']['incorrect_longer_scores_higher']:.0f}\\%"),
]
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         rf"\begin{{tabular}}{{@{{}}l{'r' * len(ext)}@{{}}}}", r"\toprule",
         "R0 score & " + " & ".join(SHORT[l] for _, l in ext) + r" \\", r"\midrule"]
for label, f in rows:
    lines.append(label + " & " + " & ".join(f(extra[k], confirm[k]) for k, _ in ext) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Length and selection checks of the released readout (exploratory). Partial correlations"
          r" of score and length given correctness, the mean within-problem Spearman correlation, and how often"
          r" a wrong solution outscores a right one to the same problem. Shorter wrong solutions usually win.}",
          r"\label{tab:length}", r"\end{table}"]
write(OUT / "length_checks.tex", lines)
print("tables:", [l for _, l in avail], "length table:", [l for _, l in ext])

# Appendix: per-benchmark results (seeds pooled).
BENCH = [("math500", "MATH500"), ("amc23", "AMC23"), ("aime24", "AIME24"), ("aime25", "AIME25"),
         ("OlympiadBench", "OlympiadBench")]
lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{6}{c}{AUROC} & \multicolumn{3}{c}{Voting accuracy} \\",
         r"\cmidrule(lr){3-8}\cmidrule(lr){9-11}",
         r"Checkpoint & Benchmark & R0 & R1 & R2 & R3 & R4 & R5 & RM@8 R0 & RM@8 R4 & Maj@8 \\", r"\midrule"]
for i, (k, l) in enumerate(avail):
    bb = confirm[k]["by_benchmark"]
    for j, (b, bl) in enumerate(BENCH):
        m = bb[b]
        cells = [num(m[c]["primary_auroc"]) for c, _ in READOUTS]
        cells += [num(m["R0_reference"]["weighted_majority_accuracy_k8"]),
                  num(m["R4_cref_none"]["weighted_majority_accuracy_k8"]),
                  num(m["R0_reference"]["unweighted_majority_accuracy_k8"])]
        lines.append((l if j == 0 else "") + f" & {bl} & " + " & ".join(cells) + r" \\")
    if i < n - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Results by benchmark, three seeds pooled. R1 and R4 match R0 on every benchmark, R2 and R3"
          r" keep at most a weak signal on the larger benchmarks, and dropping the constant lowers weighted voting"
          r" below unweighted voting on every benchmark with enough correct answers.}",
          r"\label{tab:app-bench}", r"\end{table*}"]
write(OUT / "app_benchmarks.tex", lines)

# Appendix: per-seed contrasts.
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.6pt}",
         r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Checkpoint & Seed & L1 & L2 & L3 & L4 & L5 \\", r"\midrule"]
for i, (k, l) in enumerate(avail):
    ps = confirm[k]["per_seed_contrasts"]
    for j, seed in enumerate(("0", "1", "2")):
        cells = [num(ps[seed][key], 3) for key, _, _ in CONTRASTS]
        lines.append((SHORT[l] if j == 0 else "") + f" & {seed} & " + " & ".join(cells) + r" \\")
    if i < n - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Contrasts computed on each generation seed alone. Every contrast keeps its sign and size"
          r" across seeds, except L5, whose estimate changes sign between seeds for OctoThinker, so none of the"
          r" pooled verdicts in Table~\ref{tab:contrasts} rests on a single seed.}",
          r"\label{tab:app-perseed}", r"\end{table}"]
write(OUT / "app_perseed.tex", lines)

# Appendix: output-embedding rows of the marker and the swapped tokens (exploratory).
un = json.loads((A / "unembedding_rows.json").read_text(encoding="utf-8"))
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r" & Norm & Min.\ cos. & Max.\ dist. & Change \\", r"\midrule"]
for key, label in CKPTS:
    e = un.get(key, {})
    ck = e.get("checkpoint", {})
    if "cos_marker_swap" not in ck:
        continue
    rel = 100 * e["marker_change_l2"] / ck["marker_norm"]
    lines.append(f"{SHORT[label]} & {ck['marker_norm']:.3f} & {min(ck['cos_marker_swap']):.6f} & "
                 f"{max(ck['l2_marker_swap']):.4f} & {rel:.2f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Output-embedding row of the marker in each released checkpoint: its norm, its smallest"
          r" cosine similarity and largest distance to the three swapped tokens, and how far LaSeR training moved"
          r" it from the base model, relative to its norm. The rows stay almost identical.}",
          r"\label{tab:unembed}", r"\end{table}"]
write(OUT / "app_unembed.tex", lines)
print("appendix tables written")


# Appendix E: full results with intervals, one table per checkpoint (benchmark x readout, seeds pooled).
def iv(m, key, d=3):
    ci = m.get(key + "_ci95")
    v = m.get(key)
    if v is None:
        return "--"
    if not ci or ci[0] is None:
        return num(v, d)
    return num(v, d) + r" {\tiny[" + num(ci[0], d) + ", " + num(ci[1], d) + "]}"


full = []
for k, l in avail:
    bb = confirm[k]["by_benchmark"]
    full += [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\begin{tabular}{@{}llllll@{}}", r"\toprule",
             r"Benchmark & Readout & AUROC & ECE & Brier & RM@8 \\", r"\midrule"]
    for j, (b, bl) in enumerate(BENCH):
        for i, (cond, lab) in enumerate(READOUTS):
            m = bb[b][cond]
            full.append((bl if i == 0 else "") + f" & {lab.split()[0]} & " + iv(m, "primary_auroc") + " & "
                        + iv(m, "ece_15") + " & " + iv(m, "brier") + " & " + iv(m, "weighted_majority_accuracy_k8") + r" \\")
        if j < len(BENCH) - 1:
            full.append(r"\addlinespace[2pt]")
    full += [r"\bottomrule", r"\end{tabular}",
             rf"\caption{{Full results of {l}, three seeds pooled, with 95\% cluster-bootstrap intervals over problems."
             r" Each benchmark repeats the checkpoint-level pattern: R1 and R4 match R0 in AUROC, R5 lowers the"
             r" calibration error, and R2 and R3 lose most of the separation of correct from incorrect solutions.}",
             rf"\label{{tab:app-full-{SHORT[l].lower()}}}", r"\end{table*}", ""]
write(OUT / "app_full.tex", full)

# Appendix F: every pre-specified secondary metric at checkpoint level.
SEC = [("R0_reference", "R0"), ("R1_marker_swap_mean", "R1 mean")]
lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
         r"\begin{tabular}{@{}lllllll@{}}", r"\toprule",
         r"Checkpoint & Readout & AUROC & Brier & RM@8 & RM@8, $\exp(s)$ & Kendall $\tau$ vs R0 \\", r"\midrule"]
for i, (k, l) in enumerate(avail):
    M = confirm[k]["metrics"]
    tau = M.get("kendall_tau_vs_R0", {})
    tci = M.get("kendall_tau_vs_R0_ci95", {}) or {}
    tok_conds = sorted(c for c in M if c.startswith("R1_marker_swap_token_"))
    conds = [("R0_reference", "R0"), ("R1_marker_swap_mean", "R1 mean")] + \
            [(c, "R1 token " + c.rsplit("_", 1)[1]) for c in tok_conds] + \
            [(c, lab.split()[0]) for c, lab in READOUTS[2:]]
    for j, (c, lab) in enumerate(conds):
        m = M[c]
        if c in tau and tau.get(c) is not None:
            t = num(tau[c], 4)
            if tci.get(c):
                t += r" {\tiny[" + num(tci[c][0], 4) + ", " + num(tci[c][1], 4) + "]}"
        else:
            t = "--"
        lines.append((SHORT[l] if j == 0 else "") + f" & {lab} & " + iv(m, "primary_auroc") + " & " + iv(m, "brier")
                     + " & " + iv(m, "weighted_majority_accuracy_k8") + " & " + iv(m, "exp_weighted_majority_accuracy_k8")
                     + f" & {t}" + r" \\")
    if i < n - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Pre-specified secondary metrics for every readout, with 95\% cluster-bootstrap intervals: Brier"
          r" score, accuracy of linear and exponential score-weighted voting, Kendall $\tau$ against R0, and each"
          r" R1 token on its own. Every R1 token matches R0, and exponential weights make R4 equal to R0.}",
          r"\label{tab:app-secondary}", r"\end{table*}"]
write(OUT / "app_secondary.tex", lines)

# Appendix E: AUROC per seed (all benchmarks of a seed).
lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Checkpoint & Seed & R0 & R1 & R2 & R3 & R4 & R5 \\", r"\midrule"]
for i, (k, l) in enumerate(avail):
    psm = confirm[k].get("per_seed_metrics", {})
    for j, seed in enumerate(("0", "1", "2")):
        cells = [num(psm[seed][c]["primary_auroc"]) for c, _ in READOUTS]
        lines.append((SHORT[l] if j == 0 else "") + f" & {seed} & " + " & ".join(cells) + r" \\")
    if i < n - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Primary AUROC of each readout computed on each generation seed alone, with the five benchmarks"
          r" of that seed. The three seeds of a checkpoint agree to within a few thousandths, so the pooled"
          r" estimates do not hide a seed that behaves differently.}",
          r"\label{tab:app-perseed-auroc}", r"\end{table}"]
write(OUT / "app_perseed_auroc.tex", lines)
print("appendix full/secondary/per-seed written")
