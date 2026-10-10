#!/usr/bin/env python3
"""Per-seed tables of every metric that the plan and Amendment A1 name.

For each checkpoint and each readout: primary (benchmark-weighted) AUROC, pooled-response AUROC, ECE, Brier score,
linear and exponential weighted voting, partial correlation with length and Kendall tau against R0, for each seed and
for the three seeds pooled. Writes outputs/tables/app_perseed_metrics.tex (three table* floats).
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
A = ROOT / "results/analysis"
OUT = ROOT / "outputs/tables/app_perseed_metrics.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)
CK = [("OctoThinker-3B-Short-LaSeR", "OctoThinker-3B", "octothinker"), ("Qwen2.5-7B-LaSeR", "Qwen2.5-7B", "qwen"),
      ("ORZ-7B-LaSeR", "ORZ-7B", "orz")]
RD = [("R0_reference", "$R_0$"), ("R1_marker_swap_mean", "$R_1$"), ("R2_position_eos", "$R_2$"),
      ("R3_position_answer", "$R_3$"), ("R4_cref_none", "$R_4$"), ("R5_cref_recomputed", "$R_5$")]


def f(x, d=3):
    if x is None:
        return "--"
    s = f"{x:.{d}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


L = []
for key, name, lab in CK:
    d = json.loads((A / f"{key}_confirm.json").read_text())
    L += ["\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
          "\\begin{tabular}{@{}llrrrrrrrr@{}}", "\\toprule",
          "Seed & Readout & AUROC & Pooled AUROC & ECE & Brier & Vote & Vote, $\\exp(s)$ & Partial $r$ & Kendall $\\tau$ \\\\",
          "\\midrule"]
    for si, seed in enumerate(["0", "1", "2", "all"]):
        if si:
            L.append("\\midrule")
        src = d["metrics"] if seed == "all" else d["per_seed_metrics"][seed]
        tau = (d["per_seed_metrics"][seed]["kendall_tau_vs_R0"] if seed != "all"
               else d["metrics"].get("kendall_tau_vs_R0", {}))
        for j, (r, rl) in enumerate(RD):
            m = src[r]
            L.append(("pooled" if seed == "all" else seed) * (j == 0) + " & " + rl + " & " + " & ".join([
                f(m.get("primary_auroc")), f(m.get("auroc_pooled")), f(m.get("ece_15")), f(m.get("brier")),
                f(m.get("weighted_majority_accuracy_k8")), f(m.get("exp_weighted_majority_accuracy_k8")),
                f(m.get("partial_correlation_score_length_given_correctness")), f(tau.get(r))]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          f"\\caption{{{name}: every metric that the plan and Amendment~A1 name, for each seed and for the three seeds "
          "pooled. Pooled AUROC ignores benchmarks; Vote uses linear weights. Intervals for the pooled values are in "
          "Tables~\\ref{tab:app-full-octothinker} to~\\ref{tab:app-full-orz}. The seeds agree closely.}",
          f"\\label{{tab:app-perseed-metrics-{lab}}}", "\\end{table*}", ""]
OUT.write_text("\n".join(L), encoding="utf-8")
(OUT.parent / (OUT.name + ".sha256")).write_text(hashlib.sha256(OUT.read_bytes()).hexdigest() + "  " + OUT.name + "\n")
print("written", OUT)
