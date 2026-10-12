#!/usr/bin/env python3
"""Tables for the supplementary experiments: the LaSeR checkpoints on 2026 problems and score-only training.

newdata.tex         (main text, one column): AUROC of R0, R1, R2 and length alone over all solutions, and of R0
                    and length within problems, for the LaSeR checkpoints on the 2026 problems and for the models
                    trained only on the score term.
app_newdata.tex     (appendix, two columns): the same with 95% intervals, by 2026 set, with the partial
                    correlation of R0 with length.
app_trained.tex     (appendix, two columns): every readout of the three score-only models on the five benchmarks
                    and on the 2026 problems.
app_unembed2026.tex (appendix, two columns): output rows of unused tokens in eight models released in 2026.
Reads results/analysis/newdata.json and results/analysis/unembed_2026.json. Writes outputs/tables/.
"""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
A = ROOT / "results/analysis"
OUT = Path(os.environ.get("TABLE_OUT", str(ROOT / "outputs/tables")))
OUT.mkdir(parents=True, exist_ok=True)
LASER = [("OctoThinker-3B-Short-LaSeR", "OctoThinker-3B"), ("Qwen2.5-7B-LaSeR", "Qwen2.5-7B"), ("ORZ-7B-LaSeR", "ORZ-7B")]
TRAINED = [("Qwen3-4B seed 0", "Qwen3-4B"), ("Qwen3.5-4B", "Qwen3.5-4B"), ("Gemma-4-12B", "Gemma-4-12B")]
SETS = [("aime_2026", "AIME 2026"), ("hmmt_feb_2026", "HMMT 2026")]
PC = "partial_correlation_score_length_given_correctness"


def num(x, digits=2):
    if x is None:
        return "--"
    return f"{x:.{digits}f}".replace("-", "$-$")


def ci(pair, digits=2):
    return "" if not pair else "{\\scriptsize[" + num(pair[0], digits) + ", " + num(pair[1], digits) + "]}"


def values(block):
    r, w = block["readouts"], block["within_problem"]
    return [r["R0_reference"]["primary_auroc"], r["R1_marker_swap_mean"]["primary_auroc"],
            r["R2_position_eos"]["primary_auroc"], r["length_only"]["primary_auroc"],
            w["R0_reference"]["auroc"], w["length_only"]["auroc"]]


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("wrote", OUT / name)


new = json.loads((A / "newdata.json").read_text())

# ---- main table ----
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{AUROC, all solutions} & \multicolumn{2}{c}{Within problem} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
         r" & $R_0$ & $R_1$ & $R_2$ & length & $R_0$ & length \\", r"\midrule",
         r"\multicolumn{7}{@{}l}{\emph{LaSeR models, AIME and HMMT 2026}} \\"]
for key, label in LASER:
    lines.append(label + " & " + " & ".join(num(v) for v in values(new["laser_new_problems"][key]["pooled"])) + r" \\")
lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Score term trained alone, five benchmarks}} \\"]
for key, label in TRAINED:
    lines.append(label + " & " + " & ".join(num(v) for v in values(new["trained_models"][key]["benchmarks"])) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Supplementary experiments: the LaSeR models on 2026 problems, and models trained on the "
          r"score term alone. Length: negative token count. Within problem: mean AUROC over "
          r"problems with correct and wrong solutions. Intervals in Table~\ref{tab:app-newdata}.}",
          r"\label{tab:new}", r"\end{table}"]
write("newdata.tex", lines)

# ---- appendix: the same with intervals ----
head = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{AUROC, all solutions} & \multicolumn{2}{c}{Within problem} & Partial $r$ \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
        r" & & $R_0$ & $R_1$ & $R_2$ & length & $R_0$ & length & with length \\", r"\midrule"]
lines = list(head)


def rows(label, data, block):
    r, w = block["readouts"], block["within_problem"]
    top = [num(v) for v in values(block)] + [num(r["R0_reference"][PC])]
    bottom = [ci(r[k].get("primary_auroc_ci95")) for k in ["R0_reference", "R1_marker_swap_mean", "R2_position_eos", "length_only"]]
    bottom += ["", f"{{\\scriptsize {w['R0_reference']['eligible_problems']} problems}}", ci(r["R0_reference"].get(PC + "_ci95"))]
    return [label + " & " + data + " & " + " & ".join(top) + r" \\", " & & " + " & ".join(bottom) + r" \\[2pt]"]


for key, label in LASER:
    for i, (ds, dname) in enumerate(SETS):
        lines += rows(label if i == 0 else "", dname, new["laser_new_problems"][key]["by_dataset"][ds])
lines.append(r"\midrule")
for key, label in TRAINED:
    entry = new["trained_models"][key]
    lines += rows(label, "benchmarks", entry["benchmarks"])
    if "new_problems" in entry:
        lines += rows("", "2026 sets", entry["new_problems"])
lines[-1] = lines[-1].replace(r" \\[2pt]", r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Supplementary experiments with 95\% problem-bootstrap intervals (10,000 resamples). Top: the "
          r"LaSeR checkpoints on each 2026 set, three seeds pooled. Bottom: the models trained on the score term "
          r"alone, first seed. Within-problem values average over the problems listed, those with correct and wrong "
          r"solutions.}",
          r"\label{tab:app-newdata}", r"\end{table*}"]
write("app_newdata.tex", lines)

# ---- appendix: every readout of the score-only models ----
readouts = [("$R_0$ released readout", "R0_reference"), ("$R_1$ other unused tokens", "R1_marker_swap_mean"),
            ("$R_2$ without EOS", "R2_position_eos"), ("$R_3$ at the answer", "R3_position_answer"),
            ("$R_4$ no constant", "R4_cref_none"), ("$R_5$ base reference", "R5_cref_recomputed"),
            ("length only", "length_only"), ("untrained model, $R_0$", "BASE_R0")]
lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l" + "cc" * len(TRAINED) + "@{}}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{2}{c}{" + label + "}" for _, label in TRAINED) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(TRAINED))),
         " & " + " & ".join(["benchmarks & 2026"] * len(TRAINED)) + r" \\", r"\midrule"]


def cell(key, part, fn):
    block = new["trained_models"][key].get(part)
    return fn(block) if block else "--"


for label, cond in readouts:
    vals = [cell(key, part, lambda b: num(b["readouts"][cond]["primary_auroc"], 3))
            for key, _ in TRAINED for part in ["benchmarks", "new_problems"]]
    lines.append(label + " & " + " & ".join(vals) + r" \\")
lines.append(r"\midrule")
extra = [("partial $r$ of $R_0$ with length", lambda b: num(b["readouts"]["R0_reference"][PC], 2)),
         ("ECE of $R_0$", lambda b: num(b["readouts"]["R0_reference"]["ece_15"], 3)),
         (r"vote gain, weights $\exp(s)$ (points)",
          lambda b: num(100 * (b["readouts"]["R0_reference"]["exp_weighted_majority_accuracy_k8"]
                               - b["readouts"]["R0_reference"]["unweighted_majority_accuracy_k8"]), 1)),
         ("accuracy (\\%)", lambda b: num(100 * b["accuracy"], 0))]
for label, fn in extra:
    vals = [cell(key, part, fn) for key, _ in TRAINED for part in ["benchmarks", "new_problems"]]
    lines.append(label + " & " + " & ".join(vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Models trained on LaSeR's self-score term alone (first seed), read on their own fixed "
          r"solutions to the five benchmarks and to AIME and HMMT 2026: AUROC of every readout, the partial "
          r"correlation of $R_0$ with length given correctness, its calibration error, the gain of weighted over "
          r"majority voting and the share of correct solutions. Qwen3-4B was read on the five benchmarks.}",
          r"\label{tab:app-trained}", r"\end{table*}"]
write("app_trained.tex", lines)

# ---- appendix: unused-token rows in 2026 models ----
geo = json.loads((A / "unembed_2026.json").read_text())
short = {"gemma-4-12B-it": "Gemma-4-12B", "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": "Nemotron-3-Nano-Omni-30B"}
lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Unused tokens} & \multicolumn{2}{c}{Pairwise cosine, median} & Lowest four ids \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Model & Released & count & median cosine & 5th pct. & special & ordinary & min.\ cosine \\",
         r"\midrule"]
for m in geo["models"]:
    if "error" in m:
        continue
    u, s, o = m["U"], m["S"], m["O"]
    rule = m.get("rule_cosine_to_marker")
    name = short.get(m["repo"].split("/")[1], m["repo"].split("/")[1])
    lines.append(" & ".join([
        name, m["created"], str(u["n"]), num(u.get("pairwise_median"), 3), num(u.get("pairwise_p05"), 3),
        num(s.get("pairwise_median"), 3), num(o.get("pairwise_median"), 3),
        num(min(rule[1:]), 3) if rule and len(rule) > 1 else "--"]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Output rows of unused tokens in open models first released in 2026. Unused "
          r"tokens are those named unused or reserved and the padding rows past the tokenizer's vocabulary; "
          r"special tokens are all other added tokens; ordinary tokens are 2,000 drawn at random. The last column "
          r"gives the smallest cosine between the lowest-id unused token and the next three.}",
          r"\label{tab:app-unembed2026}", r"\end{table*}"]
write("app_unembed2026.tex", lines)
