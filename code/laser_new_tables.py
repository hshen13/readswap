#!/usr/bin/env python3
"""Tables from the exploratory analyses, written to outputs/tables/.

length_checks.tex  (main, label tab:length): length controls of R0, from laser_extra.json and length_control.json
app_sweep.tex      (appendix): token sweep by cosine bin, AUROC range and raw log-probability differences
app_mech.tex       (appendix): EOS control on all benchmarks and the two terms of the log-probability
app_q3.tex         (appendix): Qwen3-4B replication of the self-reward objective
Each table gets a .sha256 next to it, as the other generated tables do.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
A = ROOT / "results/analysis"
OUT = ROOT / "outputs/tables"
CK = [("OctoThinker-3B-Short-LaSeR", "OctoThinker"), ("Qwen2.5-7B-LaSeR", "Qwen2.5"), ("ORZ-7B-LaSeR", "ORZ")]


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / (name + ".sha256")).write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + name + "\n")


def f(x, d=2):
    s = f"{x:.{d}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def ci(v, d=2):
    return "{\\footnotesize[" + f(v[0], d) + ", " + f(v[1], d) + "]}"


ext = json.loads((A / "laser_extra.json").read_text())
lc = json.loads((A / "length_control.json").read_text())
NAMES = {"OctoThinker-3B-Short-LaSeR": "OctoThinker-3B", "Qwen2.5-7B-LaSeR": "Qwen2.5-7B", "ORZ-7B-LaSeR": "ORZ-7B"}


def sci(v, d=2):
    return "{\\scriptsize[" + f(v[0], d) + ", " + f(v[1], d) + "]}"


L = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4.3pt}",
     "\\begin{tabular}{@{}lccccccccc@{}}", "\\toprule",
     " & \\multicolumn{2}{c}{Partial $r$ with length} & \\multicolumn{3}{c}{AUROC, all solutions}"
     " & \\multicolumn{2}{c}{AUROC, within problem} & Matched & Vote gain \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
     " & all & within problem & $R_0$ & length & $R_0$ w/o length & $R_0$ & length & pairs & (points) \\\\",
     "\\midrule"]
for i, (k, _) in enumerate(CK):
    o, c = lc[k]["observed"], lc[k]["ci95"]
    L.append(NAMES[k] + " & " + " & ".join([
        f(ext[k]["partial_all"]), f(ext[k]["partial_within_problem"]),
        f(o["benchmark_auroc_r0"]), f(o["benchmark_auroc_len"]), f(o["benchmark_auroc_r0_minus_loglen"]),
        f(o["within_problem_auroc_r0"]), f(o["within_problem_auroc_len"]), f(o["matched_pair_auroc"]),
        f(100 * o["vote_gain_linear"], 1)]) + " \\\\")
    L.append(" & " * 6 + " & ".join([
        sci(c["within_problem_auroc_r0"]), sci(c["within_problem_auroc_len"]), sci(c["matched_pair_auroc"]),
        sci([100 * x for x in c["vote_gain_linear"]], 1)]) + " \\\\" + ("[3pt]" if i < len(CK) - 1 else ""))
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Length controls of the released score $R_0$. Length scores a solution by its negative "
      "token count; $R_0$ w/o length removes a linear fit on log length, cross-fitted over problems; matched pairs "
      "join a correct and a wrong solution to one problem within 10\\% in length. Brackets: 95\\% problem-bootstrap "
      "intervals.}",
      "\\label{tab:length}", "\\end{table*}"]
write("length_checks.tex", L)

# token sweep
BINS = [([0.9999, 1.01], "$\\geq$0.9999"), ([0.999, 0.9999], "{}[0.999, 0.9999)"), ([0.99, 0.999], "{}[0.99, 0.999)"),
        ([0.9, 0.99], "{}[0.9, 0.99)"), ([0.5, 0.9], "{}[0.5, 0.9)"), ([-1.01, 0.5], "$<$0.5")]
L = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lrrrrrr@{}}",
     "\\toprule", " & \\multicolumn{2}{c}{OctoThinker-3B} & \\multicolumn{2}{c}{Qwen2.5-7B} & \\multicolumn{2}{c}{ORZ-7B} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
     "Cosine with the marker & AUROC range & max $|\\Delta\\ell|$ & AUROC range & max $|\\Delta\\ell|$ & AUROC range & max $|\\Delta\\ell|$ \\\\",
     "\\midrule"]
sw = {k: json.loads((A / f"token_sweep_{k}.json").read_text()) for k, _ in CK}
for b, lab in BINS:
    cells = []
    for k, _ in CK:
        t = [x for x in sw[k]["tokens"] if x["bin"] == b]
        a = [x["auroc"] for x in t]
        cells += [f(min(a), 3) + "--" + f(max(a), 3), f(max(x["max_abs_diff_lp"] for x in t), 2)]
    L.append(lab + " & " + " & ".join(cells) + " \\\\")
L += ["\\midrule", "Marker AUROC & " + " & ".join(f(sw[k]["marker_auroc"], 3) + " & " for k, _ in CK).rstrip(" &") + " \\\\",
      "\\bottomrule", "\\end{tabular}",
      "\\caption{Token sweep on seed-zero MATH500 solutions (exploratory): 20 tokens drawn at random per bin of "
      "cosine similarity between their output row and the marker's, read at the released point. "
      "$|\\Delta\\ell|$ is the largest absolute difference from the marker's log-probability over 4,000 solutions, in nat. "
      "Tokens on the shared direction read the marker's score.}",
      "\\label{tab:app-sweep}", "\\end{table*}"]
write("app_sweep.tex", L)

# mechanism
mech = json.loads((A / "mech_summary.json").read_text())
MR = [("eos", "after EOS (released)"), ("newline", "after a newline"), ("period", "after a period"),
      ("chat_special", "after a chat-template token"), ("rare_word", "after a rare word"), ("none", "after the solution ($R_2$)"),
      ("logit_marker", "marker logit after EOS"), ("neg_logz", "negative log normalizer after EOS"),
      ("base_eos", "base model, after EOS"), ("base_logit_marker", "base model, marker logit"),
      ("base_neg_logz", "base model, negative log normalizer"), ("change_logit_marker", "training change, marker logit"),
      ("change_neg_logz", "training change, negative log normalizer")]
L = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{@{}llll@{}}",
     "\\toprule", "Score read & OctoThinker-3B & Qwen2.5-7B & ORZ-7B \\\\", "\\midrule"]
for i, (k, lab) in enumerate(MR):
    if i in (6, 8):
        L.append("\\midrule")
    L.append(lab + " & " + " & ".join(f(mech[c][k]["auroc"], 3) + " " + ci(mech[c][k]["ci95"], 3) for c, _ in CK) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{End-of-sequence control and the two terms of the marker's log-probability (exploratory), "
      "benchmark-weighted AUROC on the seed-zero solutions of all five benchmarks with 95\\% problem-bootstrap "
      "intervals. Training turns the marker logit from inverted to strong; the normalizer already separates in the base model.}",
      "\\label{tab:app-mech}", "\\end{table*}"]
write("app_mech.tex", L)

# Qwen3-4B replication
q = json.loads((A / "q3_replication.json").read_text())
QR = [("R0_reference", "$R_0$ released readout"), ("R1_marker_swap_mean", "$R_1$ other marker"),
      ("R2_position_eos", "$R_2$ without EOS"), ("R3_position_answer", "$R_3$ at the answer"),
      ("R4_cref_none", "$R_4$ no constant"), ("R5_cref_recomputed", "$R_5$ base reference"),
      ("BASE_R0", "untrained base model")]
L = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lrr@{}}",
     "\\toprule", "Readout & AUROC & Partial $r$ \\\\", "\\midrule"]
for k, lab in QR:
    r = q["readouts"][k]
    L.append(lab + " & " + f(r["primary_auroc"], 3) + " " + ci(r["primary_auroc_ci95"], 3) + " & "
             + f(r["partial_corr_length"], 2) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Qwen3-4B trained only on LaSeR's self-reward term (exploratory), on its own fixed solutions to the "
      "five benchmarks; AUROC with 95\\% problem-bootstrap intervals and partial correlation with length. The released "
      "pattern recurs.}",
      "\\label{tab:app-q3}", "\\end{table}"]
write("app_q3.tex", L)
print("tables written")
