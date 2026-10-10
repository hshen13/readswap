#!/usr/bin/env python3
"""Appendix table of the exhaustive token test. Writes outputs/tables/app_exhaustive.tex."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("READSWAP_ROOT", Path(__file__).resolve().parents[1]))
A = ROOT / "results/analysis"
OUT = ROOT / "outputs/tables/app_exhaustive.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)
CK = [("OctoThinker-3B-Short-LaSeR", "OctoThinker-3B"), ("Qwen2.5-7B-LaSeR", "Qwen2.5-7B"), ("ORZ-7B-LaSeR", "ORZ-7B")]
POP = [("aligned rows", lambda t: t["aligned"]),
       ("special, unused by template", lambda t: t["special"] and not t["template_uses"]),
       ("special, used by template", lambda t: t["special"] and t["template_uses"])]
L = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{@{}llrrrr@{}}",
     "\\toprule", "Checkpoint & Tokens & Count & Within bounds & max $|\\Delta\\,\\mathrm{AUROC}|$ & max $|\\Delta s|$ \\\\",
     "\\midrule"]
for i, (key, name) in enumerate(CK):
    d = json.loads((A / f"exhaustive_tokens_{key}.json").read_text())
    mk, b = d["marker_auroc"], d["beta"]
    if i:
        L.append("\\midrule")
    first = True
    for pname, sel in POP:
        ts = [t for t in d["tokens"] if sel(t)]
        if not ts:
            continue
        ok = sum(abs(t["auroc"] - mk) <= 0.02 and b * t["max_abs_diff_lp"] <= 0.02 for t in ts)
        L.append((name if first else "") + f" & {pname} & {len(ts):,} & {ok:,} & "
                 + f"{max(abs(t['auroc'] - mk) for t in ts):.4f} & {b * max(t['max_abs_diff_lp'] for t in ts):.3f} \\\\")
        first = False
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Exhaustive token test on the seed-zero solutions of all five benchmarks (exploratory). Aligned rows have a "
      "cosine of at least 0.9999 with the marker's output row; special tokens are all added tokens of the tokenizer. Bounds "
      "fixed before the run: AUROC within 0.02 of the marker's and score within 0.02 on every solution. Every aligned row passes.}",
      "\\label{tab:app-exhaustive}", "\\end{table*}"]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
(OUT.parent / (OUT.name + ".sha256")).write_text(hashlib.sha256(OUT.read_bytes()).hexdigest() + "  " + OUT.name + "\n")
print("written")
