#!/usr/bin/env python3
"""Build the pooled confirmatory summary from the three checkpoint analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", action="append", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    analyses = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.analysis]
    contrast_names = (
        "L1_auroc_R0_minus_R1", "L2_auroc_R0_minus_R2", "L3_partial_corr_R0",
        "L4_weighted_majority_R0_minus_R4", "L5_auroc_gap_short_long_R0_minus_R3")
    holds = {key: [bool(a["contrasts"].get(key, {}).get("holds")) for a in analyses] for key in contrast_names}
    verdicts = {key: [a["contrasts"].get(key, {}).get("verdict") for a in analyses] for key in contrast_names}
    equivalence_bounds = {key: [a["contrasts"].get(key, {}).get("equivalence_bound") for a in analyses] for key in contrast_names}
    result = {
        "schema_version": 2,
        "stage": "E2c",
        "unit": "problem cluster; all 8 responses for seeds 0, 1, 2",
        "analyses": args.analysis,
        "general_claim_rule": "general only if Holm-adjusted bootstrap contrast holds for at least 2 of 3 checkpoints",
        "general_claims": {
            key: {
                "holds_by_checkpoint": vals,
                "verdicts_by_checkpoint": verdicts[key],
                "equivalence_bounds_by_checkpoint": equivalence_bounds[key],
                "holds_general": sum(vals) >= 2,
            }
            for key, vals in holds.items()
        },
        "checkpoints": [a.get("model") for a in analyses],
        "flag_counts": {str(a.get("model")): a.get("flag_counts", {}) for a in analyses},
        "r3_missingness": {str(a.get("model")): a.get("r3_missingness", {}) for a in analyses},
        "per_seed_contrasts": {str(a.get("model")): a.get("per_seed_contrasts", {}) for a in analyses},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_name(output.name + ".sha256").write_text(f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
