import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common import (
    released_last_boxed_only_string,
    released_remove_boxed,
    released_response,
)
from compare_readout_fast import flatten
from analyze_confirm_a1 import bootstrap_report_intervals


class ReleasedSurfaceTest(unittest.TestCase):
    def test_response_matches_released_strip_and_first_answer_cut(self):
        text = "  work </answer> discarded </answer>  "
        self.assertEqual(released_response(text), "work </answer>")

    def test_final_marker_and_incomplete_marker_match_released_grader(self):
        self.assertEqual(
            released_remove_boxed(released_last_boxed_only_string(r"old \boxed{1} new \boxed{2}")),
            "2",
        )
        self.assertIsNone(released_last_boxed_only_string(r"old \boxed{1} new \boxed{"))
        self.assertIsNone(
            released_remove_boxed(released_last_boxed_only_string(r"\fbox{3}"))
        )


class UnitSubsetTest(unittest.TestCase):
    def test_subset_skips_none_pred_answers(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "input.jsonl"
            output = tmp / "output.jsonl"
            row = {
                "problem": "p",
                "responses": ["bad"] + [r" good \boxed{1} "] * 20,
                "pred_answers": [None] + ["1"] * 20,
                "correctness": [False] + [True] * 20,
            }
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            subprocess.run(
                [sys.executable, "code/make_unit_subset.py", "--input", str(source), "--output", str(output)],
                cwd=root,
                env={"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root / "code")},
                check=False,
            )
            selected = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(selected), 20)
            self.assertEqual(selected[0]["unit_response_index"], 1)
            self.assertEqual(selected[0]["responses"], ["good \\boxed{1}"])


class NullR3Test(unittest.TestCase):
    def test_both_null_r3_is_retained_as_an_agreement_pair(self):
        row = {
            "correctness": [True],
            "readout_logprobs": [{
                "R0_marker": -1.0,
                "R1_markers": {"1": -2.0, "2": -3.0, "3": -4.0},
                "R2_marker": -1.5,
                "R3_marker": None,
                "R5_base_marker": -0.5,
            }],
        }
        fast = flatten([{**row, "responses": ["x"]}], True)
        reference = flatten([{**row, "responses": ["x"]}], False)
        self.assertIsNone(fast["R3"][0])
        self.assertEqual(fast["R3"], reference["R3"])


def _cluster(problem: int) -> dict:
    rows = []
    for seed in range(3):
        correctness = [bool((problem + seed + i) % 2) for i in range(8)]
        scores = []
        for i, correct in enumerate(correctness):
            score = float((2 if correct else -1) + i / 100)
            scores.append({
                "R0_reference": score,
                "R1_marker_swap_mean": score / 2,
                "R1_marker_swap": {"1": score / 2, "2": score / 2, "3": score / 2},
                "R2_position_eos": score,
                "R3_position_answer": score if i != 0 else None,
                "R4_cref_none": score + 1,
                "R5_cref_recomputed": score - 1,
            })
        rows.append({
            "responses": [f"\\boxed{{{int(x)}}}" for x in correctness],
            "pred_answers": [str(int(x)) for x in correctness],
            "correctness": correctness,
            "readout_scores": scores,
            "response_token_counts": list(range(10, 18)),
        })
    return {"dataset": "d0" if problem % 2 else "d1", "rows": rows}


class BootstrapIntervalTest(unittest.TestCase):
    def test_shared_metric_bootstrap_covers_registered_fields(self):
        clusters = [_cluster(i) for i in range(8)]
        import numpy as np

        samples = [clusters, clusters]
        metrics, _bench, _length, kendall, _benchmark_kendall = bootstrap_report_intervals(
            clusters, ["R0_reference", "R3_position_answer"], samples
        )
        self.assertEqual(len(metrics["R0_reference"]), 2)
        self.assertIn("ece_15", metrics["R0_reference"][0])
        self.assertIn("weighted_majority_accuracy_k8", metrics["R0_reference"][0])
        self.assertEqual(len(kendall["R2_position_eos"]), 2)


if __name__ == "__main__":
    unittest.main()
