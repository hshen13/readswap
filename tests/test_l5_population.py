import copy
import random
import unittest

from analyze_confirm_a1 import l5_value


def make_clusters(seed: int) -> list[dict]:
    rng = random.Random(seed)
    clusters = []
    for problem in range(40):
        rows = []
        for _ in range(3):
            correct = [rng.random() < 0.5 for _ in range(8)]
            scores = []
            for c in correct:
                r0 = rng.gauss(1.0 if c else 0.0, 1.0)
                boxed = rng.random() < 0.7
                scores.append({"R0_reference": r0,
                               "R3_position_answer": r0 + rng.gauss(0.0, 0.5) if boxed else None})
            rows.append({"correctness": correct, "readout_scores": scores,
                         "response_token_counts": [rng.randint(100, 3000) for _ in range(8)]})
        clusters.append({"dataset": f"d{problem % 2}", "rows": rows})
    return clusters


def boxed_only(clusters: list[dict]) -> list[dict]:
    out = copy.deepcopy(clusters)
    for cluster in out:
        for row in cluster["rows"]:
            keep = [i for i, item in enumerate(row["readout_scores"]) if item["R3_position_answer"] is not None]
            for key in ("correctness", "readout_scores", "response_token_counts"):
                row[key] = [row[key][i] for i in keep]
    return out


class L5PopulationTest(unittest.TestCase):
    def test_l5_uses_box_compliant_responses_for_both_readouts(self):
        # Amendment A1 §3: L5 is computed on the box-compliant responses only, for R0 as well as R3.
        for seed in range(3):
            clusters = make_clusters(seed)
            self.assertAlmostEqual(l5_value(clusters), l5_value(boxed_only(clusters)), places=12)


if __name__ == "__main__":
    unittest.main()
