import unittest

from analyze_confirm import (
    equivalence_classes,
    exp_weighted_majority,
    linear_weighted_majority,
    unweighted_majority,
)


class NegativeScoreVotingTest(unittest.TestCase):
    def test_negative_scores_keep_ranking_and_are_not_zeroed(self):
        row = {
            "answer": "right",
            "pred_answers": ["wrong", "right"],
            "correctness": [False, True],
            "readout_scores": [
                {"R0_reference": -100.0},
                {"R0_reference": -1.0},
            ],
        }
        cluster = {"rows": [row, row, row]}
        self.assertEqual(exp_weighted_majority(cluster, "R0_reference"), 1.0)

    def test_linear_weights_keep_negative_scores(self):
        row = {
            "pred_answers": ["right", "wrong"],
            "correctness": [True, False],
            "readout_scores": [{"R0_reference": -1.0}, {"R0_reference": -100.0}],
        }
        self.assertEqual(linear_weighted_majority({"rows": [row]}, "R0_reference"), 1.0)


class ConstantAndTieVotingTest(unittest.TestCase):
    def test_constant_cancels_for_exp_but_not_linear_weights(self):
        # The first class has two votes and is correct.  Adding a common
        # constant changes a linear sum by class size, but only multiplies
        # every exp weight by the same factor.
        row = {
            "pred_answers": ["right", "right", "wrong"],
            "correctness": [True, True, False],
            "readout_scores": [
                {"R0_reference": 0.0, "R4_cref_none": 1.0},
                {"R0_reference": 0.0, "R4_cref_none": 1.0},
                {"R0_reference": 0.8, "R4_cref_none": 1.8},
            ],
        }
        cluster = {"rows": [row]}
        self.assertEqual(exp_weighted_majority(cluster, "R0_reference"), 0.0)
        self.assertEqual(exp_weighted_majority(cluster, "R4_cref_none"), 0.0)
        self.assertEqual(linear_weighted_majority(cluster, "R0_reference"), 0.0)
        self.assertEqual(linear_weighted_majority(cluster, "R4_cref_none"), 1.0)

    def test_ties_use_count_then_first_response(self):
        count_tie = {
            "pred_answers": ["first", "second", "second"],
            "correctness": [False, True, True],
            "readout_scores": [{"R0_reference": 1.0}, {"R0_reference": 0.5}, {"R0_reference": 0.5}],
        }
        first_tie = {
            "pred_answers": ["first", "second"],
            "correctness": [True, False],
            "readout_scores": [{"R0_reference": 1.0}, {"R0_reference": 1.0}],
        }
        self.assertEqual(linear_weighted_majority({"rows": [count_tie]}, "R0_reference"), 1.0)
        self.assertEqual(linear_weighted_majority({"rows": [first_tie]}, "R0_reference"), 1.0)


class MathVerifyEquivalenceTest(unittest.TestCase):
    def test_equivalent_answers_share_a_class(self):
        labels = equivalence_classes([r"\frac{1}{2}", "0.5", "1"])
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_equivalent_answers_are_one_majority_class(self):
        row = {
            "pred_answers": [r"\frac{1}{2}", "0.5", "1"],
            "correctness": [True, True, False],
            "readout_scores": [
                {"R0_reference": -1.0},
                {"R0_reference": -1.0},
                {"R0_reference": -1.5},
            ],
        }
        self.assertEqual(unweighted_majority({"rows": [row]}, "R0_reference"), 1.0)


if __name__ == "__main__":
    unittest.main()
