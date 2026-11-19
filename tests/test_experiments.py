"""Exercise the controls most sensitive to baseline, split and denominator choices."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.ablation_study import ablation, modes, neighbor_votes, raw_threshold_control
from experiments.controls import differences, main
from experiments.healthcare_simulator import run_healthcare
from experiments.metrics import auroc, classification
from experiments.query_simulation import run_query
from experiments.real_embeddings import unseen_texts


class ExperimentTests(unittest.TestCase):
    def test_votes_share_neighbors_and_denominator(self):
        # A neutral item still occupies a slot. The strict boundary excludes 0.5.
        sims = np.array([[0.9, 0.8, 0.7, 0.6, 0.5], [0.5] * 5])
        labels = np.array([-0.8, 0.0, 0.9, 0.9, -0.7])
        failure, success = neighbor_votes(sims, labels, k=3)
        np.testing.assert_allclose(failure, [1 / 3, 0.0])
        np.testing.assert_allclose(success, [1 / 3, 0.0])
        self.assertEqual(classification([True, False, False], [True, True, False])["f1"], 2 / 3)

    def test_auroc_ties_and_endpoints(self):
        self.assertEqual(auroc([1, 1], [1]), 0.5)
        self.assertEqual(auroc([1, 2], [0, 1]), 0.875)
        self.assertEqual(auroc([-3], [-4]), 1)
        self.assertEqual(auroc([-3], [-2]), 0)
        for invalid in ([], [np.nan]):
            with self.assertRaises(ValueError):
                auroc([0], invalid)

    def test_geometries_and_original_positive_vote(self):
        original = modes(np.random.RandomState(42), 10, "original_frobenius")
        unit = modes(np.random.RandomState(42), 10, "unit_centers")
        self.assertAlmostEqual(np.linalg.norm(original), 1)
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1)
        row = ablation(42, "original_frobenius")
        success = row["retrieval"]["neighbor_vote"]["success"]
        self.assertEqual((success["tp"], success["fp"], success["fn"]), (71, 19, 29))
        self.assertAlmostEqual(success["f1"], 142 / 190)

    def test_fixed_cutoff_can_hide_ranking(self):
        result = raw_threshold_control()
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["thresholds"]["0.5"]["tp"], 0)
        self.assertEqual(result["thresholds"]["0.25"]["fp"], 0)
        self.assertEqual(result["thresholds"]["0.25"]["tp"], 100)

    def test_selective_feedback_changes_which_claims_agents_see(self):
        row = run_healthcare(789)
        for feedback in ("all_outcomes", "submitted_only"):
            self.assertEqual(row[feedback]["decision_disagreements"], 0)
        full = row["all_outcomes"]["bem"]["overall"]
        selective = row["submitted_only"]["bem"]["overall"]
        self.assertEqual((full["submitted"], selective["submitted"]), (1780, 1801))
        self.assertEqual((full["would_pay_referred"], selective["would_pay_referred"]), (35, 14))
        self.assertEqual(selective["referred"] - selective["would_pay_referred"],
                         selective["would_deny"] - selective["denied_submitted"])

    def test_oracle_counts_and_denominators(self):
        result = run_query()
        self.assertEqual(result["trust_errors"], 66)
        for key, counts in [("confidence_below_0.7", (109, 48)),
                            ("lowest_confidence_budget", (150, 58))]:
            row = result[key]
            self.assertEqual((row["referred"], row["errors_referred"]), counts)
            self.assertAlmostEqual(row["accuracy_automatic"], 1 - row["errors_automatic"] / row["automatic"])
            self.assertAlmostEqual(row["accuracy_with_perfect_oracle"], 1 - row["errors_automatic"] / 1000)
        self.assertIsNone(run_query(n=5, budget=5)["lowest_confidence_budget"]["accuracy_automatic"])

    def test_content_holdout(self):
        corpus = ["", "training", "first", "training ", " first", "second", " ", "second", "third"]
        groups = unseen_texts(corpus, [1], [[0, 2, 3], [4, 5, 6], [7, 8]])
        self.assertEqual([group.tolist() for group in groups], [[2], [5], [8]])

    def test_check_compares_numbers_counts_and_structure(self):
        measured = {"metric": 0.123456789, "counts": [10]}
        self.assertEqual(list(differences(measured, {"metric": 0.1234568, "counts": [10]})), [])
        self.assertTrue(list(differences(measured, {"metric": 0.1234568, "counts": [11]})))
        self.assertTrue(list(differences(measured, {"metric": 0.1234568})))
        with (TemporaryDirectory() as tmp, redirect_stdout(StringIO()), redirect_stderr(StringIO()),
              patch("experiments.controls.run_controls", return_value=measured)):
            path = Path(tmp) / "result.json"
            with patch("sys.argv", ["controls", "--output", str(path)]):
                main()
            with patch("sys.argv", ["controls", "--check", str(path)]):
                main()
                path.write_text("{}", encoding="utf-8")
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 1)
                self.assertEqual(path.read_text(encoding="utf-8"), "{}")
