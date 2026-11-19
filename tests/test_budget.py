"""Information boundaries and fixed-quota regressions for the digits replay."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from experiments import budget_benchmark as budget


class TestBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with threadpool_limits(limits=1):
            cls.cases, cls.labels, cls.train, cls.pool = budget.freeze_model()

    def test_disjoint_partitions_for_all_replays(self):
        for seed in budget.SEEDS:
            parts = budget.split_pool(self.pool, seed)
            self.assertEqual([len(p) for p in parts], [150, 150, 500, 600])
            together = np.concatenate((self.train, *parts))
            np.testing.assert_array_equal(np.sort(together), np.arange(len(self.labels)))

    def test_batch_quota_and_feedback_timing(self):
        warm, _, stream, _ = budget.split_pool(self.pool, 0)
        initial = budget.Reviews(warm, self.labels[warm])
        batches = stream.reshape(-1, budget.BATCH)
        seen = []

        def make_scorer(method, cases, history, objective, settings):
            expected = np.concatenate([warm] + seen)
            np.testing.assert_array_equal(history.ids, expected)
            np.testing.assert_array_equal(history.labels, self.labels[expected])
            return lambda ids: np.zeros(len(ids))

        def reveal(ids):
            # All scores tie, so the earlier arrivals are the only permitted reads.
            np.testing.assert_array_equal(ids, batches[len(seen)][:budget.QUOTA])
            seen.append(ids)
            return self.labels[ids]

        with patch.object(budget, "make_scorer", make_scorer):
            result = budget.collect_history("arrival", self.cases, initial, batches, {}, reveal)
        self.assertEqual(len(seen), 10)
        self.assertEqual(len(result.ids), 200)

    def test_unobserved_labels_cannot_change_decisions(self):
        _, _, stream, future = budget.split_pool(self.pool, 0)
        select = budget.select

        def replay(labels):
            choices = []

            def record(batch, scores):
                ids = select(batch, scores)
                choices.append(ids)
                return ids

            with patch.object(budget, "select", record), threadpool_limits(limits=1):
                result = budget.run_seed(self.cases, labels, self.pool, 0)
            return result, np.concatenate(choices)

        reference, choices = replay(self.labels)
        # Change ground truth on all future cases and stream cases nobody reviewed.
        hidden = np.concatenate((future, np.setdiff1d(stream, choices)))
        changed = self.labels.copy()
        changed[hidden] = self.cases.prediction[hidden]
        counterfactual, again = replay(changed)
        np.testing.assert_array_equal(choices, again)
        self.assertGreater(reference["available"]["future"]["errors"], 0)
        self.assertEqual(counterfactual["available"]["future"]["errors"], 0)
        for sign, grade in (("bem_sign", "bem_graded"), ("bem_sign_blend", "bem_graded_blend")):
            self.assertEqual(reference["methods"][sign]["stream_review_order_sha256"],
                             reference["methods"][grade]["stream_review_order_sha256"])
            self.assertEqual(reference["methods"][sign]["future"]["unit"],
                             reference["methods"][grade]["future"]["unit"])

    def test_vote_on_reviewed_digit_and_vote_on_past_error_are_distinct(self):
        x = np.array([[1., 0.], [0., 1.], [1., 0.]])
        cases = budget.PublicCases(x, np.eye(10)[[2, 1, 7]], np.array([2, 1, 7]))
        history = budget.Reviews(np.array([0, 1]), np.array([2, 1]))
        ids = np.array([2])
        for objective, cost in (("unit", 1), ("distance", 5)):
            cls = budget.vote_scores("class_vote", cases, history, ids, objective, 1)
            old = budget.vote_scores("outcome_vote", cases, history, ids, objective, 1)
            self.assertEqual(cls[0], cost)
            self.assertEqual(old[0], 0)

    def test_bem_ranking_uses_package_risk_and_feedback_scale(self):
        x = np.eye(3)
        cases = budget.PublicCases(x, np.eye(10)[[0, 0, 3]], np.array([0, 0, 3]))
        history = budget.Reviews(np.arange(3), np.array([1, 9, 3]))
        ids = np.arange(3)
        with patch.object(budget.BidirectionalExperienceMemory, "risk_signal", autospec=True) as deployed:
            deployed.side_effect = lambda mem, z: (0.125, [])
            score = budget.bem_risk("bem_graded", cases, history, "distance")
            np.testing.assert_array_equal(score(ids), [0.125] * 3)
            stored = deployed.call_args.args[0].experiences
            self.assertEqual([e.is_failure for e in stored], [True, True, False])
            self.assertEqual([e.severity for e in stored], [3, 5, 1])
        sign = budget.bem_risk("bem_sign", cases, history, "distance")(ids)
        grade = budget.bem_risk("bem_graded", cases, history, "distance")(ids)
        self.assertEqual(sign[0], sign[1])
        self.assertLess(grade[0], grade[1])
        self.assertEqual(grade[2], 0)

    def test_bem_blend_can_recover_frozen_score_on_either_objective(self):
        warm, _, _, future = budget.split_pool(self.pool, 2)
        history = budget.Reviews(warm, self.labels[warm])
        ids = future[:budget.BATCH]
        for obj in budget.OBJECTIVES:
            frozen = budget.frozen_scores(self.cases, ids, obj)
            _, w = budget.choose_weight(np.zeros_like(frozen), frozen, frozen)
            self.assertEqual(w, 0.0)
            for method in ("bem_sign_blend", "bem_graded_blend"):
                settings = {method: {obj: w}, "distance_per_error": 5.0}
                mixed = budget.make_scorer(method, self.cases, history, obj, settings)(ids)
                np.testing.assert_array_equal(mixed, frozen)
                settings[method][obj] = 0.5
                mixed = budget.make_scorer(method, self.cases, history, obj, settings)(ids)
                raw = budget.bem_risk(method.removesuffix("_blend"), self.cases, history, obj)(ids)
                scale = 5.0 if obj == "distance" else 1.0
                np.testing.assert_allclose(mixed, 0.5 * (frozen + scale * raw))

    def test_check_preserves_counts_and_allows_float_roundoff(self):
        budget.compare({"count": 50, "mean": [0.500000001]}, {"count": 50, "mean": [0.5]})
        for invalid in ({"count": 51}, {"count": 50, "mean": [0.52]},
                        {"count": 50.000000001, "mean": [0.5]}):
            with self.assertRaisesRegex(ValueError, "Benchmark mismatch"):
                budget.compare(invalid, {"count": 50, "mean": [0.5]})

    def test_joint_score_calls_all_three_package_scores_with_correct_direction(self):
        warm, _, _, future = budget.split_pool(self.pool, 0)
        ids = future[:5]
        initial = budget.Reviews(warm, self.labels[warm])
        choice = {"component_weights": dict(zip(budget.JOINT_COMPONENTS, (0.25, 0.25, 0.5))),
                  "memory_weight": 0.75}
        settings = {"bem_joint_blend": {"distance": choice}, "distance_per_error": 4.0}
        with ExitStack() as stack:
            readers = [
                stack.enter_context(patch.object(budget.BidirectionalExperienceMemory, name,
                                                 autospec=True, return_value=value))
                for name, value in (("risk_signal", (0.8, [])), ("success_signal", (0.7, [])),
                                    ("coverage_signal", 0.6))
            ]
            score = budget.make_scorer("bem_joint_blend", self.cases, initial, "distance", settings)
            observed = score(ids)
            self.assertEqual([reader.call_count for reader in readers], [len(ids)] * 3)
        expected_memory = 4.0 * (0.25 * 0.8 + 0.25 * 0.3 + 0.5 * 0.4)
        expected = 0.75 * expected_memory + 0.25 * budget.frozen_scores(self.cases, ids, "distance")
        np.testing.assert_allclose(observed, expected)

    def test_joint_calibration_selects_minimum_mse_with_bounded_grid(self):
        warm, cal, _, _ = budget.split_pool(self.pool, 0)
        initial = budget.Reviews(warm, self.labels[warm])
        calibration = budget.Reviews(cal, self.labels[cal])
        config = budget.calibrate_joint(self.cases, initial, calibration, "unit", 1.0)
        memory = budget.build_memory("bem_graded", self.cases, initial, "unit")
        components = budget.joint_components(memory, self.cases.embedding[cal])
        frozen = budget.frozen_scores(self.cases, cal, "unit")
        target = self.cases.prediction[cal] != self.labels[cal]
        candidates = [
            (round(float(np.mean((outer * (components @ inner) + (1 - outer) * frozen - target)**2)), 12),
             outer, i)
            for outer in budget.VOTE_WEIGHTS for i, inner in enumerate(budget.JOINT_WEIGHTS)
        ]
        mse, outer, i = min(candidates)
        self.assertEqual(config["memory_weight"], outer)
        self.assertEqual(config["calibration_mse"], mse)
        self.assertEqual(tuple(config["component_weights"].values()), budget.JOINT_WEIGHTS[i])


if __name__ == "__main__":
    unittest.main()
