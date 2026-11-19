"""Contracts at the package's public entry points using CPU tensors and local arrays."""
import io
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from alfm_bem import (
    ALFMBEM, ALFMConfig, Action, BidirectionalExperienceMemory, ConsensusEngine,
    ContrastiveProjection, Experience, create_alfm_bem,
)
from alfm_bem.adapters import AdapterConfig, BoundedAdapter


def parameters(adapter):
    return torch.nn.utils.parameters_to_vector(adapter.parameters()).detach().clone()


def example(dim=8):
    return Experience(np.eye(dim)[0], -1.0, "failure", datetime.now())


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = AdapterConfig(input_dim=8, hidden_dim=4, output_dim=8,
                                    max_grad_norm=0.25, max_param_norm=0.4,
                                    learning_rate=0.2, ema_decay=0.8)

    def test_initialization_and_first_through_last_step(self):
        model = BoundedAdapter(self.config)
        stats = model.get_statistics()
        self.assertAlmostEqual(stats["initial_param_norm"], self.config.max_param_norm, places=6)
        self.assertEqual(stats["total_drift"], 0)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, model.shadow[name], rtol=0, atol=0)
        path = 0
        for step in range(1, 9):
            before = parameters(model)
            model.train_step([example()])
            after = parameters(model)
            increment = (after - before).norm().item()
            path += increment
            self.assertLessEqual(increment, self.config.learning_rate * self.config.max_grad_norm + 1e-6)
            self.assertLessEqual(after.norm().item(), self.config.max_param_norm + 1e-6)
            self.assertAlmostEqual(model.total_drift, path)
            self.assertEqual(model.update_count, step)
        self.assertGreater(path, 0)

    def test_update_matches_clipped_full_gradient_then_radial_projection(self):
        m = BoundedAdapter(self.config)
        state = {name: torch.zeros_like(p) for name, p in m.named_parameters()}
        state["fc2.bias"][1] = 12  # Imported state outside the ball.
        m.load_state_dict(state)
        before = parameters(m)
        self.assertLessEqual(before.double().norm().item(), self.config.max_param_norm)
        self.assertAlmostEqual(before.norm().item(), self.config.max_param_norm, places=6)
        z = torch.tensor(example().embedding, dtype=torch.float32)
        loss = (torch.dot(m(z), z) + 0.5).relu()
        loss = loss + self.config.l2_weight * sum(p.norm() for p in m.parameters())
        grad = torch.cat([g.flatten() for g in torch.autograd.grad(loss, tuple(m.parameters()))])
        self.assertGreater(grad.norm().item(), self.config.max_grad_norm)
        clipped = grad * (self.config.max_grad_norm / (grad.norm() + 1e-6))
        proposal = before - self.config.learning_rate * clipped
        self.assertGreater(proposal.norm().item(), self.config.max_param_norm)
        expected = proposal * (self.config.max_param_norm / proposal.norm())
        m.train_step([example()])
        torch.testing.assert_close(parameters(m), expected)
        expected_shadow = before.lerp(expected, 1 - self.config.ema_decay)
        torch.testing.assert_close(torch.cat([v.flatten() for v in m.shadow.values()]), expected_shadow)

    def test_checkpoint_and_legacy_resume(self):
        model = BoundedAdapter(self.config)
        model.train_step([example()])
        checkpoint = io.BytesIO()
        model.save(checkpoint)
        checkpoint.seek(0)
        restored = BoundedAdapter.load(checkpoint, self.config)
        self.assertEqual(restored.get_statistics(), model.get_statistics())
        for use_ema in (False, True):
            np.testing.assert_array_equal(restored(example().embedding, use_ema=use_ema),
                                          model(example().embedding, use_ema=use_ema))
        self.assertFalse(np.array_equal(model(example().embedding),
                                        model(example().embedding, use_ema=True)))
        model.train_step([example()])
        restored.train_step([example()])
        torch.testing.assert_close(parameters(model), parameters(restored))

        legacy = io.BytesIO()
        outside = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
        torch.save(dict(state_dict=outside, shadow=outside, config=self.config,
                        stats=dict(tenant_id="alpha", update_count=1, total_drift=np.float64(5))), legacy)
        legacy.seek(0)
        resumed = BoundedAdapter.load(legacy, self.config)
        self.assertEqual(resumed.get_statistics()["total_drift"], 0)
        self.assertLessEqual(parameters(resumed).norm().item(), self.config.max_param_norm + 1e-6)
        self.assertLessEqual(torch.cat([v.flatten() for v in resumed.shadow.values()]).norm().item(),
                             self.config.max_param_norm + 1e-6)
        resumed.train_step([example()])
        self.assertLessEqual(resumed.total_drift, self.config.learning_rate * self.config.max_grad_norm + 1e-6)


class DecisionTests(unittest.TestCase):
    def test_thresholds_and_precedence(self):
        engine = ConsensusEngine()
        engine.add_heuristic_rule("critical", lambda ctx: ctx.get("critical"), True)
        engine.add_heuristic_rule("advisory", lambda ctx: ctx.get("warning"))
        cases = [
            (1, 1, 0, {"critical": True}, Action.ESCALATE),
            (1, 1, 0.1999, {"warning": True}, Action.QUERY),
            (0.8501, 1, 0.2, {}, Action.ESCALATE),
            (0.85, 1, 0.2, {}, Action.ABSTAIN),
            (0.6001, 1, 0.9, {}, Action.ABSTAIN),
            (0.6, 0.7001, 0.2, {}, Action.TRUST),
            (0.6, 0.7, 0.2, {}, Action.QUERY),
            (0, 1, 0.7, {"warning": True}, Action.ABSTAIN),
            (0, 0.7, 0.4999, {}, Action.QUERY),
            (0, 0.7, 0.5, {}, Action.TRUST),
        ]
        for risk, success, coverage, context, expected in cases:
            with self.subTest(risk=risk, success=success, coverage=coverage, context=context):
                decision = engine.decide(risk, success, coverage, context)
                self.assertEqual(decision.action, expected)
                self.assertEqual(decision.query_content is not None, expected == Action.QUERY)

    def test_query_region_is_a_prefix_of_increasing_coverage(self):
        engine = ConsensusEngine()
        engine.add_heuristic_rule("soft", lambda c: c.get("soft"))
        for risk in (0, 0.6, 0.8, 1):
            for success in (0, 0.7, 1):
                for context in ({}, {"soft": True}):
                    flags = [engine.decide(risk, success, c, context).action == Action.QUERY
                             for c in np.linspace(0, 1, 101)]
                    self.assertEqual(flags, sorted(flags, reverse=True))
        engine.add_heuristic_rule("required", lambda c: c["missing"], is_critical=True)
        self.assertEqual(engine.decide(0, 1, 1, {}).action, Action.ESCALATE)


class MemoryTests(unittest.TestCase):
    def test_scores_signs_dedup_and_calibrated_kernel(self):
        memory = BidirectionalExperienceMemory(dim=2)
        e, q = np.eye(2)
        for i in range(9):
            memory.add_experience(e, -1, str(i))
        self.assertEqual(memory.coverage_signal(q), 0)
        memory.add_experience(q, 1, "success")
        memory.add_experience(q, 1, "success")
        self.assertEqual(len(memory.experiences), 10)
        self.assertEqual(memory.experiences[-1].confirmation_count, 2)
        k = np.exp(-1 / (2 * 0.3**2))
        self.assertAlmostEqual(memory.coverage_signal(q), (1 + 9*k)/(9 + k), places=7)
        self.assertAlmostEqual(memory.risk_signal(e)[0], 1)
        self.assertAlmostEqual(memory.success_signal(q)[0], 1)

    def test_indexed_and_dense_retrieval_and_ttl(self):
        dense = BidirectionalExperienceMemory(dim=3)
        indexed = BidirectionalExperienceMemory(dim=3, use_knn_index=True)
        rng = np.random.default_rng(147)
        q = np.array([1., 0., 0.])
        for i, v in enumerate(rng.normal(size=(100, 3))):
            for memory in (dense, indexed):
                memory.add_experience(v, -1 if i % 2 else 1, str(i))
        for outcome_filter in (None, "failure", "success"):
            left = dense.retrieve(q, k=90, outcome_filter=outcome_filter)
            right = indexed.retrieve(q, k=90, outcome_filter=outcome_filter)
            self.assertEqual([e.context_hash for e, _ in left], [e.context_hash for e, _ in right])
            np.testing.assert_allclose([s for _, s in left], [s for _, s in right], atol=1e-7)
        self.assertAlmostEqual(dense.coverage_signal(q), indexed.coverage_signal(q))
        expired = indexed.retrieve(q, k=1)[0][0]
        expired.timestamp -= timedelta(days=120)
        expired.outcome = 0.1
        indexed.vacuum()
        self.assertNotEqual(expired.context_hash, indexed.retrieve(q, k=1)[0][0].context_hash)
        self.assertEqual(len(indexed.experiences), 99)


class WrapperTests(unittest.TestCase):
    def test_factory_query_feedback_and_isolation(self):
        model = create_alfm_bem(backbone_dim=8)
        self.assertLessEqual(model.adapter_manager.global_adapter._compute_param_norm(), 10.00001)
        z = np.arange(8.)
        ctx = {"source": 1, "ticket": "A"}
        query = model.infer(z, ctx, tenant_id="alpha")
        self.assertTrue(query.should_query())
        ctx["ticket"] = "mutated"
        self.assertTrue(model.handle_query_response(query.decision.query_id, "accepted", 1.0))
        self.assertFalse(model.handle_query_response(query.decision.query_id, "duplicate", 1.0))
        model.record_outcome(z, {"ticket": "A", "source": 1}, 1.0, "alpha")
        self.assertEqual(model.get_statistics("alpha")["bem"]["total_confirmations"], 2)
        self.assertEqual(model.get_statistics("alpha")["bem"]["total"], 1)
        self.assertEqual(model.infer(z, {}, "beta").bem_query_result["success"], 0)

    def test_replay_and_system_checkpoint_round_trip(self):
        cfg = ALFMConfig(backbone_dim=8, projection_dim=8, adapter_hidden_dim=4,
                         max_param_norm=0.4, replay_batch_size=1, replay_frequency=2,
                         similarity_threshold=0.9, risk_sensitivity=0.24)
        system = ALFMBEM(cfg)
        z = np.arange(8.)
        key = "team/a"
        system.record_outcome(z, "k1", -1, key)
        decision = system.infer(z, {}, key)
        self.assertAlmostEqual(decision.decision.risk_score, 0.24, places=5)
        self.assertEqual(system.get_statistics(key)["adapter"]["update_count"], 0)
        system.record_outcome(z, "k2", -1, key)
        adapter = system.adapter_manager.tenant_adapters[key]
        self.assertEqual(adapter.update_count, 1)
        self.assertLessEqual(adapter.total_drift, cfg.adapter_lr * cfg.max_grad_norm + 1e-6)
        before = system.infer(z, {}, key)
        with tempfile.TemporaryDirectory() as path:
            system.save(path)
            restored = ALFMBEM.load(path)
        after = restored.infer(z, {}, key)
        np.testing.assert_array_equal(before.adapted_embedding, after.adapted_embedding)
        self.assertEqual(before.decision.risk_score, after.decision.risk_score)
        self.assertEqual(restored.get_statistics(key)["adapter"], adapter.get_statistics())
        self.assertEqual(restored.get_statistics(key)["bem"]["total"], 2)
        restored.record_outcome(z, "k3", -1, key)
        self.assertEqual(restored.get_statistics(key)["adapter"]["update_count"], 1)
        restored.record_outcome(z, "k4", -1, key)
        self.assertEqual(restored.get_statistics(key)["adapter"]["update_count"], 2)

    def test_projection_fit_without_valid_pairs(self):
        proj = ContrastiveProjection(input_dim=3, hidden_dim=4, output_dim=2)
        w1 = proj.W1.copy()
        with self.assertRaisesRegex(ValueError, "polarity"):
            proj.fit(list(np.eye(3)), [1, 1, 1], n_epochs=1)
        np.testing.assert_array_equal(proj.W1, w1)
        with self.assertWarnsRegex(UserWarning, "validation pairs"):
            proj.fit(list(np.eye(3)), [1, 1, -1], validation_data=([np.ones(3)], [1]),
                     n_epochs=1)
        self.assertTrue(np.isfinite(proj.W1).all())


if __name__ == "__main__":
    unittest.main()
