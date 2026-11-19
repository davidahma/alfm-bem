"""Selective prediction on a noisy circle boundary with an immediate label oracle."""

import numpy as np


def run_query(seed=42, n=1000, budget=150):
    if n < 1 or not 0 <= budget <= n:
        raise ValueError("Require a positive sample size and a budget between 0 and n")
    rng = np.random.RandomState(seed)
    radius = np.linalg.norm(rng.randn(n, 2), axis=1)
    truth = radius <= 1.0
    confidence = 0.5 + 0.5 * np.minimum(np.abs(radius - 1.0) / 0.2, 1.0)
    prediction = np.where(rng.rand(n) < confidence, truth, ~truth)
    errors = prediction != truth
    threshold_mask = confidence < 0.7
    ranked = np.argsort(confidence, kind="stable")

    def account(mask):
        referred = int(mask.sum())
        auto = n - referred
        errors_auto = int(errors[~mask].sum())
        return {
            "referred": referred, "automatic": auto, "errors_referred": int(errors[mask].sum()),
            "errors_automatic": errors_auto,
            "accuracy_automatic": (auto - errors_auto) / auto if auto else None,
            "accuracy_with_perfect_oracle": (n - errors_auto) / n,
        }

    budget_mask = np.zeros(n, dtype=bool)
    budget_mask[ranked[:budget]] = True
    return {
        "seed": seed, "n": n, "budget": budget, "trust_errors": int(errors.sum()),
        "trust_accuracy": float((~errors).mean()),
        "confidence_below_0.7": account(threshold_mask),
        "lowest_confidence_budget": account(budget_mask),
    }
