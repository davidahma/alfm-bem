"""Outcome-voting and coverage controls for the published synthetic generator."""

import numpy as np

from alfm_bem.bem import BidirectionalExperienceMemory, CoverageMode
from .metrics import auroc, classification


def unit(vector):
    return vector / (np.linalg.norm(vector) + 1e-10)


def modes(rng, count, geometry):
    matrix = rng.randn(count, 64)
    if geometry == "original_frobenius":
        return unit(matrix)
    if geometry == "unit_centers":
        return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    raise ValueError(f"Unknown geometry {geometry}")


def draw_experiences(rng, n_per_sign, failure_modes, success_modes):
    # Preserve the original order of random draws, including the outcome magnitude.
    rows, outcomes = [], []
    groups = [
        (0.7, failure_modes, 0.05, -1.0, -0.5),
        (0.3, success_modes, 0.10, -0.5, -0.3),
        (0.7, success_modes, 0.05, 0.5, 1.0),
        (0.3, failure_modes, 0.10, 0.3, 0.5),
    ]
    for fraction, centers, noise, low, high in groups:
        for _ in range(int(fraction * n_per_sign)):
            center = centers[rng.randint(len(centers))]
            rows.append(unit(center + rng.randn(64) * noise))
            outcomes.append(rng.uniform(low, high))
    return np.array(rows), np.array(outcomes)


def neighbor_votes(similarities, outcomes, *, k=10, threshold=0.5):
    """For each query, both votes use the same mixed-outcome neighbors."""
    indices = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    eligible = np.take_along_axis(similarities, indices, axis=1) > threshold
    denominator = np.maximum(eligible.sum(axis=1), 1)
    labels = outcomes[indices]
    failure = ((labels < 0) & eligible).sum(axis=1) / denominator
    success = ((labels > 0) & eligible).sum(axis=1) / denominator
    return failure, success


def ablation(seed, geometry):
    rng = np.random.RandomState(seed)
    fm, sm = modes(rng, 10, geometry), modes(rng, 5, geometry)
    train, outcomes = draw_experiences(rng, 500, fm, sm)
    test, labels = draw_experiences(rng, 100, fm, sm)

    # The original OOD experiment uses fresh core failures as the ID population.
    heldout = [unit(fm[rng.randint(10)] + rng.randn(64) * 0.05) for _ in range(200)]
    shift = 3.0 * unit(rng.randn(64))
    clustered = [unit(shift + rng.randn(64) * 0.1) for _ in range(200)]
    uniform = [unit(rng.randn(64)) for _ in range(200)]

    memory = BidirectionalExperienceMemory(dim=64, similarity_threshold=0.5,
                                           coverage_mode=CoverageMode.KDE, kde_bandwidth=0.3)
    for i, (row, outcome) in enumerate(zip(train, outcomes)):
        memory.add_experience(row, outcome, str(i))
    # Raw dot products on normalized vectors match the cosine search in the library.
    normalized = np.stack([unit(row) for row in train])
    sims = np.stack([unit(row) for row in test]) @ normalized.T
    failure_vote, success_vote = neighbor_votes(sims, outcomes)
    max_failure = sims[:, outcomes < 0].max(axis=1)
    retrieval = {}
    for name, failure, success in [
        ("neighbor_vote", failure_vote, success_vote),
        ("failure_only", max_failure > 0.5, np.zeros(len(test))),
        ("bem", [memory.risk_signal(q)[0] for q in test],
         [memory.success_signal(q)[0] for q in test]),
    ]:
        retrieval[name] = {
            "failure": classification(labels < 0, np.asarray(failure) > 0.3),
            "success": classification(labels > 0, np.asarray(success) > 0.3),
        }

    queries = np.vstack([heldout, clustered, uniform])
    similarities = np.stack([unit(row) for row in queries]) @ normalized.T
    coverage = {}
    for name, scores in [
        ("max_all", similarities.max(axis=1)),
        ("max_failure", similarities[:, outcomes < 0].max(axis=1)),
        ("bem", [memory.coverage_signal(q) for q in queries]),
    ]:
        id_scores, clustered_scores, uniform_scores = np.split(np.asarray(scores), 3)
        coverage[name] = {
            "clustered_auroc": auroc(id_scores, clustered_scores),
            "uniform_auroc": auroc(id_scores, uniform_scores),
        }
    return {"seed": seed, "retrieval": retrieval, "coverage": coverage}


def raw_threshold_control(seed=42):
    """Archived 768-dimensional generator with its raw maximum-failure-cosine score."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(10, 768)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def sample(n):
        rows = []
        for failure in (True, False):
            for _ in range(n):
                if failure:
                    row = centers[rng.randint(10)] + 0.06 * rng.randn(768)
                else:
                    row = rng.randn(768)
                rows.append(row / np.linalg.norm(row))
                rng.random_sample()  # The original generator also drew an outcome magnitude.
        return np.stack(rows)

    train, test = sample(500), sample(100)
    scores = (test @ train[:500].T).max(axis=1)
    return {
        "seed": seed, "train_per_class": 500, "test_per_class": 100,
        "auroc": auroc(scores[:100], scores[100:]),
        "thresholds": {str(t): classification(np.arange(200) < 100, scores > t)
                       for t in (0.2, 0.25, 0.3, 0.5)},
    }
