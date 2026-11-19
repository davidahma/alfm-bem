"""Offline, fixed-capacity review of frozen digit predictions. See paper/benchmark.tex."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

# The source venv may have an editable install of another checkout.
ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))

from alfm_bem import BidirectionalExperienceMemory


TRAIN_SEED = 4701
SEEDS = tuple(range(12))
BATCH = 50
QUOTA = 5
INITIAL = 150
CALIBRATION = 150
STREAM = 500
POLICIES = ("arrival", "frozen", "outcome_vote", "class_vote", "failure_only", "bem_sign",
            "bem_sign_blend", "bem_joint_blend")
OBJECTIVES = ("unit", "distance")
K_GRID = (3, 7, 15)
VOTE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
JOINT_COMPONENTS = ("risk", "lack_of_success", "lack_of_coverage")
JOINT_WEIGHTS = ((1/3, 1/3, 1/3), (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5))
BOOTSTRAPS = 10000


@dataclass(frozen=True)
class PublicCases:
    embedding: np.ndarray
    probability: np.ndarray
    prediction: np.ndarray


@dataclass(frozen=True)
class Reviews:
    ids: np.ndarray
    labels: np.ndarray

    def append(self, ids, labels):
        return Reviews(np.concatenate((self.ids, ids)), np.concatenate((self.labels, labels)))


def freeze_model():
    pixels, labels = load_digits(return_X_y=True)
    if pixels.shape != (1797, 64) or len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError("Expected the bundled digits dataset with distinct pixel rows")
    order = np.random.default_rng(TRAIN_SEED).permutation(len(labels))
    train, pool = order[:397], order[397:]
    x = pixels / 16.0
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, tol=1e-8)
    model.fit(x[train], labels[train])
    if not np.array_equal(model.classes_, np.arange(10)) or model.n_iter_.max() >= 500:
        raise ValueError("Classifier did not fit all ten classes to convergence")
    probability = model.predict_proba(x)
    centered = x - x[train].mean(axis=0)
    embedding = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    return PublicCases(embedding, probability, probability.argmax(axis=1)), labels, train, pool


def split_pool(pool, seed):
    order = np.random.default_rng(seed).permutation(pool)
    return np.split(order, [INITIAL, INITIAL + CALIBRATION, INITIAL + CALIBRATION + STREAM])


def loss(prediction, label, objective):
    distance = np.abs(prediction - label)
    if objective == "unit":
        return (distance != 0).astype(float)
    if objective == "distance":
        return distance
    raise ValueError(f"Unknown loss {objective}")


def frozen_scores(cases, ids, objective):
    if objective == "unit":
        return 1.0 - cases.probability[ids].max(axis=1)
    costs = loss(cases.prediction[ids, None], np.arange(10), objective)
    return (cases.probability[ids] * costs).sum(axis=1)


def vote_scores(method, cases, history, ids, objective, k):
    sim = cases.embedding[ids] @ cases.embedding[history.ids].T
    nearest = np.argsort(-np.round(sim, 12), axis=1, kind="stable")[:, :k]
    weights = 1.0 / np.maximum(1.0 - np.take_along_axis(sim, nearest, axis=1), 1e-8)
    if method == "class_vote":
        # The local digit distribution assesses the CURRENT proposed digit.
        costs = loss(cases.prediction[ids, None], history.labels[nearest], objective)
    else:
        # Local outcome labels describe PAST proposals on the neighbor cases.
        costs = loss(cases.prediction[history.ids], history.labels, objective)[nearest]
    return (weights * costs).sum(axis=1) / weights.sum(axis=1)


def choose_weight(signal, frozen, target):
    errors = []
    for weight in VOTE_WEIGHTS:
        estimate = weight * signal + (1.0 - weight) * frozen
        errors.append((round(float(np.mean((estimate - target) ** 2)), 12), weight))
    return min(errors)


def calibrate(method, cases, initial, calibration, objective):
    """Choose k and the weight on the local vote using calibration labels only."""
    target = loss(cases.prediction[calibration.ids], calibration.labels, objective)
    frozen = frozen_scores(cases, calibration.ids, objective)
    candidates = []
    for k in K_GRID:
        vote = vote_scores(method, cases, initial, calibration.ids, objective, k)
        mse, weight = choose_weight(vote, frozen, target)
        candidates.append((mse, k, weight))
    _, k, weight = min(candidates)
    return k, weight


def build_memory(method, cases, history, objective):
    distance = np.abs(cases.prediction[history.ids] - history.labels)
    if method not in ("bem_sign", "bem_graded"):
        raise ValueError(f"Unknown policy {method}")
    memory = BidirectionalExperienceMemory(dim=cases.embedding.shape[1], decay_rate=0.0)
    for row_id, d in zip(history.ids, distance):
        outcome = 1.0
        if d:
            outcome = -1.0
            if method == "bem_graded" and objective == "distance":
                # Keep even the mildest mistake below the package failure threshold.
                outcome = -(0.35 + 0.65 * int(d) / 9)
        memory.add_experience(cases.embedding[row_id], outcome, str(row_id))
    return memory


def bem_risk(method, cases, history, objective):
    memory = build_memory(method, cases, history, objective)
    return lambda ids: np.array([memory.risk_signal(cases.embedding[i])[0] for i in ids])


def joint_components(memory, embeddings):
    return np.array([
        (memory.risk_signal(z)[0], 1.0 - memory.success_signal(z)[0], 1.0 - memory.coverage_signal(z))
        for z in embeddings
    ])


def calibrate_joint(cases, initial, calibration, objective, scale):
    memory = build_memory("bem_graded", cases, initial, objective)
    signals = joint_components(memory, cases.embedding[calibration.ids])
    frozen = frozen_scores(cases, calibration.ids, objective)
    target = loss(cases.prediction[calibration.ids], calibration.labels, objective)
    candidates = []
    for index, weights in enumerate(JOINT_WEIGHTS):
        mse, outer = choose_weight(scale * (signals @ weights), frozen, target)
        candidates.append((mse, outer, index))
    mse, outer, index = min(candidates)
    return {"component_weights": dict(zip(JOINT_COMPONENTS, JOINT_WEIGHTS[index])),
            "memory_weight": outer, "calibration_mse": mse}


def make_scorer(method, cases, history, objective, settings):
    """Build from reviewed labels only. Returned callable accepts public case IDs."""
    if method == "arrival":
        return lambda ids: np.zeros(len(ids))
    if method == "frozen":
        return lambda ids: frozen_scores(cases, ids, objective)
    if method == "failure_only":
        failed = cases.embedding[history.ids[cases.prediction[history.ids] != history.labels]]
        if not len(failed):
            return lambda ids: np.zeros(len(ids))
        return lambda ids: (cases.embedding[ids] @ failed.T).max(axis=1)
    if method in ("outcome_vote", "class_vote"):
        k, weight = settings[method][objective]

        def signal(ids):
            return vote_scores(method, cases, history, ids, objective, k)

    elif method == "bem_joint_blend":
        config = settings[method][objective]
        weights = [config["component_weights"][key] for key in JOINT_COMPONENTS]
        weight = config["memory_weight"]
        memory = build_memory("bem_graded", cases, history, objective)
        scale = settings["distance_per_error"] if objective == "distance" else 1.0

        def signal(ids):
            return scale * (joint_components(memory, cases.embedding[ids]) @ weights)

    elif method in ("bem_sign_blend", "bem_graded_blend"):
        weight = settings[method][objective]
        risk = bem_risk(method.removesuffix("_blend"), cases, history, objective)
        scale = settings["distance_per_error"] if objective == "distance" else 1.0

        def signal(ids):
            return scale * risk(ids)

    else:
        return bem_risk(method, cases, history, objective)
    if weight == 0.0:
        return lambda ids: frozen_scores(cases, ids, objective)
    return lambda ids: weight * signal(ids) + (1.0 - weight) * frozen_scores(cases, ids, objective)


def select(batch, scores):
    if scores.shape != batch.shape or len(batch) != BATCH or not np.isfinite(scores).all():
        raise ValueError("Expected one finite score per case in a full batch")
    # Equal scores go to the earlier arrival, independent of class or label.
    return batch[np.argsort(-np.round(scores, 12), kind="stable")[:QUOTA]]


def collect_history(method, cases, initial, batches, settings, reveal):
    history = initial
    for batch in batches:
        score = make_scorer(method, cases, history, "unit", settings)
        chosen = select(batch, score(batch))
        history = history.append(chosen, reveal(chosen))
    return history


def future_choices(method, cases, history, batches, objective, settings):
    # No label callback: neither this batch nor earlier future batches feed back.
    score = make_scorer(method, cases, history, objective, settings)
    return np.concatenate([select(batch, score(batch)) for batch in batches])


def tally(cases, labels, ids):
    distance = np.abs(cases.prediction[ids] - labels[ids])
    return {"n": len(ids), "errors": int(np.count_nonzero(distance)), "distance": int(distance.sum())}


def replacements(current, previous):
    return len(current) - len(np.intersect1d(current, previous))


def run_seed(cases, labels, pool, seed):
    warm, calibration, stream, future = split_pool(pool, seed)
    initial = Reviews(warm, labels[warm])
    calibration_set = Reviews(calibration, labels[calibration])
    settings = {
        method: {obj: calibrate(method, cases, initial, calibration_set, obj) for obj in OBJECTIVES}
        for method in ("outcome_vote", "class_vote")
    }
    # Convert a dimensionless BEM risk into comparable distance units using
    # just the initial reviewed mistakes. The same constant serves both encodings.
    distances = np.abs(cases.prediction[warm] - initial.labels)
    errors = distances[distances > 0]
    settings["distance_per_error"] = float(errors.mean()) if len(errors) else 1.0
    for method in ("bem_sign", "bem_graded"):
        weights = {}
        for obj in OBJECTIVES:
            risk = bem_risk(method, cases, initial, obj)(calibration)
            scale = settings["distance_per_error"] if obj == "distance" else 1.0
            target = loss(cases.prediction[calibration], calibration_set.labels, obj)
            _, weights[obj] = choose_weight(scale * risk, frozen_scores(cases, calibration, obj), target)
        settings[method + "_blend"] = weights
    settings["bem_joint_blend"] = {
        obj: calibrate_joint(cases, initial, calibration_set, obj,
                             settings["distance_per_error"] if obj == "distance" else 1.0)
        for obj in OBJECTIVES
    }
    stream_batches = stream.reshape(-1, BATCH)
    future_batches = future.reshape(-1, BATCH)
    histories = {
        method: collect_history(method, cases, initial, stream_batches, settings, labels.__getitem__)
        for method in POLICIES
    }
    # Recode the SAME paid reviews for the sign/grade counterfactual.
    histories["bem_graded"] = histories["bem_sign"]
    histories["bem_graded_blend"] = histories["bem_sign_blend"]
    methods = {}
    for method, history in histories.items():
        selected_stream = history.ids[len(initial.ids):]
        chosen = {
            objective: future_choices(method, cases, history, future_batches, objective, settings)
            for objective in OBJECTIVES
        }
        by_objective = {}
        for objective, after in chosen.items():
            before = future_choices(method, cases, initial, future_batches, objective, settings)
            by_objective[objective] = {
                "reviewed": tally(cases, labels, after),
                "initial_only_reviewed": tally(cases, labels, before),
                "slots_changed_by_feedback": replacements(after, before),
                "slots_changed_from_unit": replacements(after, chosen["unit"]),
            }
        methods[method] = {
            "stream_reviewed": tally(cases, labels, selected_stream),
            "stream_review_order_sha256": hashlib.sha256(selected_stream.astype("<i4").tobytes()).hexdigest(),
            "future": by_objective,
        }
    return {
        "seed": seed,
        "available": {name: tally(cases, labels, ids) for name, ids in
                      (("initial", warm), ("calibration", calibration), ("stream", stream), ("future", future))},
        "calibrated_k_and_vote_weight": {m: settings[m] for m in ("outcome_vote", "class_vote")},
        "calibrated_bem_weight": {m: settings[m] for m in ("bem_sign_blend", "bem_graded_blend")},
        "calibrated_bem_joint_blend": settings["bem_joint_blend"],
        "initial_distance_per_error": settings["distance_per_error"],
        "methods": methods,
    }


def paired_interval(differences, resamples):
    values = np.asarray(differences)
    sampled_means = values[resamples].mean(axis=1)
    lo, hi = np.quantile(sampled_means, [0.025, 0.975])
    return {"mean": float(values.mean()), "percentile_95": [float(lo), float(hi)]}


def summarize(runs):
    """One draw is a whole paired replay on this fixed classifier and finite corpus."""
    rng = np.random.default_rng(15001)
    resamples = rng.integers(len(runs), size=(BOOTSTRAPS, len(runs)))
    summary = {}
    for objective, metric in (("unit", "errors"), ("distance", "distance")):
        rows = {}
        for method in runs[0]["methods"]:
            future = [r["methods"][method]["future"][objective] for r in runs]
            full = np.array([r["available"]["future"][metric] for r in runs])
            caught = np.array([v["reviewed"][metric] for v in future])
            initial = np.array([v["initial_only_reviewed"][metric] for v in future])
            unit = np.array([r["methods"][method]["future"]["unit"]["reviewed"][metric] for r in runs])
            rows[method] = {
                "total_removed": int(caught.sum()),
                "total_remaining": int((full - caught).sum()),
                "gain_from_feedback_per_seed": paired_interval(caught - initial, resamples),
                "gain_from_policy_change_per_seed": paired_interval(caught - unit, resamples),
                "total_slots_changed_by_feedback": sum(v["slots_changed_by_feedback"] for v in future),
                "total_slots_changed_from_unit": sum(v["slots_changed_from_unit"] for v in future),
            }
        contrasts = {}
        for a, b in (("bem_graded", "bem_sign"), ("bem_graded", "class_vote"),
                     ("bem_graded", "outcome_vote"), ("bem_graded", "frozen"),
                     ("bem_graded", "failure_only"), ("class_vote", "frozen"),
                     ("bem_graded_blend", "bem_sign_blend"), ("bem_graded_blend", "frozen"),
                     ("bem_sign_blend", "frozen"), ("bem_graded_blend", "class_vote"),
                     ("bem_graded_blend", "outcome_vote"), ("bem_joint_blend", "frozen"),
                     ("bem_joint_blend", "bem_graded_blend"), ("bem_joint_blend", "outcome_vote"),
                     ("bem_joint_blend", "class_vote")):
            diffs = [r["methods"][a]["future"][objective]["reviewed"][metric]
                     - r["methods"][b]["future"][objective]["reviewed"][metric] for r in runs]
            contrasts[f"{a}_minus_{b}"] = paired_interval(diffs, resamples)
        summary[objective] = {"methods": rows, "paired_removed_per_seed": contrasts}
    return summary


def benchmark():
    with threadpool_limits(limits=1):
        cases, labels, train, pool = freeze_model()
        runs = [run_seed(cases, labels, pool, seed) for seed in SEEDS]
    return {
        "schema_version": 1,
        "protocol": {"dataset": "sklearn.datasets.load_digits", "train_seed": TRAIN_SEED,
                     "batch": BATCH, "reviews_per_batch": QUOTA, "bootstrap_draws": BOOTSTRAPS},
        "train": tally(cases, labels, train), "pool": tally(cases, labels, pool),
        "summary": summarize(runs), "runs": runs,
    }


def compare(expected, actual, path="$"):
    """Counts, membership hashes and structure are exact. Floats tolerate roundoff."""
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        for key in actual:
            compare(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, list) and len(actual) == len(expected):
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{i}]")
        return
    if isinstance(actual, float) and type(expected) in (float, int):
        # L-BFGS fits on different BLAS backends can stop at slightly different
        # probabilities. Only the descriptive calibration loss has this tolerance.
        if path.endswith(".calibration_mse") and np.isclose(
                expected, actual, rtol=1e-5, atol=1e-7):
            return
        if np.isclose(expected, actual, rtol=1e-7, atol=1e-8):
            return
    elif isinstance(actual, (str, int)) and type(expected) is type(actual) and expected == actual:
        return
    raise ValueError(f"Benchmark mismatch at {path}. Expected {expected!r}, measured {actual!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="write the full benchmark JSON (default: stdout)")
    mode.add_argument("--check", type=Path, help="rerun and compare with an existing JSON")
    args = parser.parse_args()
    result = benchmark()
    if args.check:
        try:
            expected = json.loads(args.check.read_text())
            compare(expected, result)
        except (OSError, ValueError) as err:
            parser.exit(1, f"{err}\n")
        max_delta = max(
            abs(actual["calibrated_bem_joint_blend"][obj]["calibration_mse"]
                - reference["calibrated_bem_joint_blend"][obj]["calibration_mse"])
            for actual, reference in zip(result["runs"], expected["runs"])
            for obj in OBJECTIVES
        )
        print(f"Reproduced {len(result['runs'])} paired seeds: {args.check}"
              f" (maximum absolute calibration MSE difference {max_delta:.3g})")
    else:
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output)
        else:
            sys.stdout.write(output)


if __name__ == "__main__":
    main()
