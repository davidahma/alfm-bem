"""Run or verify the release CPU controls without downloading any data."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .ablation_study import ablation, raw_threshold_control
from .healthcare_simulator import run_healthcare
from .metrics import write_json
from .query_simulation import run_query


SEEDS = (42, 123, 456, 789, 1011)


def run_controls():
    results = {}
    with threadpool_limits(limits=1):
        for geometry in ("original_frobenius", "unit_centers"):
            rows = [ablation(seed, geometry) for seed in SEEDS]
            summary = {}
            for method, metrics in rows[0]["retrieval"].items():
                scores = {}
                for sign in metrics:
                    f1s = [row["retrieval"][method][sign]["f1"] for row in rows]
                    scores[sign] = {"mean_f1": float(np.mean(f1s)), "sd_f1": float(np.std(f1s))}
                summary[method] = scores
            results[geometry] = {"by_seed": rows, "retrieval_summary": summary}
        raw = raw_threshold_control()
        health = [run_healthcare(seed) for seed in SEEDS]
    return {
        "schema_version": 1,
        "ablation": {"train_per_sign": 500, "test_per_sign": 100, "dimension": 64,
                     "id_core_failures": 200, "ood_per_family": 200, **results},
        "raw_768": raw, "healthcare": health, "query": run_query(),
    }


def differences(actual, expected, path="$"):
    """Strict JSON structure and counts, small tolerance for floating-point metrics."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(actual.keys() | expected.keys()):
            if key not in actual or key not in expected:
                yield f"{path}.{key}: missing or unexpected key"
            else:
                yield from differences(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            yield f"{path}: measured {len(actual)} items, expected {len(expected)}"
            return
        for i, (item, ref) in enumerate(zip(actual, expected)):
            yield from differences(item, ref, f"{path}[{i}]")
    elif type(actual) is float and type(expected) in (int, float):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
            yield f"{path}: measured {actual}, expected {expected}"
    elif type(actual) is not type(expected) or actual != expected:
        yield f"{path}: measured {actual!r}, expected {expected!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="write the measurements as JSON")
    group.add_argument("--check", type=Path, help="rerun and compare against a saved JSON fixture")
    args = parser.parse_args()
    ref = None
    if args.check:
        try:
            ref = json.loads(args.check.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            parser.exit(1, f"Cannot read reference: {exc}\n")
    actual = run_controls()
    if args.output:
        write_json(args.output, actual)
        print(f"Wrote {args.output}")
    else:
        mismatch = next(differences(actual, ref), None)
        if mismatch:
            parser.exit(1, f"Controls differ at {mismatch}\n")
        print(f"Controls match {args.check}")


if __name__ == "__main__":
    main()
