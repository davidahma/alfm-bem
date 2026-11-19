"""Paired, prequential decisions on generated claims under three fixed payer rules."""

from dataclasses import dataclass
import random

import numpy as np

from alfm_bem.bem import BidirectionalExperienceMemory, CoverageMode

CPT = ("99213", "90658", "25111", "99203", "71045")
DIAGNOSES = ("J01.90", "J20.9", "R05", "M67.4", "Z23")
MODIFIERS = ("", "RT", "LT", "25", "59")


@dataclass(frozen=True)
class Claim:
    cpt: str
    dx: str
    age: int
    modifier: str

    def denied(self):
        return ((self.cpt == "99213" and self.dx not in ("J01.90", "J20.9"))
                or (self.cpt == "90658" and self.age <= 3)
                or (self.cpt == "25111" and self.modifier not in ("RT", "LT")))


def generate_claim(rng):
    cpt = rng.choice(CPT)
    if cpt == "99213":
        dx = rng.choice(["J01.90", "J20.9", "R05"])
    elif cpt == "25111":
        dx = "M67.4"
    elif cpt == "90658":
        dx = "Z23"
    else:
        dx = rng.choice(DIAGNOSES)
    age = rng.randint(1, 80)
    if cpt == "25111":
        modifier = rng.choices(["RT", "LT", ""], weights=[0.4, 0.4, 0.2])[0]
    else:
        modifier = rng.choice(MODIFIERS) if rng.random() > 0.5 else ""
    rng.randint(1, 5)  # Original unused provider ID, part of the seeded stream.
    return Claim(cpt, dx, age, modifier)


def project_claims(claims):
    rng = np.random.default_rng(42)
    cpt, dx, modifier, age = [
        {key: rng.standard_normal(64) for key in keys}
        for keys in (CPT, DIAGNOSES, MODIFIERS, range(10))
    ]
    vectors = []
    for claim in claims:
        v = cpt[claim.cpt] + dx[claim.dx] + modifier[claim.modifier] + age[claim.age // 10]
        vectors.append(v / (np.linalg.norm(v) + 1e-10))
    return np.stack(vectors)


def totals(submitted, denied):
    n_sub = int(submitted.sum())
    n_rej = int((submitted & denied).sum())
    return {
        "claims": len(denied), "would_deny": int(denied.sum()), "submitted": n_sub,
        "denied_submitted": n_rej, "referred": len(denied) - n_sub,
        "would_pay_referred": int((~submitted & ~denied).sum()),
        "denial_rate_among_submitted": n_rej / n_sub if n_sub else None,
    }


def paired_decisions(claims, vectors, denied, observe_all):
    # Only risk is read. KNN_AVG skips rebuilding the unused KDE after each insertion.
    memory = BidirectionalExperienceMemory(dim=64, similarity_threshold=0.85,
                                           coverage_mode=CoverageMode.KNN_AVG)
    failure_index = []
    history = []
    for i, (claim, vector, deny) in enumerate(zip(claims, vectors, denied)):
        risk, _ = memory.risk_signal(vector)
        bem_submits = risk <= 0.6
        match = bool(failure_index) and np.max(vectors[failure_index] @ vector) > 0.85
        neighbor_submits = not match
        history.append((bem_submits, neighbor_submits))
        # Each policy sees only feedback available after its own decision.
        if observe_all or bem_submits:
            memory.add_experience(vector, -1.0 if deny else 1.0, str(claim))
        if deny and (observe_all or neighbor_submits):
            failure_index.append(i)
    bem, neighbor = np.array(history).T
    return {
        "decision_disagreements": int((bem != neighbor).sum()),
        **{name: {"overall": totals(sub, denied), "final_100": totals(sub[-100:], denied[-100:])}
           for name, sub in (("bem", bem), ("max_failure", neighbor))},
    }


def run_healthcare(seed, n=2000):
    rng = random.Random(seed)
    claims = [generate_claim(rng) for _ in range(n)]
    vectors = project_claims(claims)
    denied = np.array([claim.denied() for claim in claims])
    submit_all = np.ones(n, dtype=bool)
    return {
        "seed": seed,
        "trust_all": {"overall": totals(submit_all, denied),
                      "final_100": totals(submit_all[-100:], denied[-100:])},
        "all_outcomes": paired_decisions(claims, vectors, denied, True),
        "submitted_only": paired_decisions(claims, vectors, denied, False),
    }
