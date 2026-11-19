"""Retrieve two observed outcomes for a new embedding. Runs without downloads."""
import numpy as np

from alfm_bem import BidirectionalExperienceMemory

memory = BidirectionalExperienceMemory(dim=3, similarity_threshold=0.5)
memory.add_experience(np.array([1.0, 0.0, 0.0]), -1.0, "case-17")
memory.add_experience(np.array([0.0, 1.0, 0.0]), 1.0, "case-23")

incoming = np.array([0.8, 0.6, 0.0])
risk, failures = memory.risk_signal(incoming)
success, successes = memory.success_signal(incoming)
assert {record.context_hash for record in failures} == {"case-17"}
assert {record.context_hash for record in successes} == {"case-23"}
print(f"Risk evidence {risk:.3f}, success evidence {success:.3f}")
print("Related cases:", failures[0].context_hash, successes[0].context_hash)
