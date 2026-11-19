"""Outcome-labeled memory with cosine retrieval, failure and success scores,
kernel-based familiarity and sampling for adapter replay.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from enum import Enum
import json
import hashlib
from datetime import datetime, timedelta

from .constants import DEFAULT_FAILURE_THRESHOLD, DEFAULT_SUCCESS_THRESHOLD, KDE_BANDWIDTH


class CoverageMode(Enum):
    """Coverage signal computation method."""
    LEGACY = "legacy"      # Maximum similarity and Mahalanobis distance
    KNN_AVG = "knn_avg"    # k-NN average similarity
    KDE = "kde"            # Kernel mean on cosine dissimilarity


@dataclass
class Experience:
    """A single deployment experience stored in BEM."""
    
    embedding: np.ndarray          # Projected context embedding z ∈ R^d
    outcome: float                 # Continuous outcome score in [-1, 1]
    context_hash: str              # Caller key, or SHA-256 of context from the manager
    timestamp: datetime            # When this experience occurred
    tenant_id: str = "default"     # Tenant isolation
    domain_id: str = "default"     # Domain classification
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields for richer experiences
    reasoning_trace: Optional[str] = None   # What reasoning led to this outcome
    correction: Optional[str] = None        # Human correction if failure
    confirmation_count: int = 1             # How many times confirmed
    
    @property
    def is_failure(self) -> bool:
        return self.outcome < DEFAULT_FAILURE_THRESHOLD
    
    @property
    def is_success(self) -> bool:
        return self.outcome > DEFAULT_SUCCESS_THRESHOLD
    
    @property
    def severity(self) -> int:
        """Map outcome to severity 1-5 for compatibility with original NEP."""
        if self.outcome >= 0.5:
            return 1  # Strong success
        elif self.outcome >= 0:
            return 2  # Mild success
        elif self.outcome >= -0.5:
            return 3  # Mild failure
        elif self.outcome >= -0.8:
            return 4  # Significant failure
        else:
            return 5  # Critical failure


class BidirectionalExperienceMemory:
    """Store records under a case, tenant and domain key.

    risk_signal and success_signal return scores and contributing records.
    coverage_signal measures support from the stored embeddings.
    """

    def __init__(
        self,
        dim: int = 768,
        similarity_threshold: float = 0.5,
        risk_sensitivity: float = 0.8,
        decay_rate: float = 0.01,
        tenant_id: str = "global",
        coverage_mode: CoverageMode = CoverageMode.KDE,
        kde_bandwidth: float = KDE_BANDWIDTH,
        knn_k: int = 20,
        use_knn_index: bool = False  # Optional exact unfiltered top-k retrieval
    ):
        if dim < 1 or knn_k < 1 or not np.isfinite(kde_bandwidth) or kde_bandwidth <= 0:
            raise ValueError("dim, knn_k and kde_bandwidth must be positive")
        if not -1 <= similarity_threshold <= 1 or not np.isfinite(risk_sensitivity) or risk_sensitivity < 0:
            raise ValueError("Invalid similarity threshold or risk sensitivity")
        if not np.isfinite(decay_rate) or decay_rate < 0:
            raise ValueError("decay_rate must be finite and nonnegative")
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self.risk_sensitivity = risk_sensitivity
        self.decay_rate = decay_rate
        self.tenant_id = tenant_id

        # Coverage signal configuration
        self.coverage_mode = CoverageMode(coverage_mode)
        self.kde_bandwidth = kde_bandwidth
        self.knn_k = knn_k
        self.use_knn_index = use_knn_index

        # Unified storage — no separate failure/success stores
        self.experiences: List[Experience] = []

        # Index structures for efficient retrieval
        self._embedding_matrix: Optional[np.ndarray] = None
        self._knn_index = None
        self._kde_coverage_scale: float = 1.0

        # For legacy coverage mode
        self._mean_embedding: Optional[np.ndarray] = None
        self._cov_embedding: Optional[np.ndarray] = None

        # Anti-poisoning: velocity tracking per source
        self._feedback_velocity: Dict[str, float] = defaultdict(float)
        self._feedback_history: Dict[str, List[datetime]] = defaultdict(list)
        self._quarantine: List[Experience] = []

    def add_experience(
        self, 
        embedding: np.ndarray, 
        outcome: float, 
        context_hash: str,
        tenant_id: str = "default",
        domain_id: str = "default",
        reasoning_trace: Optional[str] = None,
        correction: Optional[str] = None
    ):
        """Add new experience to memory.

        If an experience with the same (context_hash, tenant_id, domain_id) already
        exists, we treat this as a re-confirmation: we increment confirmation count,
        refresh timestamp, and softly update the stored outcome/embedding.
        """
        if not -1 <= outcome <= 1:
            raise ValueError("outcome must lie in [-1, 1]")
        outcome = float(outcome)
        now = datetime.now()
        embedding = self._normalize(embedding)

        for exp in self.experiences:
            if (
                exp.context_hash == context_hash
                and exp.tenant_id == tenant_id
                and exp.domain_id == domain_id
            ):
                n = max(1, exp.confirmation_count)
                exp.confirmation_count = n + 1
                exp.timestamp = now
                exp.outcome = (exp.outcome * n + outcome) / (n + 1)
                exp.embedding = (exp.embedding * n + embedding) / (n + 1)
                exp.embedding = exp.embedding / (np.linalg.norm(exp.embedding) + 1e-10)
                if reasoning_trace is not None:
                    exp.reasoning_trace = reasoning_trace
                if correction is not None:
                    exp.correction = correction
                self._invalidate_index()
                return

        exp = Experience(
            embedding=embedding,
            outcome=outcome,
            context_hash=context_hash,
            timestamp=now,
            tenant_id=tenant_id,
            domain_id=domain_id,
            reasoning_trace=reasoning_trace,
            correction=correction,
        )
        self.experiences.append(exp)
        
        self._invalidate_index()

    def _normalize(self, embedding):
        embedding = np.asarray(embedding)
        if embedding.shape != (self.dim,) or not np.isfinite(embedding).all():
            raise ValueError(f"Expected a finite embedding of shape ({self.dim},)")
        return embedding / (np.linalg.norm(embedding) + 1e-10)

    def _invalidate_index(self):
        self._embedding_matrix = None
        self._knn_index = None
        self._kde_coverage_scale = 1.0
        self._mean_embedding = None
        self._cov_embedding = None

    def _ensure_index(self):
        if self._embedding_matrix is None and self.experiences:
            self._embedding_matrix = np.vstack([e.embedding for e in self.experiences])
            if self.coverage_mode == CoverageMode.KDE:
                self._calibrate_kde_coverage_scale()
            elif self.coverage_mode == CoverageMode.LEGACY and len(self.experiences) >= 10:
                self._mean_embedding = self._embedding_matrix.mean(axis=0)
                self._cov_embedding = np.cov(self._embedding_matrix, rowvar=False)

    def _calibrate_kde_coverage_scale(self):
        """
        Calibrate a per-memory linear scale that maps typical in-distribution KDE
        densities to a coverage near 1.0 without relying on a hard-coded constant.
        """
        if self._embedding_matrix is None or len(self.experiences) < 10:
            self._kde_coverage_scale = 1.0
            return

        # Deterministic subsample to keep calibration stable across runs.
        n = self._embedding_matrix.shape[0]
        sample_size = min(256, n)
        indices = np.linspace(0, n - 1, sample_size, dtype=int)
        sampled = self._embedding_matrix[indices]

        densities = []
        for x in sampled:
            sims = self._embedding_matrix @ x
            distances = 1.0 - sims
            kernel_vals = np.exp(-distances**2 / (2 * self.kde_bandwidth**2))
            densities.append(float(np.mean(kernel_vals)))

        # Map the 90th percentile of ID densities to 1.0.
        denom = max(float(np.percentile(densities, 90)), 1e-12)
        self._kde_coverage_scale = 1.0 / denom

    def _compute_similarities(self, z: np.ndarray) -> np.ndarray:
        self._ensure_index()
        if self._embedding_matrix is None:
            return np.array([])
            
        return self._embedding_matrix @ self._normalize(z)

    def risk_signal(self, z: np.ndarray) -> Tuple[float, List[Experience]]:
        """
        Compute a risk score for this context,
        based on similarity to past failures.
        
        This is the generalization of NEP's s_NEP.
        
        Returns:
            (risk_score in [0,1], list of relevant failure experiences)
        """
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return 0.0, []
        
        # Filter to failures and above threshold
        relevant_candidates = []
        
        for i, (sim, exp) in enumerate(zip(sims, self.experiences)):
            if sim > self.similarity_threshold and exp.is_failure:
                relevant_candidates.append((sim, exp))
        
        if not relevant_candidates:
            return 0.0, []
            
        # Sort by similarity (descending) and take top-k to prevent unbounded risk accumulation
        # from many weak signals in dense regions
        relevant_candidates.sort(key=lambda x: x[0], reverse=True)
        relevant_candidates = relevant_candidates[:20]  # Limit to 20 nearest failures
        
        relevant = [exp for sim, exp in relevant_candidates]
        risk_components = []
        
        for sim, exp in relevant_candidates:
            # Weight by similarity, severity, and recency
            age_days = (datetime.now() - exp.timestamp).days
            decay = np.exp(-self.decay_rate * age_days)
            weight = sim * (exp.severity / 5.0) * decay * exp.confirmation_count
            risk_components.append(
                self.risk_sensitivity * weight
            )
        
        if not risk_components:
            return 0.0, []
        
        # Combine the clipped components as a noisy-OR score.
        risk = 1.0 - np.prod([1.0 - np.clip(r, 0.0, 0.99) for r in risk_components])
        
        # Correlation boost: if multiple similar failures, increase risk
        if len(relevant) > 1:
            failure_embeddings = np.vstack([e.embedding for e in relevant])
            pairwise_sims = failure_embeddings @ failure_embeddings.T
            max_correlation = np.max(pairwise_sims[np.triu_indices(len(relevant), k=1)])
            risk = min(1.0, risk + 0.1 * max_correlation)
        
        return float(np.clip(risk, 0.0, 1.0)), relevant
    
    def success_signal(self, z: np.ndarray) -> Tuple[float, List[Experience]]:
        """
        Compute success signal: evidence that similar contexts succeeded.
        
        This enables retrieval of successful reasoning patterns.
        """
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return 0.0, []
        
        relevant = []
        success_components = []
        
        for sim, exp in zip(sims, self.experiences):
            if sim > self.similarity_threshold and exp.is_success:
                relevant.append(exp)
                age_days = (datetime.now() - exp.timestamp).days
                decay = np.exp(-self.decay_rate * age_days)
                weight = sim * ((6 - exp.severity) / 5.0) * decay
                success_components.append(weight)
        
        if not success_components:
            return 0.0, []
        
        # Average success evidence (not noisy-OR — successes are reinforcing)
        success = float(np.mean(success_components))
        return float(np.clip(success, 0.0, 1.0)), relevant
    
    def coverage_signal(self, z: np.ndarray) -> float:
        """
        Compute coverage signal: how well is this context represented
        in our experience distribution?
        
        Low coverage indicates weak support in the stored embeddings.
        
        Supports multiple modes:
        - LEGACY: max_sim + Mahalanobis
        - KNN_AVG: k-NN average similarity
        - KDE: Kernel Density Estimation on cosine distances
        """
        self._ensure_index()
        if self._embedding_matrix is None or len(self.experiences) < 10:
            return 0.0  # Not enough data for coverage estimation
        
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return 0.0
        
        if self.coverage_mode == CoverageMode.KDE:
            return self._coverage_kde(sims)
        elif self.coverage_mode == CoverageMode.KNN_AVG:
            return self._coverage_knn_avg(sims)
        else:  # LEGACY
            return self._coverage_legacy(z, sims)
    
    def _coverage_kde(self, sims: np.ndarray) -> float:
        """
        Kernel Density Estimation coverage.
        
        Computes density using Gaussian kernel over cosine distances.
        High density = well covered by experiences = in-distribution.
        """
        # Convert similarities to distances
        distances = 1.0 - sims
        
        # Gaussian kernel: K(d) = exp(-d^2 / (2 * bw^2))
        kernel_vals = np.exp(-distances**2 / (2 * self.kde_bandwidth**2))
        
        # Normalize by number of experiences for consistent scale.
        density = np.mean(kernel_vals)

        # Calibrated rescaling to [0, 1] based on typical ID density.
        coverage = np.clip(density * self._kde_coverage_scale, 0.0, 1.0)
        return float(coverage)
    
    def _coverage_knn_avg(self, sims: np.ndarray) -> float:
        """
        k-NN average similarity coverage.
        
        Average of top-k similarities. More robust than max similarity.
        """
        k = min(self.knn_k, len(sims))
        top_k_sims = np.sort(sims)[-k:]
        coverage = float(np.mean(top_k_sims))
        return np.clip(coverage, 0.0, 1.0)
    
    def _coverage_legacy(self, z: np.ndarray, sims: np.ndarray) -> float:
        """
        Original coverage method: max_sim + Mahalanobis.
        
        Kept for backward compatibility and ablation studies.
        """
        z_norm = z / (np.linalg.norm(z) + 1e-10)
        max_sim = np.max(sims) if len(sims) > 0 else 0.0
        
        try:
            diff = z_norm - self._mean_embedding
            covariance = self._cov_embedding + 1e-6 * np.eye(self.dim)
            mahal_dist = np.sqrt(max(0.0, diff @ np.linalg.solve(covariance, diff)))
            mahal_coverage = np.exp(-mahal_dist / 10.0)
        except np.linalg.LinAlgError:
            mahal_coverage = 0.5
        
        coverage = 0.6 * max_sim + 0.4 * mahal_coverage
        return float(np.clip(coverage, 0.0, 1.0))
    
    def retrieve(
        self,
        z: np.ndarray,
        k: int = 5,
        outcome_filter: Optional[str] = None  # "failure", "success", or None for all
    ) -> List[Tuple[Experience, float]]:
        """
        Retrieve k most relevant experiences with their similarities.
        
        Args:
            z: Query embedding
            k: Number to retrieve
            outcome_filter: Optionally filter to failures or successes
        """
        if k <= 0 or not self.experiences:
            return []
        if self.use_knn_index and outcome_filter is None:
            from sklearn.neighbors import NearestNeighbors
            self._ensure_index()
            if self._knn_index is None:
                self._knn_index = NearestNeighbors(metric="cosine", algorithm="brute")
                self._knn_index.fit(self._embedding_matrix)
            distances, indices = self._knn_index.kneighbors(
                self._normalize(z)[None, :], n_neighbors=min(k, len(self.experiences))
            )
            return [(self.experiences[i], float(1 - d)) for i, d in zip(indices[0], distances[0])]

        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return []
        
        # Filter by outcome if requested
        candidates = []
        for i, (sim, exp) in enumerate(zip(sims, self.experiences)):
            if outcome_filter == "failure" and not exp.is_failure:
                continue
            if outcome_filter == "success" and not exp.is_success:
                continue
            candidates.append((exp, float(sim)))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
    
    def sample_for_training(
        self,
        batch_size: int = 32,
        failure_ratio: float = 0.7
    ) -> List[Experience]:
        """
        Sample a balanced batch for adapter training.
        
        Target the requested failure fraction and fill available slots from either sign.
        """
        failures = [e for e in self.experiences if e.is_failure]
        successes = [e for e in self.experiences if e.is_success]
        
        n_failures = min(int(batch_size * failure_ratio), len(failures))
        n_successes = min(batch_size - n_failures, len(successes))
        n_failures = min(batch_size - n_successes, len(failures))
        
        # Weight by confirmation count (more confirmed = more important)
        if failures:
            failure_weights = np.array([e.confirmation_count for e in failures])
            failure_weights = failure_weights / failure_weights.sum()
            sampled_failures = list(np.random.choice(
                failures, size=n_failures, replace=False, p=failure_weights
            )) if n_failures <= len(failures) else failures
        else:
            sampled_failures = []
        
        if successes:
            success_weights = np.array([e.confirmation_count for e in successes])
            success_weights = success_weights / success_weights.sum()
            sampled_successes = list(np.random.choice(
                successes, size=n_successes, replace=False, p=success_weights
            )) if n_successes <= len(successes) else successes
        else:
            sampled_successes = []
        
        return sampled_failures + sampled_successes
    
    def vacuum(
        self,
        max_size: int = 10000,
        cluster_threshold: float = 0.95,
        ttl_days: Optional[int] = 90,
        min_confirmations_to_keep: int = 3,
        keep_abs_outcome_at_least: float = 0.8,
    ):
        """
        Prune expired, weakly confirmed, low-magnitude outcomes and cap record count.

        The legacy cluster_threshold argument is unused.
        """
        if not self.experiences:
            return

        now = datetime.now()

        # Time-based TTL pruning for low-utility entries.
        # Strongly confirmed or high-severity experiences are retained.
        if ttl_days is not None:
            cutoff = now - timedelta(days=ttl_days)
            kept: List[Experience] = []
            for exp in self.experiences:
                if exp.timestamp >= cutoff:
                    kept.append(exp)
                    continue
                if exp.confirmation_count >= min_confirmations_to_keep:
                    kept.append(exp)
                    continue
                if abs(exp.outcome) >= keep_abs_outcome_at_least:
                    kept.append(exp)
                    continue
            if len(kept) != len(self.experiences):
                self.experiences = kept
                self._invalidate_index()

        if len(self.experiences) <= max_size:
            return

        # Deterministic saturation policy: keep most confirmed, most severe, and most recent.
        self.experiences.sort(
            key=lambda e: (e.confirmation_count, abs(e.outcome), e.timestamp),
            reverse=True
        )
        self.experiences = self.experiences[:max_size]
        self._invalidate_index()

    def check_feedback_velocity(
        self,
        source_id: str,
        gamma: float = 0.95,
        threshold_sigma: float = 3.0
    ) -> bool:
        """
        Flag unusually high source activity using a decayed count per submission.

        The comparison uses the mean and standard deviation across seen sources.
        Elapsed wall-clock time does not enter the score.
        """
        now = datetime.now()
        self._feedback_history[source_id].append(now)

        # Update EMA velocity
        old_velocity = self._feedback_velocity[source_id]
        self._feedback_velocity[source_id] = gamma * old_velocity + (1 - gamma) * 1.0

        # Compute global statistics
        all_velocities = list(self._feedback_velocity.values())
        if len(all_velocities) < 5:
            return False  # Not enough data

        mean_v = np.mean(all_velocities)
        std_v = np.std(all_velocities) + 1e-10

        # Anomaly detection
        if self._feedback_velocity[source_id] > mean_v + threshold_sigma * std_v:
            return True  # Anomalous velocity

        return False

    def add_experience_with_validation(
        self,
        embedding: np.ndarray,
        outcome: float,
        context_hash: str,
        source_id: str = "unknown",
        tenant_id: str = "default",
        domain_id: str = "default",
        **kwargs
    ) -> Optional[str]:
        """
        Check source activity and sign conflicts before adding an experience.

        Returns:
            None if successful, error message if rejected/quarantined
        """
        # Check velocity
        if self.check_feedback_velocity(source_id):
            # Quarantine instead of immediate add
            exp = Experience(
                embedding=embedding / (np.linalg.norm(embedding) + 1e-10),
                outcome=outcome,
                context_hash=context_hash,
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                domain_id=domain_id,
                **kwargs
            )
            self._quarantine.append(exp)
            return f"Quarantined: High feedback velocity from {source_id}"

        # Check conflict
        conflict = self.check_conflict(embedding, outcome)
        if conflict:
            return conflict

        # Normal add
        self.add_experience(
            embedding=embedding,
            outcome=outcome,
            context_hash=context_hash,
            tenant_id=tenant_id,
            domain_id=domain_id,
            **kwargs
        )
        return None

    def process_quarantine(self, admin_approved_ids: List[str] = None):
        """
        Process quarantined experiences.

        Args:
            admin_approved_ids: List of context_hashes approved by admin
        """
        if admin_approved_ids is None:
            admin_approved_ids = []

        approved = []
        rejected = []

        for exp in self._quarantine:
            if exp.context_hash in admin_approved_ids:
                self.experiences.append(exp)
                approved.append(exp.context_hash)
            else:
                rejected.append(exp.context_hash)

        # Clear quarantine
        self._quarantine = [e for e in self._quarantine
                           if e.context_hash not in admin_approved_ids]

        if approved:
            self._invalidate_index()

        return {"approved": len(approved), "rejected": len(rejected)}

    def check_conflict(self, z: np.ndarray, outcome: float) -> Optional[str]:
        """
        Check if the proposed outcome contradicts high-confidence memory.
        
        Report conflicting strong signs above similarity 0.85 with at least three confirmations.
        
        Returns:
            Warning string if conflict found, None otherwise.
        """
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return None
            
        # Look for very similar experiences (strict threshold 0.85)
        CONFLICT_SIM_THRESHOLD = 0.85
        CONFIDENCE_MIN = 3  # Must be confirmed 3+ times to block
        
        for sim, exp in zip(sims, self.experiences):
            if sim > CONFLICT_SIM_THRESHOLD:
                # Check for sign mismatch (Opposite labels)
                # Success vs Failure
                if (outcome > 0.3 and exp.is_failure) or (outcome < -0.3 and exp.is_success):
                    if exp.confirmation_count >= CONFIDENCE_MIN:
                        return (
                            f"Conflict detected: New outcome ({outcome:.2f}) contradicts "
                            f"established experience ({exp.outcome:.2f}, confirmed {exp.confirmation_count}x) "
                            f"with similarity {sim:.2f}."
                        )
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the memory."""
        if not self.experiences:
            return {"total": 0}
        
        outcomes = np.array([e.outcome for e in self.experiences])
        return {
            "total": len(self.experiences),
            "failures": sum(1 for e in self.experiences if e.is_failure),
            "successes": sum(1 for e in self.experiences if e.is_success),
            "neutral": sum(1 for e in self.experiences if not e.is_failure and not e.is_success),
            "mean_outcome": float(np.mean(outcomes)),
            "std_outcome": float(np.std(outcomes)),
            "unique_domains": len(set(e.domain_id for e in self.experiences)),
            "total_confirmations": sum(e.confirmation_count for e in self.experiences)
        }
    
    def save(self, path: str):
        """Serialize memory to disk."""
        with open(path, 'w') as f:
            json.dump(self._as_dict(), f)

    def _as_dict(self):
        return {
            "config": {
                "dim": self.dim,
                "similarity_threshold": self.similarity_threshold,
                "risk_sensitivity": self.risk_sensitivity,
                "decay_rate": self.decay_rate,
                "tenant_id": self.tenant_id,
                "coverage_mode": self.coverage_mode.value,
                "kde_bandwidth": self.kde_bandwidth,
                "knn_k": self.knn_k,
                "use_knn_index": self.use_knn_index,
            },
            "experiences": [
                {
                    "embedding": e.embedding.tolist(),
                    "embedding_dtype": e.embedding.dtype.name,
                    "outcome": e.outcome,
                    "context_hash": e.context_hash,
                    "timestamp": e.timestamp.isoformat(),
                    "tenant_id": e.tenant_id,
                    "domain_id": e.domain_id,
                    "metadata": e.metadata,
                    "reasoning_trace": e.reasoning_trace,
                    "correction": e.correction,
                    "confirmation_count": e.confirmation_count
                }
                for e in self.experiences
            ]
        }
    
    @classmethod
    def load(cls, path: str) -> 'BidirectionalExperienceMemory':
        """Load memory from disk."""
        with open(path, 'r') as f:
            return cls._from_dict(json.load(f))

    @classmethod
    def _from_dict(cls, data):
        bem = cls(**data["config"])
        for e_data in data["experiences"]:
            exp = Experience(
                embedding=np.array(e_data["embedding"], dtype=e_data.get("embedding_dtype", "float64")),
                outcome=e_data["outcome"],
                context_hash=e_data["context_hash"],
                timestamp=datetime.fromisoformat(e_data["timestamp"]),
                tenant_id=e_data["tenant_id"],
                domain_id=e_data["domain_id"],
                metadata=e_data["metadata"],
                reasoning_trace=e_data["reasoning_trace"],
                correction=e_data["correction"],
                confirmation_count=e_data["confirmation_count"]
            )
            bem.experiences.append(exp)
        
        return bem


# Tenant-isolated memory manager
class BEMManager:
    """
    Manages multiple BEM instances with tenant isolation.
    
    Structure:
    - Global BEM: Shared records read on every managed query
    - Domain BEMs: Domain-specific experiences
    - Tenant BEMs: Per-tenant isolated experiences
    """
    
    def __init__(self, dim: int = 768, similarity_threshold: float = 0.5,
                 risk_sensitivity: float = 0.8):
        self.dim = dim
        self._memory_config = dict(dim=dim, similarity_threshold=similarity_threshold,
                                   risk_sensitivity=risk_sensitivity)
        self.global_bem = BidirectionalExperienceMemory(**self._memory_config, tenant_id="global")
        self.domain_bems: Dict[str, BidirectionalExperienceMemory] = {}
        self.tenant_bems: Dict[str, BidirectionalExperienceMemory] = {}
    
    def get_scoped_memory(
        self,
        tenant_id: str,
        domain_id: str = "default"
    ) -> List[BidirectionalExperienceMemory]:
        """
        Get all memories in scope for a tenant query.
        
        Returns [global, domain, tenant] memories for unified querying.
        """
        memories = [self.global_bem]
        
        if domain_id in self.domain_bems:
            memories.append(self.domain_bems[domain_id])
        
        if tenant_id in self.tenant_bems:
            memories.append(self.tenant_bems[tenant_id])
        
        return memories
    
    def query(
        self,
        z: np.ndarray,
        tenant_id: str,
        domain_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Unified query across all scoped memories.
        
        Returns aggregated signals from global + domain + tenant memories.
        """
        memories = self.get_scoped_memory(tenant_id, domain_id)
        
        # Aggregate risk signals (max across memories — conservative)
        risk_scores = []
        all_failure_experiences = []
        for mem in memories:
            risk, failures = mem.risk_signal(z)
            risk_scores.append(risk)
            all_failure_experiences.extend(failures)
        
        # Aggregate success signals (max across memories — optimistic)
        success_scores = []
        all_success_experiences = []
        for mem in memories:
            success, successes = mem.success_signal(z)
            success_scores.append(success)
            all_success_experiences.extend(successes)
        
        # Aggregate coverage (average — overall coverage estimate)
        coverage_scores = [mem.coverage_signal(z) for mem in memories]
        
        return {
            "risk": max(risk_scores) if risk_scores else 0.0,
            "success": max(success_scores) if success_scores else 0.0,
            "coverage": np.mean(coverage_scores) if coverage_scores else 0.0,
            "failure_experiences": all_failure_experiences,
            "success_experiences": all_success_experiences
        }

    def check_conflict(
        self,
        z: np.ndarray,
        outcome: float,
        tenant_id: str,
        domain_id: str = "default"
    ) -> Optional[str]:
        """Check for conflicts across all scoped memories."""
        # Check global, domain, and tenant memories
        memories = self.get_scoped_memory(tenant_id, domain_id)
        for mem in memories:
            conflict = mem.check_conflict(z, outcome)
            if conflict:
                return f"[{mem.tenant_id} memory] {conflict}"
        return None
    
    def add_experience(
        self,
        embedding: np.ndarray,
        outcome: float,
        context: str,
        tenant_id: str,
        domain_id: str = "default",
        **kwargs
    ):
        """Add experience to appropriate memory based on scope."""
        if not isinstance(context, str):
            context = json.dumps(context, sort_keys=True, default=str)
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()

        # Always add to tenant memory
        if tenant_id not in self.tenant_bems:
            self.tenant_bems[tenant_id] = BidirectionalExperienceMemory(
                **self._memory_config, tenant_id=tenant_id
            )
        self.tenant_bems[tenant_id].add_experience(
            embedding=embedding,
            outcome=outcome,
            context_hash=context_hash,
            tenant_id=tenant_id,
            domain_id=domain_id,
            **kwargs
        )
        

    def vacuum_all(self, max_size: int = 10000, ttl_days: Optional[int] = 90):
        """Run memory hygiene on all managed memories."""
        self.global_bem.vacuum(max_size=max_size, ttl_days=ttl_days)
        for bem in self.domain_bems.values():
            bem.vacuum(max_size=max_size, ttl_days=ttl_days)
        for bem in self.tenant_bems.values():
            bem.vacuum(max_size=max_size, ttl_days=ttl_days)
