"""Compose a projection, adapter chain, scoped experience memories and decision policy."""

import numpy as np
from typing import Dict, Any, Optional, Callable
import uuid
import io
import json
import os
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from .bem import BidirectionalExperienceMemory, BEMManager
from .projection import ContrastiveProjection
from .consensus import ConsensusEngine, ConsensusDecision, Action
from .adapters import BoundedAdapter, AdapterConfig, AdapterManager


@dataclass
class ALFMConfig:
    """Configuration for the complete ALFM-BEM system."""
    
    # Dimensions
    backbone_dim: int = 768       # Input from backbone model
    projection_dim: int = 256     # BEM embedding dimension
    adapter_hidden_dim: int = 64  # Adapter hidden layer
    
    # BEM parameters
    similarity_threshold: float = 0.5
    risk_sensitivity: float = 0.8
    
    # Adapter parameters  
    max_grad_norm: float = 1.0
    max_param_norm: float = 10.0
    adapter_lr: float = 1e-4
    
    # Experience loop
    replay_batch_size: int = 32
    replay_frequency: int = 100   # Accepted outcomes per tenant between replay attempts


@dataclass 
class InferenceResult:
    """Result of an ALFM-BEM inference."""
    decision: ConsensusDecision
    adapted_embedding: np.ndarray
    bem_query_result: Dict[str, Any]
    
    def should_trust(self) -> bool:
        return self.decision.action == Action.TRUST
    
    def should_query(self) -> bool:
        return self.decision.action == Action.QUERY
    
    def get_query(self) -> Optional[str]:
        return self.decision.query_content if self.should_query() else None


class ALFMBEM:
    """Make decisions from supplied backbone embeddings and learn from observed outcomes.

    infer returns an action, supporting records and an ID for a pending query.
    record_outcome accepts observed feedback directly. handle_query_response
    attaches feedback to the embedding that produced a specific query.
    """

    def __init__(self, config: ALFMConfig):
        if config.replay_frequency <= 0 or config.replay_batch_size <= 0:
            raise ValueError("Replay frequency and batch size must be positive")
        self.config = config
        
        # Initialize components
        self.projection = ContrastiveProjection(
            input_dim=config.backbone_dim,
            hidden_dim=512,
            output_dim=config.projection_dim
        )
        
        self.bem_manager = BEMManager(
            dim=config.projection_dim, similarity_threshold=config.similarity_threshold,
            risk_sensitivity=config.risk_sensitivity,
        )
        
        self.consensus = ConsensusEngine()
        
        adapter_config = AdapterConfig(
            input_dim=config.projection_dim,
            hidden_dim=config.adapter_hidden_dim,
            output_dim=config.projection_dim,
            max_grad_norm=config.max_grad_norm,
            max_param_norm=config.max_param_norm,
            learning_rate=config.adapter_lr
        )
        self.adapter_manager = AdapterManager(adapter_config)
        
        # Experience loop tracking
        self._inference_count = 0
        self._outcome_counts: Dict[str, int] = {}
        self._active_queries: Dict[str, Dict[str, Any]] = {}

    
    def infer(
        self,
        backbone_embedding: np.ndarray,
        context: Dict[str, Any],
        tenant_id: str,
        domain_id: str = "default"
    ) -> InferenceResult:
        """
        Run inference through the complete ALFM-BEM pipeline.
        
        Args:
            backbone_embedding: Hidden state from frozen backbone
            context: Context dict for heuristic rules
            tenant_id: Tenant identifier for isolation
            domain_id: Domain identifier
        
        Returns:
            InferenceResult with decision and all signals
        """
        # 1. Project the backbone vector into memory space
        z = self.projection(backbone_embedding)
        
        # 2. Apply adapters (tenant-specific corrections)
        z_adapted = self.adapter_manager.forward(z, tenant_id, domain_id)
        
        # 3. Query BEM for risk/success/coverage signals
        bem_result = self.bem_manager.query(z_adapted, tenant_id, domain_id)
        
        # 4. Make decision via Consensus Engine
        decision = self.consensus.decide(
            risk=bem_result["risk"],
            success=bem_result["success"],
            coverage=bem_result["coverage"],
            context=context
        )
        
        # If Query, register active query
        if decision.action == Action.QUERY:
            query_id = str(uuid.uuid4())
            decision.query_id = query_id
            self._active_queries[query_id] = {
                "embedding": z_adapted.copy(),
                "context": deepcopy(context),
                "tenant_id": tenant_id,
                "domain_id": domain_id,
            }
        
        self._inference_count += 1
        
        return InferenceResult(
            decision=decision,
            adapted_embedding=z_adapted,
            bem_query_result=bem_result
        )
    
    def record_outcome(
        self,
        backbone_embedding: np.ndarray,
        context: Any,
        outcome: float,
        tenant_id: str,
        domain_id: str = "default",
        reasoning_trace: Optional[str] = None,
        correction: Optional[str] = None
    ):
        """
        Record the outcome of an inference, closing the experience loop.
        
        This:
        1. Adds the experience to BEM
        2. Optionally triggers adapter training via experience replay
        
        Args:
            backbone_embedding: Original backbone embedding
            context: Context object (used for deduplication/hashing)
            outcome: Outcome score in [-1, 1]
            tenant_id: Tenant identifier
            domain_id: Domain identifier
            reasoning_trace: Optional reasoning that led to outcome
            correction: Optional human correction for failures
        """
        # Project embedding
        z = self.projection(backbone_embedding)
        z_adapted = self.adapter_manager.forward(z, tenant_id, domain_id)
        
        self._record_embedding(z_adapted, context, outcome, tenant_id, domain_id,
                               reasoning_trace, correction)

    def _record_embedding(self, z_adapted, context, outcome, tenant_id, domain_id,
                          reasoning_trace=None, correction=None):
        # Reject sign conflicts with well-confirmed nearby experiences.
        conflict_msg = self.bem_manager.check_conflict(
            z_adapted, outcome, tenant_id, domain_id
        )
        if conflict_msg:
            raise ValueError(f"Feedback Rejected: {conflict_msg}")

        # Add to tenant BEM
        self.bem_manager.add_experience(
            embedding=z_adapted,
            outcome=outcome,
            context=context,
            tenant_id=tenant_id,
            domain_id=domain_id,
            reasoning_trace=reasoning_trace,
            correction=correction
        )
        
        # Experience replay: periodically train adapters
        count = self._outcome_counts.get(tenant_id, 0) + 1
        self._outcome_counts[tenant_id] = count
        if count % self.config.replay_frequency == 0:
            self._run_experience_replay(tenant_id, domain_id)
            
        # Vacuum periodically to prevent unbounded growth
        if count % 1000 == 0:
            self.bem_manager.vacuum_all(max_size=10000, ttl_days=90)

    def _run_experience_replay(self, tenant_id: str, domain_id: str):
        """Train adapter using accumulated experiences."""
        if tenant_id not in self.bem_manager.tenant_bems:
            return
        
        bem = self.bem_manager.tenant_bems[tenant_id]
        if len(bem.experiences) < self.config.replay_batch_size:
            return
        
        # Train tenant adapter
        loss = self.adapter_manager.train_tenant_adapter(
            tenant_id=tenant_id,
            bem=bem,
            projection_fn=self.projection,
            batch_size=self.config.replay_batch_size
        )
        
        return loss
    
    def add_heuristic_rule(
        self,
        name: str,
        check_fn: Callable[[Dict], bool],
        is_critical: bool = False,
        description: str = ""
    ):
        """Add a heuristic rule to the Consensus Engine."""
        self.consensus.add_heuristic_rule(name, check_fn, is_critical, description)
    
    def get_statistics(self, tenant_id: str = "global") -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "inference_count": self._inference_count,
            "projection": {
                "input_dim": self.config.backbone_dim,
                "output_dim": self.config.projection_dim
            }
        }
        
        # BEM stats
        if tenant_id in self.bem_manager.tenant_bems:
            stats["bem"] = self.bem_manager.tenant_bems[tenant_id].get_statistics()
        else:
            stats["bem"] = {"total": 0}
        
        # Adapter stats
        if tenant_id in self.adapter_manager.tenant_adapters:
            stats["adapter"] = self.adapter_manager.tenant_adapters[tenant_id].get_statistics()
        else:
            stats["adapter"] = {"update_count": 0}
        
        return stats
    
    def handle_query_response(
        self,
        query_id: str,
        response: str,
        outcome: float
    ) -> bool:
        """
        Process human response to Query action.
        
        Args:
            query_id: ID returned from infer()
            response: Text response/clarification/confirmation
            outcome: Success (1.0) or Failure (-1.0) signal derived from response
            
        Returns:
            True if successfully processed
        """
        if query_id not in self._active_queries:
            return False
            
        # Retrieve context
        query_ctx = self._active_queries[query_id]
        
        # Enrich reasoning with the response
        reasoning = f"Query Response: {response}"
        
        # Record outcome
        self._record_embedding(
            z_adapted=query_ctx["embedding"],
            context=query_ctx["context"],
            outcome=outcome,
            tenant_id=query_ctx["tenant_id"],
            domain_id=query_ctx["domain_id"],
            reasoning_trace=reasoning
        )
        del self._active_queries[query_id]
        return True

    def save(self, path: str):
        """Atomically checkpoint model weights, scoped memories and replay counters.

        Pending queries are session-local. Supply application rules again on load.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "config": asdict(self.config),
            "inferences": self._inference_count,
            "outcomes": self._outcome_counts,
            "rules": [(r["name"], r["is_critical"]) for r in self.consensus.heuristic_agent.rules],
            "semantic": vars(self.consensus.semantic_agent),
            "confidence_boost_threshold": self.consensus.confidence_boost_threshold,
        }
        models = {
            "global_adapter": {"global": self.adapter_manager.global_adapter},
            "domain_adapters": self.adapter_manager.domain_adapters,
            "tenant_adapters": self.adapter_manager.tenant_adapters,
        }
        memories = {
            "global_bem": {"global": self.bem_manager.global_bem},
            "domain_bems": self.bem_manager.domain_bems,
            "tenant_bems": self.bem_manager.tenant_bems,
        }
        with NamedTemporaryFile(dir=directory, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with ZipFile(temporary, "w") as archive:
                projection = io.BytesIO()
                self.projection.save(projection)
                archive.writestr("projection.npz", projection.getvalue())
                for group, members in models.items():
                    manifest[group] = list(members)
                    for index, adapter in enumerate(members.values()):
                        weights = io.BytesIO()
                        adapter.save(weights)
                        archive.writestr(f"{group}/{index}.pt", weights.getvalue())
                for group, members in memories.items():
                    manifest[group] = list(members)
                    for index, memory in enumerate(members.values()):
                        archive.writestr(f"{group}/{index}.json", json.dumps(memory._as_dict()))
                archive.writestr("manifest.json", json.dumps(manifest))
            os.replace(temporary, directory / "system.zip")
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str, config: Optional[ALFMConfig] = None,
             consensus: Optional[ConsensusEngine] = None) -> 'ALFMBEM':
        """Restore a checkpoint. Reattach saved rule names and priorities via consensus."""
        directory = Path(path)
        if not (directory / "system.zip").exists():
            if config is None:
                raise ValueError("Legacy system directories require the original ALFMConfig")
            system = cls(config)
            system.projection = ContrastiveProjection.load(directory / "projection.npz")
            system.bem_manager.global_bem = BidirectionalExperienceMemory.load(directory / "bem_global.json")
            for file in directory.glob("bem_*.json"):
                if file.name != "bem_global.json":
                    memory = BidirectionalExperienceMemory.load(file)
                    system.bem_manager.tenant_bems[memory.tenant_id] = memory
            for file in directory.glob("adapter_*.npz"):
                adapter = BoundedAdapter.load(file, system.adapter_manager.config)
                system.adapter_manager.tenant_adapters[adapter.tenant_id] = adapter
            warnings.warn("Legacy system checkpoint omitted global adapter weights; "
                          "the global adapter has been reinitialized", stacklevel=2)
            if consensus is not None:
                system.consensus = consensus
            return system

        with ZipFile(directory / "system.zip") as archive:
            metadata = json.loads(archive.read("manifest.json"))
            if metadata["version"] != 1:
                raise ValueError("Unknown system checkpoint version")
            saved_config = ALFMConfig(**metadata["config"])
            if config is not None and config != saved_config:
                raise ValueError("Supplied ALFMConfig differs from the checkpoint")
            system = cls(saved_config)
            if consensus is not None:
                system.consensus = consensus
            rules = [[r["name"], r["is_critical"]] for r in system.consensus.heuristic_agent.rules]
            if rules != metadata["rules"]:
                raise ValueError("Supply the checkpoint's heuristic rules in consensus")
            system.consensus.confidence_boost_threshold = metadata["confidence_boost_threshold"]
            for key, value in metadata["semantic"].items():
                setattr(system.consensus.semantic_agent, key, value)
            system.projection = ContrastiveProjection.load(io.BytesIO(archive.read("projection.npz")))
            system._inference_count = metadata["inferences"]
            system._outcome_counts = metadata["outcomes"]

            for group in ("global_adapter", "domain_adapters", "tenant_adapters"):
                adapters = {}
                for index, key in enumerate(metadata[group]):
                    weights = io.BytesIO(archive.read(f"{group}/{index}.pt"))
                    adapters[key] = BoundedAdapter.load(weights, system.adapter_manager.config)
                setattr(system.adapter_manager, group, adapters["global"] if group == "global_adapter" else adapters)
            for group in ("global_bem", "domain_bems", "tenant_bems"):
                memories = {}
                for index, key in enumerate(metadata[group]):
                    data = json.loads(archive.read(f"{group}/{index}.json"))
                    memories[key] = BidirectionalExperienceMemory._from_dict(data)
                setattr(system.bem_manager, group, memories["global"] if group == "global_bem" else memories)
        return system


# Convenience function for quick setup
def create_alfm_bem(
    backbone_dim: int = 768,
    projection_dim: int = 256
) -> ALFMBEM:
    """Create an ALFM-BEM system with default configuration."""
    config = ALFMConfig(
        backbone_dim=backbone_dim,
        projection_dim=projection_dim
    )
    return ALFMBEM(config)
