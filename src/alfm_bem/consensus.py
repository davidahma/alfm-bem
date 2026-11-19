"""Deterministic arbitration of memory scores and caller-supplied rules.

Query decisions carry a suggested question. The caller supplies any response
and the corresponding observed outcome to the wrapper.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from .constants import DEFAULT_RISK_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD


class Action(Enum):
    """Possible Consensus Engine outputs."""
    TRUST = "trust"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"
    QUERY = "query"


class QueryType(Enum):
    """Types of queries the system can generate."""
    CLARIFICATION = "clarification"      # Input is ambiguous
    PRECEDENT = "precedent"              # No similar past cases
    DOMAIN_KNOWLEDGE = "domain_knowledge"  # Ask for domain-specific guidance
    VALIDATION = "validation"            # Low confidence, need check


@dataclass
class HeuristicResult:
    """Result from heuristic rule checking."""
    violations: List[str]        # List of violated rules
    is_critical: bool           # Whether any violation is critical (hard veto)
    details: Dict[str, Any]     # Additional context


@dataclass
class ConsensusDecision:
    """Full decision output from Consensus Engine."""
    action: Action
    confidence: float                    # How confident in this decision
    risk_score: float                    # From BEM
    success_score: float                 # From BEM
    coverage_score: float                # Memory familiarity
    heuristic_result: HeuristicResult
    explanation: str                     # Human-readable explanation
    query_type: Optional[QueryType] = None  # If action is QUERY
    query_content: Optional[str] = None     # Specific query to ask
    query_id: Optional[str] = None          # Unique ID for tracking response


class SemanticAgent:
    """Recommend an action using strict coverage, risk and success thresholds."""

    def __init__(
        self,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
        success_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        coverage_threshold: float = 0.2
    ):
        self.risk_threshold = risk_threshold
        self.success_threshold = success_threshold
        self.coverage_threshold = coverage_threshold
    
    def evaluate(
        self,
        risk: float,
        success: float,
        coverage: float
    ) -> Tuple[str, float, str]:
        """
        Evaluate BEM signals and return recommendation.
        
        Returns:
            (recommendation, confidence, explanation)
        """
        # Low coverage -> suggest query
        if coverage < self.coverage_threshold:
            return (
                "query",
                1.0 - coverage,
                f"Low memory coverage ({coverage:.2f})"
            )
        
        # High risk -> suggest abstain/escalate
        if risk > self.risk_threshold:
            severity = "critical" if risk > 0.85 else "high"
            return (
                "abstain",
                risk,
                f"Similar to past failures ({severity} risk: {risk:.2f})"
            )
        
        # High success evidence -> suggest trust
        if success > self.success_threshold:
            return (
                "trust",
                success,
                f"Similar to past successes (confidence: {success:.2f})"
            )
        
        # Moderate signals -> uncertain
        return (
            "uncertain",
            0.5,
            f"Mixed signals (risk: {risk:.2f}, success: {success:.2f}, coverage: {coverage:.2f})"
        )


class HeuristicAgent:
    """
    Executes deterministic rules that must be satisfied.
    
    This agent says: "These hard constraints are violated/satisfied."
    """
    
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
    
    def add_rule(
        self,
        name: str,
        check_fn,  # Callable[[Dict], bool] - returns True if violated
        is_critical: bool = False,
        description: str = ""
    ):
        """Add a heuristic rule."""
        self.rules.append({
            "name": name,
            "check": check_fn,
            "is_critical": is_critical,
            "description": description
        })
    
    def evaluate(self, context: Dict[str, Any]) -> HeuristicResult:
        """
        Check all rules against the context.
        
        Returns HeuristicResult with violations and criticality.
        """
        violations = []
        is_critical = False
        details = {}
        
        for rule in self.rules:
            try:
                violated = rule["check"](context)
                if violated:
                    violations.append(rule["name"])
                    details[rule["name"]] = rule["description"]
                    if rule["is_critical"]:
                        is_critical = True
            except Exception as e:
                details[f"{rule['name']}_error"] = str(e)
                if rule["is_critical"]:
                    violations.append(rule["name"])
                    is_critical = True
        
        return HeuristicResult(
            violations=violations,
            is_critical=is_critical,
            details=details
        )


class ConsensusEngine:
    """Apply veto, low coverage, risk, advisory rules and success in that order.

    For otherwise undecided cases, query below coverage 0.5 and trust at or
    above it. The legacy weight arguments are retained for compatibility.
    """

    def __init__(
        self,
        semantic_weight: float = 0.6,
        heuristic_weight: float = 0.4,
        confidence_boost_threshold: float = 0.9
    ):
        self.semantic_agent = SemanticAgent()
        self.heuristic_agent = HeuristicAgent()
        self.semantic_weight = semantic_weight
        self.heuristic_weight = heuristic_weight
        self.confidence_boost_threshold = confidence_boost_threshold
    
    def add_heuristic_rule(self, *args, **kwargs):
        """Delegate to heuristic agent."""
        self.heuristic_agent.add_rule(*args, **kwargs)
    
    def decide(
        self,
        risk: float,
        success: float,
        coverage: float,
        context: Dict[str, Any]
    ) -> ConsensusDecision:
        """
        Make a decision based on all signals.
        
        Args:
            risk: BEM risk signal
            success: BEM success signal
            coverage: BEM coverage signal
            context: Context dict for heuristic rules
        
        Returns:
            ConsensusDecision with action and explanation
        """
        if not all(0.0 <= score <= 1.0 for score in (risk, success, coverage)):
            raise ValueError("risk, success and coverage must each lie in [0, 1]")

        heuristic = self.heuristic_agent.evaluate(context)
        recommendation, confidence, explanation = self.semantic_agent.evaluate(
            risk, success, coverage
        )
        query_type = None
        if heuristic.is_critical:
            action, confidence = Action.ESCALATE, 1.0
            explanation = f"Critical rule violation: {heuristic.violations}"
        elif recommendation == "query":
            action = Action.QUERY
            query_type = self._diagnose_query_type(risk, success, coverage, heuristic)
        elif recommendation == "abstain":
            action = Action.ESCALATE if risk > 0.85 else Action.ABSTAIN
        elif heuristic.violations:
            action, confidence = Action.ABSTAIN, 0.7
            explanation = f"Rule violations: {heuristic.violations}"
        elif recommendation == "trust":
            action = Action.TRUST
            if confidence > self.confidence_boost_threshold:
                confidence = min(1.0, confidence * 1.2)
        elif coverage < 0.5:
            action = Action.QUERY
            query_type = QueryType.VALIDATION
            explanation = "Uncertain - requesting validation"
        else:
            action = Action.TRUST
            explanation = "Low confidence trust (proceed with caution)"

        return ConsensusDecision(
            action=action,
            confidence=confidence,
            risk_score=risk,
            success_score=success,
            coverage_score=coverage,
            heuristic_result=heuristic,
            explanation=explanation,
            query_type=query_type,
            query_content=self._generate_query(query_type, context) if query_type else None,
        )

    def _diagnose_query_type(
        self,
        risk: float,
        success: float,
        coverage: float,
        heuristic_result: HeuristicResult
    ) -> QueryType:
        """Determine what type of query would help."""
        if risk > 0.5 and success < 0.3:
            return QueryType.PRECEDENT  # Know failures but no successes
        if coverage < 0.1:
            return QueryType.DOMAIN_KNOWLEDGE  # Completely novel
        if heuristic_result.violations:
            return QueryType.CLARIFICATION  # Rule issues
        return QueryType.VALIDATION  # General uncertainty
    
    def _generate_query(
        self,
        query_type: QueryType,
        context: Dict[str, Any]
    ) -> str:
        """Generate a specific query based on type."""
        context_preview = str(context)[:100] if context else "N/A"
        
        templates = {
            QueryType.CLARIFICATION: 
                f"Input is ambiguous. Please clarify: {context_preview}...",
            QueryType.PRECEDENT: 
                f"No successful precedent found. How should this be handled? Context: {context_preview}...",
            QueryType.DOMAIN_KNOWLEDGE: 
                f"Little support in memory. What domain guidance applies here? Context: {context_preview}...",
            QueryType.VALIDATION: 
                f"Low confidence. Please verify this approach: {context_preview}..."
        }
        
        return templates.get(query_type, "Please provide additional guidance.")
