"""AgentCore Platform v1.0"""

# RET-C2-016 - Price & Margin Anomaly Detection Agent (Cat 2 domain pipeline)
#
# Two-layer nested graph: an outer backbone (AgentBaseGraph) and an inner
# domain workflow (BaseGraph - retrieve -> rerank/filter -> generate -> format).
# The fields below cover both layers.
#
# State must be a flat TypedDict, never a Pydantic model: checkpoints are
# serialized with msgpack, and model objects are silently corrupted by it.
# Extend AgentState with agent-specific fields only, and never put credentials
# or secrets in State.
#
# Every dict/list-valued field is stored as a JSON-serialized Optional[str].
# Use to_json() / from_json() at every producing and consuming node - one
# contract end-to-end. Typing such a field as a bare dict/list breaks msgpack
# serialization.

import json
import math
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse a number read back out of State: FINITE and within [lo, hi], else None.

    State survives a serialization round-trip and is seeded from caller data, so
    a value read back out is untrusted the same way the original was. The rule
    is identical to the one the caller contract applies at ingest, so the two
    boundaries cannot disagree: a number must already BE a number (a numeric
    string means the type was lost), and it must be finite. Every comparison
    against NaN is False, so a non-finite value slipped into State would
    silently disable the check it feeds. Clamping is deliberately NOT used
    here: max(lo, min(hi, nan)) returns hi, which installs the strictest bound
    instead of rejecting.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


class State(AgentState):
    """Flat TypedDict for RET-C2-016.

    All shared fields (user_input, input_context, status, session_id,
    node_history, error_log, hitl_*, ...) are inherited from AgentState.

    dict/list fields use JSON-serialized Optional[str]. Every domain field is
    annotated Optional, which is what makes it default to None when the graph
    builds its channels - tests/unit/test_graph_and_config.py pins that, so the
    convention is checked rather than assumed.

    formatted_output is NOT re-declared here - it is inherited from AgentState
    (re-declaring it with a bare type breaks the state contract).
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode (pre_process backbone slot)
    # ------------------------------------------------------------------

    # Normalised pricing question (control characters stripped, whitespace
    # collapsed, length-capped). Plain text - the caller contract travels
    # separately, on the context channel.
    validated_input: Optional[str]

    # JSON-serialised request metadata: {"source": str, "channel": str}.
    enriched_context: Optional[str]

    # ------------------------------------------------------------------
    # Runtime settings - seeded by DomainWorkflowGraph._extra_initial_state()
    # from config/config.yaml (forwarded by the outer GraphNode). Scalars, so
    # they are msgpack-safe. Read by RetrieveNode (retrieval_top_k) and
    # RerankFilterNode (both), each falling back to its module default when
    # unseeded - as on a direct unit-node call.
    # ------------------------------------------------------------------

    # Retrieval fan-out size (retrieval.top_k).
    retrieval_top_k: Optional[int]

    # Rerank/filter relevance floor (retrieval.score_threshold).
    retrieval_score_threshold: Optional[float]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # JSON-serialised normalised query object.
    # Shape: {"query": str, "terms": list[str], "category": str|None,
    #         "channel": str|None, "top_k": int|None, "pricing": dict|None}
    # Produced by InputValidateNode, which re-validates the caller contract.
    rag_query: Optional[str]

    # JSON-serialised list of candidate passages from the pricing-policy
    # knowledge base. Each: {"id": str, "title": str, "text": str,
    # "score": float}. Produced by RetrieveNode (pre-rerank, over-fetched).
    retrieved_passages: Optional[str]

    # JSON-serialised list of reranked + threshold-filtered passages.
    # Same element shape, sorted by rerank score. Produced by RerankFilterNode.
    reranked_passages: Optional[str]

    # Grounded natural-language explanation (plain text). Built only from the
    # reranked passages - no ungrounded content. Produced by GenerateAnswerNode.
    grounded_answer: Optional[str]

    # JSON-serialised list of cited passage ids. Produced by GenerateAnswerNode.
    citations: Optional[str]

    # JSON-serialised anomaly detection result computed from the caller's
    # validated pricing observation.
    # Shape: {"sku": str|None, "findings": [{"rule": str, "severity": str,
    #         "detail": str}], "highest_severity": str|None,
    #         "margin_at_risk": float|None, "evaluated": bool}
    # Produced by GenerateAnswerNode.
    anomaly_summary: Optional[str]

    # Final rendered report (plain text) with citations and the anomaly
    # findings. Assembled by the inner OutputFormatNode.
    anomaly_report: Optional[str]

    # ------------------------------------------------------------------
    # Outer layer - set by PostProcessNode (post_process backbone slot)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller: the report after the output gate.
    # formatted_output (inherited from AgentState) carries the same content.
    result: Optional[str]

    # ------------------------------------------------------------------
    # Tracing - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    error_code: Optional[str]
    # node_history inherited from AgentState
