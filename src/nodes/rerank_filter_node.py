"""AgentCore Platform v1.0"""

# RET-C2-016 - RerankFilterNode
# Inner domain node 3: rerank the retrieved passages, apply the relevance floor
# and trim to the effective retrieval depth.
#
# This build reranks deterministically - the retrieval score plus a small,
# bounded title-signal boost - then filters by score_threshold and trims to
# top_k. A real deployment swaps in a cross-encoder reranker behind the same
# filter contract.
#
# Inner node - ANONYMOUS trust (the outer boundary enforces trust).
# Node contract: execute(self, state) -> dict. Returns ONLY the state keys it
# writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.caller_contract import TOP_K_MAX, TOP_K_MIN
from src.schemas.state import finite_in_range, from_json, to_json

logger = logging.getLogger(__name__)

# Effective values when neither the caller nor config/config.yaml supplies one
# (as on a direct unit-node call).
_DEFAULT_TOP_K = 5
_DEFAULT_SCORE_THRESHOLD = 0.35

# Bounded title-signal boost: a longer, more specific rule title sorts first
# when two passages share the same raw retrieval score.
_MAX_TITLE_BOOST = 0.05
_TITLE_BOOST_DIVISOR = 2000.0


def _resolve_top_k(state: AgentState) -> int:
    """Effective retrieval depth: caller override, then config, then default.

    Every source goes through the finite+bounded parser. A value that fails is
    ignored, never clamped: clamping a non-finite value would silently install
    a bound rather than reject it.
    """
    rag_query: Dict[str, Any] = from_json(state.get("rag_query"), {}) or {}
    caller_top_k = finite_in_range(rag_query.get("top_k"), TOP_K_MIN, TOP_K_MAX)
    if caller_top_k is not None and caller_top_k == int(caller_top_k):
        return int(caller_top_k)

    configured = finite_in_range(state.get("retrieval_top_k"), TOP_K_MIN, TOP_K_MAX)
    if configured is not None and configured == int(configured):
        return int(configured)

    return _DEFAULT_TOP_K


def _resolve_score_threshold(state: AgentState) -> float:
    """Effective relevance floor from config, else the module default.

    The floor is operator-only - there is no caller override - but the value
    still crosses a serialization boundary, so it is re-validated here. A
    non-finite floor is the dangerous case: every comparison against NaN is
    False, so it would drop every candidate passage and the agent would answer
    "insufficient coverage" for every question.
    """
    configured = finite_in_range(state.get("retrieval_score_threshold"), 0.0, 1.0)
    if configured is None:
        return _DEFAULT_SCORE_THRESHOLD
    return configured


def _rerank_score(passage: Dict[str, Any]) -> float:
    """Deterministic rerank score: retrieval score plus a bounded title boost.

    The incoming score has been through State, so it is re-parsed with the
    finite+bounded parser; a passage carrying an unusable score is treated as
    score 0.0 and drops out at the relevance floor rather than being carried
    through by a NaN comparison. Pure function.
    """
    base = finite_in_range(passage.get("score"), 0.0, 1.0)
    if base is None:
        return 0.0
    title_length = len(str(passage.get("title", "")))
    boost = min(_MAX_TITLE_BOOST, title_length / _TITLE_BOOST_DIVISOR)
    return round(min(1.0, base + boost), 4)


class RerankFilterNode(FunctionNode):
    """Rerank, apply the relevance floor, and trim to the retrieval depth.

    Inner node - ANONYMOUS trust (see the module comment).

    Input state keys:
        retrieved_passages:        str   - JSON list of {id, title, text, score}
        retrieval_top_k:           int   - optional, seeded from config
        retrieval_score_threshold: float - optional, seeded from config

    Output state keys (partial dict):
        reranked_passages: str  - JSON list (<= top_k, score >= threshold)
        status:            str
        error_log:         list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        retrieved = from_json(state.get("retrieved_passages"), [])
        if not isinstance(retrieved, list):
            retrieved = []

        top_k = _resolve_top_k(state)
        score_threshold = _resolve_score_threshold(state)

        # Rerank: recompute the score and re-sort, building NEW list objects.
        reranked: List[Dict[str, Any]] = []
        for passage in retrieved:
            if not isinstance(passage, dict):
                continue
            reranked.append(
                {
                    "id": passage.get("id"),
                    "title": passage.get("title"),
                    "text": passage.get("text"),
                    "score": _rerank_score(passage),
                }
            )

        reranked.sort(key=lambda passage: passage["score"], reverse=True)

        # Apply the relevance floor, then trim to the retrieval depth.
        filtered = [passage for passage in reranked if passage["score"] >= score_threshold][:top_k]

        logger.info(
            "RerankFilterNode: in=%d reranked=%d kept=%d (threshold=%.2f, top_k=%d)",
            len(retrieved),
            len(reranked),
            len(filtered),
            score_threshold,
            top_k,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "in_count": len(retrieved),
                "kept_count": len(filtered),
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {
            "reranked_passages": to_json(filtered),
            "status": AgentStatus.SUCCESS.value,
        }
