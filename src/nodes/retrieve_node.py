"""AgentCore Platform v1.0"""

# RET-C2-016 - RetrieveNode
# Inner domain node 2: retrieve candidate passages from the pricing-policy
# knowledge base.
#
# This build scores a bundled reference corpus of pricing / margin anomaly
# rules by term overlap - deterministic and dependency-free, so the pipeline
# runs and is testable end to end out of the box. A real deployment replaces
# the corpus with its own pricing-policy store behind the same node contract.
#
# Inner node - ANONYMOUS trust (the outer boundary enforces trust).
# Node contract: execute(self, state) -> dict. Returns ONLY the state keys it
# writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.caller_contract import TOP_K_MAX, TOP_K_MIN
from src.schemas.state import finite_in_range, from_json, to_json

logger = logging.getLogger(__name__)

# Effective retrieval depth when neither the caller nor config/config.yaml
# supplies one (as on a direct unit-node call).
_DEFAULT_TOP_K = 5
# Retrieve top_k * factor candidates; RerankFilterNode trims back to top_k.
_OVERFETCH_FACTOR = 2

# Bounded SKU-match boost, applied when the caller's product code appears in a
# passage. Bounded so it can never carry an irrelevant passage over the
# relevance floor on its own.
_SKU_MATCH_BOOST = 0.25


def _resolve_top_k(state: AgentState) -> int:
    """Effective retrieval depth: caller override, then config, then default.

    Both sources are untrusted by the time they are read back out of State, so
    both go through the finite+bounded parser. A value that fails is not
    clamped into range - it is ignored, and the next source applies.
    """
    rag_query: Dict[str, Any] = from_json(state.get("rag_query"), {}) or {}
    caller_top_k = finite_in_range(rag_query.get("top_k"), TOP_K_MIN, TOP_K_MAX)
    if caller_top_k is not None and caller_top_k == int(caller_top_k):
        return int(caller_top_k)

    configured = finite_in_range(state.get("retrieval_top_k"), TOP_K_MIN, TOP_K_MAX)
    if configured is not None and configured == int(configured):
        return int(configured)

    return _DEFAULT_TOP_K


# Bundled pricing-policy knowledge base (the grounding corpus). READ-ONLY -
# never mutated inside execute(), because mutating a module-global default
# leaks across invocations. Each passage is a self-contained anomaly-rule
# reference; `category` is the value a caller may filter on.
_REFERENCE_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "KB-PRC-001",
        "category": "margin",
        "title": "Cost inversion (selling price below cost)",
        "text": (
            "A cost inversion occurs when the selling price for a product drops below its "
            "landed unit cost, producing a negative gross margin. Severity is CRITICAL: "
            "every unit sold loses money. Common causes are a mis-keyed price update, a "
            "stale cost feed, or an over-aggressive markdown."
        ),
        "tags": ["cost", "inversion", "margin", "negative", "critical", "price", "sku"],
    },
    {
        "id": "KB-PRC-002",
        "category": "margin",
        "title": "Margin bleed below target threshold",
        "text": (
            "Margin bleed is a gross margin that falls below the category target "
            "threshold without going negative. Severity is HIGH when the gap exceeds "
            "the configured tolerance. It is detected by comparing realised margin to "
            "the category margin floor in the pricing policy."
        ),
        "tags": ["margin", "bleed", "threshold", "target", "category", "high", "gross"],
    },
    {
        "id": "KB-PRC-003",
        "category": "channel",
        "title": "Channel price inversion (e-commerce below in-store)",
        "text": (
            "Channel price inversion is when the e-commerce price is lower than the "
            "in-store price for the same product, violating omnichannel price-parity "
            "policy. Severity is HIGH for key-value items. Omnichannel operators watch "
            "physical store, own e-commerce and marketplace channels for inversion."
        ),
        "tags": ["channel", "inversion", "omnichannel", "ecommerce", "instore", "parity", "sku"],
    },
    {
        "id": "KB-PRC-004",
        "category": "competitor",
        "title": "Competitor undercut on key products",
        "text": (
            "A competitor undercut flags when a monitored competitor price on a key "
            "product falls below our price. Severity is MEDIUM and escalates for "
            "traffic-driving items. This rule requires an optional competitor price "
            "feed and is advisory, not a hard error."
        ),
        "tags": ["competitor", "undercut", "keyitem", "price", "medium", "feed", "sku"],
    },
    {
        "id": "KB-PRC-005",
        "category": "compliance",
        "title": "Reference-price representation compliance",
        "text": (
            "Consumer-protection law treats an inflated original or reference price used "
            "to exaggerate a discount as a misleading representation. The agent flags a "
            "reference price that was never a genuine selling price. Severity is HIGH - "
            "this is a legal compliance exposure, not only a margin issue."
        ),
        "tags": ["reference", "compliance", "discount", "legal", "original", "high"],
    },
    {
        "id": "KB-PRC-006",
        "category": "promotion",
        "title": "Flash-sale / markdown rule violation",
        "text": (
            "A flash-sale rule violation is a markdown that breaches the promotion "
            "policy - for example a discount deeper than the approved maximum, or a "
            "promotional price left active past its end date. Severity is MEDIUM. "
            "Detection compares the active price against the approved promotion window "
            "and depth."
        ),
        "tags": ["flash", "sale", "markdown", "promotion", "discount", "medium", "rule"],
    },
    {
        "id": "KB-PRC-007",
        "category": "reporting",
        "title": "Severity scoring and anomaly record shape",
        "text": (
            "Each anomaly is emitted as a structured record: product code, rule "
            "violated, and a severity of CRITICAL, HIGH, MEDIUM or LOW. A batch summary "
            "aggregates counts by rule type and severity so pricing operations can "
            "triage same-day instead of waiting for a weekly audit."
        ),
        "tags": ["severity", "record", "anomaly", "batch", "summary", "critical", "score"],
    },
]


def _searchable_text(passage: Dict[str, Any]) -> str:
    """Lower-cased searchable blob for a passage (title + text + tags)."""
    tags = " ".join(passage.get("tags", []))
    return f"{passage.get('title', '')} {passage.get('text', '')} {tags}".lower()


def _score_passage(terms: List[str], blob: str, sku: Optional[str]) -> float:
    """Deterministic term-overlap score, normalised to [0.0, 1.0].

    score = (query terms found in the passage) / (query terms), plus a bounded
    product-code match boost. Pure function - no shared state.
    """
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in blob)
    score = hits / len(terms)
    if sku and sku.lower() in blob:
        score = min(1.0, score + _SKU_MATCH_BOOST)
    return round(score, 4)


class RetrieveNode(FunctionNode):
    """Retrieve candidate pricing-policy passages for the question.

    Inner node - ANONYMOUS trust (see the module comment).

    Input state keys:
        rag_query:         str - JSON-serialised query object
        retrieval_top_k:   int - optional, seeded from config/config.yaml

    Output state keys (partial dict):
        retrieved_passages: str  - JSON list of {id, title, text, score}
        status:             str
        error_log:          list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        rag_query: Dict[str, Any] = from_json(state.get("rag_query"), {}) or {}

        if not rag_query or not rag_query.get("query"):
            logger.error("RetrieveNode: rag_query is missing from state")
            emit_trace_event("retrieve_failed", {"reason": "missing_rag_query"}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: rag_query is missing from state"],
            }

        terms: List[str] = [str(term) for term in rag_query.get("terms", []) if isinstance(term, str)]
        category = rag_query.get("category")
        pricing = rag_query.get("pricing") or {}
        sku = pricing.get("sku") if isinstance(pricing, dict) else None

        top_k = _resolve_top_k(state)

        # Build a NEW candidate list - never mutate the reference corpus.
        candidates: List[Dict[str, Any]] = []
        for passage in _REFERENCE_CORPUS:
            if category and passage.get("category") != category:
                continue
            score = _score_passage(terms, _searchable_text(passage), sku)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "id": passage["id"],
                    "title": passage["title"],
                    "text": passage["text"],
                    "score": score,
                }
            )

        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        retrieved = candidates[: top_k * _OVERFETCH_FACTOR]

        logger.info(
            "RetrieveNode: terms=%d candidates=%d retrieved=%d top_k=%d category=%s",
            len(terms),
            len(candidates),
            len(retrieved),
            top_k,
            category,
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "term_count": len(terms),
                "retrieved_count": len(retrieved),
                "top_k": top_k,
                "category_filtered": category is not None,
            },
            state,
        )

        return {
            "retrieved_passages": to_json(retrieved),
            "status": AgentStatus.SUCCESS.value,
        }
