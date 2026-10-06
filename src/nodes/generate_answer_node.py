"""AgentCore Platform v1.0"""

# RET-C2-016 - GenerateAnswerNode
# Inner domain node 4: two things, kept separate.
#
#   1. The GROUNDED explanation, built only from the reranked knowledge-base
#      passages. When no passage clears the relevance floor, the node says the
#      knowledge base has insufficient coverage rather than inventing policy.
#   2. The ANOMALY FINDINGS, computed from the caller's validated pricing
#      observation. These are real arithmetic on real caller data - the rules
#      below decide severity, and the margin-at-risk aggregate is a genuine
#      figure, not a fixed baseline. When the caller sends no pricing
#      observation the findings are simply absent and the run degrades to the
#      knowledge-base explanation alone.
#
# The explanation never embeds the caller's question text: the answer is
# assembled from knowledge-base content and from typed, bounded caller
# numbers, so nothing free-form the caller sent can reach the report.
#
# Every number this node reads has crossed a serialization boundary, so it is
# re-parsed with the finite+bounded parser before it is compared. A NaN
# threshold compares False against everything, which would silently suppress
# every finding - exactly the decision this agent exists to make.
#
# Inner node - ANONYMOUS trust (the outer boundary enforces trust).
# Node contract: execute(self, state) -> dict. Returns ONLY the state keys it
# writes (partial-dict contract).

import logging
import math
import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.caller_contract import (
    AMOUNT_MAX,
    AMOUNT_MIN,
    MARGIN_FLOOR_MAX,
    MARGIN_FLOOR_MIN,
    UNITS_MAX,
    UNITS_MIN,
)
from src.schemas.state import finite_in_range, from_json, to_json

logger = logging.getLogger(__name__)

# Ordered severity vocabulary, strongest first.
_SEVERITY_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

# A reference ("was") price more than this multiple of the current selling
# price is treated as an inflated representation rather than a genuine former
# price.
_REFERENCE_INFLATION_RATIO = 2.0

_NO_GROUNDING_ANSWER = (
    "No pricing-policy reference passage met the relevance threshold for this question, "
    "so no grounded explanation can be given. This agent detects and explains retail "
    "pricing anomalies - cost inversion, margin bleed, channel price inversion, "
    "competitor undercut, reference-price representation, and flash-sale rule "
    "violations - grounded in a pricing-policy knowledge base. Restate the question "
    "with the anomaly type, channel or product category for a grounded explanation."
)


def _first_sentence(text: str) -> str:
    """Return the first sentence of a knowledge-base passage."""
    body = text.strip()
    if not body:
        return ""
    return re.split(r"(?<=[.。])\s", body, maxsplit=1)[0]


def _highest_severity(severities: List[str]) -> Optional[str]:
    """Return the strongest severity present in *severities*, or None."""
    present = set(severities)
    for severity in _SEVERITY_ORDER:
        if severity in present:
            return severity
    return None


def _read_observation(pricing: Any) -> Dict[str, Any]:
    """Re-parse the caller's pricing observation after its trip through State.

    Each field is re-validated against the same bounds the caller contract
    applied at ingest. A field that fails is dropped, not clamped, so the rule
    that needs it simply does not fire - the finding is withheld rather than
    computed from a value nobody validated.
    """
    if not isinstance(pricing, dict):
        return {}

    observation: Dict[str, Any] = {}

    sku = pricing.get("sku")
    if isinstance(sku, str) and sku:
        observation["sku"] = sku

    for field in ("selling_price", "unit_cost", "competitor_price", "reference_price"):
        value = finite_in_range(pricing.get(field), AMOUNT_MIN, AMOUNT_MAX)
        if value is not None:
            observation[field] = value

    floor = finite_in_range(pricing.get("margin_floor_pct"), MARGIN_FLOOR_MIN, MARGIN_FLOOR_MAX)
    if floor is not None:
        observation["margin_floor_pct"] = floor

    units = finite_in_range(pricing.get("units_sold"), UNITS_MIN, UNITS_MAX)
    if units is not None and units == int(units):
        observation["units_sold"] = int(units)

    return observation


def _required_price(unit_cost: float, margin_floor_pct: float) -> Optional[float]:
    """Selling price that just meets the category margin floor.

    required = cost / (1 - floor). The caller contract caps the floor below
    100%, so the divisor is bounded away from zero; the result is still checked
    for finiteness before it is used, because an unusable figure must withhold
    the aggregate rather than render a nonsense number.
    """
    divisor = 1.0 - (margin_floor_pct / 100.0)
    if divisor <= 0.0:
        return None
    required = unit_cost / divisor
    if not math.isfinite(required):
        return None
    return required


def _detect_anomalies(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the anomaly findings and the margin-at-risk aggregate.

    Returns the anomaly summary. `evaluated` is False when the caller supplied
    no usable pricing observation, which is how the report distinguishes "no
    anomalies found" from "nothing to evaluate".
    """
    selling_price = observation.get("selling_price")
    unit_cost = observation.get("unit_cost")
    competitor_price = observation.get("competitor_price")
    reference_price = observation.get("reference_price")
    margin_floor_pct = observation.get("margin_floor_pct")
    units_sold = observation.get("units_sold")

    findings: List[Dict[str, str]] = []

    inverted = False
    if selling_price is not None and unit_cost is not None:
        if selling_price < unit_cost:
            inverted = True
            findings.append(
                {
                    "rule": "cost_inversion",
                    "severity": "CRITICAL",
                    "detail": "The selling price is below the landed unit cost: every unit sold loses money.",
                }
            )
        elif margin_floor_pct is not None and selling_price > 0.0:
            realised_margin_pct = (selling_price - unit_cost) / selling_price * 100.0
            if math.isfinite(realised_margin_pct) and realised_margin_pct < margin_floor_pct:
                findings.append(
                    {
                        "rule": "margin_bleed",
                        "severity": "HIGH",
                        "detail": "The realised gross margin is below the category margin floor.",
                    }
                )

    if competitor_price is not None and selling_price is not None and competitor_price < selling_price:
        findings.append(
            {
                "rule": "competitor_undercut",
                "severity": "MEDIUM",
                "detail": "A monitored competitor price is below our selling price for this product.",
            }
        )

    if (
        reference_price is not None
        and selling_price is not None
        and reference_price > selling_price * _REFERENCE_INFLATION_RATIO
    ):
        findings.append(
            {
                "rule": "reference_price_representation",
                "severity": "HIGH",
                "detail": (
                    "The reference price is far above the current selling price, which reads as an "
                    "inflated former price rather than a genuine one."
                ),
            }
        )

    # -- Margin-at-risk aggregate ------------------------------------------
    # Rendered only when unit volume is known: a per-unit shortfall IS the raw
    # line item this report does not publish, so without volume there is no
    # aggregate to render.
    margin_at_risk: Optional[float] = None
    if selling_price is not None and unit_cost is not None and units_sold is not None:
        target = unit_cost if margin_floor_pct is None else _required_price(unit_cost, margin_floor_pct)
        if target is not None:
            shortfall = target - selling_price
            if shortfall > 0.0:
                exposure = shortfall * units_sold
                if math.isfinite(exposure):
                    margin_at_risk = exposure

    return {
        "sku": observation.get("sku"),
        "findings": findings,
        "highest_severity": _highest_severity([finding["severity"] for finding in findings]),
        "margin_at_risk": margin_at_risk,
        "cost_inversion": inverted,
        "evaluated": bool(observation),
    }


class GenerateAnswerNode(FunctionNode):
    """Build the grounded explanation and compute the anomaly findings.

    Inner node - ANONYMOUS trust (see the module comment).

    Input state keys:
        reranked_passages: str  - JSON list of {id, title, text, score}
        rag_query:         str  - JSON-serialised query object, carrying the
                                  validated pricing observation

    Output state keys (partial dict):
        grounded_answer: str  - plain-text grounded explanation
        citations:       str  - JSON list of cited passage ids
        anomaly_summary: str  - JSON anomaly findings + aggregate
        status:          str
        error_log:       list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        passages = from_json(state.get("reranked_passages"), [])
        if not isinstance(passages, list):
            passages = []
        passages = [passage for passage in passages if isinstance(passage, dict)]

        rag_query: Dict[str, Any] = from_json(state.get("rag_query"), {}) or {}
        observation = _read_observation(rag_query.get("pricing"))
        summary = _detect_anomalies(observation)

        # -- Grounded explanation ---------------------------------------------
        if not passages:
            logger.info("GenerateAnswerNode: no passage cleared the relevance floor")
            emit_trace_event(
                "generate_answer_no_grounding",
                {"finding_count": len(summary["findings"])},
                state,
            )
            summary["grounded"] = False
            summary["passage_count"] = 0
            return {
                "grounded_answer": _NO_GROUNDING_ANSWER,
                "citations": to_json([]),
                "anomaly_summary": to_json(summary),
                "status": AgentStatus.SUCCESS.value,
            }

        citation_ids: List[str] = [str(passage.get("id", "")) for passage in passages]

        lines: List[str] = [
            "The following pricing-policy rule references apply to this question "
            f"({len(passages)} grounded passage(s)):",
            "",
        ]
        for passage in passages:
            lines.append(
                f"- [{passage.get('id')}] {passage.get('title')}: {_first_sentence(str(passage.get('text', '')))}"
            )

        summary["grounded"] = True
        summary["passage_count"] = len(passages)
        summary["matched_rules"] = [str(passage.get("title", "")) for passage in passages]

        logger.info(
            "GenerateAnswerNode: passages=%d findings=%d highest_severity=%s",
            len(passages),
            len(summary["findings"]),
            summary["highest_severity"],
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "passage_count": len(passages),
                "finding_count": len(summary["findings"]),
                "highest_severity": summary["highest_severity"],
                "observation_evaluated": summary["evaluated"],
            },
            state,
        )

        return {
            "grounded_answer": "\n".join(lines),
            "citations": to_json(citation_ids),
            "anomaly_summary": to_json(summary),
            "status": AgentStatus.SUCCESS.value,
        }
