"""AgentCore Platform v1.0"""

# RET-C2-016 - OutputFormatNode (inner domain node 5, last in DomainWorkflowGraph)
# Renders the external report from the grounded explanation, the citations and
# the anomaly findings.
#
# The report's output schema, stated here and enforced independently by the
# output gate in src/nodes/post_process_node.py:
#
#   - Monetary figures are AGGREGATES ONLY, rounded to the nearest 1,000.
#     Per-unit selling prices, unit costs, competitor prices and reference
#     prices are never rendered - they are the caller's commercially sensitive
#     line items, and the report exists to communicate exposure and severity,
#     not to echo the input back.
#   - Product codes are rendered with the fixed SKU_ prefix. The prefix is not
#     decoration: the output gate treats a bare digit run as a monetary figure,
#     and the prefix keeps a numeric product code recognisable as an identifier
#     so it is passed through byte-for-byte instead of snapped onto the
#     rounding grid.
#
# The schema note is printed in the report itself, so a reader knows the
# figures are rounded rather than assuming they are exact.
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
from src.schemas.caller_contract import AMOUNT_MAX, UNITS_MAX
from src.schemas.state import finite_in_range, from_json

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72

# Reporting currency for the aggregate. The bundled build reports in JPY; a
# deployment reporting in another currency changes this one constant, and the
# output gate recognises the marker either way.
_REPORTING_CURRENCY = "JPY"

# Approved external precision for monetary figures (must match
# _EXTERNAL_ROUND_UNIT in src/nodes/post_process_node.py - the report RENDERS
# on this grid, the gate ENFORCES it).
_EXTERNAL_ROUND_UNIT = 1000

# Fixed prefix for a rendered product code (see the module comment).
_SKU_PREFIX = "SKU_"

_SCHEMA_NOTE = (
    f"Monetary figures in this report are aggregates rounded to the nearest "
    f"{_EXTERNAL_ROUND_UNIT:,d}. Per-unit prices, costs and margins are not rendered."
)


# Widest value the aggregate can legitimately take: the largest per-unit
# shortfall the caller contract admits, times the largest unit volume.
_MAX_AGGREGATE = AMOUNT_MAX * UNITS_MAX


def _on_grid(amount: Any) -> Optional[int]:
    """Round a monetary aggregate onto the approved external grid, or None.

    The aggregate is read back out of State, so it is untrusted the same way the
    caller's numbers were: it is re-parsed as a finite, bounded number before it
    is rendered. A value that fails is withheld rather than rendered - int() on
    a non-finite float raises, and a report that crashes on render is not a
    better outcome than a report that omits a figure it cannot vouch for.
    """
    parsed = finite_in_range(amount, 0.0, _MAX_AGGREGATE)
    if parsed is None:
        return None
    return int(round(parsed / _EXTERNAL_ROUND_UNIT) * _EXTERNAL_ROUND_UNIT)


def _render_findings(summary: Dict[str, Any]) -> List[str]:
    """Render the anomaly findings block."""
    lines: List[str] = [
        "",
        _SUBSEP,
        "ANOMALY FINDINGS",
        _SUBSEP,
    ]

    if not summary.get("evaluated"):
        lines.append("  No pricing observation was supplied, so no anomaly rules were evaluated.")
        return lines

    sku = summary.get("sku")
    if isinstance(sku, str) and sku:
        lines.append(f"  Product code:      {_SKU_PREFIX}{sku}")

    findings = summary.get("findings") or []
    if not findings:
        lines.append("  No pricing anomaly was detected in the supplied observation.")
        return lines

    lines.append(f"  Highest severity:  {summary.get('highest_severity')}")
    lines.append(f"  Rules triggered:   {len(findings)}")
    lines.append("")
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        lines.append(f"  [{finding.get('severity')}] {finding.get('rule')}")
        lines.append(f"      {finding.get('detail')}")

    exposure = _on_grid(summary.get("margin_at_risk"))
    if exposure is not None:
        lines.append("")
        lines.append(f"  Estimated margin at risk: {_REPORTING_CURRENCY} {exposure:,d}")
    return lines


class OutputFormatNode(FunctionNode):
    """Render the external price & margin anomaly report.

    Inner node - ANONYMOUS trust (see the module comment).

    Input state keys:
        grounded_answer: str  - plain-text grounded explanation
        citations:       str  - JSON list of cited passage ids
        anomaly_summary: str  - JSON anomaly findings + aggregate

    Output state keys (partial dict):
        anomaly_report: str
        result:         str  (same content - backbone convention)
        status:         str
        error_log:      list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        answer = state.get("grounded_answer") or ""
        if not isinstance(answer, str) or not answer.strip():
            logger.error("OutputFormatNode: grounded_answer is missing from state")
            emit_trace_event("output_format_failed", {"reason": "missing_grounded_answer"}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: grounded_answer is missing from state"],
            }

        citations = from_json(state.get("citations"), [])
        if not isinstance(citations, list):
            citations = []
        summary = from_json(state.get("anomaly_summary"), {})
        if not isinstance(summary, dict):
            summary = {}

        lines: List[str] = [
            _SEPARATOR,
            "PRICE & MARGIN ANOMALY REPORT",
            _SEPARATOR,
            "",
            _SCHEMA_NOTE,
            "",
            _SUBSEP,
            "GROUNDED EXPLANATION",
            _SUBSEP,
            answer.strip(),
            "",
            _SUBSEP,
            "CITATIONS",
            _SUBSEP,
        ]
        if citations:
            lines += [f"  - {citation}" for citation in citations]
        else:
            lines.append("  (none - the explanation is not grounded in the knowledge base)")

        lines += _render_findings(summary)
        lines.append(_SEPARATOR)

        document = "\n".join(lines)

        logger.info(
            "OutputFormatNode: report_chars=%d citations=%d findings=%d",
            len(document),
            len(citations),
            len(summary.get("findings") or []),
        )
        emit_trace_event(
            "output_format_complete",
            {
                "report_chars": len(document),
                "citation_count": len(citations),
                "finding_count": len(summary.get("findings") or []),
            },
            state,
        )

        return {
            "anomaly_report": document,
            "result": document,
            "status": AgentStatus.SUCCESS.value,
        }
