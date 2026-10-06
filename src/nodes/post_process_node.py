"""AgentCore Platform v1.0"""

# RET-C2-016 - PostProcessNode
# Outer backbone post_process slot: the external-output boundary for the price
# & margin anomaly report. Four passes, in this order:
#
#   (1) credential scan - an API key, JWT, bearer token or password assignment
#       anywhere in the report withholds the output entirely (sanitised stub,
#       status=ERROR);
#   (2) verbatim caller-text redaction - the report is built from
#       knowledge-base content and from typed, bounded caller numbers, so a
#       verbatim embedding of the caller's question (or a derived form of it)
#       is a leak, not a feature: any such embedding becomes [REDACTED];
#   (3) monetary precision grid - the report's stated schema expresses monetary
#       figures as aggregates on a 1,000 grid; every monetary-form token is
#       snapped onto that grid, with an audit event per redaction;
#   (4) credential re-scan - pass (1) is a PATTERN scan and pass (3) rewrites
#       digits, so the invariant is re-checked on the bytes that are actually
#       about to be surfaced. Passes (1) and (2) are order-independent against
#       the snap; a pattern scan is not, so it runs on both sides of it.
#
# The domain output gate is the module-level function `_security_gate_output`
# called from inside execute() - NOT an instance method on the node class,
# because the framework auto-wraps node instance methods on the real invoke
# path. The agent class (PriceMarginAnomalyDetectionAgent) exposes the same
# scanner as the canonical output-gate entry point and delegates to this module
# so there is a single source of truth.
#
# Node contract: execute(self, state) -> dict. Returns ONLY the state keys it
# writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG

logger = logging.getLogger(__name__)

# Credential patterns that MUST NOT appear in the rendered report.
_CREDENTIAL_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:sk|pk|ak)-[A-Za-z0-9]{16,}", "api_key_pattern"),
    (r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "jwt_pattern"),
    (r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", "bearer_token"),
    (
        r"(?:password|passwd|secret|api_key|token|access_key|private_key)" r"\s*[:=]\s*\S{8,}",
        "credential_assignment",
    ),
]

# State fields that must NEVER be embedded verbatim in the external response.
# The report is assembled from knowledge-base passages, their citations and
# typed caller numbers - caller-derived TEXT reappearing verbatim means
# caller-controlled content reached the external surface.
_BLOCKED_FIELDS = frozenset({"user_input", "validated_input", "enriched_context"})

# Approved external precision: monetary figures are aggregates expressed in
# units of 1,000 (must match _EXTERNAL_ROUND_UNIT in
# src/nodes/output_format_node.py - the report RENDERS on this grid, this gate
# ENFORCES it).
_EXTERNAL_ROUND_UNIT = 1000

# EXPLICIT output schema - monetary values are identified by FORM and by
# CURRENCY CONTEXT, never by magnitude:
#   form:    comma-grouped numbers (9,999 / 1,234,567) and unformatted runs of
#            5+ digits (a rendering-regression leak);
#   context: any bare 1-4 digit number associated with a currency marker is
#            monetary even though short - SYMMETRICALLY: a 3-letter uppercase
#            code or a currency symbol (including fullwidth ￥ and 円/₩), before
#            or after the value, attached or separated, signed or unsigned.
#            Any standalone 3-letter uppercase word counts as a code on
#            purpose: a false snap fails SAFE while a missed leak does not.
# ALL matched tokens, at ANY magnitude, must sit on the rounding grid; off-grid
# means a full-precision figure reached the external surface, so it is snapped
# and audited.
#
# IDENTIFIER GUARDS. This report renders identifiers alongside monetary
# figures - knowledge-base citation ids (KB-PRC-001) and product codes, which
# the caller contract restricts to the inert alphabet [a-z0-9_] and which this
# report renders behind a fixed SKU_ prefix. Without a guard the grammar reads
# any standalone 3-letter uppercase word as a currency marker and any long
# digit run as an amount, so "KB-PRC-001" would render as "KB-PRC0" and
# "SKU_48210" as "SKU_48,000" - the gate silently rewriting the very
# identifiers the report exists to name. The guards are single-character
# assertions on both ends of the match: a monetary token may not be adjacent to
# a letter, a digit, a hyphen or an underscore. Hyphen covers the citation ids;
# underscore covers the inert product-code alphabet and the SKU_ prefix. They
# are fixed-width, so they do not constrain the variable-width delimiter
# inside. The two ends are deliberately NOT symmetric: the leading guard also
# rejects a decimal point (see _LEADING_GUARD_CHAR), the trailing one does not.
_IDENTIFIER_CHAR = r"[A-Za-z0-9_-]"

# LEADING guard only. The decimal point joins the identifier alphabet on the way
# IN, so no match can begin part-way through a number - see _VAL_FRACTION. It
# must NOT join the trailing guard: an amount that ends a sentence ("...JPY
# 1234.") would then be followed by a `.` and escape the gate entirely.
_LEADING_GUARD_CHAR = r"[A-Za-z0-9_.-]"

_CURRENCY_MARKER = r"(?:\b[A-Z]{3}|[¥￥$€£円₩])"

# Delimiter between a currency marker and its value: horizontal whitespace and
# at most ONE newline - never a paragraph break. A plain `\s*` spans blank
# lines, so a 3-letter uppercase word ending a line would bind to the number
# that opens the next block and rewrite it ("Currency: JPY\n\n3. Cash Position"
# -> "0. Cash Position"). Every enumerated leak form (spaces, tabs, a single
# newline, signed, symmetric, comma-grouped) still matches.
_GATE_DELIM = r"[ \t]*(?:\n[ \t]*)?"

# A monetary amount may carry a decimal part, and every value alternative
# absorbs it as part of the SAME token. Without that, the fraction of a decimal
# is a free-standing digit run and the identifier guards wave it through,
# because a decimal point is not an identifier character: "9999.99999%" matches
# on its fraction alone and comes back "9999.100,000%", and "JPY 1234.56"
# snaps only the integer part and leaves the fraction dangling behind the new
# value - "JPY 1,000.56", which is neither the true amount nor a grid amount.
# Absorbing the fraction makes an off-grid decimal amount snap as ONE number
# ("JPY 1234.56" -> "JPY 1,000") and leaves a non-amount decimal - a ratio, a
# percentage, a version string - untouched.
# The absorption cannot be a plain optional. `(?:\.\d+)?` is a choice point:
# when the character after the fraction fails the trailing guard, the engine
# backtracks out of the fraction and settles for the integer part alone, and
# "JPY 1234.56m" is right back to "JPY 1,000.56m". The second arm removes the
# choice - take the fraction whole, or assert no fraction starts here.
_VAL_FRACTION = r"(?:\.\d+|(?!\.\d))"

# Group-based grammar - no variable-width lookbehinds, so the marker/value
# delimiter can be an arbitrary run of horizontal whitespace. Every value
# accepts an optional explicit +/- sign and an optional decimal part. Branch
# order matters: currency-context branches first, then the form-based branch.
_NUM_TOKEN_RE = re.compile(
    rf"(?<!{_LEADING_GUARD_CHAR})"
    # marker THEN value: "JPY 9999", "JPY  -9999", "JPY\t9999", "¥9999", "USD\n+9999".
    # The value alternatives accept the comma-grouped form FIRST: the regex is
    # leftmost-first, so without it "JPY 1,234" would match as marker + "1" and
    # the snap would mangle the number - an on-grid "JPY 1,000" must stay
    # byte-identical, and an off-grid "JPY 1,234" must snap as 1234, not as 1.
    rf"(?:(?P<pre>{_CURRENCY_MARKER}{_GATE_DELIM})"
    rf"(?P<val_after>[+-]?\d{{1,3}}(?:,\d{{3}})+{_VAL_FRACTION}|[+-]?\d{{1,4}}{_VAL_FRACTION})"
    # value THEN marker: "9999 JPY", "-9999\tJPY", "9999円", "+9999  $"
    rf"|(?P<val_before>[+-]?\d{{1,4}}{_VAL_FRACTION})"
    rf"(?P<post>{_GATE_DELIM}(?:[A-Z]{{3}}\b|[¥￥$€£円₩]))"
    # form-based, standalone at any magnitude: comma-grouped or 5+-digit runs
    rf"|(?P<val_form>[+-]?\d{{1,3}}(?:,\d{{3}})+{_VAL_FRACTION}|[+-]?\d{{5,}}{_VAL_FRACTION}))"
    rf"(?!{_IDENTIFIER_CHAR})"
)


def _enforce_precision(result: str) -> Tuple[str, int]:
    """Snap every monetary-form token onto the approved external grid.

    Returns (sanitised_result, redaction_count). A redaction means a
    full-precision monetary figure reached the external surface and the gate
    rounded it onto the grid. The currency marker, the original delimiter
    whitespace and the explicit sign of the original token are all preserved on
    the snapped replacement.
    """
    redactions = 0

    def _snap(match: "re.Match[str]") -> str:
        nonlocal redactions
        pre = match.group("pre") or ""
        post = match.group("post") or ""
        token = match.group("val_after") or match.group("val_before") or match.group("val_form")
        # float(), not int(): the token may carry a decimal part, and the WHOLE
        # amount - fraction included - is what has to land on the grid.
        value = float(token.replace(",", ""))  # float() understands a leading +/-
        if value % _EXTERNAL_ROUND_UNIT == 0:
            return match.group(0)
        redactions += 1
        snapped = round(value / _EXTERNAL_ROUND_UNIT) * _EXTERNAL_ROUND_UNIT
        plus = "+" if token.startswith("+") and snapped >= 0 else ""
        return f"{pre}{plus}{snapped:,d}{post}"

    return _NUM_TOKEN_RE.sub(_snap, result), redactions


def _security_gate_output(content: str) -> Optional[str]:
    """Scan output for disallowed credential/secret patterns.

    Returns the first violation name, or None when the output is clean.
    Module-level function rather than a node instance method - the framework
    auto-wraps node instance methods on the real invoke path, so the gate must
    live at module level.
    """
    for pattern, name in _CREDENTIAL_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return name
    return None


def _redact_blocked_fields(result: str, state: AgentState) -> Tuple[str, List[str]]:
    """Replace verbatim embeddings of caller-derived state text with [REDACTED].

    Returns (sanitised_result, redacted_field_names). Only substantial values
    (longer than 10 characters) are matched, so a short incidental overlap with
    knowledge-base wording is not redacted.
    """
    redacted: List[str] = []
    sanitised = result
    for field in sorted(_BLOCKED_FIELDS):
        value = state.get(field)
        if isinstance(value, str) and len(value) > 10 and value in sanitised:
            sanitised = sanitised.replace(value, "[REDACTED]")
            redacted.append(field)
    return sanitised, redacted


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Apply the output gate and expose the final anomaly report.

    Outer backbone post_process slot. Declared ANONYMOUS - trust was already
    enforced at PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        result:         str - rendered report from the inner OutputFormatNode
        anomaly_report: str - the same rendered report (fallback source)

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "formatted_output": message,
                "result": message,
            }
        answer = state.get("result") or state.get("anomaly_report") or ""
        if not isinstance(answer, str):
            answer = ""

        # -- Fallback for an empty report --------------------------------------
        if not answer.strip():
            logger.warning("PostProcessNode: report is empty - using the fallback message")
            answer = (
                "[Price & Margin Anomaly] No report content was generated. " "Check error_log for upstream failures."
            )

        # -- Pass 1: credential scan (withhold entirely) -----------------------
        violation = _security_gate_output(answer)
        if violation:
            return self._withhold(state, violation)

        # -- Pass 2: verbatim caller-text redaction ----------------------------
        sanitised_output, redacted_fields = _redact_blocked_fields(answer, state)
        if redacted_fields:
            logger.error(
                "PostProcessNode: caller-derived text embedded verbatim in output - %s",
                ", ".join(redacted_fields),
            )
            emit_trace_event(
                "post_process_blocked_field_redaction",
                {"fields": redacted_fields},
                state,
            )

        # -- Pass 3: monetary precision grid -----------------------------------
        sanitised_output, precision_redactions = _enforce_precision(sanitised_output)
        if precision_redactions:
            logger.warning(
                "PostProcessNode: %d off-grid monetary token(s) snapped onto the external grid",
                precision_redactions,
            )
            emit_trace_event(
                "post_process_precision_redaction",
                {"redaction_count": precision_redactions},
                state,
            )

        # -- Pass 4: credential re-scan on the bytes about to be surfaced ------
        violation = _security_gate_output(sanitised_output)
        if violation:
            return self._withhold(state, violation)

        logger.info("PostProcessNode: output gate passed - length=%d", len(sanitised_output))
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(sanitised_output)},
            state,
        )

        return {
            "formatted_output": sanitised_output,
            "result": sanitised_output,
            "status": AgentStatus.SUCCESS.value,
        }

    def _withhold(self, state: AgentState, violation: str) -> dict[str, Any]:
        """Withhold the report entirely: a credential pattern reached the output."""
        logger.error("PostProcessNode: credential pattern detected in output - %s", violation)
        emit_trace_event("post_process_credential_violation", {"violation": violation}, state)
        sanitised = (
            f"[REPORT WITHHELD: the output contained a disallowed pattern ({violation}). "
            f"Review the generated report and retry.]"
        )
        return {
            "formatted_output": sanitised,
            "result": sanitised,
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PostProcessNode: credential pattern detected - {violation}"],
        }
