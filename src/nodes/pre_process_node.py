"""AgentCore Platform v1.0"""

# RET-C2-016 - PreProcessNode
# Outer backbone pre_process slot: the ingest boundary and the owner of the
# caller-data contract.
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL trust (required_trust_level - matches the
#     manifest's declared required_trust_level in config/agent.yaml)
#   - Reject empty and over-long questions early (fail fast)
#   - Refuse instruction-override payloads before anything downstream runs
#   - Strip control characters and collapse whitespace
#   - Validate every declared input_context field - including every field of
#     the pricing observation - against explicit bounds, and fail CLOSED on a
#     violation, naming the field and never the value
#   - Write validated_input + enriched_context to State
#
# The refusals here are the template's OWN guarantees. A platform input gate
# may reject some of the same payloads first, but this node does not depend on
# that: where such a gate is absent or configured off, an unchecked payload
# would otherwise reach the answer path and come back as a success. Every
# refusal is therefore observable as behaviour - error status, nothing carried
# forward - and holds when execute() is called with no framework wrapper in
# front of it.
#
# Node contract: execute(self, state) -> dict. Returns ONLY the state keys it
# writes (partial-dict contract).

import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, TOO_LONG
from src.schemas.caller_contract import (
    MAX_QUERY_CHARS,
    is_instruction_override,
    normalise_query,
    validate_input_context,
)
from src.schemas.state import to_json

logger = logging.getLogger(__name__)


class PreProcessNode(FunctionNode):
    """Ingest validation for RET-C2-016.

    The outer backbone's pre_process slot - the only node declaring
    VERIFIED_EXTERNAL trust, so unauthenticated or anonymous callers are
    rejected here (fail fast; the inner domain nodes run behind this boundary
    and are declared ANONYMOUS).

    Input state keys:
        user_input:    str  - the caller's pricing question
        input_context: dict - optional caller parameters (channel, category,
                              top_k, pricing)

    Output state keys (partial dict):
        validated_input:  str        - normalised question
        enriched_context: str        - JSON-serialised request metadata
        status:           str        - AgentStatus.SUCCESS.value or ERROR
        error_code:       str        - set when the run COMPLETES without
                                       acting, because the caller can correct
                                       the value and send the request again
        error_log:        list[str]  - the internal reason, on either stop path
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context")

        # -- Emptiness --------------------------------------------------------
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            logger.warning("PreProcessNode: user_input is empty or missing")
            emit_trace_event("pre_process_validation_failed", {"reason": "empty_input"}, state)
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        # -- Length bound -----------------------------------------------------
        if len(user_input) > MAX_QUERY_CHARS:
            logger.warning(
                "PreProcessNode: question exceeds %d characters (%d)",
                MAX_QUERY_CHARS,
                len(user_input),
            )
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "query_too_long", "length": len(user_input)},
                state,
            )
            emit_progress(TOO_LONG)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "QUESTION_TOO_LONG",
                "error_log": [f"PreProcessNode: question exceeds {MAX_QUERY_CHARS} characters"],
            }

        # -- Instruction-override refusal (this template's own guarantee) ------
        #
        # This one terminates. The checks above complete carrying a reason
        # because the caller can correct the value; a refusal is not a value to
        # correct, and reporting it the same way would read as an invitation to
        # reword the request until it is accepted.
        if is_instruction_override(user_input):
            logger.warning("PreProcessNode: question carries an instruction-override payload")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "instruction_override"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: the question cannot be processed safely"],
            }

        # -- Caller-data contract (input_context) -----------------------------
        context, context_error = validate_input_context(input_context)
        if context_error:
            # Name the field, never the value.
            logger.warning("PreProcessNode: input_context validation failed")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "invalid_input_context"},
                state,
            )
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": [f"PreProcessNode: {context_error}"],
            }

        # -- Normalise (strip control characters, collapse whitespace) --------
        validated_input, truncation_note = normalise_query(user_input)
        if not validated_input:
            logger.warning("PreProcessNode: question is empty after normalisation")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "empty_after_normalisation"},
                state,
            )
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: question is empty after normalisation"],
            }

        logger.info(
            "PreProcessNode: validated question chars=%d pricing_fields=%d",
            len(validated_input),
            len(context.get("pricing", {})),
        )
        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "has_pricing_observation": bool(context.get("pricing")),
                "truncated": truncation_note is not None,
            },
            state,
        )

        return {
            "validated_input": validated_input,
            "enriched_context": to_json(
                {
                    "source": "PriceMarginAnomalyDetectionAgent",
                    "channel": context.get("channel", "unknown"),
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }
