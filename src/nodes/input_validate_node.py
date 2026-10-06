"""AgentCore Platform v1.0"""

# RET-C2-016 - InputValidateNode
# Inner domain node 1: the inner-graph boundary. Re-validates the caller
# contract, then tokenises the question into retrieval terms.
#
# Why re-validate: PreProcessNode owns the contract at the outer boundary, but
# the inner graph is independently invocable (unit tests, and any future reuse
# of DomainWorkflowGraph). Validating the same rules from the same module here
# means the fail-closed contract holds on BOTH entry paths and the two cannot
# drift apart - the rules live once, in src/schemas/caller_contract.py.
#
# Inner node - ANONYMOUS trust: the outer PreProcessNode (VERIFIED_EXTERNAL)
# has already enforced trust, and inner nodes must be ANONYMOUS so the outer
# invocation context passes through the GraphNode boundary without rejection.
#
# Node contract: execute(self, state) -> dict. Returns ONLY the state keys it
# writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED
from src.schemas.caller_contract import (
    is_instruction_override,
    normalise_query,
    validate_input_context,
)
from src.schemas.state import to_json

logger = logging.getLogger(__name__)

# Tokens shorter than this (after lower-casing) are dropped as stop-noise,
# EXCEPT alphanumeric codes (product codes, policy references), which are
# always kept.
_MIN_TERM_LEN = 3

# Very common words that add no retrieval signal for pricing questions.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "why",
        "was",
        "are",
        "our",
        "this",
        "that",
        "with",
        "what",
        "how",
        "does",
        "did",
        "has",
        "have",
        "from",
        "into",
        "per",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_]*")
_CODE_RE = re.compile(r"^[A-Za-z]*\d")  # contains a digit -> looks like a code


def _tokenise(text: str) -> List[str]:
    """Tokenise the question into deduplicated lower-case retrieval terms.

    Keeps short alphanumeric codes (product codes, policy references)
    regardless of length; drops stopwords and very short pure-alphabetic
    tokens.
    """
    terms: List[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        token = raw.lower()
        is_code = bool(_CODE_RE.search(token))
        if token in _STOPWORDS:
            continue
        if len(token) < _MIN_TERM_LEN and not is_code:
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


class InputValidateNode(FunctionNode):
    """Inner-graph validation and tokenisation of the pricing question.

    Inner node - ANONYMOUS trust (see the module comment).

    Input state keys:
        validated_input: str  - normalised question from PreProcessNode.
                                Falls back to user_input on a direct call.
        input_context:   dict - the caller contract, bridged in by
                                DomainWorkflowGraph._extra_initial_state()

    Output state keys (partial dict):
        rag_query: str        - JSON-serialised {query, terms, category,
                                channel, top_k, pricing}
        status:    str
        error_code: str       - set when the run COMPLETES without acting,
                                because the caller can correct the value
        error_log: list[str]  - the internal reason, on either stop path
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        if not isinstance(raw, str):
            raw = ""

        query, _note = normalise_query(raw)
        if not query:
            logger.warning("InputValidateNode: question is empty after normalisation")
            emit_trace_event("input_validate_failed", {"reason": "empty_query"}, state)
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["InputValidateNode: question is empty or missing"],
            }

        # Same refusal as the outer boundary, so a direct inner-graph
        # invocation is not a way around it.
        #
        # Terminal: a refusal, not a correctable value. Rewording the request
        # must not be presented as a route past it.
        if is_instruction_override(query):
            logger.warning("InputValidateNode: question carries an instruction-override payload")
            emit_trace_event("input_validate_failed", {"reason": "instruction_override"}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: the question cannot be processed safely"],
            }

        # Same contract as the outer boundary, from the same module.
        context, context_error = validate_input_context(state.get("input_context"))
        if context_error:
            logger.warning("InputValidateNode: input_context validation failed")
            emit_trace_event("input_validate_failed", {"reason": "invalid_input_context"}, state)
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": [f"InputValidateNode: {context_error}"],
            }

        terms = _tokenise(query)

        rag_query: Dict[str, Any] = {
            "query": query,
            "terms": terms,
            "category": context.get("category"),
            "channel": context.get("channel", "unknown"),
            "top_k": context.get("top_k"),
            "pricing": context.get("pricing") or None,
        }

        logger.info(
            "InputValidateNode: query_chars=%d term_count=%d category=%s",
            len(query),
            len(terms),
            rag_query["category"],
        )
        emit_trace_event(
            "input_validate_complete",
            {
                "term_count": len(terms),
                "has_category_filter": rag_query["category"] is not None,
                "has_pricing_observation": rag_query["pricing"] is not None,
            },
            state,
        )

        return {
            "rag_query": to_json(rag_query),
            "status": AgentStatus.SUCCESS.value,
        }
