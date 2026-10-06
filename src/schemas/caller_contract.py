"""AgentCore Platform v1.0"""

# RET-C2-016 - caller-data contract (single source of truth).
#
# Everything a caller can send is untrusted: the question text, the structured
# pricing observation, and the identifiers that select behaviour. The rules
# live here, once, because they are enforced at TWO boundaries and the two
# must never drift apart:
#
#   src/nodes/pre_process_node.py    - the ingest boundary (outer backbone)
#   src/nodes/input_validate_node.py - the inner-graph boundary, so a direct
#                                      inner-graph invocation gets the same
#                                      fail-closed contract
#
# Three rules run through all of it:
#   - fail CLOSED: an invalid value rejects the request, it is never clamped
#     into a "reasonable" one. A clamped parameter changes the answer without
#     telling anyone, and clamping is where non-finite input does its damage:
#     max(0.0, min(1.0, float("nan"))) evaluates to 1.0, silently installing
#     the strictest possible floor.
#   - every number is FINITE and BOUNDED before it is compared. float()
#     happily parses "NaN"/"Infinity", Python's json accepts bare NaN in a
#     request body, and every comparison against NaN is False - so a
#     non-finite threshold turns the exact decision this agent exists for
#     into a silent no-op.
#   - name the FIELD, never the VALUE: rejected caller data must not
#     round-trip into an error log or the response.

import math
import re
from typing import Any, List, Optional, Pattern, Tuple

# Hard cap on the accepted question length (2000 characters covers any
# realistic pricing question).
MAX_QUERY_CHARS = 2000

# Bounds for the caller-supplied retrieval-depth override.
TOP_K_MIN = 1
TOP_K_MAX = 20

# Bounds for the caller-supplied pricing observation. Prices are per unit in
# the reporting currency; the ceilings are deliberately generous but finite so
# an absurd magnitude cannot reach the arithmetic.
AMOUNT_MIN = 0.0
AMOUNT_MAX = 100_000_000.0
UNITS_MIN = 0
UNITS_MAX = 10_000_000
# A category margin floor above this is not a real retail policy, and the
# required-price arithmetic (cost / (1 - floor)) becomes ill-conditioned as the
# floor approaches 100%.
MARGIN_FLOOR_MIN = 0.0
MARGIN_FLOOR_MAX = 95.0

# Caller strings that select behaviour (the knowledge-base category filter, the
# request channel) or that are RENDERED into the report (the product code) must
# be inert identifiers: lowercase alphanumerics and underscore, bounded length.
# Free text in such a field is caller-controlled output/log injection, so it is
# rejected before it is ever compared, stored or rendered.
INERT_IDENTIFIER_RE: Pattern[str] = re.compile(r"^[a-z0-9_]{1,32}$")

# Control characters (tab and newline excepted) are stripped before processing.
CONTROL_CHARS_RE: Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE: Pattern[str] = re.compile(r"\s+")

# -- Instruction-override screen ---------------------------------------------
# Text addressed to the answering model rather than a question addressed to the
# knowledge base. This template refuses such payloads ITSELF. A platform input
# gate may refuse them too, but the template must not depend on that: where
# such a gate is absent or configured off, an unchecked payload would otherwise
# reach the answer path and come back as a success. Refusal is expressed as
# BEHAVIOUR - error status, nothing carried forward - never as a particular
# gate's wording.
#
# Deliberately narrow, and narrowed FURTHER for this domain: "rule" / "rules"
# is the central noun of the knowledge base ("disregard the previous rules for
# clearance items?" is an ordinary pricing-policy question), so the override
# target is limited to words that only ever address the model. Determiners and
# quantifiers may sit between verb and target ("ignore all these previous
# instructions") - the attack is identical, so they are absorbed rather than
# relied on being absent.
_OVERRIDE_FILLER = r"(?:(?:all|any|the|these|those|your|my)\s+)*"
_OVERRIDE_SCOPE = r"(?:previous|prior|above|earlier|preceding)"
_OVERRIDE_TARGET = r"(?:instruction|instructions|prompt|prompts)"
INJECTION_RE: Pattern[str] = re.compile(
    rf"(?:ignore|disregard|forget|discard)\s+{_OVERRIDE_FILLER}{_OVERRIDE_SCOPE}\s+"
    rf"{_OVERRIDE_FILLER}{_OVERRIDE_TARGET}"
    r"|(?:reveal|show|print|repeat|output|disclose)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+prompt|system\s+message|initial\s+prompt)"
    r"|you\s+are\s+now\s+(?:a|an)\s"
    r"|act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+|an\s+)?(?:developer|admin|root)\s+mode"
    r"|override\s+(?:your|the)\s+(?:instruction|instructions|safety)",
    re.IGNORECASE,
)


def is_instruction_override(text: str) -> bool:
    """True when *text* carries an instruction-override payload."""
    return bool(INJECTION_RE.search(text))


def finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse an untrusted numeric: a FINITE number within [lo, hi], else None.

    Rejects booleans (which are ints in Python), anything that is not already a
    number - a numeric STRING means the caller's serializer lost the type, and
    accepting one would also accept "NaN" and "Infinity" - and, the dangerous
    case, non-finite values. NaN and +/-Infinity survive float() and arrive
    intact through a raw JSON body, and every comparison against NaN is False,
    so a range check written as an inequality passes them through untouched.
    Every number read from an untrusted or serialized source comes through
    here, so the rule is the same at every boundary.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


def validate_identifier_field(name: str, value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Validate an inert-identifier caller field. Returns (value, error).

    *name* is used only to build the error message - the rejected value is
    never included.
    """
    if value is None:
        return None, None
    if not isinstance(value, str) or not INERT_IDENTIFIER_RE.match(value):
        return None, f"{name} must be a lowercase identifier (a-z, 0-9, _; 1-32 chars)"
    return value, None


def validate_top_k(name: str, value: Any) -> Tuple[Optional[int], Optional[str]]:
    """Validate the untrusted retrieval-depth override. Returns (value, error).

    Only a plain integer within [1, 20] is accepted. Booleans, floats -
    including NaN and +/-Infinity - numeric strings, and out-of-range integers
    all return a field-naming error.
    """
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"{name} must be an integer between {TOP_K_MIN} and {TOP_K_MAX}"
    if not TOP_K_MIN <= value <= TOP_K_MAX:
        return None, f"{name} must be an integer between {TOP_K_MIN} and {TOP_K_MAX}"
    return value, None


def validate_amount_field(name: str, value: Any) -> Tuple[Optional[float], Optional[str]]:
    """Validate a caller-supplied monetary amount. Returns (value, error).

    Strings are refused outright: an amount arriving as text means the caller's
    serializer lost the type, and accepting it would also accept the strings
    "NaN" and "Infinity".
    """
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{name} must be a number between {AMOUNT_MIN:,.0f} and {AMOUNT_MAX:,.0f}"
    parsed = finite_in_range(value, AMOUNT_MIN, AMOUNT_MAX)
    if parsed is None:
        return None, f"{name} must be a finite number between {AMOUNT_MIN:,.0f} and {AMOUNT_MAX:,.0f}"
    return parsed, None


def validate_percent_field(name: str, value: Any) -> Tuple[Optional[float], Optional[str]]:
    """Validate a caller-supplied margin floor, expressed in percent."""
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{name} must be a number between {MARGIN_FLOOR_MIN:g} and {MARGIN_FLOOR_MAX:g}"
    parsed = finite_in_range(value, MARGIN_FLOOR_MIN, MARGIN_FLOOR_MAX)
    if parsed is None:
        return None, f"{name} must be a finite number between {MARGIN_FLOOR_MIN:g} and {MARGIN_FLOOR_MAX:g}"
    return parsed, None


def validate_units_field(name: str, value: Any) -> Tuple[Optional[int], Optional[str]]:
    """Validate a caller-supplied unit count (whole units, bounded)."""
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"{name} must be an integer between {UNITS_MIN} and {UNITS_MAX}"
    if not UNITS_MIN <= value <= UNITS_MAX:
        return None, f"{name} must be an integer between {UNITS_MIN} and {UNITS_MAX}"
    return value, None


def normalise_query(text: str) -> Tuple[str, Optional[str]]:
    """Strip control characters, collapse whitespace, cap the length.

    Returns (normalised_text, note). The note is set when the text had to be
    truncated, so the pipeline can record that the answer was built from a
    shortened question.
    """
    cleaned = CONTROL_CHARS_RE.sub("", text)
    normalised = WHITESPACE_RE.sub(" ", cleaned).strip()
    if len(normalised) > MAX_QUERY_CHARS:
        return normalised[:MAX_QUERY_CHARS], f"query truncated to {MAX_QUERY_CHARS} characters"
    return normalised, None


# Every declared field of the pricing observation, with its validator. Ordered
# so the report and the error surface stay stable, and inventoried in ONE place
# so a new field cannot be added without a bounds rule.
_AMOUNT_FIELDS = ("selling_price", "unit_cost", "competitor_price", "reference_price")


def validate_pricing_observation(raw: Any) -> Tuple[dict[str, Any], Optional[str]]:
    """Validate the caller's `pricing` block. Returns (observation, error).

    Accepted fields (all optional; an absent block degrades the run to the
    knowledge-base answer alone):

        sku:              inert identifier, RENDERED into the report
        selling_price:    number 0 .. 100,000,000
        unit_cost:        number 0 .. 100,000,000
        competitor_price: number 0 .. 100,000,000
        reference_price:  number 0 .. 100,000,000
        margin_floor_pct: number 0 .. 95
        units_sold:       integer 0 .. 10,000,000

    Any other key is ignored. A non-mapping block is rejected.
    """
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return {}, "input_context.pricing must be an object"

    observation: dict[str, Any] = {}

    sku, sku_error = validate_identifier_field("input_context.pricing.sku", raw.get("sku"))
    if sku_error:
        return {}, sku_error
    if sku is not None:
        observation["sku"] = sku

    for field in _AMOUNT_FIELDS:
        amount, amount_error = validate_amount_field(f"input_context.pricing.{field}", raw.get(field))
        if amount_error:
            return {}, amount_error
        if amount is not None:
            observation[field] = amount

    floor, floor_error = validate_percent_field("input_context.pricing.margin_floor_pct", raw.get("margin_floor_pct"))
    if floor_error:
        return {}, floor_error
    if floor is not None:
        observation["margin_floor_pct"] = floor

    units, units_error = validate_units_field("input_context.pricing.units_sold", raw.get("units_sold"))
    if units_error:
        return {}, units_error
    if units is not None:
        observation["units_sold"] = units

    return observation, None


def validate_input_context(raw: Any) -> Tuple[dict[str, Any], Optional[str]]:
    """Validate the whole caller-supplied input_context. Returns (context, error).

    Accepted fields:

        channel:  inert identifier   (absent -> "unknown")
        category: inert identifier   (absent -> no knowledge-base filter)
        top_k:    integer 1 .. 20    (absent -> the configured retrieval depth)
        pricing:  object             (absent -> knowledge-base answer only)

    Any other key is ignored; the entry point separately caps the serialized
    size of the whole object.
    """
    if raw is None:
        return {"channel": "unknown"}, None
    if not isinstance(raw, dict):
        return {}, "input_context must be an object"

    context: dict[str, Any] = {}

    channel, channel_error = validate_identifier_field("input_context.channel", raw.get("channel"))
    if channel_error:
        return {}, channel_error
    context["channel"] = channel or "unknown"

    category, category_error = validate_identifier_field("input_context.category", raw.get("category"))
    if category_error:
        return {}, category_error
    if category is not None:
        context["category"] = category

    top_k, top_k_error = validate_top_k("input_context.top_k", raw.get("top_k"))
    if top_k_error:
        return {}, top_k_error
    if top_k is not None:
        context["top_k"] = top_k

    pricing, pricing_error = validate_pricing_observation(raw.get("pricing"))
    if pricing_error:
        return {}, pricing_error
    if pricing:
        context["pricing"] = pricing

    return context, None


def declared_numeric_fields() -> List[str]:
    """Every caller-controlled numeric field, for the non-finite test matrix.

    Kept next to the validators so a new numeric field is inventoried here
    rather than remembered.
    """
    return ["top_k", *_AMOUNT_FIELDS, "margin_floor_pct", "units_sold"]
