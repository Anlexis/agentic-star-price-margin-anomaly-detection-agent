"""The output boundary: what the report renders, and what the gate enforces.

The report's stated schema is: monetary figures are AGGREGATES ONLY, rounded to
the nearest 1,000; per-unit prices and costs are never rendered; product codes
carry the fixed SKU_ prefix. OutputFormatNode RENDERS on that grid and
PostProcessNode ENFORCES it independently, so a rendering regression cannot put
a full-precision figure on the external surface.

Both directions are pinned: every leak form snaps, and every structural token -
citation ids, product codes, counts, severities, years, versions - stays
byte-identical.
"""

import json

import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import (
    _EXTERNAL_ROUND_UNIT,
    _SKU_PREFIX,
    OutputFormatNode,
)
from src.nodes.post_process_node import (
    PostProcessNode,
    _enforce_precision,
    _security_gate_output,
)


def _format_state(summary, answer="Grounded explanation body.", citations=("KB-PRC-001",)):
    return {
        "grounded_answer": answer,
        "citations": json.dumps(list(citations)),
        "anomaly_summary": json.dumps(summary),
        "correlation_id": "unit",
    }


class TestTrustDeclarations:
    def test_output_nodes_are_anonymous(self):
        assert OutputFormatNode.required_trust_level is TrustLevel.ANONYMOUS
        assert PostProcessNode.required_trust_level is TrustLevel.ANONYMOUS


class TestRenderedSchema:
    def test_the_report_states_its_rounding_schema(self):
        result = OutputFormatNode().execute(_format_state({"evaluated": False}))
        assert f"{_EXTERNAL_ROUND_UNIT:,d}" in result["anomaly_report"]
        assert "not rendered" in result["anomaly_report"]

    def test_the_aggregate_is_rendered_on_the_grid(self):
        result = OutputFormatNode().execute(
            _format_state(
                {
                    "evaluated": True,
                    "sku": "a88421",
                    "findings": [{"rule": "cost_inversion", "severity": "CRITICAL", "detail": "d"}],
                    "highest_severity": "CRITICAL",
                    "margin_at_risk": 1_404_000.0 + 317.0,
                }
            )
        )
        assert "JPY 1,404,000" in result["anomaly_report"]
        assert "1,404,317" not in result["anomaly_report"]

    def test_raw_line_items_are_never_rendered(self):
        """Per-unit prices and costs must not appear, even though they were supplied."""
        result = OutputFormatNode().execute(
            _format_state(
                {
                    "evaluated": True,
                    "sku": "a88421",
                    "findings": [{"rule": "cost_inversion", "severity": "CRITICAL", "detail": "d"}],
                    "highest_severity": "CRITICAL",
                    "margin_at_risk": 1_404_000.0,
                    # These are carried in the summary but must not be rendered.
                    "selling_price": 1180.0,
                    "unit_cost": 1450.0,
                }
            )
        )
        report = result["anomaly_report"]
        assert "1180" not in report and "1450" not in report

    def test_the_product_code_carries_the_identifier_prefix(self):
        result = OutputFormatNode().execute(_format_state({"evaluated": True, "sku": "48210", "findings": []}))
        assert f"{_SKU_PREFIX}48210" in result["anomaly_report"]

    def test_no_observation_is_distinguished_from_no_findings(self):
        unevaluated = OutputFormatNode().execute(_format_state({"evaluated": False}))["anomaly_report"]
        clean = OutputFormatNode().execute(_format_state({"evaluated": True, "findings": []}))["anomaly_report"]
        assert "no anomaly rules were evaluated" in unevaluated
        assert "No pricing anomaly was detected" in clean

    def test_missing_explanation_fails_closed(self):
        result = OutputFormatNode().execute(_format_state({"evaluated": False}, answer=""))
        assert result["status"] == AgentStatus.ERROR.value
        assert "anomaly_report" not in result


class TestPrecisionGateSnaps:
    """Every enumerated monetary leak form is snapped onto the grid."""

    @pytest.mark.parametrize(
        "leak,expected",
        [
            # comma-grouped, marker before the value (the grouped form must win
            # over a leftmost match of the first digit)
            ("JPY 1,234", "JPY 1,000"),
            # short value in currency context, both orders
            ("JPY 9999", "JPY 10,000"),
            ("9999 JPY", "10,000 JPY"),
            # attached, and signed
            ("¥9999", "¥10,000"),
            ("JPY-9999", "JPY-10,000"),
            ("JPY +9999", "JPY +10,000"),
            ("USD\n+9999", "USD\n+10,000"),
            # arbitrary horizontal whitespace runs, including tabs
            ("JPY  9999", "JPY  10,000"),
            ("JPY\t9999", "JPY\t10,000"),
            # fullwidth and postfix symbols
            ("￥12345", "￥12,000"),
            ("9999円", "10,000円"),
            # form-based: comma-grouped or a 5+ digit run, no marker at all
            ("1,284,300", "1,284,000"),
            ("48211", "48,000"),
            # no magnitude exemption: a small off-grid value still snaps
            ("JPY 9,999", "JPY 10,000"),
            ("JPY 1", "JPY 0"),
        ],
    )
    def test_leak_forms_snap(self, leak, expected):
        snapped, redactions = _enforce_precision(leak)
        assert snapped == expected
        assert redactions == 1

    def test_an_on_grid_value_is_byte_identical(self):
        assert _enforce_precision("JPY 1,000") == ("JPY 1,000", 0)
        assert _enforce_precision("JPY 1,000,000") == ("JPY 1,000,000", 0)


class TestPrecisionGatePassesStructuralTokens:
    """Identifiers and structural numbers must survive byte-for-byte."""

    @pytest.mark.parametrize(
        "token",
        [
            # knowledge-base citation ids - the report's core deliverable
            "KB-PRC-001",
            "KB-PRC-007",
            # product codes rendered behind the fixed prefix, over the whole
            # inert alphabet including a purely numeric code
            "SKU_48210",
            "SKU_4901234567890",
            "SKU_a1b2c3",
            "SKU_sku_48210",
            "SKU_00012",
            # structural text
            "Rules triggered:   3",
            "Highest severity:  CRITICAL",
            "  [CRITICAL] cost_inversion",
            "in 2026",
            "v12",
            "12 units",
            "STAR 2026",
            "90d horizon",
            "1.0.1",
        ],
    )
    def test_structural_tokens_are_untouched(self, token):
        assert _enforce_precision(token) == (token, 0)

    def test_the_delimiter_never_spans_a_paragraph_break(self):
        """A `\\s*` delimiter would bind a trailing code to the next block's number
        and rewrite a section heading."""
        text = "Currency: JPY\n\n3. Cash Position"
        assert _enforce_precision(text) == (text, 0)

    def test_a_pattern_scan_is_not_destroyed_by_the_snap(self):
        """Identifier guards keep the snap from mangling a hyphenated pattern.

        Rewriting "SSN 123-45-6789" into "SSN 0-45-6789" would destroy the very
        shape a credential or identifier scan matches on.
        """
        for text in ("SSN 123-45-6789", "TAX 987-65-4321"):
            assert _enforce_precision(text) == (text, 0)


class TestPrecisionGateHandlesDecimals:
    """A decimal is one number, and the gate has to treat it as one number.

    The fraction of a decimal is a free-standing digit run, and the identifier
    guards do not cover it on their own because a decimal point is not an
    identifier character. Left unhandled, the gate matched fractions on their
    own: "9999.99999%" came back "9999.100,000%", and an off-grid amount was
    snapped on its integer part with the old fraction still trailing the new
    value - "JPY 1234.56" -> "JPY 1,000.56", which is neither the amount the
    caller meant nor a figure on the published grid.

    Both directions are pinned here: an amount carrying a decimal snaps as ONE
    number, and a decimal that is not an amount survives byte-for-byte.
    """

    @pytest.mark.parametrize(
        "leak,expected",
        [
            # marker before the value, plain and comma-grouped
            ("JPY 1234.56", "JPY 1,000"),
            ("JPY 1,234.56", "JPY 1,000"),
            # marker after the value, and attached symbols
            ("1234.56 JPY", "1,000 JPY"),
            ("¥1234.56", "¥1,000"),
            ("1234.56円", "1,000円"),
            # signed
            ("JPY -1234.56", "JPY -1,000"),
            ("JPY +1234.56", "JPY +1,000"),
            # form-based, no marker: a 5+-digit run carrying a fraction
            ("12345.67", "12,000"),
            ("1,284,300.75", "1,284,000"),
            # the fraction is what pushes it off the grid - the integer part is
            # already on it, so an int-only gate would have called this clean
            ("JPY 1000.49", "JPY 1,000"),
        ],
    )
    def test_a_decimal_amount_snaps_as_one_number(self, leak, expected):
        snapped, redactions = _enforce_precision(leak)
        assert snapped == expected
        assert redactions == 1
        assert "." not in snapped, "no fraction may survive the snap"

    def test_an_on_grid_decimal_amount_is_byte_identical(self):
        assert _enforce_precision("JPY 1,000.00") == ("JPY 1,000.00", 0)

    def test_a_suffixed_decimal_does_not_backtrack_into_the_integer_part(self):
        """An optional fraction can be backtracked out of; the two-arm one cannot.

        With `(?:\.\d+)?`, the "m" after the fraction fails the trailing
        guard, the engine gives ".56" back and re-matches "1234" alone, and
        the dangling-fraction corruption returns: "JPY 1,000.56m". Absorption
        is all-or-nothing, so the suffixed token is left byte-identical -
        while the same amount without the suffix still snaps as one number.
        """
        assert _enforce_precision("JPY 1234.56m") == ("JPY 1234.56m", 0)
        assert _enforce_precision("JPY 1234.56") == ("JPY 1,000", 1)

    @pytest.mark.parametrize(
        "token",
        [
            # ratios and scores - the report's own relevance figures
            "ratio 0.123456",
            "score 0.7512",
            "0.85",
            # percentages, signed and long-fractioned
            "31.4%",
            "-3.25%",
            "9999.99999%",
            "Realised margin: 12.75% against a 15.0% floor",
            # bare decimals with no currency context are not monetary
            "8.512345",
            "1180.55",
            # dotted version strings
            "1.0.1",
            "v1.0.1",
        ],
    )
    def test_a_decimal_that_is_not_an_amount_survives(self, token):
        assert _enforce_precision(token) == (token, 0)

    def test_an_amount_ending_a_sentence_does_not_escape(self):
        """The decimal point joins the LEADING guard only.

        Adding it to the trailing guard as well would let a full-precision
        amount that ends a sentence slip past the gate.
        """
        assert _enforce_precision("Exposure is JPY 1234.") == ("Exposure is JPY 1,000.", 1)
        assert _enforce_precision("Exposure is 48211.") == ("Exposure is 48,000.", 1)


class TestCredentialScan:
    @pytest.mark.parametrize(
        "leak",
        [
            "api key sk-ABCDEFGHIJKLMNOPQRST",
            "eyJhbGciOi.eyJzdWIiOi.SflKxwRJSM",
            "Authorization: Bearer abcdefgh12345678",
            "password = hunter2hunter2",
        ],
    )
    def test_credential_patterns_are_detected(self, leak):
        assert _security_gate_output(leak) is not None

    def test_a_clean_report_is_clean(self):
        assert _security_gate_output("Estimated margin at risk: JPY 1,404,000") is None


class TestOutputGateBehaviour:
    def test_a_clean_report_passes_through(self):
        report = "PRICE & MARGIN ANOMALY REPORT\nEstimated margin at risk: JPY 1,404,000"
        result = PostProcessNode().execute({"result": report, "correlation_id": "unit"})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == report

    def test_a_credential_in_the_report_withholds_the_whole_output(self):
        result = PostProcessNode().execute(
            {"result": "margin report sk-ABCDEFGHIJKLMNOPQRST", "correlation_id": "unit"}
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert "WITHHELD" in result["formatted_output"]
        assert "sk-ABCDEFGHIJKLMNOPQRST" not in result["formatted_output"]

    def test_an_off_grid_figure_is_snapped_on_the_way_out(self):
        result = PostProcessNode().execute(
            {"result": "Estimated margin at risk: JPY 1,404,317", "correlation_id": "unit"}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "JPY 1,404,000" in result["formatted_output"]
        assert "1,404,317" not in result["formatted_output"]

    def test_caller_text_embedded_verbatim_is_redacted(self):
        question = "a very distinctive caller question about winter jacket pricing"
        result = PostProcessNode().execute(
            {
                "result": f"Report body quoting {question} back at the caller.",
                "user_input": question,
                "correlation_id": "unit",
            }
        )
        assert question not in result["formatted_output"]
        assert "[REDACTED]" in result["formatted_output"]

    def test_the_two_layers_stay_independent(self):
        """A verbatim caller-text leak and an off-grid figure are handled separately."""
        question = "a very distinctive caller question about winter jacket pricing"
        result = PostProcessNode().execute(
            {
                "result": f"{question} - Estimated margin at risk: JPY 1,404,317",
                "user_input": question,
                "correlation_id": "unit",
            }
        )
        assert "[REDACTED]" in result["formatted_output"]
        assert "JPY 1,404,000" in result["formatted_output"]

    def test_an_empty_report_uses_the_fallback_message(self):
        result = PostProcessNode().execute({"result": "", "correlation_id": "unit"})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No report content" in result["formatted_output"]

    def test_the_render_and_the_gate_agree_on_the_grid(self):
        """A report rendered by OutputFormatNode passes the gate untouched."""
        rendered = OutputFormatNode().execute(
            _format_state(
                {
                    "evaluated": True,
                    "sku": "a88421",
                    "findings": [{"rule": "cost_inversion", "severity": "CRITICAL", "detail": "d"}],
                    "highest_severity": "CRITICAL",
                    "margin_at_risk": 1_404_317.0,
                }
            )
        )["anomaly_report"]
        gated, redactions = _enforce_precision(rendered)
        assert redactions == 0, "the renderer already put every figure on the grid"
        assert gated == rendered
        assert "KB-PRC-001" in gated
        assert f"{_SKU_PREFIX}a88421" in gated


class TestAggregateIsRevalidatedBeforeRendering:
    """The aggregate crosses State, so it is untrusted on the way back out.

    int() on a non-finite float raises, so an unguarded render would crash the
    report rather than omit a figure it cannot vouch for.
    """

    _FINDING = {"rule": "cost_inversion", "severity": "CRITICAL", "detail": "d"}

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "1404000", True, -1.0, None])
    def test_an_unusable_aggregate_is_withheld_not_rendered(self, bad):
        result = OutputFormatNode().execute(
            _format_state(
                {
                    "evaluated": True,
                    "findings": [self._FINDING],
                    "highest_severity": "CRITICAL",
                    "margin_at_risk": bad,
                }
            )
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        report = result["anomaly_report"]
        assert "Estimated margin at risk" not in report
        assert "[CRITICAL] cost_inversion" in report, "the finding itself is still reported"

    def test_a_valid_aggregate_still_renders(self):
        result = OutputFormatNode().execute(
            _format_state(
                {
                    "evaluated": True,
                    "findings": [self._FINDING],
                    "highest_severity": "CRITICAL",
                    "margin_at_risk": 1_404_317.0,
                }
            )
        )
        assert "Estimated margin at risk: JPY 1,404,000" in result["anomaly_report"]
