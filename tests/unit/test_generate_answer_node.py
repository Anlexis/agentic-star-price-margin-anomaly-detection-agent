"""GenerateAnswerNode: grounded explanation and the anomaly arithmetic.

Two separable guarantees:
  - the explanation is built only from passages that cleared the relevance
    floor, and says so plainly when none did;
  - the findings are real arithmetic on the caller's validated observation, so
    a caller who sends data gets a result derived from it - never a fixed
    baseline regardless of input.
"""

import json

import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode

PASSAGES = json.dumps(
    [
        {
            "id": "KB-PRC-001",
            "title": "Cost inversion (selling price below cost)",
            "text": "A cost inversion occurs when the selling price drops below cost. Severity is CRITICAL.",
            "score": 0.9,
        }
    ]
)


def _state(pricing=None, passages=PASSAGES):
    query = {
        "query": "why is margin negative",
        "terms": ["margin"],
        "category": None,
        "channel": "ec",
        "top_k": None,
        "pricing": pricing,
    }
    return {"reranked_passages": passages, "rag_query": json.dumps(query), "correlation_id": "unit"}


def _summary(result):
    return json.loads(result["anomaly_summary"])


class TestTrustDeclaration:
    def test_inner_node_is_anonymous(self):
        assert GenerateAnswerNode.required_trust_level is TrustLevel.ANONYMOUS


class TestGrounding:
    def test_explanation_is_built_from_the_passages(self):
        result = GenerateAnswerNode().execute(_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "KB-PRC-001" in result["grounded_answer"]
        assert json.loads(result["citations"]) == ["KB-PRC-001"]
        assert _summary(result)["grounded"] is True

    def test_no_grounding_says_so_instead_of_inventing_policy(self):
        result = GenerateAnswerNode().execute(_state(passages=json.dumps([])))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert (
            "insufficient" in result["grounded_answer"].lower()
            or "no pricing-policy" in result["grounded_answer"].lower()
        )
        assert json.loads(result["citations"]) == []
        assert _summary(result)["grounded"] is False

    def test_the_explanation_never_embeds_the_caller_question(self):
        """The report is assembled from knowledge-base content, not echoed input."""
        question = "a very distinctive caller question about winter jackets"
        state = _state()
        state["rag_query"] = json.dumps(
            {"query": question, "terms": ["margin"], "category": None, "channel": "ec", "top_k": None, "pricing": None}
        )
        result = GenerateAnswerNode().execute(state)
        assert question not in result["grounded_answer"]


class TestAnomalyArithmetic:
    def test_cost_inversion_is_critical(self):
        summary = _summary(
            GenerateAnswerNode().execute(_state({"sku": "a1", "selling_price": 1180.0, "unit_cost": 1450.0}))
        )
        rules = [f["rule"] for f in summary["findings"]]
        assert "cost_inversion" in rules
        assert summary["highest_severity"] == "CRITICAL"

    def test_margin_bleed_is_high(self):
        # 20% realised margin against a 30% floor, and the price is above cost.
        summary = _summary(
            GenerateAnswerNode().execute(
                _state({"selling_price": 1000.0, "unit_cost": 800.0, "margin_floor_pct": 30.0})
            )
        )
        rules = [f["rule"] for f in summary["findings"]]
        assert rules == ["margin_bleed"]
        assert summary["highest_severity"] == "HIGH"

    def test_a_healthy_margin_produces_no_finding(self):
        summary = _summary(
            GenerateAnswerNode().execute(
                _state({"selling_price": 1000.0, "unit_cost": 500.0, "margin_floor_pct": 30.0})
            )
        )
        assert summary["findings"] == []
        assert summary["highest_severity"] is None
        assert summary["evaluated"] is True

    def test_competitor_undercut_is_medium(self):
        summary = _summary(GenerateAnswerNode().execute(_state({"selling_price": 1000.0, "competitor_price": 900.0})))
        assert [f["rule"] for f in summary["findings"]] == ["competitor_undercut"]
        assert summary["highest_severity"] == "MEDIUM"

    def test_a_competitor_above_our_price_is_not_a_finding(self):
        summary = _summary(GenerateAnswerNode().execute(_state({"selling_price": 1000.0, "competitor_price": 1100.0})))
        assert summary["findings"] == []

    def test_inflated_reference_price_is_high(self):
        summary = _summary(GenerateAnswerNode().execute(_state({"selling_price": 1000.0, "reference_price": 3600.0})))
        assert [f["rule"] for f in summary["findings"]] == ["reference_price_representation"]

    def test_a_plausible_reference_price_is_not_a_finding(self):
        summary = _summary(GenerateAnswerNode().execute(_state({"selling_price": 1000.0, "reference_price": 1400.0})))
        assert summary["findings"] == []

    def test_severity_ordering_reports_the_strongest(self):
        summary = _summary(
            GenerateAnswerNode().execute(
                _state({"selling_price": 1180.0, "unit_cost": 1450.0, "competitor_price": 1099.0})
            )
        )
        assert {f["rule"] for f in summary["findings"]} == {"cost_inversion", "competitor_undercut"}
        assert summary["highest_severity"] == "CRITICAL"

    def test_the_output_is_derived_from_the_input_not_a_baseline(self):
        """Two different observations must produce two different results."""
        inverted = _summary(
            GenerateAnswerNode().execute(_state({"selling_price": 100.0, "unit_cost": 200.0, "units_sold": 10}))
        )
        healthy = _summary(
            GenerateAnswerNode().execute(_state({"selling_price": 200.0, "unit_cost": 100.0, "units_sold": 10}))
        )
        assert inverted["findings"] and not healthy["findings"]
        assert inverted["margin_at_risk"] != healthy["margin_at_risk"]


class TestMarginAtRisk:
    def test_exposure_is_shortfall_times_volume(self):
        summary = _summary(
            GenerateAnswerNode().execute(_state({"selling_price": 1180.0, "unit_cost": 1450.0, "units_sold": 5200}))
        )
        assert summary["margin_at_risk"] == pytest.approx((1450.0 - 1180.0) * 5200)

    def test_the_margin_floor_raises_the_target_price(self):
        # required = 800 / (1 - 0.30) = 1142.857...; shortfall x 100 units.
        summary = _summary(
            GenerateAnswerNode().execute(
                _state({"selling_price": 1000.0, "unit_cost": 800.0, "margin_floor_pct": 30.0, "units_sold": 100})
            )
        )
        assert summary["margin_at_risk"] == pytest.approx((800.0 / 0.7 - 1000.0) * 100)

    def test_no_exposure_when_the_price_clears_the_target(self):
        summary = _summary(
            GenerateAnswerNode().execute(
                _state({"selling_price": 2000.0, "unit_cost": 800.0, "margin_floor_pct": 30.0, "units_sold": 100})
            )
        )
        assert summary["margin_at_risk"] is None

    def test_no_aggregate_without_unit_volume(self):
        """A per-unit shortfall IS the raw line item this report does not publish."""
        summary = _summary(GenerateAnswerNode().execute(_state({"selling_price": 1180.0, "unit_cost": 1450.0})))
        assert summary["margin_at_risk"] is None


class TestUnusableNumbersWithheldNotComputed:
    """A value that fails re-validation withholds the rule; it is never clamped."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "1450", True, None])
    def test_a_bad_unit_cost_withholds_the_inversion_finding(self, bad):
        summary = _summary(GenerateAnswerNode().execute(_state({"selling_price": 1180.0, "unit_cost": bad})))
        assert [f["rule"] for f in summary["findings"]] == []
        assert summary["margin_at_risk"] is None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 200.0])
    def test_a_bad_margin_floor_withholds_the_bleed_finding(self, bad):
        summary = _summary(
            GenerateAnswerNode().execute(_state({"selling_price": 1000.0, "unit_cost": 800.0, "margin_floor_pct": bad}))
        )
        assert [f["rule"] for f in summary["findings"]] == []

    def test_no_observation_marks_the_run_unevaluated(self):
        summary = _summary(GenerateAnswerNode().execute(_state(None)))
        assert summary["evaluated"] is False
        assert summary["findings"] == []

    def test_a_non_mapping_observation_is_ignored(self):
        summary = _summary(GenerateAnswerNode().execute(_state("selling_price=1180")))
        assert summary["evaluated"] is False
