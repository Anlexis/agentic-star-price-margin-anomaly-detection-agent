"""RetrieveNode and RerankFilterNode: retrieval depth, category filter, relevance floor.

The tuning values these nodes read have crossed a serialization boundary, so
they are re-validated on the way in. The tests below pin what happens when a
value is unusable: the node falls back to its default or drops the passage - it
never lets a non-finite number decide the comparison.
"""

import json

import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import (
    _DEFAULT_SCORE_THRESHOLD,
    _DEFAULT_TOP_K,
    RerankFilterNode,
)
from src.nodes.retrieve_node import _REFERENCE_CORPUS, RetrieveNode

QUERY_TERMS = ["cost", "inversion", "margin", "negative", "price"]


def _query(**overrides):
    query = {
        "query": "cost inversion",
        "terms": QUERY_TERMS,
        "category": None,
        "channel": "unknown",
        "top_k": None,
        "pricing": None,
    }
    query.update(overrides)
    return json.dumps(query)


class TestTrustDeclarations:
    def test_inner_nodes_are_anonymous(self):
        assert RetrieveNode.required_trust_level is TrustLevel.ANONYMOUS
        assert RerankFilterNode.required_trust_level is TrustLevel.ANONYMOUS


class TestRetrieve:
    def test_retrieves_scored_passages(self):
        result = RetrieveNode().execute({"rag_query": _query(), "correlation_id": "unit"})
        assert result["status"] == AgentStatus.SUCCESS.value
        passages = json.loads(result["retrieved_passages"])
        assert passages, "the query should match the pricing-policy corpus"
        assert passages[0]["id"] == "KB-PRC-001"
        assert passages == sorted(passages, key=lambda p: p["score"], reverse=True)

    def test_missing_query_fails_closed(self):
        result = RetrieveNode().execute({"correlation_id": "unit"})
        assert result["status"] == AgentStatus.ERROR.value
        assert "retrieved_passages" not in result

    def test_category_filter_narrows_the_corpus(self):
        result = RetrieveNode().execute(
            {
                "rag_query": _query(category="compliance", terms=["reference", "price", "discount"]),
                "correlation_id": "unit",
            }
        )
        passages = json.loads(result["retrieved_passages"])
        assert [p["id"] for p in passages] == ["KB-PRC-005"]

    def test_caller_top_k_bounds_the_fan_out(self):
        result = RetrieveNode().execute(
            {"rag_query": _query(top_k=1, terms=["price", "margin", "severity", "rule"]), "correlation_id": "unit"}
        )
        # top_k 1 with the over-fetch factor of 2 caps the candidate list at 2.
        assert len(json.loads(result["retrieved_passages"])) <= 2

    @pytest.mark.parametrize("unusable", [float("nan"), float("inf"), 0, 21, "5", True])
    def test_an_unusable_top_k_falls_back_to_the_default(self, unusable):
        result = RetrieveNode().execute(
            {"rag_query": _query(top_k=unusable), "retrieval_top_k": unusable, "correlation_id": "unit"}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert len(json.loads(result["retrieved_passages"])) <= _DEFAULT_TOP_K * 2

    def test_configured_top_k_is_used_when_the_caller_sends_none(self):
        result = RetrieveNode().execute(
            {
                "rag_query": _query(terms=["price", "margin", "severity", "rule", "discount"]),
                "retrieval_top_k": 1,
                "correlation_id": "unit",
            }
        )
        assert len(json.loads(result["retrieved_passages"])) <= 2

    def test_the_reference_corpus_is_never_mutated(self):
        before = json.dumps(_REFERENCE_CORPUS, sort_keys=True)
        RetrieveNode().execute({"rag_query": _query(), "correlation_id": "unit"})
        assert json.dumps(_REFERENCE_CORPUS, sort_keys=True) == before


class TestRerankFilter:
    def _passages(self, *scores):
        return json.dumps(
            [{"id": f"KB-PRC-00{i + 1}", "title": "Rule", "text": "Body.", "score": s} for i, s in enumerate(scores)]
        )

    def test_relevance_floor_drops_weak_passages(self):
        result = RerankFilterNode().execute({"retrieved_passages": self._passages(0.9, 0.1), "correlation_id": "unit"})
        kept = json.loads(result["reranked_passages"])
        assert [p["id"] for p in kept] == ["KB-PRC-001"]

    def test_top_k_trims_after_the_floor(self):
        result = RerankFilterNode().execute(
            {"retrieved_passages": self._passages(0.9, 0.8, 0.7), "retrieval_top_k": 2, "correlation_id": "unit"}
        )
        assert len(json.loads(result["reranked_passages"])) == 2

    @pytest.mark.parametrize("unusable", [float("nan"), float("inf"), -0.1, 1.1, "0.5", True])
    def test_an_unusable_relevance_floor_falls_back_to_the_default(self, unusable):
        """A NaN floor would compare False against every score and drop everything."""
        result = RerankFilterNode().execute(
            {"retrieved_passages": self._passages(0.9), "retrieval_score_threshold": unusable, "correlation_id": "unit"}
        )
        kept = json.loads(result["reranked_passages"])
        assert kept, "a passage well above the default floor must survive"
        assert _DEFAULT_SCORE_THRESHOLD == 0.35

    @pytest.mark.parametrize("unusable", [float("nan"), float("inf"), "high", None, True])
    def test_a_passage_with_an_unusable_score_is_dropped(self, unusable):
        """Fail closed: an unscoreable passage falls to 0.0 and drops at the floor."""
        result = RerankFilterNode().execute(
            {
                "retrieved_passages": json.dumps(
                    [{"id": "KB-PRC-001", "title": "Rule", "text": "Body.", "score": unusable}]
                ),
                "correlation_id": "unit",
            }
        )
        assert json.loads(result["reranked_passages"]) == []

    def test_non_dict_entries_are_ignored(self):
        result = RerankFilterNode().execute(
            {"retrieved_passages": json.dumps(["not a passage", 42]), "correlation_id": "unit"}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert json.loads(result["reranked_passages"]) == []

    def test_absent_input_yields_an_empty_result(self):
        result = RerankFilterNode().execute({"correlation_id": "unit"})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert json.loads(result["reranked_passages"]) == []
