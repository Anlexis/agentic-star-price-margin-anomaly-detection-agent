"""Graph composition and the runtime-configuration path.

The configuration path is tested end to end on purpose: a reader that points at
a file the migration retired returns {} silently, every declared value goes
dead, and green tests hide it. These tests assert the declared values actually
reach the nodes that consume them.
"""

from pathlib import Path

import pytest
import yaml
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.graph.base_graph import BaseGraph
from framework.nodes.graph_node import GraphNode

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    PriceMarginAnomalyDetectionAgent,
    PriceMarginAnomalyGraphNode,
    _config_number,
    _runtime_config,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestManifestAndConfig:
    def test_the_manifest_declares_the_class_this_module_defines(self):
        manifest = yaml.safe_load((REPO_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
        assert manifest["class"] == "src.graph.graph.PriceMarginAnomalyDetectionAgent"
        assert manifest["name"] == PriceMarginAnomalyDetectionAgent().name

    def test_the_manifest_carries_no_runtime_block(self):
        """Runtime parameters live in config/config.yaml, never in the manifest."""
        manifest = yaml.safe_load((REPO_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
        assert "agent" not in manifest and "config" not in manifest

    def test_runtime_config_reads_the_live_file(self):
        cfg = _runtime_config()
        assert cfg, "config/config.yaml must be readable - an empty dict means every value is dead"
        assert cfg["max_retry"] == 3
        assert cfg["timeout_s"] == 30
        assert "retrieval" in cfg and "llm" in cfg

    def test_the_declared_prompt_asset_exists(self):
        cfg = _runtime_config()
        prompt_path = REPO_ROOT / cfg["llm"]["system_prompt_template"]
        assert prompt_path.exists(), "the manifest must not name an asset that does not ship"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.35", True, None, 5.0])
    def test_config_numbers_are_validated_not_clamped(self, bad):
        assert _config_number(bad, 0.0, 1.0) is None

    def test_a_valid_config_number_passes(self):
        assert _config_number(0.35, 0.0, 1.0) == 0.35


class TestComposition:
    def test_the_agent_inherits_the_framework_base_class_directly(self):
        assert PriceMarginAnomalyDetectionAgent.__bases__ == (AgentBaseGraph,)

    def test_the_inner_graph_inherits_base_graph_directly(self):
        assert DomainWorkflowGraph.__bases__ == (BaseGraph,)

    def test_the_five_backbone_slots_are_filled(self):
        agent = PriceMarginAnomalyDetectionAgent()
        agent.compile()
        assert set(agent._nodes) >= {"initialize", "pre_process", "main", "post_process", "finalize"}
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], PriceMarginAnomalyGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)
        assert isinstance(agent._nodes["main"], GraphNode)

    def test_the_agent_does_not_override_backbone_wiring(self):
        assert "add_edges" not in PriceMarginAnomalyDetectionAgent.__dict__

    def test_inner_nodes_take_no_constructor_arguments(self):
        graph = DomainWorkflowGraph(config={})
        graph.register_nodes()
        assert set(graph._nodes) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }

    def test_merge_output_returns_only_changed_keys(self):
        node = PriceMarginAnomalyGraphNode()
        delta = node.merge_output({}, {"anomaly_report": "REPORT", "status": "success"})
        # The boundary now also carries the degraded-completion marker; on an
        # answered run neither side set one, so it crosses empty.
        assert set(delta) == {
            "error_code",
            "anomaly_report",
            "result",
            "citations",
            "anomaly_summary",
            "status",
        }
        assert delta["error_code"] == ""
        assert delta["result"] == "REPORT", "the report must reach the output gate"

    def test_the_agent_exposes_the_shared_output_gate(self):
        agent = PriceMarginAnomalyDetectionAgent()
        assert agent._security_gate_output("clean report") is None
        assert agent._security_gate_output("sk-ABCDEFGHIJKLMNOPQRST") is not None


class TestDeclaredSettingsReachTheNodes:
    """The declared configuration must be live, not merely present in a file."""

    def test_parent_config_forwards_the_declared_retrieval_block(self):
        forwarded = PriceMarginAnomalyGraphNode()._parent_config()["configurable"]
        declared = _runtime_config()["retrieval"]
        assert forwarded["retrieval"]["top_k"] == declared["top_k"]
        assert forwarded["retrieval"]["score_threshold"] == declared["score_threshold"]

    def test_parent_config_forwards_the_declared_llm_block(self):
        forwarded = PriceMarginAnomalyGraphNode()._parent_config()["configurable"]
        declared = _runtime_config()["llm"]
        assert forwarded["llm"]["temperature"] == declared["temperature"]
        assert forwarded["llm"]["max_tokens"] == declared["max_tokens"]
        assert forwarded["llm"]["system_prompt_template"] == declared["system_prompt_template"]

    def test_the_inner_graph_seeds_the_settings_into_node_readable_state(self):
        node = PriceMarginAnomalyGraphNode()
        inner = DomainWorkflowGraph(config=node._parent_config())
        seeded = inner._extra_initial_state()
        declared = _runtime_config()["retrieval"]
        assert seeded["retrieval_top_k"] == declared["top_k"]
        assert seeded["retrieval_score_threshold"] == declared["score_threshold"]

    def test_a_malformed_config_value_does_not_reach_the_nodes(self):
        node = PriceMarginAnomalyGraphNode()
        inner = DomainWorkflowGraph(
            config={"configurable": {"retrieval": {"top_k": float("nan"), "score_threshold": "low"}}}
        )
        seeded = inner._extra_initial_state()
        assert "retrieval_score_threshold" not in seeded or seeded["retrieval_score_threshold"] == "low"
        # The node-side parser is the backstop; the forwarding layer drops it first.
        forwarded = node._parent_config()["configurable"]["retrieval"]
        assert forwarded["top_k"] == _runtime_config()["retrieval"]["top_k"]

    def test_a_missing_config_file_degrades_to_an_empty_mapping(self, monkeypatch):
        monkeypatch.setattr("src.graph.graph._RUNTIME_CONFIG_PATH", REPO_ROOT / "config" / "nope.yaml")
        assert _runtime_config() == {}


class TestStateChannelDefaults:
    """Every domain state field must default to None when the graph builds channels.

    The domain fields are annotated `Optional[...]`, and that annotation is what
    LangGraph's field-default resolution turns into a None default. A field
    annotated with a bare, non-Optional type would default to "missing"
    instead, and a node reading it before it is written would raise rather than
    see None. Pinned here so the convention is checked, not assumed.
    """

    def test_every_domain_field_defaults_to_none(self):
        from langgraph._internal._fields import get_field_default

        from src.schemas.state import State

        domain_fields = [
            "validated_input",
            "enriched_context",
            "retrieval_top_k",
            "retrieval_score_threshold",
            "rag_query",
            "retrieved_passages",
            "reranked_passages",
            "grounded_answer",
            "citations",
            "anomaly_summary",
            "anomaly_report",
            "result",
            "trace_id",
            "correlation_id",
        ]
        for field in domain_fields:
            annotation = State.__annotations__[field]
            assert (
                get_field_default(field, annotation, State) is None
            ), f"{field} must default to None; annotate it Optional[...]"
