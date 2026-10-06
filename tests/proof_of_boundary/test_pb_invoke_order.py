# PB-6: Invoke Execution Order Verification
#
# Verifies BaseNode.__call__() enforces: trust gate -> node_start audit ->
# input gate -> execute() -> output gate -> node_complete audit, for every
# concrete node under src/nodes/.
#
# Also verifies the full backbone invoke order for the outer
# PriceMarginAnomalyDetectionAgent (two-layer nested graph):
#   InitializeNode -> PreProcessNode (pre_process) ->
#   PriceMarginAnomalyGraphNode (main) -> PostProcessNode (post_process) ->
#   FinalizeNode
#
# The backbone invoke uses VERIFIED_EXTERNAL caller trust - the real external
# path - never for_internal(). A VERIFIED_EXTERNAL invocation context exercises
# the same code path a real deployed caller uses: it clears the outer
# PreProcessNode trust gate (required_trust_level = VERIFIED_EXTERNAL) and
# passes through the inner ANONYMOUS domain nodes. for_internal() would not
# represent a real external caller, so it is deliberately not used.

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

# -- Template-specific constants ----------------------------------------------

# Class name of the node in the `main` backbone slot.
_MAIN_SLOT_NODE = "PriceMarginAnomalyGraphNode"

# A question whose terms overlap the cost-inversion reference passage strongly
# enough to clear the relevance floor, so the pipeline produces a grounded
# explanation citing KB-PRC-001.
#
# This string MUST equal deploy/invoke_payload.json["input"], and the caller
# contract below MUST equal its "input_context": the deployment evidence
# invoke and this test must exercise an identical payload, and
# test_payload_matches_deploy_invoke_payload asserts that so the two cannot
# drift apart.
_VALID_PAYLOAD = (
    "Why is the selling price below cost producing a negative gross margin " "cost inversion on this product?"
)

# A complete, in-bounds pricing observation: selling price under unit cost
# (cost inversion, CRITICAL), a competitor below our price (MEDIUM), and unit
# volume so the margin-at-risk aggregate is computed.
_VALID_INPUT_CONTEXT = {
    "channel": "ec",
    "top_k": 5,
    "pricing": {
        "sku": "a88421",
        "selling_price": 1180,
        "unit_cost": 1450,
        "margin_floor_pct": 22,
        "competitor_price": 1099,
        "units_sold": 5200,
    },
}

_DEPLOY_PAYLOAD_PATH = Path(__file__).resolve().parents[2] / "deploy" / "invoke_payload.json"

# -----------------------------------------------------------------------------


def _discover_node_classes() -> list[type]:
    """Import every module under src/nodes/ and collect concrete BaseNode subclasses."""
    from framework.nodes.base_node import BaseNode

    try:
        pkg = importlib.import_module("src.nodes")
    except ImportError:
        return []

    discovered = []
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
        module = importlib.import_module(modname)
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNode)
                and attr is not BaseNode
                and attr.__module__ == modname
                and not inspect.isabstract(attr)
            ):
                discovered.append(attr)
    return discovered


def _patch_domain_emit(monkeypatch):
    """Patch emit_trace_event in every domain node module (avoids audit-backend calls)."""
    for mod_suffix in (
        "pre_process_node",
        "input_validate_node",
        "retrieve_node",
        "rerank_filter_node",
        "generate_answer_node",
        "output_format_node",
        "post_process_node",
    ):
        try:
            monkeypatch.setattr(
                f"src.nodes.{mod_suffix}.emit_trace_event",
                lambda *a, **k: None,
            )
        except AttributeError:
            pass  # module not yet imported / no emit symbol; fine


class TestInvokeOrder:
    """PB-6: __call__ must run trust gate -> node_start -> input gate ->
    execute() -> output gate -> node_complete."""

    def test_call_order_for_every_node(self, monkeypatch):
        node_classes = _discover_node_classes()
        if not node_classes:
            pytest.skip("no concrete BaseNode subclasses found under src/nodes/")

        import framework.nodes.base_node as base_node_module

        failures: list[str] = []
        for node_cls in node_classes:
            order: list[str] = []
            monkeypatch.setattr(
                base_node_module,
                "emit_trace_event",
                lambda event_type, _payload, _state, _o=order: _o.append(f"event:{event_type}"),
            )

            for method_name, label in (
                ("_security_gate_input", "security_gate_input"),
                ("execute", "execute"),
                ("_security_gate_output", "security_gate_output"),
            ):
                original = getattr(node_cls, method_name)

                def spy(self, arg, _o=order, _label=label, _orig=original):
                    _o.append(_label)
                    return _orig(self, arg)

                monkeypatch.setattr(node_cls, method_name, spy)

            instance = node_cls()
            # Caller trust equals the node's required level so the trust gate
            # always passes here; the denial branch is asserted in
            # TestTrustGate below.
            state = {
                "caller_trust_level": node_cls.required_trust_level.value,
                "correlation_id": "pb6-invoke-order-test",
            }
            instance(state)

            expected = [
                "event:node_start",
                "security_gate_input",
                "execute",
                "security_gate_output",
                "event:node_complete",
            ]
            if order != expected:
                failures.append(
                    f"{node_cls.__name__}: invoke order violation.\n" f"expected: {expected}\nactual:   {order}"
                )

        assert not failures, "\n\n".join(failures)


class TestTrustGate:
    """PB-6: the trust gate in BaseNode.__call__ runs BEFORE execute() and denies
    a caller whose trust is below the node's required_trust_level."""

    def test_pre_process_denies_anonymous_caller(self, monkeypatch):
        """PreProcessNode (required VERIFIED_EXTERNAL) must refuse an ANONYMOUS caller."""
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.trust_level import TrustLevel
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
                "user_input": _VALID_PAYLOAD,
                "correlation_id": "pb6-trust-denial",
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any(
            "trust gate" in entry.lower() for entry in result.get("error_log", [])
        ), f"expected a trust-gate denial, got error_log={result.get('error_log')}"
        assert result.get("validated_input") is None, "nothing may be carried forward on denial"

    def test_pre_process_admits_verified_external_caller(self, monkeypatch):
        """The same node admits a VERIFIED_EXTERNAL caller and runs execute() to SUCCESS."""
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.trust_level import TrustLevel
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "user_input": _VALID_PAYLOAD,
                "input_context": _VALID_INPUT_CONTEXT,
                "correlation_id": "pb6-trust-admit",
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None


class TestBackboneInvokeOrder:
    """PB-6 backbone: a full Graph().invoke() runs the 5-node backbone in order.

    Backbone order: InitializeNode -> PreProcessNode (pre_process) ->
                    PriceMarginAnomalyGraphNode (main) ->
                    PostProcessNode (post_process) -> FinalizeNode
    """

    def _invoke(self, monkeypatch, input_context=None):
        _patch_domain_emit(monkeypatch)
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel
        from src.graph.graph import Graph

        agent = Graph()
        agent.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        return agent.invoke(
            _VALID_PAYLOAD,
            ctx=ctx,
            input_context=_VALID_INPUT_CONTEXT if input_context is None else input_context,
        )

    def test_backbone_invoke_succeeds_and_returns_output(self, monkeypatch):
        from framework.schemas.agent_status import AgentStatus

        result = self._invoke(monkeypatch)
        assert result.get("status") == AgentStatus.SUCCESS.value, (
            f"Expected status={AgentStatus.SUCCESS.value!r}, got: {result.get('status')!r}\n"
            f"error_log: {result.get('error_log')}"
        )
        assert result.get("output") is not None, "output must be set after a successful invoke"
        # The rendered report must be surfaced, citing the top-scoring
        # cost-inversion reference passage.
        assert "PRICE & MARGIN ANOMALY REPORT" in result["output"]
        assert "KB-PRC-001" in result["output"]

    def test_backbone_node_history_matches_expected_order(self, monkeypatch):
        result = self._invoke(monkeypatch)
        history = result.get("node_history", [])
        assert history == [
            "InitializeNode",
            "PreProcessNode",
            "PriceMarginAnomalyGraphNode",
            "PostProcessNode",
            "FinalizeNode",
        ], f"unexpected backbone node_history: {history}"

    def test_main_slot_is_the_domain_graph_node(self):
        """The `main` backbone slot must be PriceMarginAnomalyGraphNode (a GraphNode)."""
        from framework.nodes.graph_node import GraphNode
        from src.graph.graph import (
            PriceMarginAnomalyDetectionAgent,
            PriceMarginAnomalyGraphNode,
        )

        agent = PriceMarginAnomalyDetectionAgent()
        agent.compile()
        main_node = agent._nodes.get("main")
        assert main_node is not None, "main slot must be registered"
        assert isinstance(
            main_node, PriceMarginAnomalyGraphNode
        ), f"main slot must be PriceMarginAnomalyGraphNode, got {type(main_node).__name__}"
        assert isinstance(main_node, GraphNode), "main slot node must subclass GraphNode"
        assert main_node.__class__.__name__ == _MAIN_SLOT_NODE

    def test_payload_matches_deploy_invoke_payload(self):
        """The deployment payload must be exactly what PB-6 asserts yields SUCCESS.

        The deployment evidence run POSTs deploy/invoke_payload.json as the
        /invoke request body, so its `input` and `input_context` must equal the
        constants this test exercises.
        """
        assert _DEPLOY_PAYLOAD_PATH.exists(), "deploy/invoke_payload.json is required for deployment"
        body = json.loads(_DEPLOY_PAYLOAD_PATH.read_text(encoding="utf-8"))
        assert body.get("input") == _VALID_PAYLOAD
        assert body.get("input_context") == _VALID_INPUT_CONTEXT
