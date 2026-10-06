"""AgentCore Platform v1.0"""

# RET-C2-016 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph of the Cat 2 two-layer nested architecture. It
# encapsulates the whole price & margin anomaly workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by PriceMarginAnomalyGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - _extra_initial_state() seeds the caller's input_context (context bridge)
#   - get_output() designed together with PriceMarginAnomalyGraphNode.merge_output()
#   - No platform-SDK imports (framework/ and shared/ only)
#   - Not placed under src/subagents/

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for RET-C2-016.

    Inherits BaseGraph directly for a fully custom node topology. Called by
    PriceMarginAnomalyGraphNode.get_subgraph() in graph.py, which passes the
    validated runtime settings (_parent_config()) into the constructor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - re-validate contract, tokenise
          -> retrieve        (RetrieveNode)       - score the pricing-policy corpus
          -> rerank_filter   (RerankFilterNode)   - rerank, threshold, top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded explanation + findings
          -> output_format   (OutputFormatNode)   - render the external report
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates
    with the canonical execute(self, state) -> dict signature. initialize and
    finalize are outer backbone concerns and are not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ret_c2_016_price_margin_anomaly_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across the inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate the inner graph config before compilation.

        The forwarded `retrieval` block (top_k / score_threshold) is already
        type-, finiteness- and range-validated by
        PriceMarginAnomalyGraphNode._parent_config(), and is read per call by
        the domain nodes with safe defaults, so absence is non-fatal.
        Validation is permissive here rather than raising.
        """
        pass

    # -- Config + caller-context seeding into inner state ----------------------

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed inner state with the runtime settings and the caller's context.

        Two independent hand-offs happen here:

        1. Runtime settings: PriceMarginAnomalyGraphNode._parent_config()
           forwards the validated `retrieval` block under
           config["configurable"]; this hook republishes it as SCALAR state
           fields, because the node contract is execute(self, state) - no
           config parameter, so tuning knobs travel through State.
           RetrieveNode / RerankFilterNode read retrieval_top_k and
           retrieval_score_threshold from state, falling back to their module
           defaults when unseeded.

        2. Caller context: GraphNode.execute() does not forward the outer
           state's input_context into subgraph.invoke(), so the outer graph
           stashes it in a ContextVar (PriceMarginAnomalyGraphNode
           .extract_input) and this hook reads it back - see
           src/graph/context_bridge.py. Without this, inner-node reads of
           state["input_context"] (the caller's pricing observation and the
           per-invocation retrieval overrides) would always see {}.
        """
        configurable = (self.config or {}).get("configurable", {}) or {}
        retrieval_raw = configurable.get("retrieval") or {}
        retrieval: dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
        extra: dict[str, Any] = {"input_context": get_caller_input_context()}
        if "top_k" in retrieval:
            extra["retrieval_top_k"] = retrieval["top_k"]
        if "score_threshold" in retrieval:
            extra["retrieval_score_threshold"] = retrieval["score_threshold"]
        return extra

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract. Do NOT
        register initialize or finalize; those are outer backbone concerns
        handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments: FunctionNode
        subclasses take no __init__, tuning config flows in via State (see
        _extra_initial_state() above), and execute(self, state) takes no config
        parameter. Every key registered here is referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear domain topology.

        Each step passes its partial-dict output into the shared State. The
        topology is intentionally linear - there is no conditional branching
        between domain nodes. route() is implemented as the ABC requires, but
        add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing - required by the BaseGraph ABC.

        For this linear topology add_conditional_edges() is not used, so this
        method is never called at runtime. It is implemented to satisfy the ABC
        contract, and returns END on error so an unexpected call does not
        re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return str(END)
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by PriceMarginAnomalyGraphNode.merge_output() in
        graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()   emits: "anomaly_report", "citations",
                                        "anomaly_summary", "status", ...
            Outer merge_output() reads: sub_result.get(...) for each of them

        trace_id, correlation_id and node_history are surfaced for
        observability; merge_output() maps the first four into the outer state
        delta and the rest are available for a future extension without an
        inner-graph change.
        """
        return {
            # the reason must leave the subgraph or the outer graph cannot report it
            "error_code": state.get("error_code"),
            "anomaly_report": state.get("anomaly_report"),
            "citations": state.get("citations"),
            "anomaly_summary": state.get("anomaly_summary"),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
