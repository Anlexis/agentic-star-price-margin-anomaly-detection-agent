"""Shared fixtures for the unit suite."""

import pytest

_DOMAIN_NODE_MODULES = (
    "pre_process_node",
    "input_validate_node",
    "retrieve_node",
    "rerank_filter_node",
    "generate_answer_node",
    "output_format_node",
    "post_process_node",
)


@pytest.fixture(autouse=True)
def _silence_audit(monkeypatch):
    """Neutralise audit emission so unit tests never reach an audit backend.

    Autouse: every node emits audit events, and a test that forgets to patch
    would otherwise depend on the ambient environment.
    """
    for module in _DOMAIN_NODE_MODULES:
        try:
            monkeypatch.setattr(f"src.nodes.{module}.emit_trace_event", lambda *a, **k: None)
        except (AttributeError, ModuleNotFoundError):
            pass
