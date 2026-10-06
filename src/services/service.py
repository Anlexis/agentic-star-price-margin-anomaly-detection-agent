"""AgentCore Platform v1.0"""

# Service layer: domain queries, external API wrappers, data aggregation.
# Must NOT contain business logic, routing, or credentials.
# Nodes call this; this calls shared/services/ for external integrations.
#
# Contract: RET-C2-016 runs a fully offline pipeline - RetrieveNode scores the
# bundled pricing-policy knowledge base and the anomaly rules operate on the
# caller's own observation, so no external domain service is called. `Service`
# is the seam where a real deployment wires its pricing-policy store or its
# merchandising system client. `fetch()` therefore has a concrete no-op
# contract: it returns an empty result set rather than raising, so a caller
# that does reach it degrades gracefully instead of erroring.

from __future__ import annotations

from typing import Any


class Service:
    """Domain data-access seam for RET-C2-016 (offline build; no external service)."""

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return domain data for the query.

        The bundled build has no external source, so this returns an empty
        result set. A deployment that wires a real pricing-policy store
        replaces the body and keeps the signature.
        """
        return {"query": query, "results": [], "context": context or {}}
