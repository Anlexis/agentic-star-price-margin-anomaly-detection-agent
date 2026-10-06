"""AgentCore Platform v1.0"""

# Package-root export: the manifest entry point resolves through this package,
# so `import src.graph` exposes the agent class here. The manifest's `class:`
# field carries the full dotted path (src.graph.graph.<Class>); this re-export
# additionally keeps `src.graph.PriceMarginAnomalyDetectionAgent` importable
# for callers that address the package root.
from .graph import PriceMarginAnomalyDetectionAgent

__all__ = ["PriceMarginAnomalyDetectionAgent"]
