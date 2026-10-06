# Price & Margin Anomaly Detection Agent

AI agent for detecting retail price and margin anomalies, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Retail
> **Template ID**: RET-C2-016

## Overview

Detects and explains retail pricing anomalies — cost inversion, margin bleed, channel price
inversion, competitor undercut, inflated reference prices and flash-sale rule violations. A
question is validated and normalised, matching pricing-policy passages are retrieved, reranked
and filtered by a relevance threshold, and the explanation is assembled **only** from passages
that clear that threshold, with the knowledge-base passage ids cited. When no passage is relevant
enough, the agent says the knowledge base has insufficient coverage instead of inventing policy.

A caller may also send a pricing observation on the request's context channel — a product code,
selling price, unit cost, category margin floor, competitor price, reference price and unit
volume. Each field is validated against explicit bounds and refused outright if it is malformed
or non-finite; the anomaly rules then run on the validated numbers and the report names the rules
triggered, their severity, and the estimated margin at risk. The report deliberately publishes
**aggregates only, rounded to the nearest 1,000** — per-unit prices, costs and margins never
appear in it, and a separate output boundary enforces that grid independently of the renderer.

The bundled knowledge base is a small sample of pricing-policy reference material so the pipeline
runs and tests end-to-end out of the box; a real deployment replaces it with its own pricing
policy corpus behind the same node contract.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at
graph compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit and boundary tests
config/       registration manifest and runtime parameters
prompts/      answer-synthesis prompt for a model-backed build
docs/         design and test documentation
```

See `docs/` for the design specification and the test specification.

## Customising

1. Adjust `config/config.yaml` for your own retrieval depth and relevance floor.
2. Replace the bundled pricing-policy corpus in `src/nodes/retrieve_node.py` with your own.
3. Review the anomaly rules in `src/nodes/generate_answer_node.py` and the caller-data bounds in
   `src/schemas/caller_contract.py` against your own pricing policy.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
