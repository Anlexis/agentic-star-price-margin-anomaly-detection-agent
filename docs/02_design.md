# Template Design Specification — RET-C2-016 Price & Margin Anomaly Detection Agent

## Position in the framework architecture

| Aspect | Value |
|---|---|
| Agent class | `PriceMarginAnomalyDetectionAgent` (`src/graph/graph.py`) |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Pattern | Cat 2 — a domain pipeline behind the fixed outer backbone |
| State | flat `TypedDict` composition (never a model object — msgpack incompatible); dict/list fields are JSON-serialised strings |
| Node | framework inheritance; a node overrides `execute(self, state) -> dict` and nothing else |
| Graph | composition — `register_nodes()` substitutes nodes; the nested inner graph rides in the `main` slot via `GraphNode` |

## Domain context

A pricing analyst for a multi-channel retailer. The agent answers pricing and
margin questions grounded in a pricing-policy knowledge base, and — when the
caller supplies a pricing observation — evaluates that observation against the
anomaly rules and reports what it found.

Anomaly categories: cost inversion, margin bleed, channel price inversion,
competitor undercut, reference-price representation, and flash-sale / markdown
rule violations.

## Architecture overview

### Outer backbone (AgentBaseGraph — fixed five-node pipeline)

```
START → initialize → pre_process → main(GraphNode) → post_process → finalize → END
                                         ↓ (retry, max_retry from config/config.yaml)
                                       pre_process
```

### Inner domain workflow (DomainWorkflowGraph — linear five-node pipeline)

```
START → input_validate → retrieve → rerank_filter → generate_answer
          → output_format → END
```

### Node configuration

| Slot | Class | File | Trust | Responsibility |
|---|---|---|---|---|
| pre_process | `PreProcessNode` | `src/nodes/pre_process_node.py` | VERIFIED_EXTERNAL | trust boundary; owns the caller-data contract |
| main | `PriceMarginAnomalyGraphNode` | `src/graph/graph.py` | ANONYMOUS | wraps the inner graph; bridges `input_context` |
| post_process | `PostProcessNode` | `src/nodes/post_process_node.py` | ANONYMOUS | the external-output boundary |
| input_validate | `InputValidateNode` | `src/nodes/input_validate_node.py` | ANONYMOUS | re-applies the contract at the inner boundary; tokenises |
| retrieve | `RetrieveNode` | `src/nodes/retrieve_node.py` | ANONYMOUS | scores the pricing-policy corpus |
| rerank_filter | `RerankFilterNode` | `src/nodes/rerank_filter_node.py` | ANONYMOUS | reranks, applies the relevance floor, trims to depth |
| generate_answer | `GenerateAnswerNode` | `src/nodes/generate_answer_node.py` | ANONYMOUS | grounded explanation + anomaly findings |
| output_format | `OutputFormatNode` | `src/nodes/output_format_node.py` | ANONYMOUS | renders the external report |

`initialize` and `finalize` are the framework defaults, injected by
`super().register_nodes()`. `add_edges()` is never overridden on the outer
graph.

## The caller-data contract

`/invoke` accepts the question as `input` and structured parameters as
`input_context`. Every field is validated against explicit bounds by
`src/schemas/caller_contract.py`, which is the single source of truth and is
applied at **two** boundaries — `PreProcessNode` (outer) and
`InputValidateNode` (inner), so a direct inner-graph invocation gets the same
rules and the two cannot drift apart.

| Field | Rule | Absent |
|---|---|---|
| `channel` | inert identifier `^[a-z0-9_]{1,32}$` | `"unknown"` |
| `category` | inert identifier | no knowledge-base filter |
| `top_k` | integer 1..20 | the configured retrieval depth |
| `pricing.sku` | inert identifier (rendered into the report) | no product code in the report |
| `pricing.selling_price` | finite number 0 .. 100,000,000 | the rules needing it do not fire |
| `pricing.unit_cost` | finite number 0 .. 100,000,000 | as above |
| `pricing.competitor_price` | finite number 0 .. 100,000,000 | as above |
| `pricing.reference_price` | finite number 0 .. 100,000,000 | as above |
| `pricing.margin_floor_pct` | finite number 0 .. 95 | the margin-bleed rule does not fire |
| `pricing.units_sold` | integer 0 .. 10,000,000 | no margin-at-risk aggregate |

Three rules run through all of it:

- **Fail closed.** An invalid value stops the request from being acted on; it is
  never clamped into a "reasonable" one. Clamping is where non-finite input does
  its damage: `max(0.0, min(1.0, float("nan")))` evaluates to `1.0`, which
  installs the strictest possible bound without telling anyone. Fail-closed says
  the value is not used — it does not say the invocation terminates; which of
  the two happens is *The refusal contract* below.
- **Every number is finite and bounded before it is compared.** `float()`
  parses `"NaN"` and `"Infinity"`, a raw JSON body carries them intact, and
  every comparison against NaN is `False` — so a non-finite threshold turns the
  exact decision this agent exists to make into a silent no-op. A number must
  also already *be* a number: a numeric string means the type was lost
  upstream, and the same door admits `"NaN"`.
- **Name the field, never the value.** Rejected caller data never round-trips
  into an error log or the response.

An absent `pricing` block is not an error: the run degrades to the
knowledge-base explanation alone, and the report says the rules were not
evaluated — which is distinct from evaluating them and finding nothing.

**The context channel carries no free text.** Every accepted field is either an
inert identifier or a bounded number, and undeclared keys are ignored — so there
is no route by which free-form caller text can enter State alongside the
question. That is deliberate: a sanitiser applied only to the question channel
would leave a second door open, and closing the door is stronger than sanitising
what comes through it.

## The refusal contract

A refusal that stops the pipeline is not always the same kind of event. One
question separates them: **can the caller fix this by sending a different
request?**

### Correctable — the run COMPLETES without acting

`status = AgentStatus.SUCCESS.value`, plus an `error_code` marker in State.

| Trigger | Marker | Node |
|---|---|---|
| question absent, empty, whitespace-only, or a non-string | `EMPTY_INPUT` | `PreProcessNode` |
| question past the character ceiling | `QUESTION_TOO_LONG` | `PreProcessNode` |
| question empty after normalisation (control characters only) | `EMPTY_INPUT` | `PreProcessNode`, again at `InputValidateNode` |
| any `input_context` field failing the caller contract — non-finite, out of range, wrong type, non-inert identifier, non-mapping | `INVALID_REQUEST` | `PreProcessNode`, again at `InputValidateNode` |

Terminating on these would end the calling surface's turn and surface an
exception type in place of the reason, leaving the correction reachable only
from the audit trail. Completing lets the caller repair the request and send it
again on the same conversation.

**The marker never leaves the process.** `error_code` is a State field and is
not part of `get_output()`'s envelope. `PostProcessNode` maps it onto one of the
fixed sentences in `src/services/failure_message.py` and returns that as the
caller-facing body. Each sentence names what to correct and nothing else — it
never echoes the rejected value, names a field path, or quotes a gate message;
those stay in `error_log`.

No domain work runs on this route. `PriceMarginAnomalyGraphNode.execute()`
checks the marker before `super().execute()` and skips the inner graph
entirely, so a second, vaguer reason cannot overwrite the specific one already
settled; inside the inner graph the same check in each node carries the marker
to `output_format` untouched. `merge_output()` keeps the outer reason ahead of
any inner one for the same purpose.

### Terminating — the run is ABANDONED

`status = AgentStatus.ERROR.value`, reason in `error_log` only.

| Trigger | Node |
|---|---|
| caller trust below `VERIFIED_EXTERNAL` | framework trust gate, before `PreProcessNode` |
| instruction-override payload in the question | `PreProcessNode`, again at `InputValidateNode` |
| a credential pattern in the assembled report (either scan pass) | `PostProcessNode` |

**These do not degrade to a completed run.** A refusal is not a value to
correct, and reporting it the same way as a correctable one would read as an
invitation to reword the request until it is accepted. The credential case
withholds the report outright; the caller receives a notice that the output was
withheld and no report content.

An oversized `input_context` is refused earlier still — `src/api/server.py`
returns HTTP 413 before a graph invocation exists, so it belongs to neither
class.

### Bridging the context channel into the inner graph

`GraphNode.execute()` invokes the inner graph without forwarding the outer
state's `input_context`, so inner-node reads of it would always see `{}`.
`src/graph/context_bridge.py` closes that gap through the sanctioned subclass
hooks: `PriceMarginAnomalyGraphNode.extract_input()` stashes the context in a
`ContextVar` before the inner invoke, and
`DomainWorkflowGraph._extra_initial_state()` seeds it back into the inner
state. The `ContextVar` keeps the hand-off per thread and task, so concurrent
invocations in one process cannot see each other's context.

## Runtime configuration

`config/agent.yaml` is the registration manifest and carries no runtime values.
Every runtime parameter lives in `config/config.yaml`, which the platform
registry loads and passes as `Graph(config=...)`; the standalone server reads
the same file through `_runtime_config()`, so both deployments see identical
configuration.

| Key | Consumer |
|---|---|
| `max_retry` | the outer backbone's retry routing |
| `timeout_s` | invocation timeout |
| `retrieval.top_k` | `RetrieveNode`, `RerankFilterNode` (a caller may narrow it) |
| `retrieval.score_threshold` | `RerankFilterNode` — operator-only, no caller override |
| `retrieval.hybrid_search` | declared for a deployment wiring a real hybrid store |
| `llm.*` | forwarded for a model-backed build; the bundled build reads none of it |

Every forwarded number is validated for type, finiteness and range before it is
forwarded. An invalid value is not forwarded at all, and the consuming node
falls back to its own default — a malformed configuration file can neither
crash graph construction nor silently weaken the relevance floor.

## Anomaly rules

Computed in `GenerateAnswerNode` from the validated observation. Each rule fires
only when every input it needs survived re-validation; a rule whose input is
unusable is withheld, never computed from an unvalidated number.

| Rule | Condition | Severity |
|---|---|---|
| `cost_inversion` | selling price < unit cost | CRITICAL |
| `margin_bleed` | realised gross margin % < `margin_floor_pct` (and not inverted) | HIGH |
| `reference_price_representation` | reference price > 2 × selling price | HIGH |
| `competitor_undercut` | competitor price < selling price | MEDIUM |

**Margin at risk** = `max(0, required_price − selling_price) × units_sold`,
where `required_price = unit_cost / (1 − margin_floor_pct/100)` when a floor is
declared and `unit_cost` otherwise. The caller contract caps the floor below
100%, so the divisor stays bounded away from zero, and the result is checked
for finiteness before it is rendered.

The aggregate is rendered **only** when unit volume is known: a per-unit
shortfall is itself a raw line item, and this report does not publish those.

## The output boundary

The report's stated schema, printed in the report itself:

- monetary figures are **aggregates only, rounded to the nearest 1,000**;
  per-unit selling prices, unit costs, competitor prices and reference prices
  are never rendered;
- product codes are rendered with the fixed `SKU_` prefix.

`OutputFormatNode` **renders** on that grid. `PostProcessNode` **enforces** it
independently, in four passes:

1. **credential scan** — an API key, JWT, bearer token or password assignment
   anywhere in the report withholds the output entirely (`status=error`);
2. **verbatim caller-text redaction** — the report is assembled from
   knowledge-base content and typed caller numbers, so a verbatim embedding of
   the caller's question is a leak, not a feature;
3. **monetary precision grid** — every monetary-form token is snapped onto the
   1,000 grid, with an audit event per redaction;
4. **credential re-scan** — pass 1 is a pattern scan and pass 3 rewrites
   digits, so the invariant is re-checked on the bytes actually surfaced.

Monetary values are identified by **form** and **currency context**, never by
magnitude: comma-grouped numbers and runs of five or more digits are monetary
by form; a bare 1–4 digit number is monetary when a three-letter uppercase code
or a currency symbol sits on either side of it, attached or separated by
horizontal whitespace, signed or unsigned. A false snap fails safe; a missed
leak does not.

Two details are load-bearing:

- **The marker-to-value delimiter stays inside one line.** A `\s*` delimiter
  spans paragraph breaks, so a three-letter uppercase word ending a line would
  bind to the number opening the next block and rewrite it — the gate
  rewriting document structure rather than protecting it.
- **Identifier guards on both ends of the match.** This report renders
  identifiers next to monetary figures. Without a guard the grammar reads any
  standalone three-letter uppercase word as a currency marker and any long
  digit run as an amount, so `KB-PRC-001` would render as `KB-PRC0` and
  `SKU_48210` as `SKU_48,000`. A monetary token may therefore not be adjacent
  to a letter, a digit, a hyphen or an underscore. The guards are
  single-character assertions, so they do not constrain the variable-width
  delimiter inside the match.

## Security posture

| Layer | Mechanism |
|---|---|
| Trust enforcement | `PreProcessNode.required_trust_level = VERIFIED_EXTERNAL`, matching the manifest |
| Entry-point auth | `src/api/server.py` requires a bearer token when `INVOKE_AUTH_TOKEN` is set; middleware-established trust is never demoted |
| Input validation | `src/schemas/caller_contract.py`, enforced at the outer and inner boundaries |
| Instruction-override refusal | the template's own guarantee, asserted by calling `execute()` directly with no wrapper in front of it |
| Output gate | module-level `_security_gate_output()` in `post_process_node.py`, also exposed on the agent class |
| Audit | `emit_trace_event()` in every node's `execute()`, including every validation decision |
| Credentials | none in State; no secrets are required (`requires.secrets: []` in the manifest) |

The instruction-override screen is deliberately narrow and narrowed further for
this domain: "rule" and "rules" are the central nouns of the knowledge base, so
an unanchored pattern would refuse ordinary pricing questions. That is the
fail-closed direction, and the one that blocks real work.

## Design decisions

| Decision | Rationale |
|---|---|
| State dict/list fields as JSON strings | checkpoints are msgpack-serialized; bare dicts corrupt silently |
| Inner nodes declared ANONYMOUS | the outer boundary enforces trust; an inner node requiring more would reject the passed-through context |
| Output gate as a module-level function | the framework auto-wraps node instance methods on the real invoke path |
| Contract module shared by both boundaries | two enforcement points, one rule set — they cannot drift apart |
| Deterministic retrieval and synthesis | the bundled build calls no model and no vector store, so the pipeline is runnable and testable as shipped |
