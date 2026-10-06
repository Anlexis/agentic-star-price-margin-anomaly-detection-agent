# Test Specification — RET-C2-016 Price & Margin Anomaly Detection Agent

## 1. Test strategy

- **Agent:** RET-C2-016 — Price & Margin Anomaly Detection Agent (Cat 2, a
  two-layer nested graph: an outer `AgentBaseGraph` backbone and an inner
  `DomainWorkflowGraph`).
- **Test types:** unit (per node, the caller contract, graph wiring, the
  configuration path) · proof-of-boundary (framework security and serialization
  contracts) · end-to-end through the real ASGI `/invoke` entry point.
- **Framework provisioning:** `framework` (`agenticstar-agentcore`) is supplied
  by the environment. The suite imports the real modules; there are no stub
  nodes.
- **Audit:** `emit_trace_event` is patched at the node module level by an
  autouse fixture in `tests/unit/conftest.py`, never through a `sys.modules`
  stub, which would break the real `shared` package the framework loads at
  import time.
- **Assertions are behavioural.** A refusal is asserted by the status it is
  contracted to produce and by nothing being carried forward, never by a
  particular gate's wording — wording changes between framework versions, the
  guarantee does not.
- **Refusals are asserted per class.** A correctable refusal must complete
  (`status=success` with an `error_code` marker); an unfixable one must
  terminate (`status=error`). Both are asserted explicitly, so a regression that
  collapsed the two into one status would fail here rather than pass quietly.

### Test file map

| File | Scope |
|---|---|
| `tests/unit/conftest.py` | autouse audit-silencing fixture |
| `tests/unit/test_caller_contract.py` | bounds, inert identifiers, the non-finite matrix, the instruction-override screen (both directions) |
| `tests/unit/test_pre_process_node.py` | the ingest boundary: refusals proved by calling `execute()` directly |
| `tests/unit/test_input_validate_node.py` | the inner boundary re-applies the same contract |
| `tests/unit/test_retrieve_and_rerank.py` | retrieval depth, category filter, relevance floor, unusable-value fallbacks |
| `tests/unit/test_generate_answer_node.py` | grounding, the anomaly rules, the margin-at-risk arithmetic |
| `tests/unit/test_output_schema_and_gate.py` | what the report renders and what the output gate enforces, both directions |
| `tests/unit/test_graph_and_config.py` | composition, and the declared configuration reaching the nodes |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | TC-06 / TC-07 — the framework gates are non-bypassable |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6 per-node and backbone invoke order, the trust gate, payload alignment |
| `tests/proof_of_boundary/test_invoke_e2e.py` | end-to-end through the real ASGI `/invoke` with bearer auth |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4 platform-SDK import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2 / PB-5 State msgpack and credential safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7 interrupt propagation (skip stub — not enabled here) |

### Canonical payload

The question and the caller contract used by the backbone invoke test are
identical to `deploy/invoke_payload.json`, and
`test_payload_matches_deploy_invoke_payload` asserts that equality so the two
cannot drift apart:

```json
{
  "input": "Why is the selling price below cost producing a negative gross margin cost inversion on this product?",
  "input_context": {
    "channel": "ec",
    "top_k": 5,
    "pricing": {
      "sku": "a88421",
      "selling_price": 1180,
      "unit_cost": 1450,
      "margin_floor_pct": 22,
      "competitor_price": 1099,
      "units_sold": 5200
    }
  }
}
```

The question's terms overlap the `KB-PRC-001` cost-inversion passage strongly
enough to clear the relevance floor, so the pipeline produces a grounded
explanation citing it; the observation is cost-inverted with a competitor below
our price, so it also produces a CRITICAL and a MEDIUM finding and a
margin-at-risk aggregate.

## 2. Framework compliance

| TC-ID | Test | Expected result | Where |
|---|---|---|---|
| TC-01 | State is a flat `TypedDict` with no model objects | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Empty or whitespace-only input is refused at the ingest boundary | `status=success` with an `error_code` marker — correctable; nothing carried forward | `test_pre_process_node.py` |
| TC-03 | No credential-shaped field in State | AST scan: 0 violations | `test_state_safety.py` |
| TC-04 | Node contract is `execute(self, state) -> dict` | signature asserted per node | `test_pb_invoke_order.py` |
| TC-06 | The default input gate cannot be overridden | raises at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-07 | The default output gate cannot be overridden | raises at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | `required_trust_level` is enforced before `execute()` | ANONYMOUS refused, VERIFIED_EXTERNAL admitted | `test_pb_invoke_order.py` |
| TC-09 | Only the ingest node requires VERIFIED_EXTERNAL; every other node is ANONYMOUS | trust levels asserted per node | per-node test modules |

## 3. Caller-data contract

| TC-ID | Test | Expected result |
|---|---|---|
| TC-10 | Every declared numeric field refuses NaN, ±Infinity, their string forms, booleans and numeric strings | field-naming error; request refused |
| TC-11 | Every declared numeric field refuses an out-of-range magnitude | field-naming error |
| TC-12 | The numeric-field inventory in the contract matches the fields the rules read | asserted directly, so a new field cannot skip its bounds rule |
| TC-13 | Caller strings that select behaviour or are rendered are locked to `^[a-z0-9_]{1,32}$` | free text refused |
| TC-14 | A rejected value is never echoed into the error | asserted on a distinctive value |
| TC-15 | One bad field refuses the whole request | nothing is dropped or clamped |
| TC-16 | An absent `pricing` block degrades to the knowledge-base answer | `status=success`, report says the rules were not evaluated |
| TC-17 | The same contract holds at the inner-graph boundary | inner node refuses the same payloads, in the same class as the outer boundary |
| TC-18 | The instruction-override screen refuses attack forms and passes ordinary pricing questions containing the same words | both directions asserted |
| TC-19 | An oversized context is refused at the entry point | HTTP 413 |
| TC-19a | A caller-contract rejection at either boundary completes rather than terminating | `status=success` with an `error_code` marker, at `PreProcessNode` and at `InputValidateNode` |
| TC-19b | An instruction-override payload at either boundary terminates rather than completing | `status=error`, no marker, nothing carried forward |

## 4. Domain behaviour

| TC-ID | Test | Expected result |
|---|---|---|
| TC-20 | A cost-inverted observation produces a CRITICAL finding | `cost_inversion` reported |
| TC-21 | A margin below the declared floor produces a HIGH finding | `margin_bleed` reported |
| TC-22 | A competitor below our price produces a MEDIUM finding | `competitor_undercut` reported |
| TC-23 | An inflated reference price produces a HIGH finding | `reference_price_representation` reported |
| TC-24 | A healthy observation produces no finding | empty findings, evaluated true |
| TC-25 | The margin-at-risk aggregate equals shortfall × volume, using the declared floor | exact arithmetic asserted |
| TC-26 | No aggregate without unit volume | withheld — a per-unit shortfall is a raw line item |
| TC-27 | A value that fails re-validation withholds its rule | the rule does not fire; nothing is clamped |
| TC-28 | Two different observations produce two different results | not a stub-only path |
| TC-29 | No passage clears the relevance floor | explicit no-coverage answer as a success outcome |
| TC-30 | The category filter and the retrieval-depth override reach the inner pipeline | citation sets differ accordingly |

## 5. Output boundary

| TC-ID | Test | Expected result |
|---|---|---|
| TC-31 | Every enumerated monetary leak form snaps onto the grid | comma-grouped; 5+ digit runs; short values in currency context; marker before or after; attached or separated by spaces, tabs or a newline; signed; fullwidth and postfix symbols |
| TC-32 | An on-grid figure is byte-identical | zero redactions |
| TC-33 | No magnitude exemption | a small off-grid value still snaps |
| TC-34 | The delimiter never spans a paragraph break | `"Currency: JPY\n\n3. Cash Position"` byte-identical |
| TC-35 | Citation ids and product codes survive byte-for-byte | `KB-PRC-001`, `SKU_48210`, `SKU_4901234567890`, `SKU_a1b2c3` |
| TC-36 | Structural tokens are untouched | counts, years, versions, severities, horizons |
| TC-37 | A hyphenated pattern is not mangled by the snap | `"SSN 123-45-6789"` byte-identical, so a pattern scan still sees it |
| TC-38 | A credential pattern withholds the whole output | `status=error`, the pattern absent from the response |
| TC-39 | Verbatim caller text in the report is redacted | `[REDACTED]`, independently of the grid pass |
| TC-40 | The renderer and the gate agree | a rendered report passes the gate with zero redactions |
| TC-41 | Raw line items never reach the report | supplied per-unit prices absent from the output |

## 6. Boundary and end-to-end

| PB-ID | Test | Expected result |
|---|---|---|
| PB-2 / PB-5 | State msgpack and credential safety | AST scan: 0 violations |
| PB-4 | No direct platform-SDK import under `src/` | AST scan: 0 violations |
| PB-6 | Per-node invoke order: trust gate → node_start → input gate → `execute()` → output gate → node_complete | asserted for every node under `src/nodes/` |
| PB-6b | Backbone order: Initialize → PreProcess → the domain graph node → PostProcess → Finalize | `node_history` asserted |
| PB-6c | The deployment payload equals the payload PB-6 asserts succeeds | asserted directly |
| PB-7 | Interrupt propagation | skipped — not enabled for this template |
| PB-E2E | The full path through the real ASGI `/invoke` with bearer auth | non-zero output, each severity path, on-grid scan; a correctable rejection (empty input, malformed or non-finite caller data, non-finite retrieval depth) returns `status=success`, while an instruction-override payload returns `status=error` |
| PB-AUTH | A missing or wrong bearer token | HTTP 401 with a generic body |

## 7. Error handling

A condition the caller can correct completes the run; one that cannot be
corrected by rewording terminates it. See *The refusal contract* in
`docs/02_design.md`.

| Condition | Handled by | Result |
|---|---|---|
| empty or over-long question | `PreProcessNode` | `status=success` with an `error_code` marker, no domain work; the body names what to correct |
| question empty after normalisation | `PreProcessNode`, again at `InputValidateNode` | as above |
| malformed or non-finite caller field | `caller_contract`, at both boundaries | `status=success` with an `error_code` marker; `error_log` names the field, never the value |
| instruction-override payload | `PreProcessNode`, again at `InputValidateNode` | `status=error`, nothing carried forward |
| oversized `input_context` | `src/api/server.py` | HTTP 413 |
| no passage clears the relevance floor | `GenerateAnswerNode` | explicit no-coverage answer, `status=success` |
| malformed `config/config.yaml` | `_runtime_config()` / `_config_number()` | the value is not forwarded; nodes use their defaults |
| credential pattern in the report | `PostProcessNode` | output withheld, `status=error` |
| off-grid monetary figure | `PostProcessNode` | snapped onto the grid, audit event emitted |
