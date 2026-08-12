# Stage F Bounded Diagnostic Capability Design

## Authorization boundary

Stage E produced a **conditional synthetic-only** result rather than a real-corpus quality go decision. The user has explicitly approved Stage F only under that recorded limitation. Consequently, this stage implements **deterministic, local, dry-run diagnostic modules** around the frozen DCSS core. It does not make a capability, autonomy, natural-language-quality, safety, consciousness, or external-deployment claim.

## Capability matrix

| Module | Stage F implementation | Explicit boundary |
|---|---|---|
| Episodic memory | Bounded compact records, explicit thresholded writes, deduplication, LRU eviction, sparse candidate retrieval, and provenance. | No hidden-state history and no silent per-token writes. |
| Retriever | Local document index using token-inverted candidates with source ID and character offsets. | No network, external document fetch, or unprovenanced response. |
| Tool registry | Typed local arithmetic and local lookup tools with schema, timeout, permission, and dry-run controls. | No shell, network, file mutation, payment, account, post, delete, or external action. |
| Planner / executor | Deterministic typed plans with schema/precondition/budget validation, dry-run execution, cancellation, and ordered hashed audit events. | No self-directed planning; only explicitly supplied typed actions execute. |
| Verifier | Safe arithmetic checks, citation/provenance checks, and explicit `UNKNOWN` abstention. | It cannot convert missing evidence into a verified pass. |
| Orchestrator | Composition wrapper which visibly labels retrieved, generated, and verified results. | It does not alter DCSS parameters or mandatory core execution semantics. |

## Preregistered diagnostic thresholds

| Gate | Requirement |
|---|---:|
| Optional core path | Core DCSS output/state equality with and without injected capability modules: exact within `1e-6`. |
| Memory boundedness | Capacity eight records; retrieval scores only a sparse candidate set; deterministic LRU eviction. |
| Memory provenance | Retrieved source identifier, offset, namespace, timestamp/index, retention policy, and content hash all present. |
| Retrieval | Needle retrieval returns correct provenance; contradiction is surfaced as conflict rather than silently resolved. |
| Tool safety | Invalid schemas and unapproved permissions reject before execution; all registered tools default to dry-run. |
| Timeout / cancellation | Timeout simulation and cancellation terminate deterministically with a recoverable audit state. |
| Planning | Cycles, missing preconditions, unregistered tool actions, and over-budget plans reject before executor invocation. |
| Verification | True arithmetic passes, false arithmetic fails, ambiguous/evidence-free claims return `UNKNOWN`. |
| Injection containment | Retrieved and tool-output instruction strings remain untrusted data and cannot alter registry permissions or trigger action. |
| Auditability | Every write, retrieval, tool validation/invocation, plan validation/execution, and verification result has an ordered event ID and hash. |

## Frozen test fixtures

The retrieval corpus consists only of committed local records, including a known needle, a contradicted pair of claims, and adversarial prompt-injection strings. The tool suite contains a deterministic arithmetic operation and a deterministic local lookup. A `mutating_demo` tool is registered only to prove that its permission fails under the global dry-run policy; it never executes an external side effect.

## Capability status policy

All modules can at most receive **Experimental** status after this stage. No module is eligible for internal-pilot or deployment status because no external side effects, real document corpus, real task distribution, or human usability study is included. The final manifest must retain `external_side_effects_enabled: false` and record all known limitations and rollback conditions.
