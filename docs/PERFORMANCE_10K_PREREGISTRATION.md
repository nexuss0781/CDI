# Performance Pre-Registration: 10,000 Token-Positions per Second

> **Objective:** raise retained CDI training throughput toward **10,000 token positions per second** while preserving causal semantics, the DCSS state trajectory, quality loss, parameter inventory, and the validated CCT evidence contract.

## Benchmark Definition

The primary benchmark is the complete CPU training step used by the CCT pilot: EthioBBPE token IDs, batch size 2, chunk length 16, causal loss, backward pass, and AdamW update. Throughput is active causal token positions divided by elapsed step time. The benchmark records one-thread CPU performance, warm-up steps, median and mean step time, peak host memory, and environment details. Secondary measurements use chunk lengths 64 and 256 to determine whether an optimization only helps the shortest case.

The 10,000 target is a **training-throughput target**, not a forward-only microbenchmark. It is compared against the same measured baseline and the matched GRU/Transformer references. A forward-only result may be reported as diagnostic data but cannot satisfy the gate.

## Current Baseline

The retained CCT-G3.4 residual CDI measured approximately 327.6 token positions per second at length 16 after the first bounded runtime repair, while GRU measured approximately 3,635.3 and Transformer approximately 7,308.7 under the same local one-thread audit. The 10,000 target therefore requires a major runtime improvement and must not be assumed reachable.

## Optimization Ladder

Only the following semantics-preserving optimizations may be attempted in this performance sprint:

| Tier | Candidate optimization | Required invariant |
|---|---|---|
| P1 | Dense four-vertex weighted Laplacian kernel using the exact factored `SᵀWS` equation | Same edge-weight gradients, state output, geometry energy, and runtime guards within float32 equivalence tolerance |
| P2 | Fused three-band selective-gate projections or equivalent batched linear algebra | Same gate tensors, generator parameters, state, readout features, and gradients |
| P3 | Fixed-length CPU graph compilation or scripted recurrence execution | Same logits, causal loss, state trajectory, and gradients; compilation overhead excluded from steady-state throughput |
| P4 | Memory-allocation and tensor-layout reductions in the token loop | Same outputs, masks, checkpoint metadata, and state serialization |

A tier is retained only if its equivalence and regression gates pass and its measured steady-state training throughput improves. Failed tiers are reverted or disabled; no optimization may alter the quality protocol or introduce a new architecture mechanism.

## Equivalence and Safety Gates

Every retained tier must satisfy all of the following:

1. Full CDI causal logits and token-weighted loss match the pre-optimization path within `atol=1e-6`, `rtol=1e-5` for CPU float32 reference fixtures.
2. All three structured state bands match within the same tolerance after every token in a fixed short sequence.
3. Active gradients remain finite and match the reference for representative trainable parameters within `atol=1e-5`, `rtol=1e-4`.
4. Padded and dense attention masks remain behaviorally equivalent.
5. Geometry quadratic-form and state-norm safety guards remain fail-closed.
6. The full repository regression suite passes with no test deletion or weakened assertion.
7. Peak host memory remains below the 11 GiB ceiling.

## Performance Decision

| Outcome | Decision |
|---|---|
| CDI reaches 10,000 training token positions/s at length 16 and all gates pass | `EARNED_10K_PERFORMANCE_EVIDENCE` |
| CDI improves materially but remains below 10,000 | `PARTIAL_PERFORMANCE_EVIDENCE`; retain only measured improvements and continue only by separate review |
| Any equivalence, gradient, stability, regression, or memory gate fails | `PERFORMANCE_REPAIR_REJECTED`; revert the failing tier |

No speed result authorizes changing the corpus, context, capacity, optimizer, precision, language domain, or quality comparison. The validated CCT-G3.6 quality result remains the reference.

## References

[1]: [Performance readiness report](PERFORMANCE_READINESS.md)  
[2]: [CCT-G3.6 decision](CCT_G3_6_DECISION.md)  
[3]: [Authoritative CCT checklist](../Todo.md)
