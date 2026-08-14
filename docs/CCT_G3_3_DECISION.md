# CCT-G3.3 Decision: Harmonic-Memory-Band Contribution

> **Decision:** `EARNED_HARMONIC_EVIDENCE`. **Global quality status:** `REDESIGN_BEFORE_SCALE`. This is a bounded mechanism result under the frozen CCT-G3 protocol. It does **not** authorize scaling, a quality claim against GRU, longer context, capacity changes, corpus expansion, throughput work, or fluency claims.

## Decision Summary

The submitted CCT-G3.3 Colab result satisfies every pre-registered mechanism gate. Full CDI had lower token-weighted held-out validation loss than its exact parameter-preserving harmonic-disabled counterpart in every seed. The harmonic-disabled control retained the harmonic module parameters and readout slots but deterministically zeroed the harmonic state path; full CDI therefore isolates the predictive contribution of the active harmonic 16–64 time-constant band under the current 16-token, reset-state protocol. [1]

| Gate | Result | Evidence |
|---|---|---|
| Complete finite five-model, three-seed matrix | **PASS** | All 15 records are present and finite. |
| Full CDI learning | **PASS** | Training loss decreased for full CDI in seeds 11, 29, and 47. |
| Parameter fairness | **PASS** | Full CDI, harmonic-disabled CDI, and geometry-free CDI each have 80,510 parameters; maximum five-model spread is 0.49%, below the 1% limit. |
| State and gradient diagnostics | **PASS** | Fixed-held-out traces are finite; harmonic-disabled state norm, energy, and harmonic gradient are zero throughout its diagnostic trace. |
| Harmonic contribution | **PASS** | Full CDI has lower held-out validation loss than harmonic-disabled CDI in every seed. |
| Geometry re-confirmation | **PASS** | Full CDI has lower held-out validation loss than geometry-free CDI in every seed. |
| Transformer tolerance | **PASS** | Full CDI is 0.126% lower in mean validation loss than the matched Transformer. |
| Per-seed GRU quality relation | **FAIL** | Full CDI remains above GRU validation loss in every seed. |
| Host-memory guard | **PASS** | Peak recorded host memory is 2.51995 GiB, below the 11 GiB ceiling. |

## Harmonic-Band Result

| Seed | Full CDI validation loss | Harmonic-disabled validation loss | Full-CDI improvement |
|---:|---:|---:|---:|
| 11 | 6.818117 | 6.840314 | 0.022197 |
| 29 | 6.885216 | 6.943294 | 0.058078 |
| 47 | 6.839128 | 6.853095 | 0.013967 |
| **Mean** | **6.847487** | **6.878901** | **0.031414** |

The positive difference is repeated across all three declared seeds. Under the pre-registration, this earns **`EARNED_HARMONIC_EVIDENCE`** and supports retaining the harmonic band in the current bounded CDI configuration. It does not establish a long-context benefit, because the protocol uses 16-token chunks and resets recurrent state between chunks. [1]

## Re-confirmed Geometry Result

| Seed | Full CDI validation loss | Geometry-free validation loss | Full-CDI improvement |
|---:|---:|---:|---:|
| 11 | 6.818117 | 6.845003 | 0.026886 |
| 29 | 6.885216 | 6.899266 | 0.014050 |
| 47 | 6.839128 | 6.841657 | 0.002529 |
| **Mean** | **6.847487** | **6.861976** | **0.014489** |

CCT-G3.3 independently re-confirms the CCT-G3.1 sparse-geometry finding under the five-model matrix. The value remains mechanistic evidence, not a scale authorization. [2]

## Matched Aggregate Results

| Model | Parameters | Mean validation loss | Mean test loss | Mean validation accuracy | Mean tokens/sec |
|---|---:|---:|---:|---:|---:|
| `dcss_cdi` | 80,510 | 6.847487 | 6.877744 | 0.088383 | 219.5 |
| `dcss_harmonic_disabled` | 80,510 | 6.878901 | 6.909749 | 0.087804 | 308.5 |
| `dcss_geometry_free` | 80,510 | 6.861976 | 6.894169 | 0.087355 | 263.4 |
| `gru_baseline` | 80,120 | 6.800372 | 6.828304 | 0.088686 | 1,769.1 |
| `transformer` | 80,172 | 6.856126 | 6.880668 | 0.091536 | 2,892.5 |

The matched GRU remains the binding quality reference. Full CDI is 0.6928% above GRU in mean validation loss, and it is above GRU in each seed. Its causal serial implementation also remains slower than both baselines. These findings preserve the global decision `REDESIGN_BEFORE_SCALE`.

## Reproducibility Record

| Field | Submitted value |
|---|---|
| Run status | `COMPLETE` |
| Result format | `dcss-cdi-cct-g3-3-harmonic-ablation-v1` |
| Result fingerprint | `8178477fa9730455b81ffa7b08e982d89f97594d8920a12489823cfad0be3ae2` |
| Code revision | `4a8402d2e234b4d45a721ac108eda9dab8b56a4e` |
| Dataset | `Nexuss0781/synaxarium`; 321 documents; MIT asserted by source card |
| Data manifest | `947d152f129f2fd91433fa9b64574c674f9aea8e472ef3cb4fd7dafa5a2bd0d9` |
| Tokenizer artifact | EthioBBPE fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds | `[11, 29, 47]` |
| Training budget | 1,000 steps/model/seed; 30,000 causal positions/model/seed |
| Context and batches | Chunk length 16; batch size 2; 32 chunks/document |
| Optimizer and precision | AdamW; learning rate 0.01; CPU float32 |
| Training/evaluation | Deterministic per-epoch shuffle; all held-out validation and test batches |
| Environment | Python 3.12.13; PyTorch 2.11.0+cpu |
| Host-memory guard | 11 GiB maximum; 2.51995 GiB peak |

## Boundary and Next Action

CCT-G3 now has repeated held-out evidence for the three tested CDI-specific components: fixed contrast readout, sparse geometry, and the harmonic memory band. The current full configuration must retain those components for any later reviewed comparison.

However, the full CDI configuration still fails the pre-existing GRU quality relation in every seed. The 3,000-step ladder, larger corpus, context ladder, capacity changes, speed work, and production or fluency claims remain blocked. **No new experiment is authorized automatically by this decision.** The next action is a user-reviewed, separately pre-registered architecture-selection or quality-recovery proposal that preserves the established component evidence and directly addresses the GRU gap.

## References

[1]: [CCT-G3.3 pre-registration](CCT_G3_3_PREREGISTRATION.md)  
[2]: [CCT-G3.1 decision](CCT_G3_1_DECISION.md)  
[3]: [CCT-G3.2 decision](CCT_G3_2_DECISION.md)  
[4]: [Authoritative CCT checklist](../Todo.md)
