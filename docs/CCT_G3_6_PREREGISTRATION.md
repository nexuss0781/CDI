# CCT-G3.6 Pre-Registration: Bounded Quality Continuation

> **Purpose:** test whether the retained CCT-G3.4 residual CDI continues improving under a modestly larger optimization budget after performance readiness passed. This is a bounded quality approach, not CCT-G2.2 scale authorization.

## Question

CCT-G3.4 selected `dcss_residual_cdi` because it beat GRU in all three seeds at 1,000 steps, but its mean validation loss of 6.743546 did not reach the 2% material target of 6.664364. CCT-G3.5 fusion did not add value and is closed. CCT-G3.6 asks whether a **small, controlled training continuation** improves the retained model without changing its architecture or data contract.

## No Architecture Change

The model is the retained CCT-G3.4 `dcss_residual_cdi` configuration. It retains the earned fixed contrast readout, sparse geometry, harmonic memory band, and selective token residual. No fusion gate, new module, new tokenizer, new corpus, context increase, parameter increase, or optimizer change is allowed.

## Frozen Continuation Contract

| Field | Locked value |
|---|---|
| Candidate | `dcss_residual_cdi` |
| Baselines | `gru_baseline`, `transformer` |
| Corpus | `Nexuss0781/synaxarium`; 321 governed documents |
| Tokenizer | EthioBBPE 2.0.0; existing artifact fingerprint |
| Seeds | `[11, 29, 47]` |
| Total training steps | **1,500** per model/seed; 50% continuation beyond the 1,000-step CCT-G3.4 rung |
| Chunks and context | 32 chunks/document; chunk length 16; batch size 2 |
| Optimizer | AdamW; learning rate 0.01 |
| Precision and device | CPU float32 |
| Shuffle and evaluation | Deterministic per-epoch batch shuffle; all held-out batches |
| Memory guard | 11 GiB maximum host memory |
| Parameter contract | Existing matched counts; no parameter changes |
| Primary metric | Token-weighted held-out validation cross-entropy |

## Decision Gates

| Gate | Requirement |
|---|---|
| Complete evidence | All 9 model/seed records finite and complete. |
| Learning | CDI training loss decreases in all three seeds. |
| Quality relation | CDI matches or beats GRU in every seed. |
| Progress | CDI mean validation loss is no worse than the 1,000-step retained-CDI reference of 6.743546. |
| Material target | CDI mean validation loss reaches at or below 6.664364; this is reported separately and is not assumed. |
| Runtime | 11 GiB guard and finite state/gradient behavior pass. |

## Interpretation

A run that passes learning, quality relation, and progress is **bounded continuation evidence**. It does not authorize 3,000 steps, a larger corpus, longer context, capacity changes, or English-scale training. A run that reaches the material target may support a separately reviewed proposal for the next rung; it does not authorize that rung automatically.

If CDI loses to GRU in any seed, the continuation fails the quality gate and no larger training run is allowed. If CDI does not improve over 6.743546, the result shows no continuation evidence even if it remains ahead of GRU.

## References

[1]: [CCT-G3.4 decision](CCT_G3_4_DECISION.md)  
[2]: [Performance readiness report](PERFORMANCE_READINESS.md)  
[3]: [Authoritative CCT checklist](../Todo.md)
