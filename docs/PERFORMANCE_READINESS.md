# Performance Readiness Before Training

> **Performance decision:** `BOUNDED_RUNTIME_READY`. **Quality/scale decision:** `SCALE_NOT_AUTHORIZED`.

This report evaluates execution cost before authorizing another training approach. It does not claim language quality, English fluency, long-context capability, or production performance. The retained model under review is the selected CCT-G3.4 `dcss_residual_cdi`; the rejected CCT-G3.5 fusion extension is not included in the training boundary. [1] [2]

## Audit Method

The audit used CPU float32, one PyTorch thread, the same 80,550-parameter retained residual CDI, and matched GRU and Transformer reference implementations. It measured complete optimization steps—causal loss, backward pass, and AdamW update—after two warm-up steps over eight timed repeats at sequence lengths 16, 64, and 256. Inputs were synthetic token IDs used only to isolate execution cost; these measurements are **not** quality or training evidence.

A CPU profiler was also run on three retained-CDI causal-loss forward passes at sequence length 16. The code path was inspected separately to identify the Python-token-serial recurrence, per-token state masking, fixed-contrast projection, and matrix-free geometry operations.

## Semantics-Preserving Repairs

The following changes were made without changing model equations, parameter values, state layout, tokenizer behavior, loss alignment, padding semantics, or the frozen CCT protocol:

| Repair | Preserved behavior |
|---|---|
| Dense attention-mask fast path | When every position is active, the implementation directly carries the candidate state and hidden output; padded batches retain the original state-selection and masking path. |
| Fixed contrast projection | Replaced the equivalent small `einsum` expression with batched matrix multiplication; the fixed zero-sum basis and feature ordering are unchanged. |
| Geometry scale cache | Caches the fixed maximum edge-weight scalar as a non-persistent buffer while preserving the bounded-sigmoid weight equation. |

The full post-repair regression suite passed **297 tests**. Fourteen focused performance, runtime, geometry, and residual-control tests passed, including dense-mask logit equivalence, padded-mask logit equivalence, fixed-basis feature equivalence, finite gradients, and existing state-safety guards. Peak audit RSS was **1.256 GiB**, well below the 11 GiB ceiling.

## Before-and-After Throughput

| Model and length | Before repair tok/s | After repair tok/s | Change |
|---|---:|---:|---:|
| Retained CDI, 16 | 270.9 | 327.6 | **+20.9%** |
| Retained CDI, 64 | 242.5 | 294.3 | **+21.4%** |
| Retained CDI, 256 | 247.4 | 252.0 | **+1.9%** |
| GRU, 16 | 3,667.0 | 3,635.3 | within audit variation |
| GRU, 64 | 3,603.4 | 3,570.1 | within audit variation |
| GRU, 256 | 2,628.3 | 2,622.1 | within audit variation |
| Transformer, 16 | 6,626.6 | 7,308.7 | reference variation |
| Transformer, 64 | 11,541.7 | 11,858.7 | reference variation |
| Transformer, 256 | 5,601.2 | 5,648.3 | reference variation |

The retained CDI remains materially slower than both matched baselines: after repair it reaches approximately 327.6 token positions per second at length 16 and 252.0 at length 256, compared with 3,635.3 and 2,622.1 for GRU. The performance repair is therefore useful for reducing waiting time but does **not** establish a speed advantage.

## Profiler Finding

The retained CDI profiler’s largest self-time categories were `aten::mul` at 12.74% self CPU time, `aten::addmm` at 6.38%, conversion/copy operations (`aten::_to_copy`, `aten::copy_`, and `aten::to`) totaling a substantial fraction of the hot path, and indexing/view operations. The remaining cost is structural: the model advances a Python loop over each token and performs repeated stable-band, geometry, and readout operations. The current repair removes avoidable work but does not eliminate the serial recurrence.

## Training Eligibility Decision

The performance gate passes for a **bounded, explicitly reviewed CPU training approach**: execution is finite, the memory headroom is large, semantics-equivalence tests pass, the full regression suite passes, and short-context throughput improved by approximately 21%. The quality and scale gate does not pass for a larger data or step ladder: the selected CCT-G3.4 residual CDI is `QUALITY_RECOVERY_PARTIAL` because it missed the pre-registered 2% material-quality margin, and CCT-G3.5 fusion earned `NO_FUSION_EVIDENCE`. [1] [2]

Accordingly, the following remain unauthorized: a 3,000-step run, larger corpus, English scaling, longer context, capacity expansion, throughput claims, and fluency claims. A future training cell may be prepared only for a user-reviewed, bounded quality approach that preserves the retained CCT-G3.4 model and explicitly states its limited evidence scope. It must not be presented as a scale transition.

## References

[1]: [CCT-G3.4 decision](CCT_G3_4_DECISION.md)  
[2]: [CCT-G3.5 decision](CCT_G3_5_DECISION.md)  
[3]: [Authoritative CCT checklist](../Todo.md)
