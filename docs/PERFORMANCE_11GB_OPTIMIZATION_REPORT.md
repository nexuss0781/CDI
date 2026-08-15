# CDI 11 GiB Kernel Optimization Report

**Branch:** `optimize-11gb-throughput`  
**Scope:** Semantics-preserving eager CPU execution improvements for the retained `dcss_residual_cdi` model.  
**Author:** **Manus AI**  
**Date:** 2026-08-15

## Executive result

The optimization removes repeated per-token construction from the CDI recurrence while preserving the existing equations, public state contract, runtime guards, and numerical equivalence tests. Under the repository’s eager diagnostic protocol, CDI improved from the retained baseline of **849.92 / 764.94 / 702.05 token-positions/s** at lengths 16 / 64 / 256 to **3,422.10 / 2,962.18 / 2,696.02 token-positions/s**. This is a **3.84–4.03× throughput improvement** over the retained CDI eager path.

The implementation remains well inside the requested memory ceiling. Peak resident memory was **0.586 GiB**, or approximately **5.3% of the 11 GiB guard**. The matched Transformer was still faster in this local eager CPU run, so this report does **not** claim that CDI has surpassed Transformer performance. The result is a validated kernel improvement and a stronger foundation for the next optimization stage, not a quality or superiority claim.

## Measurement contract

| Field | Value |
|---|---:|
| Device | CPU |
| CPU threads | 1 |
| Precision | Float32 |
| Batch size | 2 |
| Vocabulary | 16,000 |
| CDI parameters | 80,550 |
| Optimizer | AdamW, learning rate 0.01 |
| Warm-up steps | 2 |
| Measured steps | 8 |
| Memory guard | 11 GiB |
| Tokenizer fingerprint | `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |

The retained baseline contract and its original eager measurements are recorded in [`PERFORMANCE_10K_DECISION.md`](./PERFORMANCE_10K_DECISION.md) [1]. The optimized run used the same model family, tokenizer, batch size, optimizer, precision, thread count, and sequence lengths.

## Throughput results

| Model / implementation | Length 16 | Length 64 | Length 256 |
|---|---:|---:|---:|
| CDI retained eager baseline | 849.92 | 764.94 | 702.05 |
| CDI optimized eager path | **3,422.10** | **2,962.18** | **2,696.02** |
| CDI speedup | **4.03×** | **3.87×** | **3.84×** |
| Matched Transformer, optimized-run audit | 15,442.75 | 17,272.11 | 11,043.68 |
| CDI / Transformer throughput | 22.2% | 17.2% | 24.4% |

The improvement is attributable to execution overhead reduction rather than a changed parameter count or a relaxed numerical contract. The optimized CDI path still has a recurrent Python-visible token loop in eager mode, and the full tied 16,000-way vocabulary projection remains part of the measured training step. Those two items are the dominant remaining barriers to Transformer-level eager throughput.

## Implemented kernel changes

### Chunk-reused geometry operator

`MatrixFreeLaplacian.operator()` now constructs the differentiable vertex Laplacian once per active chunk. The operator remains connected to `edge_log_weights`, so optimizer gradients are preserved, but recurrent tokens reuse the same chunk-local operator rather than rebuilding it repeatedly. `apply()` still accepts the original no-argument form and validates supplied operator shapes.

### Chunk-reused generator tensors

The three-band generator parameters—base log-timescales, rotation biases, and input-injection tensors—are stacked once per chunk by `CohomodynamicCell.fused_kernel_tensors()`. The existing fused step accepts those tensors optionally, preserving compatibility for direct callers while eliminating repeated per-token `torch.stack` work in the active language-model path.

### Packed-state fused recurrence

`step_fused_stacked()` carries the three memory bands in one stacked tensor throughout the token loop. The public `CohomodynamicState` is reconstructed only after the chunk completes. This removes repeated per-token state packing and unpacking while keeping the same exact diagonal-plus-pairwise-skew update, geometry correction, energy guard, norm guard, and deferred diagnostic metrics.

### Profiling utility

`scripts/profile_cdi_hotpath.py` records CPU time, operator call counts, tensor shapes, and allocation behavior for a single eager training step. The optimized profile shows lower `aten::mm`, `aten::select`, `aten::stack`, and reshape overhead than the pre-refactor profile, while the vocabulary softmax and recurrent arithmetic remain visible as the next bottlenecks.

## Memory result

| Measurement | Result |
|---|---:|
| Peak RSS, optimized eager diagnostic | **0.586 GiB** |
| Configured guard | 11.000 GiB |
| Guard utilization | **5.3%** |
| Headroom | **10.414 GiB** |

The current diagnostic already uses constant-size recurrent state with respect to logical history. The optimization does not allocate a sequence-by-sequence attention matrix or any quadratic token-pair buffer. The 11 GiB ceiling should nevertheless remain a fail-closed process guard during future capacity, context, and vocabulary scaling.

## Validation

The complete repository regression suite passed **304 tests**. The focused performance-equivalence suite passed **7 tests**, including dense-path versus optimized-path logits, recurrent state, loss, and gradient agreement. `git diff --check` also passed.

The first local compiled rerun was not used as evidence: the environment initially lacked Python development headers, and after that dependency was installed the three-length Inductor compilation exceeded the local five-minute execution window. A clean compiled benchmark on a provisioned host is required before making any compiled-throughput claim.

## Next high-value upgrades

The next optimization tier should preserve the current correctness branch and target the two remaining measured bottlenecks. First, add a fused training-only output-loss path that computes exact cross-entropy in vocabulary tiles and does not retain full `[batch, sequence, vocabulary]` logits when the caller needs only a scalar loss. This should be exposed separately from `causal_loss()` so evaluation and diagnostic APIs retain their full-logit contract. Second, move the recurrent token loop behind a stable fixed-shape compiled or C++/CUDA kernel boundary, with compilation performed outside timed windows and explicit fallback to the validated eager kernel.

A third tier should introduce a memory planner that accounts for parameter, optimizer, activation, vocabulary-tile, and recurrent-state bytes before execution. It should select sequence microchunks and vocabulary tiles under a configurable budget such as `11 GiB × 0.85`, leaving operational headroom for the Python runtime and allocator fragmentation. Mixed precision should be added only with a numerical-equivalence gate: BF16 or FP16 activations may reduce memory, but geometry guards, state norms, and loss/gradient parity must be measured rather than assumed.

Finally, the CDI architecture should be evaluated at longer contexts where its constant-state design can plausibly offset Transformer quadratic attention costs. The correct comparison is not the current short eager diagnostic alone; it is a matched sweep reporting throughput, peak memory, quality, and scaling exponents at lengths 256, 512, 1,024, 2,048, and beyond. The repository’s own proposal defines this as the appropriate path for a falsifiable beyond-Transformer claim [2].

## References

[1]: `PERFORMANCE_10K_DECISION.md` — retained CDI performance baseline and decision record.

[2]: `../PROPOSAL_BEYOND_TRANSFORMER_FINAL.md` — DCSS-CDI architecture proposal, sparse recurrence rationale, and matched scaling gates.
