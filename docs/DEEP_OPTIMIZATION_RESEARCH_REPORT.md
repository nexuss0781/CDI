# CDI Deep Optimization Research Report

**Status:** `TRAINING_BLOCKED_PENDING_OPTIMIZATION`

**Repository:** `nexuss0781/CDI`

**Audited revision:** `c737da6319f253a4975829c3802f6a323eb58f11`

**Scope:** Active `cdi.v3` language-model path only. The legacy `cdi.engine` path is not used in the current CDI language benchmark and is excluded from the optimization decision.

## Executive conclusion

CDI is not currently throughput-competitive with the repository's matched GRU or Transformer baselines. The central problem is not the asymptotic state-space idea. CDI has a constant-size recurrent state and avoids a sequence-by-sequence attention matrix, but its implementation still executes a **Python-serial token loop containing hundreds of small tensor operations and autograd nodes**. The current implementation therefore fails to convert its theoretical linear-time structure into hardware-efficient throughput.

The latest matched CPU measurements show CDI at **3,273 / 2,788 / 2,547 token-positions/s** for lengths 16 / 64 / 256. The matched repository GRUCell adapter reaches **9,971 / 8,587 / 6,328**, while the tiny Transformer reaches **14,987 / 17,315 / 11,000**. A separately added fused `torch.nn.GRU` reference reaches **8,500 / 10,627 / 8,994**. CDI is therefore approximately **3.0–3.1× slower than the GRUCell adapter**, **2.5–3.8× slower than fused GRU**, and **4.6–6.2× slower than the tiny Transformer** across the measured lengths.

The decisive optimization is to replace the Python recurrence with a **single fused fixed-shape scan kernel** that combines gate projection, pairwise integration, geometry correction, readout, and recurrent-state storage. Secondary gains come from prepacking the five gate projections into one projection, moving readout out of the token loop, eliminating per-token diagnostics and temporary allocations, and implementing an exact tiled vocabulary-loss kernel. No further CDI training should begin until the fused-kernel path passes equivalence, regression, memory, and matched-throughput gates.

## 1. Current benchmark evidence

All local measurements below use CPU, one PyTorch thread, float32, batch size 2, vocabulary size 16,000, AdamW, and the active EthioBBPE tokenizer. The eager benchmark times a complete training step: forward, causal loss, backward, and AdamW update. The model sizes are intentionally matched at approximately 80k parameters.

### 1.1 Eager training throughput

| Model | Parameters | Length 16 tok/s | Length 64 tok/s | Length 256 tok/s |
|---|---:|---:|---:|---:|
| CDI `dcss_residual_cdi` | 80,550 | 3,273 | 2,788 | 2,547 |
| GRUCell adapter | 80,120 | 9,971 | 8,587 | 6,328 |
| Transformer | 80,172 | 14,987 | 17,315 | 11,000 |

| CDI relative rate | Length 16 | Length 64 | Length 256 |
|---|---:|---:|---:|
| Versus GRUCell adapter | 32.8% | 32.5% | 40.3% |
| Versus Transformer | 21.8% | 16.1% | 23.2% |

The refreshed official eager artifact is `results/deep_optimization_eager/latest.json`. The earlier official decision record reported a lower CDI eager baseline of **849.92 / 764.94 / 702.05 tok/s** because it used an older eager path before the retained kernel refactors. The current measurements should be treated as the post-refactor baseline; the older numbers remain historical evidence, not a contradictory benchmark.

### 1.2 Matched matrix with fused GRU reference

The repository's GRUCell adapter is useful for architectural continuity but is not a best-case library-kernel reference. A second benchmark therefore compares it with `torch.nn.GRU`, which maps the same width-four recurrence to the framework's fused sequence implementation.

| Model | Parameters | Length 16 tok/s | Length 64 tok/s | Length 256 tok/s |
|---|---:|---:|---:|---:|
| CDI | 80,550 | 2,913 | 2,786 | 2,361 |
| GRUCell adapter | 80,120 | 10,080 | 7,918 | 6,134 |
| Fused `torch.nn.GRU` | 80,120 | 8,500 | 10,627 | 8,994 |
| Tiny Transformer | 80,172 | 14,415 | 16,148 | 11,309 |

The fused GRU is faster than the Python GRUCell adapter at lengths 64 and 256, demonstrating that recurrence alone is not the problem. **Kernel mapping and operation fusion are the problem.** CDI has more structured dynamics than GRU, but it currently exposes those dynamics as many independent small operations instead of one hardware-friendly recurrent kernel.

The raw matrix artifact is `results/deep_optimization_matrix/latest.json`, and the reproducible benchmark source is `benchmarks/deep_optimization_matrix.py`.

### 1.3 Compiled CDI evidence

The retained fixed-shape compiled benchmark previously measured **3,448.83 tok/s** at length 16 under the canonical protocol, with 0.979 GiB peak RSS. A refreshed short diagnostic with only one warmup and three measured steps reached **7,433 tok/s** at length 16 and 0.555 GiB peak RSS, but this is not directly comparable to the canonical eight-step result. The attempt to compile lengths 16, 64, and 256 in one run was terminated by the available execution window before producing a complete artifact.

| Compiled evidence | Length | CDI tok/s | Protocol note |
|---|---:|---:|---|
| Canonical retained result | 16 | 3,449 | Three warmups, eight measured steps |
| Refreshed short diagnostic | 16 | 7,433 | One warmup, three measured steps; not a replacement gate |
| Multi-length compile attempt | 16/64/256 | Incomplete | Compile overhead exceeded local execution window |

The correct conclusion is that compilation can help, but **current compilation is not yet a dependable long-length production path**. Compile time, graph capture, and fixed-shape restrictions must be measured separately from steady-state step throughput.

### 1.4 Real-corpus matched pilot

The repository's governed real-corpus pilot used three seeds, 30 documents, document-isolated splits, 5 training steps, 16-token chunks, and the shared tokenizer and optimizer contract. It is a small diagnostic, not a language-quality claim.

| Model | Mean validation loss | Mean test loss | Mean tok/s | Peak RSS |
|---|---:|---:|---:|---:|
| CDI | 9.6276 | 9.6322 | 2,228 | 0.523 GiB |
| GRUCell adapter | 9.5926 | 9.5995 | 5,787 | 0.523 GiB |
| Transformer | 9.4716 | 9.4875 | 6,775 | 0.523 GiB |

The pilot decision was `REDESIGN_BEFORE_SCALE`: CDI learned, but did not match the GRU relation in any seed. This is a useful warning that a speed optimization cannot be separated from training stability and quality evaluation.

## 2. CDI execution-path dissection

The active path is:

```text
input IDs
  -> four-dimensional embedding
  -> five batched gate projections
  -> Python token loop
  -> three-band exact pairwise recurrence
  -> dense four-vertex Laplacian correction
  -> per-token state readout
  -> stacked hidden outputs
  -> tied 16,000-way projection
  -> causal cross-entropy
  -> backward through the entire token loop
```

The active state is compact: three bands × four vertices × four channels = **48 scalar state elements**, or **192 bytes** in float32. That persistent state is not the memory problem. The problem is the transient computation graph and the number of small operations used to update it.

### 2.1 Python-serial recurrence is the primary systems bottleneck

`DCSSLanguageModel._forward_active_embeddings` loops over sequence positions and calls `step_fused_stacked` once per token. Each token performs state reshaping, exponentials, trigonometric functions, elementwise products, geometry application, runtime metrics, readout, and autograd bookkeeping. The list of token outputs is stacked only after the loop.

This creates three costs:

- Python dispatch and interpreter overhead for every token.
- A long reverse-mode autograd graph with many tiny nodes.
- Poor hardware occupancy because the tensors are too small for efficient matrix kernels.

The current code is mathematically linear in sequence length but not **kernel-linear** in the hardware sense. Mamba's published speed advantage comes from a hardware-aware recurrent scan, not from writing a Python recurrence around an SSM equation [1].

### 2.2 The gate path uses five projections instead of one packed projection

`CohomodynamicCell.fused_gate_tensors` makes five separate `F.linear` calls: forcing, input gate, transport gate, timescale offset, and geometry gate. Each call concatenates weights and biases from the three bands at runtime. The code is semantically correct, but it repeatedly creates concatenated parameter views and launches multiple small linear operations.

The upgrade is straightforward: create one packed gate projection with output width equal to the sum of all gate widths, run one `F.linear`, and split the result into forcing, input, transport, timescale, and geometry slices. The nonlinearities can then be applied to contiguous slices. This should reduce dispatches and remove repeated concatenation from the forward path.

### 2.3 Per-token geometry correction is small mathematically but expensive operationally

The production path reuses one differentiable dense four-by-four Laplacian per chunk, which is much better than rebuilding it for every token. Nevertheless, every token still performs a matrix multiply for `Lx`, followed by geometry energy and state-norm calculations. The state is tiny, so these operations are dominated by launch and dispatch overhead rather than arithmetic intensity.

The geometry operator is not a full-state dense matrix and the allocation audit correctly finds no forbidden sequence-square or full-state-square allocation. The relevant optimization is therefore not “remove quadratic memory”; it is **fuse the fixed four-vertex correction into the recurrent kernel** or use a compact edge-difference form with no intermediate operator application.

### 2.4 Runtime safety metrics remain in the compiled recurrence

`step_fused_stacked` computes spectral, geometry-energy, and state-norm metrics for every token even when the compiled benchmark uses deferred guards. The guards are checked outside the compiled recurrence, but the metric reductions still contribute to the graph and backward path.

The safe optimization is a two-mode design:

- `strict`: check every token, used for equivalence and adversarial safety tests.
- `fast`: accumulate inexpensive bounds in the fused kernel and perform full diagnostics at chunk boundaries or a configurable interval, with fail-closed fallback to strict mode when a bound is approached.

Removing checks entirely is not acceptable. Moving them out of every-token execution while preserving a conservative envelope is the correct optimization.

### 2.5 Per-token readout should be moved after the scan

`step_fused_stacked` calls the readout linear layer for each token. `_forward_active_embeddings` then stacks the returned hidden vectors and performs the tied vocabulary projection after the loop. The state-to-width-four readout can instead be applied to the entire stacked state tensor in one batched operation after the scan.

This change would reduce per-token small linear calls and improve the shape seen by BLAS/Inductor. It also makes the recurrence kernel responsible only for state evolution, which is the correct abstraction for a scan implementation.

### 2.6 State storage and output accumulation create avoidable temporaries

The current path stores token outputs in a Python list and calls `torch.stack` after the loop. It also uses repeated `select`, `reshape`, `clone`, `slice`, and `copy_` operations to maintain the packed state and readout layout.

The profiler at length 64 records the following hot operations:

| Operation | Self CPU share | Calls |
|---|---:|---:|
| `aten::mm` | 7.04% | 333 |
| `aten::mul` | 6.61% self / 9.64% total | 2,782 |
| `aten::select` | 4.31% | 2,593 |
| `aten::copy_` | 3.77% | 2,645 |
| `aten::slice` | 3.50% | 413 |
| `aten::zeros` | 3.46% | 693 |
| `aten::clone` | 3.27% | 192 |
| `aten::sum` | 2.59% | 832 |
| `aten::_to_copy` | 2.22% | 1,693 |

The profiler reports 57.513 ms self CPU time for the profiled step. The pattern is characteristic of an allocation/dispatch-bound small-tensor kernel, not a large-matrix compute-bound model.

### 2.7 Vocabulary projection and loss are the second major scaling surface

The model produces a full `[batch, length, 16,000]` logit tensor and then computes cross-entropy. At length 64 the profiler records approximately 7.69 MB each for log-softmax forward and backward buffers. This cost scales with batch × sequence length × vocabulary and becomes significant as context or batch grows.

The correct exact optimization is a tiled vocabulary-loss path that computes the target logit and log-sum-exp in vocabulary blocks without materializing all logits, while retaining exact gradients. A sampled or adaptive softmax would change the learning objective and should not be substituted into the primary equivalence path without a separate quality protocol.

### 2.8 Compile barriers and fixed-shape restrictions remain

The official PyTorch compilation documentation states that unsupported Python code creates graph breaks and loses optimization opportunities [4]. CDI's token loop, state object manipulation, runtime branching, and safety checks make the current path difficult to compile into one efficient graph. The existing compiled wrapper is fixed-shape and full-graph, which explains both its potential and its long compile time.

A real production scan needs a fixed-shape tensor state, no Python dictionaries in the loop, no per-token `.item()` calls, and a dedicated strict/deferred guard boundary. Without those changes, `torch.compile` is an optimization experiment rather than a stable kernel strategy.

## 3. Comparison with SSM, GRU, and Transformer families

### 3.1 CDI versus conventional SSMs and Mamba

CDI shares the SSM property of carrying a bounded recurrent state, so its inference state memory is constant with respect to sequence length. However, the comparison must distinguish **algorithmic complexity** from **realized throughput**.

Mamba reports linear-time sequence modeling with input-dependent selective SSM parameters and a hardware-aware parallel algorithm for recurrent mode [1]. CDI currently has input-dependent gates and a stable exact pairwise integrator, but it does not yet have an equivalent fused scan kernel. Therefore CDI currently receives the cost of a recurrent model without receiving the hardware benefit of a fused recurrent implementation.

The correct CDI target is not to copy Mamba's architecture. It is to adopt the same systems principle: represent the recurrence as a packed fixed-shape state transition that can be scanned or fused, then benchmark it on the target CPU/GPU. Literature speedup claims must not be transplanted into CDI's result table.

### 3.2 CDI versus GRU

A GRU uses reset, update, and new gates and can be implemented as a compact recurrent transition [2][3]. The repository's GRUCell adapter is Python-serial, but PyTorch's `torch.nn.GRU` reference maps the same recurrence to a fused sequence implementation and is substantially faster at lengths 64 and 256.

CDI performs more work per token: five gate families, pairwise rotations, exponential decay, geometry correction, state norm, energy metrics, and a state readout. With width four, those operations are too small to amortize dispatch costs. CDI must therefore reduce operation count and fuse the complete transition; merely tuning learning rate or adding data will not solve the throughput gap.

### 3.3 CDI versus Transformer

A causal Transformer has quadratic attention work and memory in sequence length in the standard formulation. FlashAttention shows that exact attention wall-clock behavior is strongly determined by memory I/O and that tiling/fusion can deliver substantial speedups while retaining exact attention [5]. The tiny Transformer baseline is fast here because the sequence lengths are only 16–256, the width is four, and the implementation benefits from batched matrix operations.

The current Transformer comparison does not prove that Transformer wins at long context. It proves that CDI loses in the measured short-context CPU regime. A fair long-context study must add lengths 1,024, 4,096, and 16,384, use a memory-safe attention implementation, and report both training-step throughput and inference decode throughput.

## 4. Prioritized major optimization program

### P0 — Measurement and correctness instrumentation

**Goal:** Make every optimization attributable and reversible.

- Add labeled timers for gate projection, scan, geometry, state-to-readout, vocabulary projection, loss, backward, and optimizer.
- Record `torch.compile` graph count, graph breaks, compile time, steady-state time, and peak RSS separately.
- Add allocation counts and tensor-shape histograms to the benchmark artifact.
- Keep the current exact eager path as the correctness oracle.

**Promotion gate:** identical logits, state, loss, and gradients within the existing tolerances; no regression in the full test suite.

### P1 — Pack and fuse the gate projection

**Goal:** Replace five runtime-concatenated linear calls with one packed projection.

- Store a single packed gate weight and bias or create a compile-time packed view.
- Produce all three bands in one batched projection.
- Apply nonlinearities to contiguous slices.
- Preserve parameter fingerprints through a migration adapter if checkpoint compatibility is required.

**Expected impact:** lower dispatch count and lower Python overhead; likely meaningful at short lengths but insufficient alone for the full gap.

### P2 — Remove per-token readout and temporary allocation traffic

**Goal:** Keep the scan as a state-only operation.

- Return packed state trajectories from the scan.
- Apply the fixed state-to-width-four readout once over the complete `[B,T,...]` tensor.
- Replace Python output lists with preallocated or compiler-visible buffers.
- Eliminate per-token `select`, `clone`, `slice`, `zeros`, and `copy_` where views or fused writes are safe.

**Expected impact:** better CPU vectorization and fewer autograd nodes; likely the highest-value pure-Python refactor after the scan itself.

### P3 — Fused exact recurrence kernel

**Goal:** Replace the Python token loop with one fixed-shape scan kernel.

Candidate implementations, in increasing systems complexity:

1. A fully tensorized fixed-length `torch.compile` path with packed state and no Python dictionaries.
2. A custom C++/ATen CPU operator with forward and backward kernels, using contiguous `[B,T,W]` gate tensors.
3. A CUDA/Triton kernel for the scan, with a CPU reference path retained for equivalence testing.
4. A parallel-prefix formulation where the linear part of the stable recurrence is scanned and the input-dependent gates are fused into the scan.

The kernel must fuse gate application, pairwise decay/rotation, geometry correction, and state writes. It must expose a strict safety mode and a deferred fast mode.

**Expected impact:** largest throughput improvement and the only credible path to closing the multi-x gap.

### P4 — Fused geometry correction

**Goal:** Make the fixed four-vertex graph correction part of the recurrence kernel.

- Precompute compact edge indices and signs once.
- Compute edge differences and scatter contributions in registers or local temporaries.
- Avoid materializing a separate four-by-four operator application per token.
- Preserve the dense reference Laplacian for exact tests only.

**Expected impact:** moderate in the nano model, larger when vertices/channels grow; especially important because geometry currently adds small tensor operations to every token.

### P5 — Exact tiled vocabulary loss

**Goal:** Remove full `[B,T,V]` logit materialization without changing the objective.

- Compute vocabulary blocks with `F.linear` or a custom matmul.
- Accumulate exact log-sum-exp across blocks.
- Gather target logits from the correct block.
- Implement a custom autograd function or compile-friendly reduction so backward does not retain the entire vocabulary tensor.
- Keep the existing full-logit path as an equivalence oracle.

**Expected impact:** reduces memory growth and can improve long-context throughput; it will not eliminate the recurrent-loop bottleneck at short lengths.

### P6 — Safety and compiler boundary redesign

**Goal:** Preserve fail-closed runtime behavior without paying full diagnostic cost in every token.

- Run conservative scalar bounds inside the kernel.
- Accumulate maxima and violation bits without Python `.item()` calls.
- Check the accumulated metrics at chunk boundaries.
- Fall back to strict per-token checks for adversarial or debug mode.
- Require `fullgraph=True` in the fast benchmark and fail the benchmark if graph breaks occur.

**Expected impact:** removes graph breaks and per-token reduction overhead while preserving safety.

### P7 — Precision, optimizer, and memory policy

**Goal:** Use the 11 GiB budget to increase useful model capacity after kernel work is complete.

- GPU: benchmark bf16 autocast and fused optimizer paths after exact fp32 parity.
- CPU: benchmark float32 first; do not infer GPU performance from CPU results.
- Use gradient accumulation and activation checkpointing only when they improve the intended throughput/memory objective.
- Consider 8-bit optimizer state only after measuring optimizer share; current profiler shows optimizer time is small relative to recurrence/backward.

**Expected impact:** memory headroom and capacity scaling, not a substitute for kernel fusion.

### P8 — Long-context evaluation

**Goal:** Prove the claimed state-space advantage instead of extrapolating from 16–256 tokens.

Required lengths:

```text
16, 64, 256, 1,024, 4,096, 16,384
```

Required metrics:

- Training-step token-positions/s.
- Single-stream decode token/s.
- Peak RSS or GPU allocated/reserved memory.
- Compile time and steady-state time separately.
- State bytes per active sequence.
- Exactness and finite-value checks.
- Quality loss on a held-out corpus at each length.

## 5. Recommended implementation order and gates

| Stage | Work | Required gate |
|---|---|---|
| O0 | Add instrumentation and keep current oracle | Artifact complete; no behavior change |
| O1 | Pack gates and move readout after scan | Exact logits/loss/gradient parity; measurable dispatch reduction |
| O2 | Fused geometry and buffer layout | No forbidden allocations; state parity; no regression |
| O3 | Fixed-shape compiled or custom scan | Zero graph breaks; compile and steady-state metrics recorded |
| O4 | Exact tiled vocabulary loss | Loss/gradient parity; memory reduction at long context |
| O5 | Matched architecture re-benchmark | CDI reaches at least 2× current eager CDI and closes at least half the gap to fused GRU |
| O6 | Long-context gate | 1k/4k/16k evidence; no unmeasured extrapolation |
| O7 | Resume training review | Only after O5/O6 and quality revalidation pass |

Suggested numeric performance gates are deliberately staged rather than presented as guaranteed outcomes:

| Gate | Length 16 | Length 64 | Length 256 |
|---|---:|---:|---:|
| Current eager CDI baseline | 3,273 | 2,788 | 2,547 |
| O1 target | ≥4,500 | ≥4,000 | ≥3,500 |
| O3 target | ≥8,000 | ≥8,000 | ≥6,000 |
| O5 comparison target | ≥50% of fused GRU | ≥50% of fused GRU | ≥50% of fused GRU |

A target is not a pass merely because throughput rises. Every target requires exact equivalence, full regression, memory compliance, and complete benchmark artifacts.

## 6. Final decision

**Training remains blocked.** The current CDI implementation has a strong mathematical stability and state-memory foundation, but it is not yet an efficient sequence kernel. The immediate work should be P0–P3: instrument the path, pack gates, move readout out of the token loop, and implement a fused scan. P4 and P5 should follow because they become increasingly important as context and vocabulary budgets grow.

The comparison evidence is clear:

- CDI is state-memory efficient but dispatch-inefficient.
- GRU wins because a simple recurrence is implemented by optimized kernels.
- The Transformer wins at short context because tiny batched matrix operations dominate the quadratic attention disadvantage only at longer sequences.
- Mamba/SSM results show what a hardware-aware scan can achieve, not what an unfused Python SSM automatically achieves.

No further data or continual-learning session can correct this systems bottleneck. The next approved action should be an optimization implementation sprint with the current exact path retained as the oracle.

## References

[1]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

[2]: https://d2l.ai/chapter_recurrent-modern/gru.html "Dive into Deep Learning: Gated Recurrent Units"

[3]: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html "PyTorch GRU documentation"

[4]: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html "PyTorch torch.compile tutorial"

[5]: https://arxiv.org/abs/2205.14135 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

[6]: `results/deep_optimization_eager/latest.json` "Current eager matched benchmark"

[7]: `results/deep_optimization_matrix/latest.json` "Current matrix including fused GRU"

[8]: `results/deep_optimization_pilot/latest.json` "Current real-corpus pilot"

[9]: `results/deep_optimization_components/latest.json` "Current CDI component timing profile"

[10]: `results/deep_optimization_scaling/latest.json` "Current sequence scaling probe"
