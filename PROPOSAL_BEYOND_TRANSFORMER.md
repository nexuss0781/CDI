# CDI v3 Proposal: A Trainable, Efficient NLP Engine Beyond the Transformer

**Prepared for approval — implementation has not started.**

## Executive position

The CDI repository contains an interesting **physics-inspired and geometry-inspired research prototype**, but it is not yet an efficient language model and it does not currently justify a claim of superintelligence. The correct next step is not to add more mathematical terminology around the existing dense operators. It is to turn CDI into a falsifiable, scalable sequence-modeling system whose mathematical structure produces measurable advantages in memory, throughput, stability, long-context retention, and language quality.

I propose evolving CDI into **DCSS-CDI: Dissipative Cohomodynamic Selective State Space**, a recurrent/scan-compatible engine that preserves CDI’s strongest ideas—cochains, sparse geometry, harmonic memory, Dirac-like transport, and energy diagnostics—while replacing the dense global linear-algebra path with **sparse, factorized, state-space dynamics**. The engine will be beyond the conventional Transformer in its core sequence mechanism: no quadratic token-token attention, no dense sequence-by-sequence operator, and no per-step rebuilding of a full global matrix.

> **Important scientific boundary:** this project can aim to discover a better computational architecture for general-purpose language modeling. It cannot honestly promise superintelligence from architecture alone. Superintelligence would require advances in data, optimization, memory, planning, tool use, verification, alignment, and large-scale empirical validation. The proposal therefore defines a route toward a powerful and efficient research engine, not an unsupported guarantee of AGI.

## 1. What the current CDI engine actually is

The public description says that CDI replaces attention with Hodge-theoretic inference, uses a Dirac operator, and learns through a heat equation [0]. The implementation is differentiable and contains useful mathematical tests, but the current computational behavior is materially different from the claimed asymptotic picture.

| Current component | What the code does now | Consequence |
|---|---|---|
| State | A flat global state of size `N = n_points × spinor_dim × total_belief_dim` | State size grows rapidly with geometric and belief dimensions. |
| Dirac operator | Materializes a dense global matrix from graph-edge blocks | Local graph structure is converted into a global dense object. |
| Belief Laplacian | Builds `D²`, lifted cochain Laplacian, connection coupling, and `A²` as dense matrices | Forward application is a dense `N × N` matrix-vector product, not an `O(n)` matrix-free operation [1]. |
| Heat dynamics | Runs `K` explicit Euler steps for every token | Cost is multiplied by sequence length and heat steps; stability depends on the step size and largest eigenvalue. |
| Operator updates | Rebuilds all operator matrices after every optimizer step | Training repeatedly pays expensive graph construction and dense matrix assembly [2]. |
| Batch processing | Calls the single-sequence recurrent path once per batch item | Batching is not vectorized through the core recurrence. |
| Context | Sets `seq_len = n_points` | Context length is tied to the number of manifold points rather than being independently scalable. |
| Readout | Injects each token into the same repeated `B_0` slots, averages them, then reads out | Token-position structure and multi-scale memory are weakly represented. |
| Numerical mode | Defaults to CPU and `float64` | Useful for diagnostics, but expensive for language-model training. |
| Evaluation | WikiText-2, SciQ, and 30 handwritten questions | Adequate for a prototype smoke test, not sufficient for architecture claims. |

The main issue is therefore not that CDI lacks mathematical structure. The main issue is that the current implementation **materializes the very global dense operators that an efficient engine must avoid**. A dense `N × N` matrix requires quadratic memory in the state dimension and quadratic work per application. Rebuilding those matrices after every optimizer update makes the training path especially costly.

## 2. Proposed engine: DCSS-CDI

The proposed engine will use a **stable selective state-space recurrence over a sparse cochain graph**. It combines four ideas:

1. **Selective state dynamics** so the model can decide what to retain, forget, or route based on the current token. Selective state-space models are a strong evidence-based alternative to attention because they provide linear sequence scaling and content-dependent state updates [3].
2. **Sparse cohomological geometry** so local incidence and transport operators remain sparse and preserve the interpretation of degrees, boundaries, and cycles.
3. **Dissipative-plus-conservative physics-inspired dynamics** so the state has a controlled energy budget rather than relying on unconstrained explicit Euler updates.
4. **Multi-timescale harmonic memory** so fast lexical patterns, medium-range syntax, and slow semantic structure are represented by separate stable modes.

### 2.1 State representation

Instead of one dense global vector containing every point, spinor, and belief degree, the engine will maintain a structured state:

\[
 z_t = \left[z_t^{(-m)}, \ldots, z_t^{(0)}, \ldots, z_t^{(h)}\right],
 \qquad z_t^{(k)} \in \mathbb{R}^{r_k \times d_k},
\]

where each cochain degree has a compact channel state, `r_k` is a small number of learned or fixed modes, and `d_k` is the feature width. The total state is factorized; it is not a Kronecker-expanded dense tensor over every position and spinor coordinate.

The model will retain a sparse geometric scaffold with incidence matrix `S` and sparse edge weights `W`. The graph Laplacian is represented by its factorization:

\[
 L_g = S^\top W S,
\]

and applied as `Sᵀ(W(Sx))`. The matrix is never assembled as a dense global array. Cochain maps will be stored as sparse or low-rank maps between adjacent degrees.

### 2.2 Stable selective dynamics

For token embedding `x_t`, a small gating network produces input-dependent forcing and timescales:

\[
 u_t = \phi(W_u x_t), \qquad
 \tau_t = \operatorname{softplus}(W_\tau x_t) + \tau_{\min},
\]

\[
 B_t = B_0 \operatorname{diag}(g_t), \qquad
 C_t = C_0 \operatorname{diag}(q_t),
\]

where `g_t` and `q_t` are bounded gates. The generator is parameterized as

\[
 A_t = -R_tR_t^\top - \Lambda_t + \Omega_t,
\]

where `R_tR_tᵀ` and `Λ_t` are positive semidefinite dissipative terms and `Ω_t` is skew-symmetric. This separates **decay and mixing** from **energy-preserving transport**. The gate may alter the input, output, and timescale, but it cannot arbitrarily make the dynamical system unstable.

The baseline discretization will be the bilinear/Cayley update:

\[
 \left(I - \frac{\Delta t}{2}A_t\right)z_t
 = \left(I + \frac{\Delta t}{2}A_t\right)z_{t-1}
 + \Delta t\,B_tu_t.
\]

For the first implementation, `A_t` will be diagonal-plus-low-rank or block-sparse so the update can be evaluated in linear time in the feature width. A matrix-free preconditioned solve may be added later for larger structured blocks. This is preferable to using explicit Euler as the only update because it makes the stability condition testable and reduces sensitivity to an unknown maximum eigenvalue.

### 2.3 CDI-specific geometric field

The state update will include a sparse geometric field term:

\[
 z_t \leftarrow z_t - \alpha_t\,\mathcal{L}_{\text{sparse}}(z_t)
                  + \beta_t\,\mathcal{T}_{\text{cochain}}(z_t),
\]

where `L_sparse` is a matrix-free graph/cochain Laplacian and `T_cochain` is a low-rank transport operator across selected degrees. The implementation will enforce the cochain identity structurally where possible:

\[
 \delta_{k+1}\delta_k = 0.
\]

The preferred construction is to derive learnable maps from a fixed sparse incidence skeleton and learn only admissible edge weights and low-rank feature transforms. A penalty will remain as a diagnostic, but correctness should not depend entirely on an optimizer discovering an exact algebraic identity.

### 2.4 Multi-timescale harmonic memory

The engine will use three memory bands:

| Band | Purpose | Mechanism |
|---|---|---|
| Fast | Character, token, and local syntax | Short decay constants and local sparse transport. |
| Middle | Phrase, sentence, and discourse features | Intermediate decay constants with degree-1 cochain routing. |
| Slow/harmonic | Topics, entities, and persistent facts | Long decay constants and a low-dimensional approximately harmonic subspace. |

The bands will share the same sparse geometric scaffold but use different stable spectra. This is the central CDI hypothesis: **language structure can be represented more efficiently by a controlled hierarchy of dynamical modes than by repeatedly comparing every token with every other token**. This hypothesis must be tested rather than assumed.

A small optional content-addressed memory may be added after the base recurrence works. It will use sparse top-`k` retrieval over compressed memory slots, never a full `L × L` attention matrix. This protects exact recall for names and rare facts without sacrificing linear streaming behavior.

### 2.5 Parallel training and constant-state inference

Training will expose both a recurrent API and a chunk-parallel API. The recurrence will be written in an associative-scan-compatible form wherever the selected dynamics permit it, following the practical lesson of modern state-space models: parallel training and recurrent inference should be separate execution modes [3] [5].

At inference time, the model will carry only its compact state, sparse memory summaries, and optional top-`k` memory slots. It will not retain the full token history. At training time, it will process chunks with vectorized batch operations and checkpointed backpropagation.

## 3. Required repository changes

The implementation should be a **v3 branch or feature flag**, not an immediate destructive rewrite. The current mathematical engine and tests will remain available as `legacy_v2`, while the new engine is introduced behind a clean interface.

| Area | Required change | Acceptance condition |
|---|---|---|
| Parameter system | Convert raw learnable tensors into `torch.nn.Module` and `torch.nn.Parameter` objects with `state_dict()` support. | Checkpoints, device movement, mixed precision, and optimizer grouping work without custom tensor lists. |
| Geometry | Add sparse incidence graphs and matrix-free `apply(x)` methods. Cache topology and index structures only. | No dense `N × N` operator is created in the forward path. |
| Cohain structure | Replace dense per-edge and dense lifted maps with sparse/low-rank degree-to-degree operators. | `δ²` is structurally zero for the fixed topological part and numerically monitored for the learned feature part. |
| Dynamics | Add `SelectiveCohomodynamicSSM` with stable generator parameterization and bilinear/Cayley or exact diagonal updates. | Norm and energy behavior are bounded under zero input and controlled under bounded input. |
| Memory | Add fast, middle, and harmonic state bands with configurable mode counts. | Long-context tests can increase sequence length without increasing recurrent state memory linearly with history. |
| Engine | Add `CDIEngineV3` with `forward_chunk(x, state=None) -> (y, state)` and `step(x, state)`. | Batch, chunk, and token-by-token execution produce equivalent outputs within tolerance. |
| Training | Remove mandatory `rebuild_operators()` after every optimizer step. Recompute only inexpensive parameter-dependent local terms during forward. | Per-step wall time and peak memory are measured before and after the change. |
| Precision/device | Support `float32`, `bfloat16` where available, CPU, and CUDA when available. Retain `float64` for theorem diagnostics. | Numerical diagnostics and training kernels use separate precision policies. |
| Tokenizer | Make vocabulary, special tokens, padding, sequence length, and embedding initialization explicit. Resolve the README/training tokenizer inconsistency. | The same tokenizer configuration is reproducible from a saved checkpoint. |
| Data pipeline | Add deterministic train/validation/test splits, streaming datasets, packed sequences, and configurable context lengths. | No evaluation example can enter training or fine-tuning data. |
| Evaluation | Replace the 30-question hand test as the primary claim with standardized language-model and long-context evaluations. | Every reported number includes dataset version, token count, parameter count, hardware, and seed. |
| Diagnostics | Preserve spectral gap, energy, harmonic dimension, cochain residual, gradient flow, and operator symmetry where mathematically applicable. | Diagnostics do not slow the main training path and are clearly labeled as mathematical checks, not intelligence proofs. |

## 4. What will not be claimed

The project should not claim that a spectral gap, Euler convergence, harmonic dimension, or topological index is an intelligence metric until the metric has demonstrated predictive validity across independent tasks. A theorem about a numerical operator is not a theorem about cognition.

Likewise, the project should not claim “beyond Transformer” merely because it uses a different vocabulary. The claim will be restricted to measurable properties: **attention-free core sequence processing, subquadratic or linear scaling in sequence length, compact streaming state, trainability, and competitive quality under matched compute and parameter budgets**.

Mamba demonstrates that input-dependent state-space parameters can recover content-sensitive behavior while retaining linear sequence scaling [3]. Hyena demonstrates that long convolutions and data-controlled gating can replace attention with subquadratic computation [4]. RWKV demonstrates a trainable architecture with parallelizable training and recurrent inference [5]. Samba shows the value of hybridizing selective state compression with a small recent-memory mechanism for long-context recall [6]. DCSS-CDI should be compared against these families rather than presenting them as irrelevant competitors.

## 5. Implementation sequence after approval

### Phase A — Reproducible baseline

First, freeze the current v2 behavior, add a deterministic benchmark harness, record parameter count, throughput, peak memory, forward scaling, gradient norms, and language-model loss, and make the existing test suite runnable in a clean environment. This phase may fix packaging and test dependencies but will not alter the v2 algorithm.

### Phase B — Sparse operator substrate

Second, implement sparse incidence/cochain maps and matrix-free graph operators. The first target is functional equivalence on small random systems: sparse application must agree with the existing dense operator within a declared tolerance. Dense construction will remain available only as a reference oracle for small dimensions.

### Phase C — Stable selective recurrence

Third, implement `SelectiveCohomodynamicSSM` with diagonal-plus-low-rank stable generators, bounded input-dependent gates, three memory bands, and both token-step and chunk-scan execution. This is the first point at which a new language-model engine exists.

### Phase D — NLP training integration

Fourth, connect the new engine to the tokenizer and packed dataset pipeline, add causal next-token loss, checkpoint/resume, gradient accumulation, validation, and mixed-precision support. The v2 engine will remain as a matched baseline.

### Phase E — Ablation and scale study

Fifth, compare the following variants under matched parameter count and training tokens: a small Transformer, legacy CDI v2, sparse ungated CDI, selective CDI without geometry, selective CDI without harmonic memory, and full DCSS-CDI. This prevents the project from attributing gains to geometry when they actually come from gating, normalization, or parameter count.

### Phase F — Long-context and capability expansion

Only after the base language model is stable should the project add optional sparse episodic memory, retrieval, tool interfaces, planning modules, or self-verification. These are separate capabilities; they should not be hidden inside the sequence operator or described as consequences of the physics-inspired core.

## 6. How the engine will be tested

### 6.1 Mathematical and numerical tests

The sparse substrate will be checked against dense reference implementations at small sizes. Tests will verify shape preservation, sparse/dense agreement, symmetry where intended, positive semidefinite dissipation, cochain composition, gauge/transport consistency where retained, and preservation of the harmonic subspace.

The dynamical system will be tested under zero input, bounded constant input, impulses, alternating inputs, and random bounded inputs. The tests will check that the state remains finite, that the dissipative component does not increase the declared energy, that the conservative component preserves energy up to numerical tolerance, and that long rollouts do not silently diverge.

### 6.2 Training tests

The model must overfit a tiny synthetic corpus, pass a one-batch gradient check, and produce finite gradients for every intended parameter group. Chunk-parallel and token-recurrent execution must agree. Saving and restoring a checkpoint must reproduce the same next-token logits under the same seed.

The training harness will include tests for causal masking by construction, padding exclusion from the loss, deterministic shuffling, mixed-precision loss scaling, gradient clipping, and resume-from-checkpoint behavior.

### 6.3 Scaling tests

Sequence lengths will be swept across at least `256, 512, 1,024, 2,048, 4,096, 8,192`, subject to hardware. For each length, the benchmark will report forward time, training-step time, peak memory, tokens per second, and state bytes per token. Log-log regression will estimate the empirical scaling exponent.

| Proposed gate | Target for prototype acceptance |
|---|---:|
| Forward memory scaling exponent | ≤ 1.20 over the measured range |
| Forward time scaling exponent | ≤ 1.25 over the measured range |
| No dense sequence-length square tensor | Required by code inspection and runtime tracing |
| Streaming state growth with history length | Approximately constant for fixed model width |
| Numerical non-finites in a standard run | Zero |
| Recurrent/chunk output disagreement | ≤ `1e-5` in `float32` reference mode |
| Speed versus legacy dense CDI | At least 4× at a context where both can run |
| Speed versus matched small Transformer | Target ≥ 2× at 4k context; report if not met |
| Peak memory versus matched small Transformer | Target ≤ 60% at 4k context; report if not met |

These are **engineering gates, not guaranteed results**. If the architecture fails them, the design will be revised or rejected rather than renamed.

### 6.4 Language quality tests

The first quality study will use a fixed tokenizer and matched parameter budgets. It will report validation perplexity or bits-per-byte, loss versus training tokens, convergence speed, and downstream task results. Candidate datasets should include WikiText-2 for continuity, WikiText-103 or another larger language-model corpus for scale, and a clean held-out corpus for leakage control.

Long-context behavior will be tested with synthetic associative recall, passkey/needle retrieval, entity tracking, document continuation, and length extrapolation. Standardized task suites will be preferred over hand-written questions. The evaluation will distinguish **memorization**, **retrieval**, **composition**, and **reasoning** rather than merging them into one score.

### 6.5 Ablations

The study must isolate the contribution of each claimed innovation. At minimum, it will remove selective gates, remove sparse geometry, remove the harmonic band, replace the stable discretization with explicit Euler, replace structural cochain maps with unconstrained maps, and vary the memory width. Each ablation will use the same optimizer, data, parameter budget, and training token count.

## 7. Final goal of the evolution

The final goal is to produce a **research-grade, efficient, trainable general-purpose sequence engine** with the following properties:

| Goal | Definition of success |
|---|---|
| Efficient | Linear or near-linear sequence scaling, compact streaming state, and substantially lower memory than full attention at long context. |
| Trainable | Stable gradients, reproducible convergence, checkpoint/resume, mixed precision, and vectorized chunk training. |
| Geometric | Sparse cochain structure, interpretable degree-wise state, measurable harmonic and transport diagnostics, and no unnecessary dense operator materialization. |
| Capable | Competitive language modeling and strong long-context retention at matched parameter and compute budgets. |
| Extensible | Clear interfaces for retrieval, tools, planning, verification, and multimodal inputs without contaminating the core recurrence. |
| Scientifically honest | Every claim tied to a benchmark, baseline, ablation, hardware report, and reproducible configuration. |

The intended long-term architecture is therefore not “a magic equation that becomes superintelligent.” It is a **stable cognitive field substrate**: an efficient sequence processor that can maintain persistent state, route information across multiple temporal scales, expose interpretable internal structure, and later support memory, planning, tool use, and verification. Those higher-level capabilities must be built and tested separately.

## 8. Approval requested

I recommend approving implementation in the staged order **A → B → C → D**, with no destructive rewrite and no superintelligence claim during the prototype period. The first implementation deliverable after approval will be a working sparse/stable DCSS-CDI prototype plus a benchmark report against legacy CDI v2 and a small Transformer baseline.

Approval should mean permission to introduce the new **physics-inspired, efficient NLP engine** as an experimental v3 path while preserving v2 as a reference implementation. The go/no-go decision for continued development will be made from measured scaling, stability, and language-quality results—not from the elegance of the theory.

## References

[0]: https://github.com/nexuss0781/CDI "CDI repository and current architecture"

[1]: https://github.com/nexuss0781/CDI/blob/master/cdi/operators/laplacian.py "CDI dense belief Laplacian implementation"

[2]: https://github.com/nexuss0781/CDI/blob/master/cdi/engine.py "CDI engine rebuild and recurrent forward path"

[3]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

[4]: https://proceedings.mlr.press/v202/poli23a.html "Hyena Hierarchy: Towards Larger Convolutional Language Models"

[5]: https://aclanthology.org/2023.findings-emnlp.936/ "RWKV: Reinventing RNNs for the Transformer Era"

[6]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling"
