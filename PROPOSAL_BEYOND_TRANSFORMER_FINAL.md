# CDI v3 Proposal: Dissipative Cohomodynamic Selective State Space

**Approval document — implementation has not started.**

## Executive recommendation

I recommend evolving CDI into **DCSS-CDI: Dissipative Cohomodynamic Selective State Space**, a physics-inspired, sparse, recurrent language engine designed to remove the current dense global-operator bottleneck while preserving CDI’s cochain, geometric, harmonic, and energy-based ideas.

The proposal is intentionally falsifiable. It does **not** claim that changing the sequence operator alone creates superintelligence. It proposes a research path toward a more efficient and trainable foundation for language, with later extensions for memory, planning, tools, and verification. Those higher-level capabilities would require independent engineering and evaluation.

The first implementation should be a non-destructive **v3 experimental path**. CDI v2 remains as a reference oracle and baseline. The project should proceed only when measured scaling, stability, and language quality justify the next phase.

## 1. What CDI currently does

The repository describes CDI as a post-neural engine based on sheaf cohomology, Dirac operators, Hodge inference, and heat-equation learning [1]. The current implementation is differentiable and has meaningful mathematical tests, but its NLP path is still a short-context prototype and its principal operators are dense.

| Current path | Actual implementation | Main limitation |
|---|---|---|
| State | Flat vector of size `N = n_points × spinor_dim × total_belief_dim` | State size grows quickly with geometric and belief dimensions. |
| Dirac operator | Builds a dense global matrix from graph-edge blocks | Local geometry is materialized globally. |
| Belief Laplacian | Forms `D²`, lifted cochain terms, connection coupling, and `A²` as dense matrices [2] | Forward application is dense `N × N` multiplication. |
| Heat dynamics | Applies `K` explicit Euler steps for every token | Per-token cost is multiplied by heat steps and depends on stability-sensitive `dt`. |
| Optimizer update | Rebuilds operator matrices after every optimizer step [3] | Training pays repeated graph construction and dense assembly. |
| Batch path | Loops over batch items and runs one sequence recurrence per item | The main recurrence is not fully vectorized. |
| Context | Sets context length equal to `n_points` | Geometry resolution and NLP context are coupled. |
| Position/readout | Repeats each token’s injection across slots, averages `B_0`, and reads out | Position-specific and multi-timescale memory are weak. |
| Runtime | Defaults to CPU and `float64` | Appropriate for diagnostics, inefficient for LM training. |
| Evaluation | WikiText-2, SciQ, and 30 handwritten questions [1] | Not sufficient for broad architecture claims. |

The central diagnosis is precise: **CDI’s current implementation is mathematically structured but not computationally sparse**. A dense `N × N` matrix has quadratic memory and matvec cost in the state dimension. The next engine must keep the topology and dynamics in factored form and apply them matrix-free.

## 2. Proposed architecture

DCSS-CDI will replace the dense global heat-flow path with a **stable selective state-space recurrence over a sparse cochain graph**. The design combines four mechanisms.

### 2.1 Factorized cochain state

The state will be represented degree by degree:

\[
 z_t = [z_t^{(-m)},\ldots,z_t^{(0)},\ldots,z_t^{(h)}],
 \qquad z_t^{(k)}\in\mathbb{R}^{r_k\times d_k}.
\]

Each degree uses a compact number of modes rather than a Kronecker-expanded dense tensor over all points, spinor coordinates, and belief channels. A fixed sparse incidence skeleton will represent the graph and degree structure.

The geometric Laplacian will be applied in factored form:

\[
 L_gx = S^\top W(Sx),
\]

where `S` is sparse incidence and `W` contains admissible edge weights. The global `L_g` matrix will not be assembled in the forward path.

### 2.2 Selective content-dependent dynamics

For token embedding `x_t`, the engine will produce bounded input-dependent gates and timescales:

\[
 u_t=\phi(W_ux_t),\qquad
 \tau_t=\operatorname{softplus}(W_\tau x_t)+\tau_{\min}.
\]

The state generator will be parameterized as

\[
 A_t=-R_tR_t^\top-\Lambda_t+\Omega_t,
\]

with positive-semidefinite dissipation `R_tR_tᵀ + Λ_t` and skew-symmetric transport `Ω_t`. This gives the model a controlled separation between decay/mixing and energy-preserving motion.

The first update will use a stable bilinear/Cayley discretization:

\[
 \left(I-\frac{\Delta t}{2}A_t\right)z_t
 =\left(I+\frac{\Delta t}{2}A_t\right)z_{t-1}
 +\Delta t B_tu_t.
\]

The initial generator will be diagonal-plus-low-rank or block-sparse, so the update remains efficient. The implementation will avoid a generic dense solve. This is a deliberate change from the current explicit-Euler-plus-dense-Laplacian path.

Selective state-space models provide the relevant external precedent: Mamba makes state parameters depend on the current input to recover content-sensitive propagation and forgetting while retaining linear sequence scaling [4]. RWKV demonstrates the complementary training/inference design in which parallelizable training can coexist with recurrent constant-state inference [6].

### 2.3 CDI geometric field

After the stable state update, the engine will apply a sparse geometric field:

\[
 z_t\leftarrow z_t-\alpha_t\mathcal{L}_{\mathrm{sparse}}(z_t)
 +\beta_t\mathcal{T}_{\mathrm{cochain}}(z_t).
\]

`L_sparse` will be a matrix-free graph/cochain Laplacian. `T_cochain` will route information across adjacent degrees using sparse or low-rank maps. Wherever possible, the cochain identity

\[
 \delta_{k+1}\delta_k=0
\]

will be guaranteed structurally by construction. A residual penalty remains useful for learned feature maps, but the optimizer should not be solely responsible for discovering the topological identity.

### 2.4 Multi-timescale harmonic memory

The engine will maintain three state bands.

| Band | Intended information | Design |
|---|---|---|
| Fast | Characters, tokens, local syntax | Short decay constants and local sparse transport. |
| Middle | Phrases, sentences, discourse | Intermediate decay constants and degree-1 routing. |
| Harmonic/slow | Topics, entities, persistent facts | Long decay constants and a compact approximately harmonic subspace. |

This is the core CDI hypothesis: useful language structure may be represented more efficiently by controlled dynamical modes than by repeated all-pairs token comparison. It must be tested against matched baselines.

An optional sparse episodic memory may be added later for rare names and exact facts. It must use compressed slots and top-`k` retrieval, not a hidden quadratic attention matrix.

### 2.5 Training and inference interfaces

The engine will expose both:

```text
forward_chunk(x, state=None) -> (y, state)
step(x, state) -> (y, state)
```

Training will use vectorized chunk execution and checkpointing. Streaming inference will carry only the compact recurrent state and optional compressed memory. The model will not retain the full token history.

## 3. Required repository changes

The changes should be introduced behind a v3 feature boundary.

| Area | Change | Acceptance condition |
|---|---|---|
| Parameters | Convert raw tensors to `nn.Module`/`nn.Parameter`; provide `state_dict()`. | Device movement, optimizer grouping, and checkpoint resume work natively. |
| Sparse geometry | Add fixed sparse incidence and matrix-free `apply(x)` operators. | No dense `N × N` operator is created in forward. |
| Cohains | Replace dense edge maps with sparse/low-rank adjacent-degree maps. | Structural cochain composition is zero up to declared numerical tolerance. |
| Dynamics | Add `SelectiveCohomodynamicSSM` with stable generator and bounded gates. | Zero-input rollouts remain finite and controlled. |
| Discretization | Add Cayley/bilinear or exact diagonal updates; retain Euler only as a baseline. | Stability is measured rather than assumed. |
| Memory | Add fast, middle, and harmonic bands. | Context length can increase without storing all prior tokens. |
| Engine | Add `CDIEngineV3` with chunk and step APIs. | Chunk and step outputs agree within tolerance. |
| Training | Remove mandatory full `rebuild_operators()` after every step. | Per-step time and peak memory improve over v2. |
| Precision | Support `float32`, `bfloat16` where available, CPU, and CUDA; keep `float64` diagnostics. | Training and theorem-checking use separate precision policies. |
| Tokenizer | Make vocabulary, special tokens, padding, context, and embedding setup explicit. | A checkpoint reproduces its tokenizer configuration. |
| Data | Add deterministic splits, streaming, packed sequences, and leakage checks. | Validation/test data never enters training. |
| Diagnostics | Preserve energy, spectral, harmonic, cochain, symmetry, and gradient diagnostics. | Diagnostics are out of the hot path and are not presented as intelligence proofs. |

The current README/training tokenizer mismatch and the custom tensor-list parameter handling should be resolved before scaling. They are reproducibility risks independent of the new architecture.

## 4. Implementation sequence

### Phase A — Freeze and benchmark v2

Preserve v2 behavior, make the clean-environment test command work, and record parameter count, throughput, peak memory, forward scaling, gradient statistics, and LM loss. This is a baseline phase, not an architecture rewrite.

### Phase B — Build the sparse operator substrate

Implement sparse incidence/cochain maps and matrix-free graph operators. For small systems, compare sparse application against the current dense implementation as a reference oracle.

### Phase C — Implement the stable selective recurrence

Add the stable generator, bounded gates, three memory bands, and token-step/chunk execution. This is the first new NLP engine.

### Phase D — Integrate NLP training

Add causal next-token loss, packed datasets, validation, checkpoint/resume, gradient accumulation, mixed precision, and reproducible configuration logging.

### Phase E — Run matched ablations

Compare a small Transformer, legacy CDI v2, ungated sparse CDI, selective CDI without geometry, selective CDI without harmonic memory, explicit-Euler CDI, and full DCSS-CDI under matched parameters and training tokens.

### Phase F — Add higher-level capabilities only after the core wins

Sparse episodic memory, retrieval, tool use, planning, self-verification, and multimodal input are separate modules. They should be added only after the sequence core demonstrates reliable scaling and language quality.

## 5. Test and evaluation plan

### 5.1 Mathematical correctness

For small random systems, verify sparse/dense agreement, shapes, intended symmetry, positive-semidefinite dissipation, cochain composition, transport consistency, and harmonic-subspace behavior. Dense matrices may exist only in these small reference tests.

### 5.2 Dynamical stability

Run zero-input, constant-input, impulse, alternating-input, and bounded-random-input tests. Measure state norms and the declared energy. The dissipative component must not increase energy under zero input; the conservative component must preserve it within numerical tolerance; long rollouts must remain finite.

### 5.3 Training correctness

The new engine must overfit a tiny synthetic corpus, pass a one-batch gradient test, produce finite gradients for intended parameter groups, preserve causality, exclude padding from loss, resume checkpoints deterministically, and match chunk versus recurrent outputs.

### 5.4 Scaling

Sweep sequence lengths such as `256, 512, 1,024, 2,048, 4,096, 8,192` where hardware allows. Record wall time, peak memory, tokens/second, state bytes, and empirical log-log scaling exponents.

| Prototype gate | Target |
|---|---:|
| Forward memory exponent | ≤ 1.20 over measured range |
| Forward time exponent | ≤ 1.25 over measured range |
| Dense sequence-length-square tensor | None in runtime trace |
| Streaming state growth | Approximately constant with history length |
| Non-finite values in standard run | Zero |
| Step/chunk disagreement | ≤ `1e-5` in `float32` reference mode |
| Speed versus legacy CDI | At least 4× where both fit |
| Speed versus matched small Transformer | Target ≥ 2× at 4k context |
| Peak memory versus matched Transformer | Target ≤ 60% at 4k context |

These are go/no-go engineering targets, not guaranteed outcomes. A failed target must be reported and trigger redesign rather than relabeling.

### 5.5 Language quality and long context

Use WikiText-2 for continuity, a larger language-model corpus such as WikiText-103 or an equivalent licensed corpus, and a clean held-out set. Report loss/perplexity or bits-per-byte, tokens seen, parameter count, optimizer, hardware, wall time, seed, and memory.

Long-context tests must distinguish associative recall, passkey/needle retrieval, entity tracking, document continuation, and length extrapolation. Use standardized suites where possible; do not rely on the current 30-question handcrafted set for major claims. Hyena shows that long convolutions plus data-controlled gating can provide a meaningful attention-free comparison point [5]. Samba shows why a hybrid of selective state compression and limited recent-memory access can improve long-context recall and throughput [7].

### 5.6 Required ablations

Remove selective gates, sparse geometry, harmonic memory, stable discretization, and structural cochain construction one at a time. Keep optimizer, data, parameter count, and training tokens fixed. This separates gains from gating, recurrence, normalization, parameter count, and geometry.

## 6. Final goal

The final goal is a **research-grade efficient general-purpose sequence substrate** with these measurable properties:

| Goal | Success definition |
|---|---|
| Efficient | Linear or near-linear sequence scaling and compact streaming state. |
| Trainable | Stable gradients, reproducible convergence, chunk-parallel training, and checkpoint/resume. |
| Geometric | Sparse cochain structure, interpretable degree-wise state, and measurable harmonic/transport diagnostics. |
| Capable | Competitive LM quality and strong long-context retention at matched compute and parameter budgets. |
| Extensible | Clean interfaces for retrieval, tools, planning, and verification. |
| Scientifically honest | Every claim tied to a baseline, ablation, dataset, hardware report, and reproducible configuration. |

The long-term vision is a **stable cognitive-field substrate**: an efficient sequence processor that preserves information across multiple timescales, exposes interpretable internal structure, and can support memory, planning, tool use, and verification. The physics-inspired mathematics is valuable only if it produces better measured behavior.

## 7. Approval requested

Approve implementation of DCSS-CDI as an experimental v3 path in the order **A → B → C → D**, while keeping CDI v2 intact. The first implementation deliverable after approval will be:

1. a sparse/matrix-free operator layer;
2. a stable selective cohomodynamic recurrence;
3. a working `CDIEngineV3` chunk/step interface; and
4. a benchmark report against legacy CDI v2 and a small Transformer baseline.

The decision to continue toward larger models will be based on measured scaling, stability, and language-quality results—not on theoretical elegance alone.

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and current architecture"

[2]: https://github.com/nexuss0781/CDI/blob/master/cdi/operators/laplacian.py "CDI dense belief Laplacian implementation"

[3]: https://github.com/nexuss0781/CDI/blob/master/cdi/engine.py "CDI engine rebuild and recurrent forward path"

[4]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

[5]: https://proceedings.mlr.press/v202/poli23a.html "Hyena Hierarchy: Towards Larger Convolutional Language Models"

[6]: https://aclanthology.org/2023.findings-emnlp.936/ "RWKV: Reinventing RNNs for the Transformer Era"

[7]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling"
