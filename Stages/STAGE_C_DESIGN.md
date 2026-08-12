# Stage C Customized Design — Stable Selective Cohomodynamic Recurrence

## Scope

Stage C introduces the first executable **DCSS-CDI sequence engine** while preserving the Stage B invariant that production geometry is matrix-free. It implements causal token stepping, chunk execution by the same causal fold, three frequency-cascade memory bands, stable selective dynamics, sparse Laplacian coupling, state serialization, diagnostics, and a reproducible gate harness. It does not introduce a tokenizer, a language-model corpus, or a language-quality claim; those remain Stage D work.

## CPU-safe `nano` configuration

The Stage C `nano` tier uses four topology vertices, a width of four per band, and three bands. The structured recurrent state consequently contains `4 × 4 × 3 = 48` scalar elements per leading batch item, satisfying the rapid-iteration constraint without using a Kronecker-expanded state. Production recurrence and geometry use float32; float64 is reserved for algebra/reference checks.

| Parameter | Value | Rationale |
|---|---:|---|
| Vertices | 4 | Reuses the deterministic Stage B nano topology. |
| Per-band channel width | 4 | Supports pairwise conservative rotations and remains compact. |
| Structured state elements | 48 | Below the 64-element nano limit. |
| Input / output width | 4 | Enables direct synthetic-memory probes without a tokenizer. |
| Frequency bands | fast, middle, harmonic | Required explicit separated memory modes. |
| Timescale ranges | `[0.25, 1]`, `[2, 8]`, `[16, 64]` | Logarithmically separated frequency cascade. |
| Geometry step cap | `0.02` | Bounded explicit sparse-Laplacian correction for the nano graph. |

## State layout and update order

The canonical state is an immutable tuple `CohomodynamicState(fast, middle, harmonic)`. Each element has shape `(..., n_vertices, band_width)` and supports arbitrary leading batch dimensions. The update order is fixed for both `step` and `forward_chunk`:

1. Select content-dependent bounded forcing, input gate, transport gate, and log-timescale offset from the input token.
2. Apply the exact diagonal-plus-pairwise-skew discretization inside each band.
3. Apply the bounded matrix-free Stage B Laplacian correction to the updated band state.
4. Concatenate vertex-averaged band states and apply the learned readout.

The cochain transport term is intentionally set to zero in this first state layout because Stage B degree-zero cochains map vertices to edges, while every Stage C band is a vertex state. The Laplacian is the degree-preserving geometric correction. A future degree-structured state may add the cochain term only with a declared edge-state layout and new equivalence gates.

## Stable selective generator

Each band uses a diagonal dissipative component and pairwise two-dimensional skew rotation. For content-dependent parameters from a `SelectiveGate`, the continuous generator is represented without a per-token dense state matrix as

\[
A_t = -\lambda_t I + \Omega_t,
\]

where `lambda_t > 0` is formed from a bounded gate divided by a positive, finite timescale and `Omega_t` is represented by pairwise rotations. The exact closed form per pair is

\[
z_{t+1}=e^{-\Delta t\lambda_t}R(\Delta t\omega_t)z_t + \Delta t\,g_tu_t.
\]

This is algebraically the exact integration of the block-diagonal `-lambda I + Omega` homogeneous update. It avoids dense per-token matrices, guarantees non-amplifying zero-input dynamics when dissipation is active, and permits an exact conservative diagnostic when dissipation is set to zero.

## Frequency-cascade initialization

Band timescales are initialized as logarithmically spaced values within their declared intervals, rather than copied from one common initializer. Content-dependent log-timescale offsets are clipped to a finite interval and reported through diagnostics. This makes band separation a measurable gate, not a label.

## Stage C gates

| Gate | Threshold |
|---|---:|
| Causality | Maximum pre-causal output difference `≤ 1e-6`. |
| Step/chunk output and state agreement | Float32 maximum absolute error `≤ 1e-5`. |
| Step/chunk gradient agreement | Float32 maximum absolute error `≤ 1e-5`. |
| State serialization continuation | Float32 maximum absolute error `≤ 1e-6`. |
| Zero-input stability | 10,000 steps, finite state, non-increasing total band energy. |
| Bounded-input stability | 10,000 bounded random steps, finite state and norm `≤ 1e4`. |
| Conservative diagnostic | Relative energy drift `≤ 1e-5` over the diagnostic rollout. |
| Dissipative diagnostic | Zero-input energy is non-increasing within `1e-6`. |
| Memory-band separation | Median impulse retention satisfies harmonic > middle > fast at the declared probe delay. |
| Sparse allocation guard | No `torch.kron` or dense matrix with shape `(48, 48)` in the production recurrence path. |
| Geometry ablation | Both enabled and ablated paths execute; ablation emits exact zero geometry correction. |
| Synthetic memory probe | Harmonic retention at delay 16 is at least twice fast-band retention. |

## Transition discipline

A Stage C gate report must explicitly retain `stage_d_implementation_allowed: false`. Stage D remains blocked until an explicit user approval follows the Stage C review.
