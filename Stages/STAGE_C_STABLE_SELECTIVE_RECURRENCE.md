# Stage C — Stable Selective Cohomodynamic Recurrence

**Dependency:** Stage B must pass, including dense-reference equivalence, gradient equivalence, and the production no-dense guard.

**Status:** Specification only. No implementation is authorized by this document.

## Stage objective

Stage C introduces the first new DCSS-CDI sequence engine. It combines the Stage B matrix-free geometric field with a content-dependent, stable state-space recurrence and three explicitly separated memory bands. The stage must demonstrate that the recurrence is numerically stable, differentiable, causal, and equivalent between token-step and chunk execution before any large NLP training is attempted.

The output is a module-level engine that can process one token at a time for streaming inference and a chunk of tokens for vectorized training.

## Scope and non-goals

Stage C includes stable generator parameterization, bounded selective gates, bilinear/Cayley or exact diagonal discretization, fast/middle/harmonic memory bands, sparse geometric coupling, causal step execution, chunk execution, state serialization, and dynamical diagnostics.

Stage C does not include a tokenizer rewrite, a full language-model dataset, retrieval, tool use, planning, self-verification, or claims of language quality. A synthetic sequence task is allowed and required for functional validation, but it is not an NLP benchmark.

## State and update contract

The canonical state is a structured tuple:

\[
 z_t=(z_t^{\mathrm{fast}},z_t^{\mathrm{mid}},z_t^{\mathrm{harm}}),
\]

with each band carrying a compact feature state. The exact tensor layout must support arbitrary leading batch dimensions and must be recorded in the module documentation.

For an input embedding `x_t`, compute bounded gates and forcing:

\[
 u_t=\phi(W_ux_t),\qquad
 g_t=\sigma(W_gx_t),\qquad
 q_t=\sigma(W_qx_t),
\]

and positive timescales:

\[
 \tau_t=\operatorname{softplus}(W_\tau x_t)+\tau_{\min}.
\]

The generator must separate dissipation and conservative transport:

\[
 A_t=-R_tR_t^\top-\Lambda_t+\Omega_t,
\qquad \Omega_t^\top=-\Omega_t.
\]

The implementation may use a diagonal-plus-low-rank approximation, block-sparse approximation, or a mathematically equivalent stable form. It may not silently materialize a dense per-token state matrix.

The default discrete update is:

\[
 \left(I-\frac{\Delta t}{2}A_t\right)z_t
 =\left(I+\frac{\Delta t}{2}A_t\right)z_{t-1}
 +\Delta t B_tu_t.
\]

If the implementation uses a diagonal closed form rather than a solve, it must document the equivalence and test both the algebra and the gradients. The geometric correction is then applied through Stage B operators:

\[
 z_t\leftarrow z_t-\alpha_t\mathcal{L}_{\mathrm{sparse}}(z_t)
 +\beta_t\mathcal{T}_{\mathrm{cochain}}(z_t).
\]

The stage must specify whether geometric correction is applied before or after readout and must keep the order fixed across step and chunk paths.

## Required modules and interfaces

| Module | Required interface | Contract |
|---|---|---|
| `StableGenerator` | `parameters_from_input(x)`, `apply(z, params)`, `energy_terms(z)` | Generator has declared dissipative and conservative components. |
| `SelectiveGate` | `forward(x)` | Gates/timescales are bounded or lower-bounded as declared. |
| `CayleyIntegrator` | `step(z, u, params, dt)` | Stable discrete update with differentiable gradients. |
| `MemoryBand` | `step`, `reset`, `state_size`, `diagnostics` | Each band has its own timescale range and state shape. |
| `CohomodynamicCell` | `step(x, state)` | One causal update including sparse geometry. |
| `SelectiveCohomodynamicSSM` | `step(x, state)`, `forward_chunk(x, state=None)` | Step and chunk APIs return output and new state. |
| `StateCodec` | `pack`, `unpack`, `fingerprint` | State can be serialized without losing dtype/device semantics. |
| `DynamicsDiagnostics` | `energy`, `norms`, `spectral_estimates`, `gate_stats` | Diagnostics are detached from the hot path. |

The implementation must use `nn.Module` and `nn.Parameter` for all trainable quantities. All parameters must have explicit initialization ranges and names. A parameter inventory must include each band, gate, generator, sparse-field coupling, and readout.

## Stable parameterization requirements

The dissipative term must be positive semidefinite by construction or be accompanied by a proof/test that the chosen parameterization is equivalent. A skew-symmetric term must be constructed as `M - Mᵀ` or another exact skew-symmetric parameterization.

Gates must not be able to produce unbounded amplification. A sigmoid gate, bounded tanh gate, clipped log-timescale, or an equivalent documented mechanism is acceptable. Any clipping must expose the unclipped statistics so saturation can be detected.

The minimum timescale must be positive. The maximum timescale must be finite in the first implementation. The initialization must distribute time constants across the fast, middle, and harmonic bands rather than initialize every band identically.

The recurrence must support zero-state initialization and learned-state initialization as separate modes. The default benchmark must use zero state for equivalence and a configured learned state only after the zero-state path passes.

## Step/chunk equivalence

The chunk implementation must be causally equivalent to repeated calls to `step`. For a sequence `x[0:L]` and initial state `s0`, compare:

```text
(y_chunk, s_chunk) = forward_chunk(x, s0)
(y_step, s_step) = fold(step, x, s0)
```

The comparison must cover outputs, final states, intermediate states when requested, and gradients with respect to inputs and parameters. Random lengths, batch sizes, padding patterns, and state values must be included.

If an optimized scan path cannot exactly reproduce the recurrent path due to floating-point order, the tolerance must be measured across lengths and reported. The result must not be described as exact equivalence unless the test supports that claim.

## Evaluation harness

Expose commands equivalent to:

```text
python -m benchmarks.stage_c cell --config tiny --seed 42
python -m benchmarks.stage_c equivalence --lengths 1,2,7,32,128 --seed 42
python -m benchmarks.stage_c stability --steps 10000 --inputs zero,impulse,random
python -m benchmarks.stage_c gradients --lengths 1,8,64 --seed 42
python -m benchmarks.stage_c synthetic_memory --task delayed_copy --steps 1000
```

### Required test groups

| Test group | Procedure | Required result |
|---|---|---|
| Shape/type/device | Exercise scalar, batch, and sequence inputs across supported dtypes/devices | No shape drift or unintended promotion. |
| Causality | Perturb token `x_j` and verify outputs before `j` do not change | Zero pre-causal influence within tolerance. |
| Step/chunk | Compare recurrent fold and chunk scan | Output/state/gradient error report. |
| State serialization | Pack/unpack and continue the sequence | Continuation matches original execution. |
| Zero-input stability | Run long zero-input rollout | Finite norms and declared energy decay/bound. |
| Bounded-input stability | Drive bounded random inputs for long rollouts | No NaN/Inf; state norm remains within predeclared bound. |
| Impulse response | Apply one impulse then zero input | Decay profile for each memory band. |
| Conservative test | Disable dissipation and apply conservative dynamics | Energy preserved within tolerance. |
| Dissipative test | Disable conservative transport and apply decay | Energy non-increasing under zero input. |
| Gate behavior | Sweep token amplitudes and distributions | Gate/timescale bounds and saturation statistics. |
| Gradient test | Backpropagate through long sequences | Finite gradients and no silent disconnections. |
| Synthetic memory | Delayed copy, selective recall, and distractor rejection | Predeclared accuracy and retention curves. |
| Allocation guard | Trace production forward | No dense per-token state matrix. |

## Pass/fail gates

| Gate | Pass condition | Failure consequence |
|---|---|---|
| Causal correctness | No output before token `j` responds to a perturbation at `j` beyond tolerance | Stop; the model is not a valid causal engine. |
| Step/chunk agreement | `float32` output/state error ≤ `1e-5` for reference lengths, or a documented length-dependent tolerance is approved | Stop optimization of the scan path. |
| Gradient equivalence | Step and chunk gradients agree within declared tolerance | Stop; training would optimize a different function from inference. |
| Stable zero-input rollout | No non-finite state over 10,000 steps; energy is non-increasing or bounded by the declared contract | Stop and repair parameterization/integration. |
| Bounded-input rollout | No non-finite values and norm remains under the configured bound for the standard stress distribution | Stop. |
| Memory-band separation | Fast, middle, and harmonic impulse responses exhibit distinct measured timescale ranges | Fail the design or revise initialization; labels alone are insufficient. |
| State serialization | Pack/unpack continuation matches within dtype tolerance | Stop. |
| Sparse integration | Stage B production guard remains clean | Stop if dense state operators reappear. |
| Synthetic memory | Meets the declared delayed-copy and distractor-rejection thresholds | If it fails, diagnose recurrence capacity before NLP integration. |

The stage must publish stability envelopes rather than one cherry-picked rollout. The envelope must vary step size, input norm, sequence length, and random seed.

## Transition test to Stage D

Stage D may begin only after the following transition protocol passes:

```text
1. Load a Stage B topology and operator checkpoint.
2. Instantiate the recurrence in zero-state mode.
3. Pass identical random embeddings through step and chunk APIs.
4. Compare outputs, final states, and gradients.
5. Run the long-rollout stability suite for every memory band.
6. Run the allocation guard and verify no dense state matrix.
7. Serialize and restore a mid-sequence state.
8. Produce a signed Stage C manifest with all gate results.
```

The transition manifest must include the exact generator parameterization, integrator, gate bounds, timescale ranges, state layout, memory-band widths, and numerical tolerances. Stage D cannot silently change these settings while claiming to use the Stage C engine.

## Exit artifacts

Stage C exits with the stable recurrence modules, step/chunk equivalence tests, state codec, stability envelope plots or CSVs, synthetic-memory results, parameter inventory, and a reproducible transition manifest. The deliverable must identify which stability properties are guaranteed structurally and which are empirical observations.
