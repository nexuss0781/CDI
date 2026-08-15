# CDI Dominant Performance Upgrade Report

## Executive verdict

The CDI repository now contains the validated performance-upgrade stack on the canonical `master` branch at commit `315efd3`. The work preserves the existing DCSS-CDI equations, factorized state layout, checkpoint contracts, Colab/Drive entry points, and model parameter count. The complete regression suite passes: **305 tests passed**.

The engineering and correctness gates pass, but the strict dominant-throughput gate does **not** yet pass on the eager CPU benchmark. Eager CDI improved from the recorded pre-upgrade baseline of approximately **3,273 / 2,788 / 2,547 token-positions per second** at lengths 16 / 64 / 256 to **3,924 / 3,547 / 3,348**, corresponding to approximately **1.20× / 1.27× / 1.31×** improvement. The acceptance target required at least 2× eager CDI and thresholds of 8,000 / 8,000 / 6,000 token-positions per second. Training should therefore remain blocked under the previously approved policy until the eager gate is revised or another kernel generation closes the remaining gap.

The compiled fixed-shape path is materially stronger: it measured **8,462 / 8,216 / 7,913 token-positions per second** at lengths 16 / 64 / 256, with peak RSS of approximately **0.66 GiB**, well below the 8.5 GiB operating target and the 11 GiB hard ceiling. This confirms that the exact scan is compiler-friendly, but it does not substitute for the eager acceptance gate.

## Implemented upgrades

| Upgrade | Implementation | Correctness status | Performance or memory effect |
|---|---|---:|---|
| Packed gate projection | Five small gate projections are concatenated into one differentiable `F.linear` operation and split into the original forcing, input, transport, timescale, and geometry slices. | Pass; maximum observed output error below 1e-6 and gradient error below 1e-6 in the focused probe. | Removes four projection dispatches from the dense path while preserving the original parameter layout. |
| Post-scan readout | The production dense path carries packed state through the token loop and computes the readout once over the stacked trajectory. | Pass; dense logits, state, loss, and gradients remain equivalent within repository tolerances. | Removes per-token readout work and the Python hidden-output append/readout pattern. |
| Fixed-shape fused scan | The dense deferred/disabled path uses an exact higher-order scan over time-major gate tensors and reused chunk kernels. | Pass; deferred guard, state, logit, loss, and gradient equivalence tests pass. | Compiled path reaches 7.9k–8.5k token-positions per second across the tested lengths. |
| Fused geometry correction | The scan applies the reused matrix-free geometry operator inside the recurrence, with deferred safety metrics reduced over the trajectory. | Pass; geometry allocation and safety tests pass. | Avoids rebuilding geometry per token and preserves the factorized 48-element state. |
| Exact tiled vocabulary loss | `causal_loss(return_logits=False)` accumulates target logits and log-sum-exp values over 4,096-token vocabulary tiles. | Pass; loss error was 0.0 and maximum gradient error was approximately 1.1e-8 in the focused parity probe. | Avoids materializing the full `[batch, time, vocabulary]` logits tensor during tiled training. |
| Eager/compiled routing | Eager causal loss remains on the faster validated fused Python loop; fixed-shape scan remains available to compiler-oriented forward execution. | Pass; full suite remains green. | Prevents the prototype scan operator from slowing eager training while retaining compiled acceleration. |
| Allocation audit and entry-point compatibility | The Stage E audit now distinguishes true square allocations from legal `[batch, time, state]` trajectories; `run.sh` retains Colab/Drive behavior and expected safe-route markers. | Pass; production-route and Stage E tests pass. | Preserves the training pipeline and prevents false dense-allocation failures. |

## Benchmark evidence

The canonical eager benchmark uses CPU float32, one thread, batch size two, 8 measured steps after two warm-up steps, and the 16,000-token EthioBBPE vocabulary. The compiled benchmark uses full-graph compilation with `reduce-overhead` and the same sequence lengths.

| Path | Length 16 | Length 64 | Length 256 | Peak RSS |
|---|---:|---:|---:|---:|
| Eager CDI, pre-upgrade baseline | 3,273 tok/s | 2,788 tok/s | 2,547 tok/s | approximately 0.6 GiB |
| Eager CDI, upgraded | **3,924 tok/s** | **3,547 tok/s** | **3,348 tok/s** | **0.61 GiB** |
| Compiled CDI, upgraded | **8,462 tok/s** | **8,216 tok/s** | **7,913 tok/s** | **0.66 GiB** |
| Eager CDI, tiled vocabulary loss | 3,444 tok/s | 3,180 tok/s | 2,961 tok/s | 0.63 GiB high-water RSS |

### Before and after alongside matched reference models

The following table places the **pre-upgrade CDI** and **upgraded eager CDI** results beside the matched GRUCell, fused `torch.nn.GRU`, and Transformer reference measurements from the same CPU float32 matrix benchmark family. The reference rows are included for context; the upgrade decision remains based on CDI’s own before-versus-after gate.

| Model or path | Length 16 | Length 64 | Length 256 |
|---|---:|---:|---:|
| CDI, pre-upgrade matched baseline | 2,913 tok/s | 2,786 tok/s | 2,361 tok/s |
| **CDI, upgraded eager** | **3,924 tok/s** | **3,547 tok/s** | **3,348 tok/s** |
| CDI eager improvement | **+34.7%** | **+27.3%** | **+41.8%** |
| GRUCell adapter reference | 10,080 tok/s | 7,918 tok/s | 6,134 tok/s |
| Fused `torch.nn.GRU` reference | 8,500 tok/s | 10,627 tok/s | 8,994 tok/s |
| Transformer reference | 14,415 tok/s | 16,148 tok/s | 11,309 tok/s |
| **CDI, upgraded compiled** | **8,462 tok/s** | **8,216 tok/s** | **7,913 tok/s** |

The matched matrix artifact reports CDI-to-fused-GRU ratios of approximately **0.34× / 0.26× / 0.26×** before optimization and **0.46× / 0.33× / 0.37×** after optimization when comparing upgraded eager CDI against the saved fused-GRU reference rows. The compiled CDI path reaches approximately **0.996× / 0.773× / 0.880×** of those fused-GRU reference throughputs at lengths 16 / 64 / 256, respectively. These figures show substantial progress in compiled execution but confirm that eager CDI has not yet become dominant.

The tiled path trades throughput for memory behavior because it performs multiple vocabulary projections. Its primary benefit is bounded vocabulary-logit workspace. At length 256, the full projection would represent 8.16 million float32 logits for batch two and 255 causal positions, approximately 32.6 MiB before autograd overhead. A 4,096-token tile represents approximately 2.09 million float32 logits, approximately 8.4 MiB, or about a **3.9× reduction in the vocabulary projection workspace**. The persistent CDI state remains only **192 bytes** in float32.

## Exactness and safety evidence

The focused equivalence suite passes all eight tests, including dense-mask parity, padded-mask parity, packed-gate parity, deferred-guard parity, and gradient parity. The complete repository suite passes all **305 tests**. The tiled-loss probe reports zero loss error and a maximum gradient error of approximately `1.1e-8`. The full model readiness gate remains green, including checkpoint reload equivalence, finite inference, one-step training, integrity rejection, and memory protection.

The repository contains no new dense full-state operator, no `torch.kron` production path, and no sequence-square allocation. The scan state remains factorized as three bands × four vertices × four channels. Runtime deferred guard metrics are reduced over the scan trajectory and checked outside the compiled recurrence, preserving fail-closed behavior.

## Acceptance-gate verdict

| Gate | Result | Verdict |
|---|---:|---|
| Exact equations and parameter contract | Pass | No architecture or state-layout change was introduced. |
| Focused equivalence suite | Pass | 8/8 tests passed. |
| Full regression suite | Pass | 305/305 tests passed. |
| Memory hard ceiling | Pass | Measured RSS stayed below 0.7 GiB in the final runs, far below 11 GiB. |
| Persistent state size | Pass | 192 bytes float32. |
| Compiled fixed-shape throughput | Pass for measured target | 8.46k / 8.22k / 7.91k tok/s exceeded the 8k / 8k / 6k compiled-style thresholds. |
| Eager dominant-throughput target | **Fail** | 3.92k / 3.55k / 3.35k tok/s did not reach the required 8k / 8k / 6k thresholds or 2× eager improvement. |
| Tiled-loss memory objective | Pass functionally; throughput trade-off recorded | Full vocabulary logits are avoided, but the current tile loop is slower in eager mode. |

## Training decision

The optimization work is **validated but not dominant under the strict eager acceptance contract**. The repository is safe to use for further kernel work and benchmark-driven iteration, but the previously blocked CDI training should remain blocked if the acceptance policy is unchanged. The next optimization sprint should focus on an eager fused kernel that avoids both Python dispatch and the higher-order scan’s eager overhead, preferably through a dedicated compiled extension or a stable low-level fused operator for the fixed nano state shape. The tiled loss should remain available as the memory-safe training route, with its tile size selected according to the target memory-throughput trade-off.

## Reproducibility artifacts

The implementation and regression evidence are in the canonical repository at [nexuss0781/CDI](https://github.com/nexuss0781/CDI). The key local benchmark artifacts used for this report are `results/final_performance_eager_restored/latest.json` and `results/final_performance_compiled/latest.json`. The focused probes are `scripts/test_packed_gates.py`, `scripts/probe_cdi_scan.py`, and `scripts/test_tiled_loss.py`.

## References

[1]: https://github.com/nexuss0781/CDI "Canonical CDI repository"
[2]: https://pytorch.org/docs/stable/generated/torch.compile.html "PyTorch compilation documentation"
