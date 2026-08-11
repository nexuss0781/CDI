# Stage B — Sparse Incidence and Matrix-Free Cohain Operator Substrate

**Dependency:** Stage A must pass and its v2 reference artifacts must be archived.

**Status:** Specification only. No implementation is authorized by this document.

## Stage objective

Stage B replaces CDI’s dense global geometric and cochain operator construction with a sparse, factorized substrate. The stage is successful only if the new operators agree with dense reference operators on small systems while avoiding dense global matrices in normal forward execution.

The purpose is to make geometry computationally local and composable. This stage does not yet introduce selective token-dependent dynamics or a new language-model training loop. It establishes the operator layer on which Stage C will build the stable recurrent engine.

## Scope and non-goals

Stage B includes sparse graph topology, sparse incidence maps, sparse degree-to-degree cochain maps, matrix-free graph Laplacian application, admissible transport/connection application, device-aware sparse execution, and dense-reference test oracles for small dimensions.

Stage B does not include selective gates, Cayley integration, multi-timescale memory, tokenizer changes, language-model quality claims, retrieval, or planning. It may add a small operator-only benchmark, but it must not be presented as an NLP result.

## Mathematical contract

For a graph with incidence operator `S` and edge-weight operator `W`, the geometric Laplacian must be applied as

\[
\mathcal{L}_g x = S^\top W(Sx),
\]

without forming `SᵀWS` as a dense matrix in the production path. For adjacent cochain degrees, maps must satisfy the declared composition law

\[
\delta_{k+1}\delta_k = 0
\]

for the structural topological component. Learned feature maps may have a residual, but the residual must be explicitly measured and must not be confused with the topological identity.

The sparse implementation must declare whether operators are symmetric, self-adjoint, positive semidefinite, skew-symmetric, or merely bounded. Tests must verify only properties that the implementation actually promises.

## Required modules and interfaces

| Module | Required interface | Contract |
|---|---|---|
| `SparseTopology` | `from_config`, `edge_index`, `incidence`, `to(device)`, `fingerprint()` | Topology is deterministic, immutable after construction, and serializable. |
| `SparseIncidence` | `apply(x)`, `transpose_apply(y)`, `nnz`, `shape` | Applies incidence and transpose without dense conversion. |
| `SparseCochainMap` | `apply_degree(k, x)`, `compose(k, x)`, `residual(k)` | Adjacent-degree maps have declared shapes and structural composition. |
| `MatrixFreeLaplacian` | `apply(x)`, `quadratic_form(x)`, `energy(x)` | Production forward path uses factored applications. |
| `SparseTransport` | `apply(x, parameters)`, `adjoint_apply(x)` | Connection/transport is applied edge-wise or block-sparse. |
| `DenseReferenceOperators` | `build_small()`, `apply(x)` | Test-only oracle; refuses dimensions above a configured safety limit. |
| `OperatorDiagnostics` | `check_symmetry`, `check_psd`, `check_cochain`, `estimate_spectrum` | Diagnostics run outside the training hot path. |

The operators must support leading batch dimensions. The canonical tensor layout must be documented, and all implementations must preserve dtype and device unless a diagnostic explicitly requests promotion.

## Topology construction

The topology builder must be deterministic under a seed and must produce a stable fingerprint based on ordered vertices, edges, faces or higher cells, orientation signs, and configuration. The fingerprint must be stored in checkpoints and benchmark results.

For the first implementation, topology may remain fixed while edge weights and low-rank feature transforms are learned. Rebuilding topology during every optimizer step is prohibited. If a later experiment learns topology, it must use an explicit proposal and acceptance mechanism rather than mutating sparse indices invisibly.

The builder must reject self-loops, duplicate directed edges, invalid orientation indices, disconnected configurations where connectivity is required, and inconsistent higher-order cell boundaries. A disconnected topology may be allowed only when explicitly requested by a test.

## Matrix-free implementation requirements

The production operator must never call `torch.kron` or allocate a dense global matrix proportional to the full state dimension. It must apply local edge contributions using indexed gathers/scatters, sparse matrix multiplication, block-sparse kernels, or equivalent factorizations.

The implementation must include an instrumentation mode that records tensor shapes and flags any allocation whose element count exceeds a configurable fraction of the full state-square size. The standard forward benchmark must run with this instrumentation enabled at least once per configuration.

The operator must support `float32` and `float64` initially. `bfloat16` support may be deferred if the selected sparse backend does not provide reliable kernels, but the interface must make the limitation explicit.

## Dense reference oracle

For small dimensions only, construct dense equivalents from the same topology and parameters. The oracle must use identical orientation conventions, weight normalization, boundary ordering, and parameter values. It must compare:

```text
matrix_free.apply(x)
reference_dense_matrix @ x
```

for random vectors, batched vectors, basis vectors, and gradients with respect to edge weights and feature parameters.

The dense oracle must refuse to run above a declared limit, such as `N <= 2,048`, to prevent accidental production use. The refusal itself must be tested.

## Evaluation harness

The Stage B harness must expose commands equivalent to:

```text
python -m benchmarks.stage_b topology --config tiny --seed 42
python -m benchmarks.stage_b equivalence --config tiny --trials 100 --seed 42
python -m benchmarks.stage_b gradients --config tiny --seed 42
python -m benchmarks.stage_b sparse_scaling --sizes 16,32,64,128,256
python -m benchmarks.stage_b production_guard --config small
```

### Test matrix

| Test | Procedure | Required measurement |
|---|---|---|
| Topology determinism | Build topology twice from the same configuration | Identical fingerprint and sparse indices. |
| Topology validity | Check orientation, duplicates, dimensions, boundary ordering, and connectivity policy | Complete validation report. |
| Incidence transpose | Compare inner products `<Sx,y>` and `<x,Sᵀy>` | Relative error. |
| Laplacian equivalence | Compare matrix-free and dense outputs over random/basis inputs | Maximum absolute and relative error. |
| Batch equivalence | Compare arbitrary leading batch dimensions | Shape and numerical errors. |
| Gradient equivalence | Compare parameter gradients through both paths | Per-parameter relative error. |
| Cochain composition | Apply `δ_{k+1}(δ_k(x))` to random and basis states | Residual norm and relative residual. |
| Symmetry/PSD | Test declared properties with quadratic forms and small eigensystems | Symmetry error, minimum eigenvalue, negative-energy count. |
| Device transfer | Run CPU and available CUDA paths | Output error and device consistency. |
| Production guard | Trace allocations and inspect operator objects | No dense full-state matrix in forward. |
| Sparse scaling | Sweep graph/state sizes | Time, peak memory, nonzero count, and scaling exponent. |

### Numerical tolerances

For `float64`, sparse/dense output and gradient agreement should use `rtol=1e-9, atol=1e-11` unless accumulation order requires a documented relaxation. For `float32`, use `rtol=1e-5, atol=1e-6`. The cochain residual threshold must be reported relative to `||x||`; a default structural target is `≤1e-10` in `float64` and `≤1e-5` in `float32`.

## Pass/fail gates

| Gate | Pass condition | Failure consequence |
|---|---|---|
| Topology integrity | All invalid-topology tests reject and all valid fixtures reproduce fingerprints | Stop and repair topology construction. |
| Dense equivalence | Outputs agree for all reference trials within dtype tolerance | Stop. No dynamics may be built on an unverified sparse operator. |
| Gradient equivalence | Intended gradients agree within declared tolerance | Stop. A numerically correct forward path with incorrect gradients is a failure. |
| Cohain identity | Structural `δ²` residual meets the declared tolerance on random and basis tests | Stop or explicitly narrow the mathematical claim. |
| Production no-dense guard | No full-state dense matrix or Kronecker lift appears in the standard forward path | Mandatory fail. Do not proceed with a disguised dense implementation. |
| Device correctness | CPU path passes; CUDA path passes when available or is explicitly marked unavailable | CPU is mandatory; missing backend support cannot be hidden. |
| Memory scaling | Peak memory grows with sparse nonzeros/state width, not full operator square | If it fails, redesign before Stage C. |
| Serialization | Topology fingerprint and parameters restore exactly | Stop. Sparse indices are part of the model state. |

A Stage B failure is not repaired by loosening tolerances without an error analysis. If sparse and dense paths disagree because the dense v2 operator was itself inconsistent, the discrepancy must be documented and the chosen reference convention must be approved before proceeding.

## Transition test to Stage C

The Stage C transition harness must:

```text
1. Load the archived Stage A v2 reference.
2. Construct a small sparse topology from the same deterministic fixture.
3. Compare sparse and dense outputs, gradients, energy, and diagnostics.
4. Run production guard tracing on the small and small-production configurations.
5. Serialize and restore topology plus parameters.
6. Generate a Stage B manifest with all gates and raw metrics.
```

Stage C may begin only if all mandatory gates pass, the production guard proves that no full dense operator is used, and the sparse operator can be applied to batched states on the target device.

## Exit artifacts

Stage B exits with sparse topology serialization, matrix-free operator modules, dense reference oracles, equivalence tests, allocation instrumentation, sparse scaling reports, and a transition manifest. The artifact must include the exact mathematical properties that passed and the properties intentionally not claimed.

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and current v2 architecture"
