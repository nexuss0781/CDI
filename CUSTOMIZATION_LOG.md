# DCSS-CDI Customization Log

## Scope and baseline preservation

This log records the user-authorized customization of CDI into **DCSS-CDI**. The repository remains anchored at the frozen Stage A commit `8be410c`; the Stage A v2 implementation is retained as a reference baseline rather than rewritten. Stage B introduces a new `cdi.v3` sparse substrate and does not alter the v2 recurrent language-model semantics. This separation is deliberate because Stage A is the reproducibility reference against which later engineering work must be measured.

> **Stage B boundary.** Stage B implements only the sparse, factorized operator substrate. Selective recurrence, memory bands, tokenizer replacement, and language-model training are specified here as later-stage customizations but are not implemented before explicit approval for their respective stages.

| Customization | Original specification | DCSS-CDI customization | Rationale | Gate-threshold effect |
|---|---|---|---|---|
| CPU-first production path | Generic CPU/CUDA device-aware sparse implementation | Treat CPU and float32 as the required development path; report CUDA as unavailable when it is not present | The requested environment is CPU-only and needs reproducible, memory-safe iteration | None. CPU remains mandatory; unavailable CUDA is explicit. |
| `nano` tier | Stage B used a `tiny` example configuration | Add `nano` as the default Stage B configuration with four vertices, eight feature channels, and total factorized state width of 32 | Enables a complete gate run in less than 30 seconds while keeping a nontrivial two-simplex topology | None. The required float32 equivalence tolerance remains `rtol=1e-5`, `atol=1e-6`. |
| Factorized state layout | Dense global state was flattened as `N = n_points × spinor_dim × total_belief_dim` | Use the canonical matrix-free layout `(..., n_vertices, channels)` and apply geometry independently over channels | Avoids Kronecker lifting and full-state `N × N` matrices | None. The dense reference remains a small, test-only oracle. |
| `cohomological_health_score` | Separate spectral, cochain, and energy diagnostics | Add a bounded scalar in `[0, 1]` that multiplies normalized spectral-gap, cochain-residual, and energy-validity components; any failed critical condition returns `0` | Provides one actionable diagnostic without hiding its component metrics | No gate tolerance is loosened. The score is diagnostic, not a substitute for individual gates. |
| `frequency_cascade` initialization | No custom memory-band initialization | Specify logarithmically spaced fast, middle, and harmonic time constants for the Stage C selective recurrence | Makes later memory bands measurably distinct and reproducible | Not applicable to Stage B; it will be verified in Stage C before use in NLP training. |
| Zero-dependency tokenizer | v2 imports `EthioBBPE` at runtime | Specify a pure-Python deterministic character tokenizer, with optional BPE extension implemented only from saved vocabulary and merge files, for Stage D | Removes the external tokenizer installation and download dependency while preserving the frozen Stage A reference | Not applicable to Stage B. The v2 tokenizer is not modified in this stage so the baseline remains auditable. |
| `geometry_ablation` | No flag to isolate recurrent dynamics from geometry | Add `geometry_ablation: bool = False` to `CDIConfig` and the v3 matrix-free Laplacian; when true, it returns a zero geometric contribution without constructing a dense substitute | Allows Stage C and later studies to isolate selective recurrence cleanly | No gate tolerance is loosened. Both enabled and disabled modes are included in Stage B equivalence and guard evidence. |
| Sparse topology | Topology was derived inside dense operator construction | Use deterministic fan-triangulated topology with immutable ordered vertices, oriented edges, faces, and SHA-256 fingerprint | Guarantees reproducible incidence orientation and a non-vacuous structural `δ²=0` check at small size | None. Topology must reproduce exactly for a fixed configuration and seed. |
| Dense reference safety | Dense equivalence was required only for small systems | Enforce a hard full-state size limit (`N <= 2,048`) in the test-only dense oracle | Prevents accidental promotion of the oracle into a production execution path | This strengthens the production safety requirement. |
| Allocation guard | No existing sparse-path allocation instrumentation | Trace dispatcher outputs during production application and reject `torch.kron` plus dense matrices at or above the configured full-state-square threshold | Converts the no-dense requirement into executable evidence | None. Dense allocation remains an unconditional Stage B failure. |

## Stage-by-stage customized implementation plan

| Stage | Customized objective | Status at the end of Stage B |
|---|---|---|
| A | Preserve v2, reproducibility harness, and known-defect reporting | Complete at `8be410c`; rerun locally during this task to recreate its missing JSON artifact. |
| B | Implement CPU-safe float32 sparse topology, incidence, cochains, matrix-free Laplacian, dense oracle, diagnostics, guard, benchmark harness, and gates | Implemented and evaluated in this task. |
| C | Add stable content-selective recurrence, `frequency_cascade` memory bands, chunk/streaming equivalence, and `geometry_ablation` studies | Explicitly deferred pending user approval after the Stage B gate review. |
| D | Integrate the pure-Python tokenizer, causal LM training, checkpoints, deterministic data handling, and CPU-focused reproducibility | Explicitly deferred pending Stage C approval. |
| E | Run matched ablations and scale studies, including geometry, selectivity, and memory-band ablations | Explicitly deferred pending Stage D completion. |
| F | Add independently verified capability modules only after the core LM is stable | Explicitly deferred pending Stage E completion. |

## Known Stage A defects carried forward

The historical `clifford_negative_signature_d4` float64 relation error of `4.0` remains unmodified. It is outside the Stage B scalar cochain substrate and is reported as unaffected rather than silently repaired. The historical `sheaf_parameters_inactive_in_lm_path` issue also remains unchanged because Stage B does not invoke the v2 language-model forward path. The v3 operator report identifies its active parameters directly: edge-weight parameters participate when geometry is enabled and intentionally receive no gradient when `geometry_ablation=True`.

## Gate policy

All Stage B pass/fail gates retain the user-provided float32 tolerance of `rtol=1e-5` and `atol=1e-6`, with structural cochain residual at most `1e-5` relative. The `nano` tier changes workload size only; it does not weaken the mathematical or production-safety acceptance criteria. The Stage B report keeps `stage_c_implementation_allowed` set to `false` until the user explicitly authorizes Stage C.

## Stage A baseline correction addendum

The user required that Stage A have no carried limitations before any Stage C work. The historical d=4 real gamma construction was corrected rather than waived: it now uses verified real `Cl(0,d)` generator templates for dimensions one through eight, with a negative Clifford signature and an explicit real spinor-dimension table. The prior template used symmetric blocks that square to `+I`, causing the measured relation error of `4.0`; the corrected contract is tested both for every supported flat dimension and for the contravariant curved-frame relation.

The original `tiny` configuration assumed the smaller complex-style spinor dimension and became unsafe under the correct real d=4 representation. It was changed into a compact float64 test profile with four points and a 24-dimensional belief complex. This changes the test workload only, not the mathematical acceptance threshold: it remains v2-axiom compliant and keeps dense test operators within the CPU development memory envelope.

The v2 LM path now adds the live observation-sheaf injection to `W_iota @ e_t` and adds the live sheaf projection to `W_out @ b0_mean`. `sheaf.embedding_matrix` and `sheaf.output_matrix` are consequently mandatory finite, nonzero entries in the Stage A LM-gradient gate. Finally, checkpoint restoration rebuilds live dense operators after parameter copy, eliminating stale-operator output mismatches. These changes remove the two documented baseline limitations and do not relax any gate threshold.
