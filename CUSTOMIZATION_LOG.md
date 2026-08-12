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

## Stage C selective recurrence customization

Stage C uses a new `nano` recurrent configuration with four vertices, three memory bands, and four channels per band. Its structured state has 48 scalar elements per leading batch item, below the requested 64-element nano limit. This differs from the generic Stage C specification only by selecting a CPU-safe concrete tier; it does not weaken the numerical gate thresholds.

The approved `frequency_cascade` customization is implemented as logarithmically separated, finite timescale intervals: fast `[0.25, 1]`, middle `[2, 8]`, and harmonic `[16, 64]`. The original Stage C specification required distinct finite ranges but did not prescribe a concrete initializer. The change makes this requirement deterministic and testable without changing its pass condition.

Stage C represents its generator as diagonal dissipation plus pairwise skew rotations and uses the exact closed form for those two-dimensional blocks. This is mathematically equivalent to exponentiating the declared block-diagonal stable generator but avoids per-token dense state matrices. It strengthens, rather than relaxes, the no-dense production guard.

The Stage B Laplacian is applied after each band update as the fixed degree-preserving geometric correction. The Stage B cochain transport is explicitly zero for this first vertex-only state layout because degree-zero cochains map vertices to edges. This is an honest layout restriction, recorded in the Stage C design, and does not alter any Stage C gate threshold. A future degree-structured state must add edge state and a separate cochain-equivalence gate before enabling that term.

The Stage C 10,000-step stability envelope is executed under `torch.no_grad()` because it is a forward-only empirical stability measurement, while gradient equivalence is a separate mandatory gate. This avoids retaining a 30,000-update autograd graph across the three stress modes and keeps the complete nano gate below 30 seconds on CPU. It does not change the state update, stability thresholds, or gradient gate.

The synthetic-memory command exposes `--task delayed_copy --steps 1000` as specified. The first Stage C probe measures delayed impulse retention with zero distractors: after a fixed delay, harmonic retention must exceed middle retention, middle must exceed fast retention, and harmonic retention must be at least twice fast retention. This is an explicit Stage C recurrence-capacity probe, not a claim of tokenizer or language-model performance; full data-based delayed-copy training remains outside the Stage C non-goals.

## Stage D tokenizer and reproducible training customization

The external EthioBBPE tokenizer dependency is replaced by a pure-Python, Unicode-NFC character tokenizer with a deterministic vocabulary built from a documented local corpus. The original code imported `ethiobbpe` and downloaded a pretrained tokenizer at construction time. The replacement removes all external runtime dependencies and network behavior, adds a versioned tokenizer artifact and fingerprint, and does not weaken any causal, masking, or checkpoint gate.

The default Stage D corpus is a small repository-local deterministic synthetic corpus. It is intentionally labeled as synthetic debugging data rather than a primary language-quality corpus; its purpose is reproducible causal alignment, tokenizer, masking, resume, and matched-baseline validation under the CPU nano budget. This is a hardware- and dependency-safe adaptation of the Stage D corpus requirement, and the report will not present its metrics as WikiText/SciQ or real-corpus quality.

Stage D uses float32 on CPU, with no AMP path claimed in the default environment. The precision gate verifies finite float32 computation and records CUDA/bfloat16 as unavailable when applicable. This does not alter the precision-safety threshold.

The Stage D matched v2, v3, and Transformer comparison is restricted to the same tokenizer, local synthetic corpus, token budget, optimizer family, and evaluation code. It validates fair comparison plumbing only; it does not claim that the systems have been evaluated on a real language-model benchmark.

## Stage E controlled-ablation customization

Stage E uses the frozen Stage D local synthetic corpus, tokenizer, CPU float32 precision, and deterministic 100-step token budget for all matrix members. This supports controlled engineering and synthetic-sequence diagnostics only. The study does not label these results as real-corpus quality, transfer, WikiText, SciQ, or natural-language claims.

The nano scaling range is frozen at lengths 8–256 and the streaming range at 512 tokens. The Stage E report will calculate empirical exponents over that measured range but will retain the specification's 4k–8k memory and legacy-speed targets as `NOT_MEASURED`; it will not extrapolate or mark them passed. The no-harmonic, no-geometry, ungated, Euler, and unconstrained-cochain variants are isolated named ablations. The unconstrained-cochain diagnostic is explicitly exempt from the full-production no-dense claim; allocation auditing applies to the full DCSS path.

## Stage F bounded diagnostic capability customization

Stage F proceeds only as an explicitly user-approved diagnostic extension of the Stage E conditional synthetic-only result. The implementation is restricted to deterministic local memory, retrieval, tools, planning, execution, and verification fixtures. It does not claim general language capability, retrieval quality on real documents, operational autonomy, or external-system safety.

All tools are typed, local, and dry-run by default. No shell, network, account, payment, posting, deletion, transfer, external file mutation, or real side effect is registered or enabled. A mutating demo tool is retained only as a negative permission-boundary fixture and must reject before execution. Retrieved and tool-generated instruction-like strings are treated strictly as untrusted data.

The optional capability modules are not wired into the mandatory DCSS recurrence training path. Stage F verifies exact core-path optionality and labels every capability at most `Experimental`. The final manifest will retain `external_side_effects_enabled: false` and Stage G does not exist; any follow-up requires an explicit new user request.

## Production inference hardening customization

The production inference path is isolated from the training path and restores only checkpoints accepted by the existing atomic SHA-256 verification contract. It consumes `load_verified(path)["stage_d_payload"]` without changing the Stage D payload format, validates the embedded tokenizer artifact and tokenizer-lineage binding, reconstructs the DCSS language model, and requires strict state-dictionary loading plus topology and model-fingerprint agreement. Any missing, malformed, inconsistent, or tampered artifact is rejected rather than silently repaired or partially loaded.

Architecture reconstruction now derives the embedding width, vertex count, and even band width from checkpoint tensors and validates the three-band pooled-readout contract before model creation. This supports both the current 4-vertex, 4-channel-per-band production topology and compatible earlier nano checkpoints without weakening the nano state-size constraint or assuming a fixed historical architecture. The inference interface carries the cohomodynamic recurrent state token by token through `forward_chunk`, applies deterministic seedable top-k/top-p sampling, and suppresses padding, beginning-of-sequence, and document sentinel tokens from decoding.

The public `DCSSInferenceEngine`, `GenerationConfig`, `InferenceMetadata`, and `interactive_chat` interfaces remain importable from both `cdi.v3` and `cdi.v3.production`. Their exports are lazy to keep `python -m cdi.v3.production.inference` free of package pre-import warnings and to avoid coupling inference initialization to the production training import path. Focused regression coverage includes deterministic greedy and sampled decoding, invalid-control rejection, tamper rejection, special-token suppression, and architecture-derived restoration; the complete repository regression suite passes with 244 tests.

The one-shot inference command now prints only `DCSSInferenceEngine.complete(...)`, i.e., the newly generated continuation. It no longer prefixes displayed evaluation samples with the user-supplied prompt. The lower-level `generate()` method retains its explicit full-sequence contract for callers that need prompt-plus-continuation token accounting, while `complete()` is the required presentation method for honest qualitative evaluation.
