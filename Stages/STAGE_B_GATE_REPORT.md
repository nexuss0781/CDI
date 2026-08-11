# Stage B Gate Report — DCSS-CDI

## Result

The Stage B gate status is **PASS** on the configured CPU float32 path. Stage C remains blocked: `stage_c_implementation_allowed` is `false`.

| Gate | Status | Evidence summary |
|---|---|---|
| topology_integrity | PASS | `{"config": {"allocation_fraction_limit": 0.5, "cover_k": 2, "dense_reference_limit": 2048, "device": "cpu", "dtype_str": "float32", "energy_limit": 100000000.0, "geometry_ablation": false, "n_vertices": 4, "name": "nano", "seed": 42, "spectral_target": 0.1, "s` |
| dense_equivalence | PASS | `{"atol": 1e-06, "modes": {"false": {"basis_trials": 32, "batch": {"max_abs": 9.5367431640625e-07, "max_rel": 1.6465733096993063e-06, "mean_abs": 9.108529042123337e-08}, "batch_shape": [2, 3, 4, 8], "geometry_ablation": false, "max_abs": 9.5367431640625e-07, "m` |
| gradient_equivalence | PASS | `{"atol": 1e-06, "modes": {"false": {"comparison": {"max_abs": 4.76837158203125e-07, "max_rel": 1.2715068464785872e-07, "mean_abs": 1.4603138254187797e-07}, "dense_gradient_norm": 7.295492172241211, "geometry_ablation": false, "passed": true, "sparse_gradient_n` |
| operator_diagnostics | PASS | `{"cochain": {"basis_max_relative_residual": 0.0, "max_relative_residual": 6.539087138435207e-08, "passed": true, "random_max_relative_residual": 6.539087138435207e-08, "structural": true, "threshold": 1e-05}, "cohomological_health_score": {"cochain_residual": ` |
| production_no_dense_guard | PASS | `{"modes": {"false": {"forbidden_operations": [], "geometry_ablation": false, "large_allocations": [], "metadata": {"factorization": "S_transpose @ diag(softplus(edge_log_weights)) @ S", "forbidden_operations": ["torch.kron", "dense_full_state_operator"], "full` |
| device_correctness | PASS | `{"cpu": {"output_device": "cpu", "status": "PASS"}, "cuda": {"reason": "torch.cuda.is_available() is false in the configured CPU development environment.", "status": "UNAVAILABLE"}}` |
| memory_scaling | PASS | `{"full_state_square_not_allocated": true, "method": "Exact structural storage accounting plus process peak RSS. The production count is O(E \u00d7 channels + V \u00d7 channels + E), while the forbidden dense count is O((V \u00d7 channels)^2).", "records": [{"d` |
| serialization | PASS | `{"parameter_fingerprint": "925ae9af6f8b7622a62a5edba416acea02047c833562b5f1ee6b7df42d8685ef", "restored_parameter_fingerprint": "925ae9af6f8b7622a62a5edba416acea02047c833562b5f1ee6b7df42d8685ef", "topology_fingerprint": "84080201a65516f8985d3201e78492f1ce45ef3` |

## Customization evidence

The `nano` tier uses a factorized state dimension of `32`, below the required 64. The report contains both `geometry_ablation=false` and `geometry_ablation=true` evidence. The cohomological health score is reported within the operator diagnostics gate.

## Corrected Stage A baseline

Stage A is now a fully validated baseline: its real negative-signature Clifford templates pass through d=8, both LM sheaf maps are mandatory active paths, and checkpoint restoration rebuilds live operators. Stage B identifies `edge_log_weights` as active only when geometry is enabled.

## Transition rule

This report does not authorize Stage C. The user must explicitly respond with **approved** or **proceed to Stage C** before selective recurrence, frequency-cascade memory bands, tokenizer replacement, or NLP training integration can begin.
