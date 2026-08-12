# Stage E Gate Report — Controlled Ablations and Nano Scale Study

## Result

**Status:** `PASS` with decision **`CONDITIONAL_GO_SYNTHETIC_ONLY`**. This is a controlled, multi-seed, CPU nano study on the frozen repository-local synthetic corpus. It separates engineering, synthetic-quality, and scientific evidence and makes no real-corpus or natural-language capability claim.

| Gate | Status | Evidence summary |
|---|---:|---|
| configuration_and_data_audit | PASS | {"classification": "local_synthetic_engineering_study", "data_manifest_fingerprint": "2cf28ed17829b7c61bd21cad433a677dcfa46f811ee36fb132597eb99559f363", "matrix_manifest_fingerprints": {"C": "c98837413e2d3fc7fde2034cad9c |
| parameter_audit | PASS | {"matching_note": "Exact matching is impossible for heterogeneous tiny baselines; counts are reported rather than hidden.", "records": {"C": {"class": "DCSSLanguageModel", "manifest": {"definition": {"allocation_claim_sc |
| matched_multi_seed_training | PASS | {"raw_runs": [{"cursor": 100, "data_manifest_fingerprint": "2cf28ed17829b7c61bd21cad433a677dcfa46f811ee36fb132597eb99559f363", "elapsed_seconds": 0.1370689090108499, "final_loss": 1.5243035554885864, "finite": true, "id" |
| sequence_scaling | PASS | {"forward_time_fit": {"exponent": 0.9586029511170231, "intercept": -7.571694545683421, "residual_rmse": 0.03658994643483563}, "measured_lengths": [8, 16, 32, 64, 128, 256], "persistent_memory_fit": {"exponent": 0.0, "int |
| streaming_and_allocation_audit | PASS | {"continuation_max_abs": 0.0, "forbidden_kron_operations": [], "runtime_dense_full_state_allocations": [], "runtime_dense_sequence_allocations": [], "source_guard": {"to_dense": false, "torch_kron_call": false}, "state_b |
| long_context_harmonic_retention | PASS | {"delay": 32, "full_harmonic_initial_norm": 0.0009498791769146919, "full_harmonic_norm": 0.0005072717322036624, "full_retained_ratio": 0.5340381645709243, "no_harmonic_norm": 0.0, "scope": "synthetic_memory_diagnostic",  |
| fresh_reproducibility_rerun | PASS | {"loss_max_abs": 0.0, "parameter_fingerprint": "e6a35499d940aecd9da44b8c2fec225dfaa301f7790557964c55b5a6c1fec25c", "seed": 1, "steps": 40, "validation_loss_abs": 0.0} |
| separate_engineering_quality_scientific_analysis | PASS | {"ablation_validation_losses": {"C": 3.8839126030604043, "E": 3.6517706314722695, "G": 3.603725870450338, "H": 3.5601075490315757, "U": 3.666372458140055}, "attribution": {"full_minus_geometry_free_validation_loss": -3.1 |

## Required negative-result record

The 4k–8k scaling range, real-corpus quality, transfer evaluation, matched dense CDI-v2 speed ratio, and 4k Transformer memory ratio are **not measured** in the fixed CPU nano study. They are not inferred from the 8–256 measured range and are not marked passed.

## Transition state

```json
{
  "stage_e": "PASS",
  "decision": "CONDITIONAL_GO_SYNTHETIC_ONLY",
  "stage_f_implementation_allowed": false,
  "required_action": "explicit user review and approval before Stage F"
}
```

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and DCSS-CDI Stage E implementation"
