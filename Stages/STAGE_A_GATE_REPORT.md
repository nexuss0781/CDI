# Stage A Gate Report — Corrected CDI v2 Baseline

## Result

The corrected Stage A reproducibility harness completed with **`PASS`** on the CPU float32 micro configuration. The report contains **no known defects and no baseline limitations**. The frozen `8be410c` reference was used to identify the original defects; this corrected baseline supersedes its limitation-bearing report and provides the validated v2 foundation for the already implemented Stage B sparse substrate.

> **Validation status:** Every mandatory Stage A gate passed. The two previously carried limitations—an invalid d=4 Clifford representation and inactive observation-sheaf parameters in the recurrent LM path—are now mandatory, tested properties of the baseline.

| Gate | Status | Evidence |
|---|---:|---|
| Deterministic forward | **PASS** | Identical repeated parameter fingerprints, loss `12.670943`, and output error `0.0`. |
| Construction and LM loss | **PASS** | Finite total loss `12.670943`; CE `3.465624`; perplexity `31.9964`. |
| Complete gradient flow | **PASS** | Manifold, metric, theta, injection, readout, connection, belief, **sheaf embedding**, and **sheaf output** paths are finite and nonzero. |
| Optimizer / rebuild | **PASS** | Post-step operator reconstruction completed at global step `1`. |
| Checkpoint parameters | **PASS** | Exact saved/restored parameter fingerprint match. |
| Checkpoint logits | **PASS** | Restored forward output maximum absolute error `0.0`. |
| Deterministic overfit | **PASS** | Loss fell from `11.977951` to `0.809814`, a `93.24%` relative reduction in 60 steps. |
| Scaling finiteness | **PASS** | Lengths `1, 2, 4, 8, 16` were finite on CPU. |
| Clifford signature | **PASS** | Real negative-signature templates for dimensions 1–8 satisfy `γᵢγⱼ + γⱼγᵢ = -2δᵢⱼI`; corrected d=4 test passes. |
| Curved Clifford contract | **PASS** | Curved gamma matrices satisfy the negative relation against the contravariant metric implied by the supplied frame. |
| Complete repository suite | **PASS** | `193 passed` with no warnings. |

## Corrected contracts

The baseline now uses verified real `Cl(0,d)` templates for dimensions one through eight. The previous d=4 construction used symmetric matrices that square to `+I`, contradicting the declared negative Clifford convention. The replacement uses real anticommuting generators that square to `-I`; the corresponding real spinor dimensions are represented explicitly by `CDIConfig.spinor_dim`. The unit-test profile was compacted to preserve a safe CPU memory envelope under the mathematically correct d=4 representation.

The recurrent LM path now combines the dedicated injection/readout maps with `ObservationSheaf.embedding_matrix` and `ObservationSheaf.output_matrix`. Both sheaf maps are included in the token-level computation and are mandatory finite, nonzero gradient paths. This removes the former disconnect between the sheaf subsystem and the language-model loss.

Checkpoint restoration now rebuilds the live dense Dirac and Laplacian operators after copying saved parameter tensors. This guarantees that restored outputs reflect restored parameters rather than construction-time operator matrices.

## Resource envelope

The Stage A micro run uses float32 with `n_points=4`, state width `384`, and a CPU-only device. Its length-16 forward run completed in `0.001338` seconds. The separate float64 `tiny` profile has been reduced to a valid test-only configuration of state width at most `2048`, preventing the real d=4 spinor correction from creating an unsafe dense test allocation.

## Transition state

Stage A is **validated**. Stage B remains independently committed and its own transition report continues to set `stage_c_implementation_allowed: false`. No Stage C work begins until the user explicitly grants the next approval.

```json
{
  "stage_a": "PASS",
  "baseline_limitations": [],
  "known_v2_defects": [],
  "stage_b_implementation_allowed": false,
  "stage_c_implementation_allowed": false,
  "required_action": "user approval before Stage C"
}
```

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and corrected Stage A baseline"
