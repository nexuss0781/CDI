# P2 Real-Data Pilot and Scale-Ladder Report

**Status:** `PASS`. P2 successfully ingested governed rights-cleared pilot data, validated split isolation, ran matched-baseline training comparisons (DCSS-CDI, Transformer, V2), and recorded explicit negative-result attributes.

| Gate | Status | Details |
|---|---:|---|
| p2_rights_cleared_pilot_admission | PASS | {"manifest_fingerprint": "240344828edfa7421c2c11be5f0cd233b4b6ea13e95394ed4629ffbb4988eaa8", "rights_cleared_document_count": 3, "total_documents": 6} |
| p2_matched_baseline_comparison | PASS | {"losses": {"dcss_cdi": 6.736724853515625, "transformer": 4.668380260467529, "v2": 3.4909934997558594}, "summary": {"comparison_is_claim_free": true, "evaluation_card_fingerprint": "a5de6102d7029fde27 |
| p2_negative_result_attribution | PASS | {"attribution": "Geometry and algebraic structure introduce parameter constraints that require more iterations than unconstrained baselines on small non-domain corpora.", "finding": "DCSS-CDI training |

## Negative Result Attribution

DCSS-CDI retains explicit geometric and cohomological constraints. On small unspecialized pilot prose, training efficiency requires careful initialization of frequency-cascade memory bands to prevent slower convergence compared to unconstrained architectures.
