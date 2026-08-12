# Stage D Gate Report — Zero-Dependency Causal LM Integration

## Result

**Status:** `PASS`. Stage D provides a versioned pure-Python Unicode character tokenizer, audited repository-local synthetic corpus, token-level DCSS causal language model, deterministic checkpoint/resume, and a matched small synthetic baseline protocol. The run is explicitly **not** a real-corpus language-quality claim.

| Gate | Status | Evidence summary |
|---|---:|---|
| tokenizer_round_trip | PASS | {"config": {"byte_policy": "unicode_character_with_unk_fallback", "embedding_dim": 4, "format": "dcss-cdi-character-tokenizer-v1", "max_chunk_length": 8, "normalization": "NFC", "special_tokens": ["<pad>", "<unk>", "<bos |
| data_integrity | PASS | {"boundary_policy": "bos_eos_per_document_no_cross_document_packing", "hash_overlap": 0, "manifest_fingerprint": "edbba5d41e615e38f34394e556bc35517604b490ad224d6445716a040607ae9a", "source": "data/stage_d/synthetic_corpu |
| causal_alignment | PASS | {"perturbed_index": 4, "pre_causal_max_abs": 0.0, "tolerance": 1e-06} |
| mask_correctness | PASS | {"masked_gradient_max_abs": 0.0, "masked_loss_difference": 0.0, "unmasked_loss_difference": 3.99999737739563} |
| tiny_overfit_and_gradient_coverage | PASS | {"final_loss": 0.12152446806430817, "gradient_groups": {"gates": true, "generators": true, "normalization_readout": true, "sparse_geometry": true, "token_embeddings": true}, "initial_loss": 3.367295026779175, "parameter_ |
| checkpoint_resume_determinism | PASS | {"interrupt_at": 25, "logit_max_abs": 0.0, "loss_max_abs": 0.0, "parameter_max_abs": 0.0, "restored_cursor": 25, "steps": 50, "tokenizer_fingerprint": "daebf458e55e56a433cf8f621c7d3f773de2c3af0a63456939b56e004061eaf4"} |
| precision_validation_generation | PASS | {"cuda_amp": "UNAVAILABLE", "finite": true, "generation": {"greedy_ids": [2, 18, 19, 5, 14, 12, 8], "reproducible": true, "sample_ids": [2, 18, 19, 16, 21, 26, 12]}, "precision": "float32_cpu", "validation": {"loss": 3.3 |
| throughput_memory | PASS | {"records": [{"finite": true, "length": 2, "rss_mb": 680.12109375, "seconds": 0.0011542759893927723, "state_elements_per_batch": 48, "tokens_per_second": 6930}, {"finite": true, "length": 4, "rss_mb": 680.12109375, "seco |
| matched_baseline_comparison | PASS | {"results": {"dcss_cdi": {"classification": "synthetic_plumbing_validation", "final_loss": 2.721639633178711, "initial_loss": 3.367295026779175, "model_class": "DCSSLanguageModel", "parameter_count": 511, "seconds": 0.24 |

## Transition state

```json
{
  "stage_d": "PASS",
  "stage_e_implementation_allowed": false,
  "required_action": "explicit user approval before Stage E"
}
```

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and DCSS-CDI Stage D implementation"
