# Error Cycles

## EthioBBPE tokenizer migration

The legacy CDI path tokenized training data with EthioBBPE while the active `CDITokenizer` compatibility wrapper used a small character vocabulary and silently clamped input IDs. This destroyed token identity for IDs outside the character vocabulary and made the data/model contract invalid.

The corrective work replaces the active tokenizer adapter with an EthioBBPE-backed artifact, validates every ID range instead of clamping, and makes checkpoint restoration use the exact serialized tokenizer snapshot.

The first focused Stage D run exposed two obsolete character-tokenizer assertions: one prohibited the EthioBBPE dependency and another expected a Unicode snowman to become an unknown token. Both were replaced with EthioBBPE dependency, range, round-trip, and artifact-restoration checks. The focused Stage D suite then passed: 9 tests passed.
