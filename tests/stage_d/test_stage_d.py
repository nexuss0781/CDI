import inspect

import cdi.tokenizer as legacy_tokenizer_module
import torch

from cdi.v3 import (
    EthioBBPETokenizer,
    DCSSLanguageModel,
    LocalSyntheticCorpus,
    StageDConfig,
    build_model,
    checkpoint_payload,
    optimizer_for,
    restore_checkpoint,
    train_steps,
)
from cdi.v3.training import collate_examples, deterministic_batches, pack_documents, seed_everything


def resources(seed: int = 42):
    config = StageDConfig.nano(seed=seed)
    corpus = LocalSyntheticCorpus.default()
    tokenizer = corpus.tokenizer(config)
    splits = corpus.split(seed)
    examples, _ = pack_documents(splits["train"], tokenizer, config.chunk_length)
    batches = deterministic_batches(examples, tokenizer, config)
    return config, corpus, tokenizer, splits, batches


def test_tokenizer_declares_ethiobbpe_dependency():
    assert "ethiobbpe" in inspect.getsource(legacy_tokenizer_module).lower()


def test_ethiobbpe_tokenizer_round_trip_unicode_and_artifact(tmp_path):
    _, corpus, tokenizer, _, _ = resources()
    text = "a  b\t\ne\u0301"
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded.ids) == tokenizer.normalize(text)
    empty = tokenizer.encode("")
    assert empty.ids == (tokenizer.bos_id, tokenizer.eos_id)
    unknown = tokenizer.encode("☃")
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in unknown.ids)
    assert tokenizer.decode(unknown.ids) == tokenizer.normalize("☃")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = EthioBBPETokenizer.load(path)
    assert restored.fingerprint == tokenizer.fingerprint
    assert restored.decode(restored.encode(text).ids) == tokenizer.normalize(text)
    assert corpus.manifest(tokenizer, StageDConfig.nano())["tokenizer_fingerprint"] == tokenizer.fingerprint


def test_explicit_truncation_and_padding_contract():
    config, corpus, tokenizer, splits, _ = resources()
    long_text = splits["train"][0].text * 10
    try:
        tokenizer.encode(long_text, max_length=config.chunk_length)
        assert False, "Overlength text must require explicit truncation."
    except ValueError:
        pass
    truncated = tokenizer.encode(long_text, max_length=config.chunk_length, truncate=True)
    assert truncated.truncated
    ids, mask, count = tokenizer.pad([truncated], max_length=config.chunk_length)
    assert ids.shape == mask.shape == (1, config.chunk_length)
    assert count == 0


def test_data_manifest_has_deterministic_nonoverlapping_splits():
    config, corpus, tokenizer, splits, _ = resources()
    manifest = corpus.manifest(tokenizer, config)
    hashes = [set(manifest["splits"][split]["document_hashes"].values()) for split in ("train", "validation", "test")]
    assert not hashes[0].intersection(hashes[1])
    assert not hashes[0].intersection(hashes[2])
    assert not hashes[1].intersection(hashes[2])
    assert manifest["source_url_or_local_source"] == "data/stage_d/synthetic_corpus.jsonl"
    assert manifest["fingerprint"]


def test_token_level_forward_shape_and_causality():
    config, _, tokenizer, _, batches = resources()
    seed_everything(config.seed)
    model = DCSSLanguageModel(tokenizer)
    ids = batches[0]["input_ids"]
    mask = batches[0]["attention_mask"]
    logits, state = model.forward_chunk(ids, attention_mask=mask)
    assert logits.shape == (*ids.shape, tokenizer.vocab_size)
    perturbed = ids.clone()
    perturbed[:, -1] = tokenizer.unk_id
    other, _ = model.forward_chunk(perturbed, attention_mask=mask)
    assert (logits[:, :-1] - other[:, :-1]).abs().max().item() <= 1e-6
    assert all(torch.isfinite(tensor).all() for tensor in state.tensors())


def test_masked_targets_do_not_change_loss_or_gradient():
    _, _, tokenizer, _, _ = resources()
    seed_everything(42)
    first = DCSSLanguageModel(tokenizer)
    second = DCSSLanguageModel(tokenizer)
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        first.output_bias[tokenizer.token_to_id["a"]] = 2.0
        first.output_bias[tokenizer.token_to_id["b"]] = -2.0
        second.output_bias.copy_(first.output_bias)
    ids = torch.tensor([[tokenizer.bos_id, tokenizer.token_to_id["a"], tokenizer.token_to_id["b"], tokenizer.pad_id]], dtype=torch.long)
    mask = torch.tensor([[True, True, False, False]])
    changed = ids.clone()
    changed[:, 2] = tokenizer.unk_id
    left = first.causal_loss(ids, mask)
    right = second.causal_loss(changed, mask)
    left.loss.backward()
    right.loss.backward()
    assert torch.allclose(left.loss, right.loss, atol=1e-7, rtol=0.0)
    assert torch.allclose(first.embedding.weight.grad, second.embedding.weight.grad, atol=1e-7, rtol=0.0)
    unmasked = ids.clone()
    unmasked[:, 1] = tokenizer.token_to_id["b"]
    assert not torch.allclose(left.loss.detach(), first.causal_loss(unmasked, mask).loss.detach())


def test_tiny_dcss_overfit_and_gradient_inventory():
    config, _, tokenizer, _, batches = resources()
    seed_everything(config.seed)
    model = build_model("dcss_cdi", tokenizer, config)
    losses, _, _ = train_steps(model, batches, config, steps=40)
    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] < losses[0]
    inventory = model.parameter_inventory()
    assert inventory["groups"]["token_embeddings"] > 0
    assert inventory["groups"]["gates"] > 0
    assert inventory["groups"]["generators"] > 0
    assert inventory["groups"]["sparse_geometry"] > 0


def test_checkpoint_resume_matches_uninterrupted():
    config, corpus, tokenizer, _, batches = resources()
    manifest = corpus.manifest(tokenizer, config)
    seed_everything(config.seed)
    uninterrupted = build_model("dcss_cdi", tokenizer, config)
    full_losses, _, _ = train_steps(uninterrupted, batches, config, steps=12)
    probe = batches[0]
    full_logits, _ = uninterrupted.forward_chunk(probe["input_ids"], attention_mask=probe["attention_mask"])

    seed_everything(config.seed)
    partial = build_model("dcss_cdi", tokenizer, config)
    first_losses, optimizer, cursor = train_steps(partial, batches, config, steps=6)
    payload = checkpoint_payload(partial, optimizer, tokenizer, manifest, config, step=6, cursor=cursor)
    resumed = build_model("dcss_cdi", tokenizer, config)
    resumed_optimizer = optimizer_for(resumed, config)
    step, restored_cursor = restore_checkpoint(
        payload,
        resumed,
        resumed_optimizer,
        tokenizer,
        expected_data_manifest=manifest,
        expected_config=config,
    )
    second_losses, _, _ = train_steps(resumed, batches, config, steps=6, optimizer=resumed_optimizer, start_cursor=restored_cursor)
    resumed_logits, _ = resumed.forward_chunk(probe["input_ids"], attention_mask=probe["attention_mask"])
    assert step == 6
    assert full_losses == first_losses + second_losses
    assert torch.allclose(full_logits, resumed_logits, atol=1e-6, rtol=1e-5)
    for left, right in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)


def test_shuffled_checkpoint_resume_matches_uninterrupted():
    config, _, tokenizer, _, batches = resources()
    seed_everything(config.seed)
    uninterrupted = build_model("dcss_cdi", tokenizer, config)
    full_losses, _, full_cursor = train_steps(uninterrupted, batches, config, steps=12, shuffle_each_epoch=True)
    probe = batches[0]
    full_logits, _ = uninterrupted.forward_chunk(probe["input_ids"], attention_mask=probe["attention_mask"])

    seed_everything(config.seed)
    partial = build_model("dcss_cdi", tokenizer, config)
    first_losses, optimizer, cursor = train_steps(partial, batches, config, steps=6, shuffle_each_epoch=True)
    resumed = build_model("dcss_cdi", tokenizer, config)
    resumed_optimizer = optimizer_for(resumed, config)
    resumed.load_state_dict(partial.state_dict())
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    second_losses, _, resumed_cursor = train_steps(
        resumed,
        batches,
        config,
        steps=6,
        optimizer=resumed_optimizer,
        start_cursor=cursor,
        shuffle_each_epoch=True,
    )
    resumed_logits, _ = resumed.forward_chunk(probe["input_ids"], attention_mask=probe["attention_mask"])
    assert full_cursor == resumed_cursor
    assert full_losses == first_losses + second_losses
    assert torch.allclose(full_logits, resumed_logits, atol=1e-6, rtol=1e-5)
    for left, right in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)


def test_deterministic_generation_and_matched_baselines_forward():
    config, _, tokenizer, _, batches = resources()
    seed_everything(config.seed)
    dcss = build_model("dcss_cdi", tokenizer, config)
    prefix = batches[0]["input_ids"][0, :3]
    greedy_one = dcss.generate(prefix, mode="greedy", max_new_tokens=4)
    greedy_two = dcss.generate(prefix, mode="greedy", max_new_tokens=4)
    sample_one = dcss.generate(prefix, mode="sample", seed=7, max_new_tokens=4)
    sample_two = dcss.generate(prefix, mode="sample", seed=7, max_new_tokens=4)
    assert torch.equal(greedy_one, greedy_two)
    assert torch.equal(sample_one, sample_two)
    assert int(sample_one.max()) < tokenizer.vocab_size
    for name in ("v2", "dcss_cdi", "transformer"):
        model = build_model(name, tokenizer, config)
        report = model.causal_loss(batches[0]["input_ids"], batches[0]["attention_mask"])
        assert torch.isfinite(report.loss)
