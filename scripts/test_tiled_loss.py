from __future__ import annotations

import torch

from benchmarks.ethiobbpe_synaxarium_pilot import build_model
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


torch.manual_seed(17)
tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=32, embedding_dim=4))
full = build_model("dcss_residual_cdi", tokenizer, seed=17).train()
tiled = build_model("dcss_residual_cdi", tokenizer, seed=17).train()
tiled.load_state_dict(full.state_dict())
ids = torch.randint(1, tokenizer.vocab_size, (2, 32), dtype=torch.long)
mask = torch.ones_like(ids, dtype=torch.bool)
full_report = full.causal_loss(ids, mask, return_logits=True)
tiled_report = tiled.causal_loss(ids, mask, return_logits=False, vocab_tile_size=1024)
full_report.loss.backward()
tiled_report.loss.backward()
loss_error = float((full_report.loss - tiled_report.loss).abs())
gradient_error = max(
    float((left.grad - right.grad).abs().max())
    for left, right in zip(full.parameters(), tiled.parameters())
    if left.grad is not None and right.grad is not None
)
print({"loss_error": loss_error, "gradient_error": gradient_error, "tiled_logits_numel": tiled_report.logits.numel(), "full_logits_numel": full_report.logits.numel()})
assert loss_error < 1e-6
assert gradient_error < 1e-5
assert tiled_report.logits.numel() == 0
assert full_report.logits.shape == (2, 31, tokenizer.vocab_size)
