from __future__ import annotations

import argparse

import torch

from benchmarks.ethiobbpe_synaxarium_pilot import build_model
from cdi.v3 import EthioBBPETokenizer, TokenizerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--sort", default="self_cpu_time_total")
    parser.add_argument("--rows", type=int, default=30)
    args = parser.parse_args()

    torch.set_num_threads(1)
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    model = build_model("dcss_residual_cdi", tokenizer, seed=11).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ids = torch.randint(1, tokenizer.vocab_size, (2, args.length), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model.causal_loss(ids, mask).loss.backward()
        optimizer.step()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        optimizer.zero_grad(set_to_none=True)
        loss = model.causal_loss(ids, mask).loss
        loss.backward()
        optimizer.step()

    print(prof.key_averages().table(sort_by=args.sort, row_limit=args.rows))


if __name__ == "__main__":
    main()
