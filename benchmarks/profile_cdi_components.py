from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.ethiobbpe_synaxarium_pilot import build_model
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


def timed(fn, warmup: int, measured: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(measured):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return sum(values) / measured, values


def profile_length(tokenizer, length: int, warmup: int, measured: int) -> dict[str, object]:
    torch.manual_seed(11)
    model = build_model("dcss_residual_cdi", tokenizer, seed=11).train()
    ids = torch.randint(1, tokenizer.vocab_size, (2, length), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    components: dict[str, object] = {}

    def forward_only():
        model.forward_chunk_active(ids[:, :-1], return_state=False, runtime_guard_mode="deferred")

    def forward_with_loss():
        model.causal_loss(ids, mask).loss

    def forward_and_backward():
        optimizer.zero_grad(set_to_none=True)
        model.causal_loss(ids, mask).loss.backward()

    def optimizer_step():
        optimizer.step()

    for name, fn in (("forward_only", forward_only), ("forward_with_loss", forward_with_loss), ("forward_backward", forward_and_backward), ("optimizer_step", optimizer_step)):
        mean, values = timed(fn, warmup, measured)
        components[name] = {"mean_seconds": mean, "median_seconds": sorted(values)[len(values) // 2], "raw_seconds": values}

    return {"length": length, "batch_size": 2, "parameters": sum(p.numel() for p in model.parameters()), "components": components}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--output-dir", default="results/deep_optimization_components")
    args = parser.parse_args()
    torch.set_num_threads(1)
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    result = {"benchmark": "profile_cdi_components", "threads": 1, "lengths": [profile_length(tokenizer, length, args.warmup, args.measured) for length in args.lengths]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
