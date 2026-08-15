"""Eager CPU secondary benchmark for the registered CDI performance sprint."""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import torch

from benchmarks.ethiobbpe_synaxarium_pilot import build_model, parameter_count
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


def run_length(tokenizer: EthioBBPETokenizer, model_name: str, length: int, warmup: int, measured: int, cdi_tiled: bool = False) -> dict[str, object]:
    model = build_model(model_name, tokenizer, seed=11).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ids = torch.randint(1, tokenizer.vocab_size, (2, length), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)

    def loss_fn():
        if model_name == "dcss_residual_cdi" and cdi_tiled:
            return model.causal_loss(ids, mask, return_logits=False).loss
        return model.causal_loss(ids, mask).loss

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss_fn().backward()
        optimizer.step()
    timings = []
    for _ in range(measured):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        timings.append(time.perf_counter() - start)
    elapsed = sum(timings)
    return {
        "parameters": parameter_count(model),
        "length": length,
        "batch_size": int(ids.shape[0]),
        "warmup_steps": warmup,
        "measured_steps": measured,
        "mean_step_seconds": elapsed / measured,
        "median_step_seconds": sorted(timings)[len(timings) // 2],
        "tokens_per_second": ids.numel() * measured / elapsed,
        "last_loss": float(loss.detach()),
        "raw_step_seconds": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/performance_10k_eager")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--lengths", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--cdi-tiled", action="store_true", help="Use exact tiled vocabulary loss for CDI training steps.")
    args = parser.parse_args()

    torch.set_num_threads(1)
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    result = {
        "benchmark": "performance_10k_eager",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "dtype": "float32",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cdi_tiled_loss": bool(args.cdi_tiled),
        "models": {
            model_name: {
                str(length): run_length(tokenizer, model_name, length, args.warmup, args.measured, cdi_tiled=args.cdi_tiled)
                for length in args.lengths
            }
            for model_name in ("dcss_residual_cdi", "gru_baseline", "transformer")
        },
    }
    result["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    if result["peak_rss_gib"] >= 11.0:
        raise MemoryError(f"Peak RSS exceeded the 11 GiB guard: {result['peak_rss_gib']:.3f} GiB")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
