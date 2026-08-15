"""Registered CPU training-throughput benchmark for the compiled CDI fast path.

The benchmark keeps the frozen CCT workload: EthioBBPE, batch size 2,
float32 CPU execution, AdamW, and fixed lengths 16/64/256. Compilation is
performed before timing; measured steps include forward, backward, and the
optimizer update. Deferred runtime metrics are checked outside the compiled
recurrence and remain fail-closed.
"""
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


class CompiledDenseCDI(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        return self.model.forward_chunk_active(
            input_ids,
            return_state=False,
            runtime_guard_mode="deferred",
        )


def _guard_or_raise(metrics: tuple[torch.Tensor, torch.Tensor, torch.Tensor], model: torch.nn.Module) -> None:
    spectral_violation, max_geometry_energy, max_state_norm = metrics
    cell = model.ssm.cell
    if bool(spectral_violation.detach().item()):
        raise FloatingPointError("Deferred spectral-envelope guard failed.")
    if bool((max_geometry_energy > cell.stage_b_config.energy_limit).detach().item()):
        raise FloatingPointError("Deferred geometry-energy guard failed.")
    if bool((max_state_norm > cell.config.state_norm_bound).detach().item()):
        raise FloatingPointError("Deferred state-norm guard failed.")


def _run_length(tokenizer: EthioBBPETokenizer, length: int, warmup: int, measured: int) -> dict[str, object]:
    model = build_model("dcss_residual_cdi", tokenizer, seed=11).train()
    compiled = torch.compile(
        CompiledDenseCDI(model),
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=True,
    ).train()
    optimizer = torch.optim.AdamW(compiled.parameters(), lr=0.01)
    ids = torch.randint(1, tokenizer.vocab_size, (2, length), dtype=torch.long)

    def loss_fn() -> torch.Tensor:
        logits, _, metrics = compiled(ids)
        _guard_or_raise(metrics, model)
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, tokenizer.vocab_size),
            ids[:, 1:].reshape(-1),
        )

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
        "length": length,
        "batch_size": int(ids.shape[0]),
        "parameters": parameter_count(model),
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
    parser.add_argument("--output-dir", default="results/performance_10k_compiled")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--lengths", nargs="+", type=int, default=[16, 64, 256])
    args = parser.parse_args()

    torch.set_num_threads(1)
    tokenizer = EthioBBPETokenizer.from_pretrained(
        TokenizerConfig(max_chunk_length=16, embedding_dim=4)
    )
    result = {
        "benchmark": "performance_10k_compiled",
        "target_tokens_per_second": 10000,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "dtype": "float32",
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": True,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "models": {
            "dcss_residual_cdi": {
                str(length): _run_length(tokenizer, length, args.warmup, args.measured)
                for length in args.lengths
            }
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
