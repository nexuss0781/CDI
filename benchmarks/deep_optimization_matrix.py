from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.ethiobbpe_synaxarium_pilot import build_model, parameter_count
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


class FusedGRULanguageModel(nn.Module):
    """Matched-width GRU using PyTorch's fused sequence GRU rather than GRUCell loops."""

    def __init__(self, tokenizer: EthioBBPETokenizer, width: int = 4, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(tokenizer.vocab_size, width, padding_idx=tokenizer.pad_id, dtype=dtype)
        self.gru = nn.GRU(width, width, num_layers=1, batch_first=True, dtype=dtype)
        self.output_bias = nn.Parameter(torch.zeros(tokenizer.vocab_size, dtype=dtype))
        nn.init.normal_(self.embedding.weight, std=0.02)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def causal_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        source = self.embedding(input_ids[:, :-1])
        hidden, _ = self.gru(source)
        logits = F.linear(hidden, self.embedding.weight, self.output_bias)
        targets = input_ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))


class TimedModel:
    def __init__(self, name: str, model: nn.Module, loss_fn):
        self.name = name
        self.model = model
        self.loss_fn = loss_fn


def build_timed_models(tokenizer: EthioBBPETokenizer, seed: int) -> list[TimedModel]:
    torch.manual_seed(seed)
    cdi = build_model("dcss_residual_cdi", tokenizer, seed=seed).train()
    torch.manual_seed(seed)
    cell_gru = build_model("gru_baseline", tokenizer, seed=seed).train()
    torch.manual_seed(seed)
    fused_gru = FusedGRULanguageModel(tokenizer).train()
    torch.manual_seed(seed)
    transformer = build_model("transformer", tokenizer, seed=seed).train()
    return [
        TimedModel("dcss_residual_cdi", cdi, lambda ids: cdi.causal_loss(ids, torch.ones_like(ids, dtype=torch.bool)).loss),
        TimedModel("gru_cell_adapter", cell_gru, lambda ids: cell_gru.causal_loss(ids, torch.ones_like(ids, dtype=torch.bool)).loss),
        TimedModel("torch_nn_gru_fused", fused_gru, fused_gru.causal_loss),
        TimedModel("transformer", transformer, lambda ids: transformer.causal_loss(ids, torch.ones_like(ids, dtype=torch.bool)).loss),
    ]


def time_model(timed: TimedModel, ids: torch.Tensor, warmup: int, measured: int) -> dict[str, object]:
    optimizer = torch.optim.AdamW(timed.model.parameters(), lr=0.01)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = timed.loss_fn(ids)
        loss.backward()
        optimizer.step()
    timings: list[float] = []
    for _ in range(measured):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = timed.loss_fn(ids)
        loss.backward()
        optimizer.step()
        timings.append(time.perf_counter() - started)
    elapsed = sum(timings)
    return {
        "parameters": parameter_count(timed.model),
        "length": int(ids.shape[1]),
        "batch_size": int(ids.shape[0]),
        "warmup_steps": warmup,
        "measured_steps": measured,
        "mean_step_seconds": elapsed / measured,
        "median_step_seconds": sorted(timings)[len(timings) // 2],
        "tokens_per_second": ids.numel() * measured / elapsed,
        "last_loss": float(loss.detach()),
        "raw_step_seconds": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--lengths", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--output-dir", default="results/deep_optimization_matrix")
    args = parser.parse_args()
    torch.set_num_threads(1)
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    result: dict[str, object] = {
        "benchmark": "deep_optimization_matrix",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "dtype": "float32",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "models": {},
    }
    models = build_timed_models(tokenizer, seed=11)
    for timed in models:
        by_length: dict[str, object] = {}
        for length in args.lengths:
            ids = torch.randint(1, tokenizer.vocab_size, (2, length), dtype=torch.long)
            by_length[str(length)] = time_model(timed, ids, args.warmup, args.measured)
        result["models"][timed.name] = by_length
    result["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
