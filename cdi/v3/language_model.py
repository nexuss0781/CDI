"""Token-level causal language-model adapters for Stage D."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .ssm import CohomodynamicState, SelectiveCohomodynamicSSM, StageCConfig
from .tokenizer import EthioBBPETokenizer


@dataclass(frozen=True)
class LossReport:
    loss: torch.Tensor
    token_count: int
    logits: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor


class DCSSLanguageModel(nn.Module):
    """Causal token-level adapter around :class:`SelectiveCohomodynamicSSM`.

    The public ``forward_chunk`` API consumes token IDs directly. Padding keeps
    the recurrent state unchanged, and loss alignment is explicit: logits at
    position ``t`` are trained against token ``t+1``.
    """

    def __init__(self, tokenizer: EthioBBPETokenizer, config: StageCConfig | None = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config or StageCConfig.nano()
        self.config.validate()
        if tokenizer.config.embedding_dim != self.config.input_width or self.config.output_width != self.config.input_width:
            raise ValueError("Tokenizer embedding_dim and Stage C input/output widths must match for tied projection.")
        self.embedding = nn.Embedding(tokenizer.vocab_size, self.config.input_width, padding_idx=tokenizer.pad_id, dtype=self.config.dtype, device=self.config.device)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[tokenizer.pad_id].zero_()
        self.ssm = SelectiveCohomodynamicSSM(self.config)
        self.output_bias = nn.Parameter(torch.zeros(tokenizer.vocab_size, dtype=self.config.dtype, device=self.config.device))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def _select_state(self, old: CohomodynamicState, new: CohomodynamicState, active: torch.Tensor) -> CohomodynamicState:
        selector = active.unsqueeze(-1).unsqueeze(-1)
        return CohomodynamicState(*(torch.where(selector, new_tensor, old_tensor) for old_tensor, new_tensor in zip(old.tensors(), new.tensors())))

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        state: CohomodynamicState | None = None,
        attention_mask: torch.Tensor | None = None,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, CohomodynamicState] | torch.Tensor:
        squeezed = input_ids.ndim == 1
        if squeezed:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, length) or (length,).")
        input_ids = input_ids.to(dtype=torch.long, device=self.embedding.weight.device)
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_id)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
            if tuple(attention_mask.shape) != (batch, length):
                raise ValueError("attention_mask must match input_ids shape.")
        current = state if state is not None else self.ssm.initial_state(batch_shape=(batch,), mode="zero")
        embeddings = self.embedding(input_ids)
        hidden_steps = []
        for index in range(length):
            hidden, candidate = self.ssm.step(embeddings[:, index], current)
            active = attention_mask[:, index]
            current = self._select_state(current, candidate, active)
            hidden_steps.append(hidden * active.unsqueeze(-1).to(dtype=hidden.dtype))
        hidden_chunk = torch.stack(hidden_steps, dim=1)
        logits = F.linear(hidden_chunk, self.embedding.weight, self.output_bias)
        if squeezed:
            logits = logits.squeeze(0)
        if return_state:
            return logits, current
        return logits

    def causal_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> LossReport:
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("causal_loss expects (batch, length >= 2) token IDs.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_id)
        logits, _ = self.forward_chunk(input_ids[:, :-1], attention_mask=attention_mask[:, :-1], return_state=True)
        targets = input_ids[:, 1:].to(device=logits.device)
        loss_mask = attention_mask[:, 1:].to(dtype=torch.bool, device=logits.device) & attention_mask[:, :-1].to(dtype=torch.bool, device=logits.device)
        raw = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none")
        weights = loss_mask.reshape(-1).to(dtype=raw.dtype)
        count = int(loss_mask.sum().item())
        loss = (raw * weights).sum() / weights.sum().clamp_min(1.0)
        return LossReport(loss=loss, token_count=count, logits=logits, targets=targets, loss_mask=loss_mask)

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: Sequence[int] | torch.Tensor,
        max_new_tokens: int = 8,
        mode: Literal["greedy", "sample"] = "greedy",
        seed: int = 42,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        prefix = torch.as_tensor(prefix_ids, dtype=torch.long, device=self.embedding.weight.device).reshape(1, -1)
        if prefix.numel() == 0:
            prefix = torch.tensor([[self.tokenizer.bos_id]], dtype=torch.long, device=self.embedding.weight.device)
        logits, state = self.forward_chunk(prefix, return_state=True)
        generated = prefix[0].tolist()
        generator = torch.Generator(device=self.embedding.weight.device).manual_seed(seed)
        next_logits = logits[:, -1, :]
        for _ in range(max_new_tokens):
            if mode == "greedy":
                token = next_logits.argmax(dim=-1)
            elif mode == "sample":
                probabilities = torch.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                token = torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)
            else:
                raise ValueError("generation mode must be 'greedy' or 'sample'.")
            generated.append(int(token.item()))
            logits, state = self.forward_chunk(token.view(1, 1), state=state, attention_mask=torch.ones((1, 1), dtype=torch.bool, device=token.device), return_state=True)
            next_logits = logits[:, -1, :]
        return torch.tensor(generated, dtype=torch.long, device=prefix.device)

    def parameter_inventory(self) -> Dict[str, Any]:
        groups: Dict[str, int] = {"token_embeddings": 0, "output_projection_tied": 0, "gates": 0, "generators": 0, "memory_bands": 0, "initial_state_optional": 0, "sparse_geometry": 0, "cochain_maps": 0, "normalization_readout": 0}
        entries = []
        for name, parameter in self.named_parameters():
            count = parameter.numel()
            if name.startswith("embedding"):
                group = "token_embeddings"
            elif name.endswith("learned_initial_state"):
                group = "initial_state_optional"
            elif name == "output_bias":
                group = "normalization_readout"
            elif ".gate." in name:
                group = "gates"
            elif ".generator." in name:
                group = "generators"
            elif "geometry" in name:
                group = "sparse_geometry"
            elif "readout" in name:
                group = "normalization_readout"
            else:
                group = "memory_bands"
            groups[group] += count
            entries.append({"name": name, "group": group, "count": count, "shape": list(parameter.shape), "requires_grad": parameter.requires_grad})
        groups["output_projection_tied"] = self.embedding.weight.numel()
        return {"total_parameters": sum(parameter.numel() for parameter in self.parameters()), "groups": groups, "entries": entries, "tied_output": "embedding.weight"}


class LegacyCDIV2Adapter(nn.Module):
    """Compact legacy-style causal baseline for Stage D plumbing comparison.

    It is intentionally marked as an adapter, not a claim that dense CDI v2
    has been retrained at scale. It shares tokenizer, batches, loss, optimizer
    family, and evaluation code with the DCSS model.
    """

    def __init__(self, tokenizer: EthioBBPETokenizer, width: int = 4, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(tokenizer.vocab_size, width, padding_idx=tokenizer.pad_id, dtype=dtype)
        self.cell = nn.GRUCell(width, width, dtype=dtype)
        self.output_bias = nn.Parameter(torch.zeros(tokenizer.vocab_size, dtype=dtype))
        nn.init.normal_(self.embedding.weight, std=0.02)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def forward_chunk(self, input_ids: torch.Tensor, state: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, return_state: bool = True):
        if input_ids.ndim != 2:
            raise ValueError("Legacy adapter expects (batch, length) IDs.")
        batch, length = input_ids.shape
        active_mask = input_ids.ne(self.tokenizer.pad_id) if attention_mask is None else attention_mask.bool()
        hidden = torch.zeros(batch, self.embedding.embedding_dim, dtype=self.embedding.weight.dtype, device=input_ids.device) if state is None else state
        outputs = []
        for index in range(length):
            candidate = self.cell(self.embedding(input_ids[:, index]), hidden)
            active = active_mask[:, index].unsqueeze(-1)
            hidden = torch.where(active, candidate, hidden)
            outputs.append(hidden * active.to(hidden.dtype))
        features = torch.stack(outputs, dim=1)
        logits = F.linear(features, self.embedding.weight, self.output_bias)
        return (logits, hidden) if return_state else logits

    def causal_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> LossReport:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_id)
        logits, _ = self.forward_chunk(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
        targets = input_ids[:, 1:]
        mask = attention_mask[:, 1:] & attention_mask[:, :-1]
        raw = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none")
        weights = mask.reshape(-1).to(raw.dtype)
        return LossReport((raw * weights).sum() / weights.sum().clamp_min(1.0), int(mask.sum()), logits, targets, mask)


class TinyTransformerBaseline(nn.Module):
    """Small causal Transformer baseline used only for the matched synthetic protocol."""

    def __init__(self, tokenizer: EthioBBPETokenizer, width: int = 4, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(tokenizer.vocab_size, width, padding_idx=tokenizer.pad_id, dtype=dtype)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=1, dim_feedforward=width * 2, dropout=0.0, batch_first=True, dtype=dtype)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output_bias = nn.Parameter(torch.zeros(tokenizer.vocab_size, dtype=dtype))
        nn.init.normal_(self.embedding.weight, std=0.02)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def forward_chunk(self, input_ids: torch.Tensor, state: None = None, attention_mask: torch.Tensor | None = None, return_state: bool = True):
        if input_ids.ndim != 2:
            raise ValueError("Transformer baseline expects (batch, length) IDs.")
        _, length = input_ids.shape
        active = input_ids.ne(self.tokenizer.pad_id) if attention_mask is None else attention_mask.bool()
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1)
        features = self.encoder(self.embedding(input_ids), mask=causal_mask, src_key_padding_mask=~active)
        logits = F.linear(features, self.embedding.weight, self.output_bias)
        return (logits, None) if return_state else logits

    def causal_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> LossReport:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_id)
        logits, _ = self.forward_chunk(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
        targets = input_ids[:, 1:]
        mask = attention_mask[:, 1:] & attention_mask[:, :-1]
        raw = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none")
        weights = mask.reshape(-1).to(raw.dtype)
        return LossReport((raw * weights).sum() / weights.sum().clamp_min(1.0), int(mask.sum()), logits, targets, mask)
