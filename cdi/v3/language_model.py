"""Token-level causal language-model adapters for Stage D."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .ssm import CohomodynamicState, GateValues, SelectiveCohomodynamicSSM, StageCConfig
from .tokenizer import EthioBBPETokenizer


@dataclass(frozen=True)
class LossReport:
    loss: torch.Tensor
    token_count: int
    logits: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor


def _tiled_cross_entropy(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    output_bias: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """Compute exact causal cross-entropy in vocabulary tiles."""
    if tile_size <= 0:
        raise ValueError("vocab tile_size must be positive")
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(dtype=hidden.dtype)
    target_logits = torch.zeros(flat_targets.shape, dtype=hidden.dtype, device=hidden.device)
    log_partition: torch.Tensor | None = None
    vocabulary = embedding_weight.shape[0]
    for start in range(0, vocabulary, tile_size):
        end = min(start + tile_size, vocabulary)
        tile_logits = F.linear(flat_hidden, embedding_weight[start:end], output_bias[start:end])
        tile_log_partition = torch.logsumexp(tile_logits, dim=-1)
        log_partition = tile_log_partition if log_partition is None else torch.logaddexp(log_partition, tile_log_partition)
        in_tile = (flat_targets >= start) & (flat_targets < end)
        local_targets = (flat_targets - start).clamp(min=0, max=end - start - 1)
        gathered = tile_logits.gather(1, local_targets.unsqueeze(-1)).squeeze(-1)
        target_logits = torch.where(in_tile, gathered, target_logits)
    if log_partition is None:
        raise ValueError("vocabulary must contain at least one token")
    raw = log_partition - target_logits
    return (raw * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)


class SelectiveTokenResidual(nn.Module):
    """Bounded causal source-token residual for the CCT-G3.4 readout candidate."""

    def __init__(self, width: int, *, dtype: torch.dtype, device: str) -> None:
        super().__init__()
        self.value_projection = nn.Linear(width, width, dtype=dtype, device=device)
        self.gate_projection = nn.Linear(width, width, dtype=dtype, device=device)
        for layer in (self.value_projection, self.gate_projection):
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, source_embedding: torch.Tensor, *, ablated: bool) -> torch.Tensor:
        if ablated:
            return torch.zeros_like(source_embedding)
        return torch.sigmoid(self.gate_projection(source_embedding)) * torch.tanh(self.value_projection(source_embedding))


class StateConditionedResidualFusion(nn.Module):
    """Bounded causal fusion of the DCSS readout and CCT-G3.4 token residual."""

    def __init__(self, width: int, *, dtype: torch.dtype, device: str) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(width * 2, width, dtype=dtype, device=device)
        nn.init.xavier_uniform_(self.gate_projection.weight, gain=0.25)
        nn.init.constant_(self.gate_projection.bias, 2.0)

    def forward(self, state_readout: torch.Tensor, token_residual: torch.Tensor, *, ablated: bool) -> torch.Tensor:
        if ablated:
            gate = torch.ones_like(state_readout)
        else:
            gate = torch.sigmoid(self.gate_projection(torch.cat((state_readout, token_residual), dim=-1)))
        return state_readout + gate * token_residual


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
        self.token_residual = (
            SelectiveTokenResidual(self.config.input_width, dtype=self.config.dtype, device=self.config.device)
            if self.config.token_residual_enabled
            else None
        )
        self.residual_fusion = (
            StateConditionedResidualFusion(self.config.input_width, dtype=self.config.dtype, device=self.config.device)
            if self.config.residual_fusion_enabled
            else None
        )
        self.output_bias = nn.Parameter(torch.zeros(tokenizer.vocab_size, dtype=self.config.dtype, device=self.config.device))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def expected_inactive_trainable_parameters(self) -> frozenset[str]:
        """Return every parameter intentionally disconnected by an exact ablation."""

        inactive: set[str] = set()
        if self.config.geometry_ablation or self.config.contrast_readout_ablation:
            inactive.add("ssm.cell.geometry.edge_log_weights")
        if self.config.harmonic_ablation:
            inactive.update(
                name
                for name, parameter in self.named_parameters()
                if name.startswith("ssm.cell.bands.harmonic.") and parameter.requires_grad
            )
        if self.config.token_residual_ablation:
            inactive.update(
                name
                for name, parameter in self.named_parameters()
                if name.startswith("token_residual.") and parameter.requires_grad
            )
        if self.config.residual_fusion_ablation:
            inactive.update(
                name
                for name, parameter in self.named_parameters()
                if name.startswith("residual_fusion.") and parameter.requires_grad
            )
        return frozenset(inactive)

    def _select_state(self, old: CohomodynamicState, new: CohomodynamicState, active: torch.Tensor) -> CohomodynamicState:
        selector = active.unsqueeze(-1).unsqueeze(-1)
        return CohomodynamicState(*(torch.where(selector, new_tensor, old_tensor) for old_tensor, new_tensor in zip(old.tensors(), new.tensors())))

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        state: CohomodynamicState | None = None,
        attention_mask: torch.Tensor | None = None,
        return_state: bool = True,
        *,
        return_logits: bool = True,
        runtime_guard_mode: Literal["python", "tensor", "deferred", "disabled"] = "python",
    ) -> Tuple[torch.Tensor, CohomodynamicState] | torch.Tensor | Tuple[torch.Tensor, CohomodynamicState, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        all_active = bool(attention_mask.all().item())
        if all_active:
            dense_result = self._forward_active_embeddings(
                embeddings,
                current,
                return_state=return_state,
                squeezed=squeezed,
                runtime_guard_mode=runtime_guard_mode,
                return_hidden=not return_logits,
            )
            if runtime_guard_mode == "deferred":
                dense_logits, dense_state, metrics = dense_result
                spectral_violation, max_geometry_energy, max_state_norm = metrics
                if bool(spectral_violation.detach().item()):
                    raise FloatingPointError("Deferred spectral-envelope guard failed.")
                if bool((max_geometry_energy > self.ssm.cell.stage_b_config.energy_limit).detach().item()):
                    raise FloatingPointError("Deferred geometry-energy guard failed.")
                if bool((max_state_norm > self.ssm.cell.config.state_norm_bound).detach().item()):
                    raise FloatingPointError("Deferred state-norm guard failed.")
                if return_state:
                    return dense_logits, dense_state, metrics
                return dense_logits, None, metrics
            dense_logits, dense_state = dense_result
            if return_state:
                return dense_logits, dense_state
            return dense_logits
        fused_gate_sequence = self.ssm.cell.fused_gate_values(embeddings)
        hidden_steps = []
        for index in range(length):
            source_embedding = embeddings[:, index]
            step_gates = {
                name: GateValues(
                    forcing=values.forcing[:, index],
                    input_gate=values.input_gate[:, index],
                    transport_gate=values.transport_gate[:, index],
                    log_timescale_offset=values.log_timescale_offset[:, index],
                    geometry_gate=values.geometry_gate[:, index],
                )
                for name, values in fused_gate_sequence.items()
            }
            hidden, candidate = self.ssm.step(source_embedding, current, fused_gates=step_gates)
            if self.token_residual is not None:
                residual = self.token_residual(source_embedding, ablated=self.config.token_residual_ablation)
                if self.residual_fusion is not None:
                    hidden = self.residual_fusion(hidden, residual, ablated=self.config.residual_fusion_ablation)
                else:
                    hidden = hidden + residual
            if all_active:
                current = candidate
                hidden_steps.append(hidden)
            else:
                active = attention_mask[:, index]
                current = self._select_state(current, candidate, active)
                hidden_steps.append(hidden * active.unsqueeze(-1).to(dtype=hidden.dtype))
        hidden_chunk = torch.stack(hidden_steps, dim=1)
        logits = hidden_chunk if not return_logits else F.linear(hidden_chunk, self.embedding.weight, self.output_bias)
        if squeezed:
            logits = logits.squeeze(0)
        if return_state:
            return logits, current
        return logits

    def _forward_active_embeddings(
        self,
        embeddings: torch.Tensor,
        state: CohomodynamicState,
        *,
        return_state: bool,
        squeezed: bool = False,
        runtime_guard_mode: Literal["python", "tensor", "deferred", "disabled"] = "python",
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, CohomodynamicState | None] | Tuple[torch.Tensor, CohomodynamicState, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Fast exact path for dense causal chunks with no padding."""
        current = state
        deferred_guards = runtime_guard_mode == "deferred"
        if deferred_guards:
            max_spectral_violation = torch.zeros((), dtype=torch.bool, device=embeddings.device)
            max_geometry_energy = torch.zeros((), dtype=embeddings.dtype, device=embeddings.device)
            max_state_norm = torch.zeros((), dtype=embeddings.dtype, device=embeddings.device)
        token_residual_sequence = (
            self.token_residual(embeddings, ablated=self.config.token_residual_ablation)
            if self.token_residual is not None
            else None
        )
        flat_fused = not self.ssm.cell.disable_harmonic and self.ssm.cell.unconstrained_cochain is None
        if flat_fused:
            forcing, input_gate, transport_gate, offsets, geometry = self.ssm.cell.fused_gate_tensors(embeddings)
            kernel_tensors = self.ssm.cell.fused_kernel_tensors()
            geometry_operator = self.ssm.cell.geometry.operator(
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            stacked_current = torch.stack(current.tensors(), dim=-3)
            use_scan = runtime_guard_mode in ("deferred", "disabled")
            if use_scan:
                stacked_current, trajectory_time_major = self.ssm.cell.scan_fused_stacked(
                    forcing,
                    input_gate,
                    transport_gate,
                    offsets,
                    geometry,
                    stacked_current,
                    kernel_tensors=kernel_tensors,
                    geometry_operator=geometry_operator,
                )
                stacked_trajectory = trajectory_time_major.transpose(0, 1)
                if deferred_guards:
                    spectral_violation = (
                        self.config.geometry_step_cap
                        * geometry
                        * (2.0 * (self.config.n_vertices - 1) * self.config.max_geometry_edge_weight)
                        > 1.0
                    ).any()
                    trajectory_energy = self.ssm.cell.geometry.energy(trajectory_time_major)
                    trajectory_norm = torch.linalg.vector_norm(trajectory_time_major, dim=(-2, -1))
                    max_spectral_violation = torch.logical_or(max_spectral_violation, spectral_violation)
                    max_geometry_energy = torch.maximum(max_geometry_energy, trajectory_energy.max())
                    max_state_norm = torch.maximum(max_state_norm, trajectory_norm.max())
            else:
                state_steps = []
                for index in range(embeddings.shape[1]):
                    step_result = self.ssm.cell.step_fused_stacked(
                        forcing[:, index],
                        input_gate[:, index],
                        transport_gate[:, index],
                        offsets[:, index],
                        geometry[:, index],
                        stacked_current,
                        runtime_guard_mode=runtime_guard_mode,
                        return_runtime_metrics=deferred_guards,
                        store_diagnostics=runtime_guard_mode in ("python", "tensor"),
                        kernel_tensors=kernel_tensors,
                        geometry_operator=geometry_operator,
                        return_output=False,
                    )
                    if deferred_guards:
                        _, stacked_current, step_metrics = step_result
                        spectral_violation, geometry_energy, state_norm = step_metrics
                        max_spectral_violation = torch.logical_or(max_spectral_violation, spectral_violation)
                        max_geometry_energy = torch.maximum(max_geometry_energy, geometry_energy.max())
                        max_state_norm = torch.maximum(max_state_norm, state_norm.max())
                    else:
                        _, stacked_current = step_result
                    state_steps.append(stacked_current)
                stacked_trajectory = torch.stack(state_steps, dim=1)
            hidden_chunk = self.ssm.cell.readout(self.ssm.cell._readout_features_stacked(stacked_trajectory))
            if token_residual_sequence is not None:
                if self.residual_fusion is not None:
                    hidden_chunk = self.residual_fusion(
                        hidden_chunk,
                        token_residual_sequence,
                        ablated=self.config.residual_fusion_ablation,
                    )
                else:
                    hidden_chunk = hidden_chunk + token_residual_sequence
            current = CohomodynamicState(*(stacked_current.select(-3, index) for index in range(len(current.tensors()))))
        else:
            hidden_steps = []
            fused_gate_sequence = self.ssm.cell.fused_gate_values(embeddings)
            for index in range(embeddings.shape[1]):
                source_embedding = embeddings[:, index]
                step_gates = {
                    name: GateValues(
                        forcing=values.forcing[:, index],
                        input_gate=values.input_gate[:, index],
                        transport_gate=values.transport_gate[:, index],
                        log_timescale_offset=values.log_timescale_offset[:, index],
                        geometry_gate=values.geometry_gate[:, index],
                    )
                    for name, values in fused_gate_sequence.items()
                }
                step_result = self.ssm.step(
                    source_embedding,
                    current,
                    fused_gates=step_gates,
                    runtime_guard_mode=runtime_guard_mode,
                    return_runtime_metrics=deferred_guards,
                )
                if deferred_guards:
                    hidden, current, step_metrics = step_result
                    spectral_violation, geometry_energy, state_norm = step_metrics
                    max_spectral_violation = torch.logical_or(max_spectral_violation, spectral_violation)
                    max_geometry_energy = torch.maximum(max_geometry_energy, geometry_energy.max())
                    max_state_norm = torch.maximum(max_state_norm, state_norm.max())
                else:
                    hidden, current = step_result
                if self.token_residual is not None:
                    residual = token_residual_sequence[:, index]
                    if self.residual_fusion is not None:
                        hidden = self.residual_fusion(hidden, residual, ablated=self.config.residual_fusion_ablation)
                    else:
                        hidden = hidden + residual
                hidden_steps.append(hidden)
            hidden_chunk = torch.stack(hidden_steps, dim=1)
        logits = hidden_chunk if return_hidden else F.linear(hidden_chunk, self.embedding.weight, self.output_bias)
        if squeezed:
            logits = logits.squeeze(0)
        if deferred_guards:
            return logits, current if return_state else None, (max_spectral_violation, max_geometry_energy, max_state_norm)
        return logits, current if return_state else None

    def forward_chunk_active(
        self,
        input_ids: torch.Tensor,
        *,
        state: CohomodynamicState | None = None,
        return_state: bool = False,
        runtime_guard_mode: Literal["python", "tensor", "deferred", "disabled"] = "python",
    ) -> Tuple[torch.Tensor, CohomodynamicState | None] | Tuple[torch.Tensor, CohomodynamicState, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compile-friendly exact path for fixed dense chunks."""
        input_ids = input_ids.to(dtype=torch.long, device=self.embedding.weight.device)
        embeddings = self.embedding(input_ids)
        current = state if state is not None else self.ssm.initial_state(batch_shape=(input_ids.shape[0],), mode="zero")
        return self._forward_active_embeddings(
            embeddings,
            current,
            return_state=return_state,
            runtime_guard_mode=runtime_guard_mode,
        )

    def causal_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_logits: bool = True,
        vocab_tile_size: int = 4096,
    ) -> LossReport:
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("causal_loss expects (batch, length >= 2) token IDs.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_id)
        source_mask = attention_mask[:, :-1]
        forward_result = self.forward_chunk(
            input_ids[:, :-1],
            attention_mask=source_mask,
            return_state=True,
            return_logits=return_logits,
            runtime_guard_mode="deferred",
        )
        hidden_or_logits = forward_result[0]
        targets = input_ids[:, 1:].to(device=hidden_or_logits.device)
        loss_mask = attention_mask[:, 1:].to(dtype=torch.bool, device=hidden_or_logits.device) & source_mask.to(dtype=torch.bool, device=hidden_or_logits.device)
        count = int(loss_mask.sum().item())
        if return_logits:
            raw = F.cross_entropy(hidden_or_logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none")
            weights = loss_mask.reshape(-1).to(dtype=raw.dtype)
            loss = (raw * weights).sum() / weights.sum().clamp_min(1.0)
            logits = hidden_or_logits
        else:
            loss = _tiled_cross_entropy(
                hidden_or_logits,
                targets,
                loss_mask,
                self.embedding.weight,
                self.output_bias,
                vocab_tile_size,
            )
            logits = torch.empty(0, dtype=hidden_or_logits.dtype, device=hidden_or_logits.device)
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
        groups: Dict[str, int] = {"token_embeddings": 0, "output_projection_tied": 0, "gates": 0, "generators": 0, "memory_bands": 0, "initial_state_optional": 0, "sparse_geometry": 0, "cochain_maps": 0, "normalization_readout": 0, "selective_token_residual": 0, "state_conditioned_fusion": 0}
        entries = []
        for name, parameter in self.named_parameters():
            count = parameter.numel()
            if name.startswith("residual_fusion."):
                group = "state_conditioned_fusion"
            elif name.startswith("token_residual."):
                group = "selective_token_residual"
            elif name.startswith("embedding"):
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
