"""Stage C: stable selective cohomodynamic state-space recurrence.

The production path stores a factorized vertex state for each memory band and
uses no dense ``(total_state_dim, total_state_dim)`` state operator.  Each
band integrates a diagonal dissipative plus pairwise skew generator exactly,
then receives a bounded matrix-free Stage B Laplacian correction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from hashlib import sha256
from math import log
from typing import Any, Dict, Iterable, Literal, Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import DCSSConfig
from .laplacian import MatrixFreeLaplacian
from .topology import SparseTopology


BAND_NAMES: Tuple[str, str, str] = ("fast", "middle", "harmonic")


@dataclass(frozen=True)
class StageCConfig:
    """CPU-first Stage C configuration for the structured selective recurrence.

    States have shape ``(..., n_vertices, band_width)`` per band. The nano
    configuration therefore has ``4 * 4 * 3 = 48`` factorized elements, below
    the project-wide rapid-iteration limit of 64 while avoiding a dense lift.
    """

    name: str = "nano"
    n_vertices: int = 4
    input_width: int = 4
    output_width: int = 4
    band_width: int = 4
    seed: int = 42
    dtype_str: str = "float32"
    device: str = "cpu"
    geometry_ablation: bool = False
    contrast_readout_ablation: bool = False
    harmonic_ablation: bool = False
    token_residual_enabled: bool = False
    token_residual_ablation: bool = False
    residual_fusion_enabled: bool = False
    residual_fusion_ablation: bool = False
    dt: float = 0.10
    geometry_step_cap: float = 0.02
    max_geometry_edge_weight: float = 2.0
    tau_min: float = 0.05
    tau_max: float = 64.0
    max_log_timescale_offset: float = 0.35
    rotation_limit: float = 1.0
    input_bound: float = 1.0
    state_norm_bound: float = 1.0e4
    band_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.25, 1.0),
        (2.0, 8.0),
        (16.0, 64.0),
    )

    @property
    def dtype(self) -> torch.dtype:
        try:
            return getattr(torch, self.dtype_str)
        except AttributeError as exc:
            raise ValueError(f"Unsupported Stage C dtype: {self.dtype_str}") from exc

    @property
    def total_state_dim(self) -> int:
        return self.n_vertices * self.band_width * len(BAND_NAMES)

    @property
    def per_band_state_dim(self) -> int:
        return self.n_vertices * self.band_width

    def validate(self) -> None:
        if self.token_residual_ablation and not self.token_residual_enabled:
            raise ValueError("token_residual_ablation requires token_residual_enabled.")
        if self.residual_fusion_enabled and not self.token_residual_enabled:
            raise ValueError("residual_fusion_enabled requires token_residual_enabled.")
        if self.residual_fusion_ablation and not self.residual_fusion_enabled:
            raise ValueError("residual_fusion_ablation requires residual_fusion_enabled.")
        if self.name != "nano":
            raise ValueError("Stage C currently exposes only the CPU-safe 'nano' tier.")
        if self.n_vertices < 3:
            raise ValueError("Stage C requires at least three topology vertices.")
        if self.input_width <= 0 or self.output_width <= 0:
            raise ValueError("Input and output widths must be positive.")
        if self.band_width <= 0 or self.band_width % 2:
            raise ValueError("band_width must be a positive even integer for pairwise skew rotations.")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Stage C supports float32 production and float64 reference checks only.")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("Stage C accepts only CPU or CUDA device strings.")
        if not 0.0 < self.dt <= 1.0:
            raise ValueError("dt must be in (0, 1].")
        if not 0.0 <= self.geometry_step_cap <= 0.05:
            raise ValueError("geometry_step_cap must be in [0, 0.05] for the nano stability envelope.")
        if not 0.0 < self.max_geometry_edge_weight <= 2.0:
            raise ValueError("max_geometry_edge_weight must lie in (0, 2] for the nano stability envelope.")
        # A graph with at most n_vertices - 1 incident edges has lambda_max(L)
        # bounded by 2 * (n_vertices - 1) * max_edge_weight. The explicit
        # correction must remain non-expansive in that conservative envelope.
        if self.geometry_step_cap * 2.0 * (self.n_vertices - 1) * self.max_geometry_edge_weight > 1.0:
            raise ValueError("geometry_step_cap and max_geometry_edge_weight exceed the explicit correction stability envelope.")
        if not 0.0 < self.tau_min <= self.tau_max:
            raise ValueError("Timescale bounds must be finite and positive.")
        if len(self.band_ranges) != len(BAND_NAMES):
            raise ValueError("Every declared memory band needs one timescale range.")
        previous_upper = 0.0
        for lower, upper in self.band_ranges:
            if not self.tau_min <= lower <= upper <= self.tau_max:
                raise ValueError("Each band range must be ordered and within global timescale bounds.")
            if lower <= previous_upper:
                raise ValueError("Frequency-cascade bands must have strictly separated timescale ranges.")
            previous_upper = upper
        if self.total_state_dim >= 64:
            raise ValueError("The Stage C nano tier must remain below total_state_dim 64.")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def nano(
        cls,
        seed: int = 42,
        geometry_ablation: bool = False,
        contrast_readout_ablation: bool = False,
        harmonic_ablation: bool = False,
        token_residual_enabled: bool = False,
        token_residual_ablation: bool = False,
        residual_fusion_enabled: bool = False,
        residual_fusion_ablation: bool = False,
    ) -> "StageCConfig":
        if token_residual_ablation and not token_residual_enabled:
            raise ValueError("token_residual_ablation requires token_residual_enabled.")
        if residual_fusion_enabled and not token_residual_enabled:
            raise ValueError("residual_fusion_enabled requires token_residual_enabled.")
        if residual_fusion_ablation and not residual_fusion_enabled:
            raise ValueError("residual_fusion_ablation requires residual_fusion_enabled.")
        config = cls(
            seed=seed,
            geometry_ablation=geometry_ablation,
            contrast_readout_ablation=contrast_readout_ablation,
            harmonic_ablation=harmonic_ablation,
            token_residual_enabled=token_residual_enabled,
            token_residual_ablation=token_residual_ablation,
            residual_fusion_enabled=residual_fusion_enabled,
            residual_fusion_ablation=residual_fusion_ablation,
        )
        config.validate()
        return config

    def stage_b_config(self) -> DCSSConfig:
        config = DCSSConfig(
            name="nano",
            n_vertices=self.n_vertices,
            state_width=self.band_width,
            cover_k=2,
            seed=self.seed,
            dtype_str=self.dtype_str,
            device=self.device,
            geometry_ablation=self.geometry_ablation,
            max_geometry_edge_weight=self.max_geometry_edge_weight,
        )
        config.validate()
        return config


@dataclass(frozen=True)
class CohomodynamicState:
    """Structured three-band state; each tensor has ``(..., vertices, width)``."""

    fast: torch.Tensor
    middle: torch.Tensor
    harmonic: torch.Tensor

    def by_name(self, name: str) -> torch.Tensor:
        if name == "fast":
            return self.fast
        if name == "middle":
            return self.middle
        if name == "harmonic":
            return self.harmonic
        raise KeyError(name)

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.fast, self.middle, self.harmonic

    def with_band(self, name: str, value: torch.Tensor) -> "CohomodynamicState":
        values = {band: self.by_name(band) for band in BAND_NAMES}
        values[name] = value
        return CohomodynamicState(values["fast"], values["middle"], values["harmonic"])

    def detach(self) -> "CohomodynamicState":
        return CohomodynamicState(*(tensor.detach() for tensor in self.tensors()))


@dataclass(frozen=True)
class GateValues:
    """Bounded content-dependent values emitted by one selective gate."""

    forcing: torch.Tensor
    input_gate: torch.Tensor
    transport_gate: torch.Tensor
    log_timescale_offset: torch.Tensor
    geometry_gate: torch.Tensor


@dataclass(frozen=True)
class GeneratorParameters:
    """Stable per-token coefficients for one memory band."""

    forcing: torch.Tensor
    dissipation: torch.Tensor
    omega: torch.Tensor
    geometry_gate: torch.Tensor
    tau: torch.Tensor


class SelectiveGate(nn.Module):
    """Bounded content-dependent forcing, gates, and finite timescale offsets."""

    def __init__(self, input_width: int, band_width: int, max_log_timescale_offset: float) -> None:
        super().__init__()
        pair_width = band_width // 2
        self.forcing_projection = nn.Linear(input_width, band_width)
        self.input_gate_projection = nn.Linear(input_width, pair_width)
        self.transport_projection = nn.Linear(input_width, pair_width)
        self.timescale_projection = nn.Linear(input_width, band_width)
        self.geometry_projection = nn.Linear(input_width, 1)
        self.max_log_timescale_offset = float(max_log_timescale_offset)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> GateValues:
        if x.ndim < 1:
            raise ValueError("SelectiveGate expects an input tensor with a feature axis.")
        forcing = torch.tanh(self.forcing_projection(x))
        input_gate = torch.sigmoid(self.input_gate_projection(x))
        transport_gate = torch.sigmoid(self.transport_projection(x))
        offset = torch.tanh(self.timescale_projection(x)) * self.max_log_timescale_offset
        geometry_gate = torch.sigmoid(self.geometry_projection(x)).squeeze(-1)
        return GateValues(
            forcing=forcing * input_gate.repeat_interleave(2, dim=-1),
            input_gate=input_gate,
            transport_gate=transport_gate,
            log_timescale_offset=offset,
            geometry_gate=geometry_gate,
        )


class StableGenerator(nn.Module):
    """Diagonal dissipative plus pairwise skew generator for one memory band.

    The trainable ``log_tau_base`` is initialized by the configured logarithmic
    frequency cascade. The conservative component is represented only by one
    angular value per pair, avoiding an ``O(width²)`` per-token matrix.
    """

    def __init__(self, band_width: int, tau_range: Tuple[float, float], config: StageCConfig, gate: SelectiveGate) -> None:
        super().__init__()
        self.band_width = band_width
        self.pair_width = band_width // 2
        self.config = config
        self.gate = gate
        self.tau_range = tau_range
        lower, upper = tau_range
        initial = torch.linspace(log(lower), log(upper), self.band_width, dtype=config.dtype, device=config.device)
        self.log_tau_base = nn.Parameter(initial)
        self.rotation_bias = nn.Parameter(torch.zeros(self.pair_width, dtype=config.dtype, device=config.device))
        injection = torch.empty(config.n_vertices, band_width, dtype=config.dtype, device=config.device)
        nn.init.xavier_uniform_(injection)
        self.input_injection = nn.Parameter(injection)

    def parameters_from_gates(self, gates: GateValues, dissipation_scale: float = 1.0) -> GeneratorParameters:
        lower, upper = self.tau_range
        # Content can adapt timescales only within the finite, disjoint band
        # interval declared by the frequency cascade.
        log_tau = (self.log_tau_base + gates.log_timescale_offset).clamp(
            min=max(log(self.config.tau_min), log(lower)),
            max=min(log(self.config.tau_max), log(upper)),
        )
        tau = torch.exp(log_tau)
        # A positive floor makes the normal operating mode strictly dissipative.
        dissipation = (0.05 + gates.input_gate) / tau[..., ::2]
        dissipation = dissipation * float(dissipation_scale)
        omega = (2.0 * gates.transport_gate - 1.0 + self.rotation_bias) * self.config.rotation_limit
        return GeneratorParameters(
            forcing=gates.forcing,
            dissipation=dissipation,
            omega=omega,
            geometry_gate=gates.geometry_gate,
            tau=tau,
        )

    def parameters_from_input(self, x: torch.Tensor, dissipation_scale: float = 1.0) -> GeneratorParameters:
        return self.parameters_from_gates(self.gate(x), dissipation_scale=dissipation_scale)

    def apply(self, z: torch.Tensor, params: GeneratorParameters) -> torch.Tensor:
        """Apply the continuous generator without materializing its matrix."""
        pairs = z.reshape(*z.shape[:-1], self.pair_width, 2)
        omega = params.omega.unsqueeze(-2)
        lam = params.dissipation.unsqueeze(-2)
        first = -lam * pairs[..., 0] - omega * pairs[..., 1]
        second = omega * pairs[..., 0] - lam * pairs[..., 1]
        return torch.stack((first, second), dim=-1).reshape_as(z)

    def energy_terms(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        energy = 0.5 * z.square().sum(dim=(-2, -1))
        return {"energy": energy, "norm": torch.linalg.vector_norm(z, dim=(-2, -1))}


class CayleyIntegrator(nn.Module):
    """Exact discretization of diagonal-plus-pairwise-skew stable blocks.

    The exact pair update is equivalent to exponentiating the block generator
    ``-lambda I + [[0,-omega],[omega,0]]``. It is used instead of a Cayley
    solve because it preserves the same stability contract without a per-token
    dense state matrix.
    """

    def __init__(self, band_width: int) -> None:
        super().__init__()
        self.band_width = band_width
        self.pair_width = band_width // 2

    def step(self, z: torch.Tensor, u: torch.Tensor, params: GeneratorParameters, dt: float) -> torch.Tensor:
        pairs = z.reshape(*z.shape[:-1], self.pair_width, 2)
        angle = (params.omega * dt).unsqueeze(-2)
        decay = torch.exp((-params.dissipation * dt).unsqueeze(-2))
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotated_first = cosine * pairs[..., 0] - sine * pairs[..., 1]
        rotated_second = sine * pairs[..., 0] + cosine * pairs[..., 1]
        homogeneous = torch.stack((rotated_first, rotated_second), dim=-1) * decay.unsqueeze(-1)
        return homogeneous.reshape_as(z) + dt * u

    def conservative_step(self, z: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
        """Diagnostic-only zero-dissipation rotation used to test energy preservation."""
        pairs = z.reshape(*z.shape[:-1], self.pair_width, 2)
        angle = (omega * dt).unsqueeze(-2)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        result = torch.stack(
            (cosine * pairs[..., 0] - sine * pairs[..., 1], sine * pairs[..., 0] + cosine * pairs[..., 1]),
            dim=-1,
        )
        return result.reshape_as(z)


class MemoryBand(nn.Module):
    """One selective stable memory band with an independently initialized timescale range."""

    def __init__(self, name: str, tau_range: Tuple[float, float], config: StageCConfig) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.tau_range = tau_range
        self.gate = SelectiveGate(config.input_width, config.band_width, config.max_log_timescale_offset)
        self.generator = StableGenerator(config.band_width, tau_range, config, self.gate)
        self.integrator = CayleyIntegrator(config.band_width)

    @property
    def state_size(self) -> int:
        return self.config.per_band_state_dim

    def reset(self, batch_shape: Sequence[int] = (), device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.zeros(
            *batch_shape,
            self.config.n_vertices,
            self.config.band_width,
            device=device or self.config.device,
            dtype=dtype or self.config.dtype,
        )

    def step(self, x: torch.Tensor, state: torch.Tensor, dissipation_scale: float = 1.0) -> Tuple[torch.Tensor, GeneratorParameters]:
        params = self.generator.parameters_from_input(x, dissipation_scale=dissipation_scale)
        forcing = params.forcing.unsqueeze(-2) * self.generator.input_injection
        updated = self.integrator.step(state, forcing, params, self.config.dt)
        return updated, params

    def diagnostics(self, state: torch.Tensor, params: GeneratorParameters | None = None) -> Dict[str, torch.Tensor]:
        report = self.generator.energy_terms(state)
        if params is not None:
            report.update({
                "tau_min": params.tau.amin(),
                "tau_max": params.tau.amax(),
                "gate_mean": params.geometry_gate.mean(),
                "dissipation_min": params.dissipation.amin(),
            })
        return report


class StateCodec:
    """Lossless structured-state pack/unpack and deterministic fingerprinting."""

    @staticmethod
    def pack(state: CohomodynamicState) -> Dict[str, Any]:
        return {
            "format": "dcss-cdi-stage-c-state-v1",
            "bands": {name: state.by_name(name).detach().clone() for name in BAND_NAMES},
            "dtypes": {name: str(state.by_name(name).dtype) for name in BAND_NAMES},
            "devices": {name: str(state.by_name(name).device) for name in BAND_NAMES},
        }

    @staticmethod
    def unpack(payload: Mapping[str, Any], device: torch.device | str | None = None) -> CohomodynamicState:
        if payload.get("format") != "dcss-cdi-stage-c-state-v1":
            raise ValueError("Unrecognized Stage C state serialization format.")
        bands = payload.get("bands")
        if not isinstance(bands, Mapping) or any(name not in bands for name in BAND_NAMES):
            raise ValueError("Serialized Stage C state is missing one or more bands.")
        tensors = tuple(bands[name].clone().to(device=device) if device is not None else bands[name].clone() for name in BAND_NAMES)
        return CohomodynamicState(*tensors)

    @staticmethod
    def fingerprint(state: CohomodynamicState) -> str:
        digest = sha256()
        for name in BAND_NAMES:
            tensor = state.by_name(name).detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()


class DynamicsDiagnostics:
    """Detached diagnostics; never invoked from the production recurrence hot path."""

    @staticmethod
    def energy(state: CohomodynamicState) -> Dict[str, float]:
        values = {name: float((0.5 * state.by_name(name).detach().square().sum()).item()) for name in BAND_NAMES}
        values["total"] = sum(values.values())
        return values

    @staticmethod
    def norms(state: CohomodynamicState) -> Dict[str, float]:
        values = {name: float(torch.linalg.vector_norm(state.by_name(name).detach()).item()) for name in BAND_NAMES}
        values["total"] = float(sum(value * value for value in values.values()) ** 0.5)
        return values

    @staticmethod
    def spectral_estimates(parameters: Mapping[str, GeneratorParameters]) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "dissipation_min": float(params.dissipation.detach().amin().item()),
                "dissipation_max": float(params.dissipation.detach().amax().item()),
                "omega_abs_max": float(params.omega.detach().abs().amax().item()),
                "tau_min": float(params.tau.detach().amin().item()),
                "tau_max": float(params.tau.detach().amax().item()),
            }
            for name, params in parameters.items()
        }

    @staticmethod
    def gate_stats(parameters: Mapping[str, GeneratorParameters]) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "geometry_gate_min": float(params.geometry_gate.detach().amin().item()),
                "geometry_gate_max": float(params.geometry_gate.detach().amax().item()),
                "forcing_abs_max": float(params.forcing.detach().abs().amax().item()),
            }
            for name, params in parameters.items()
        }


class CohomodynamicCell(nn.Module):
    """One causal selective update with matrix-free sparse geometric correction."""

    def __init__(self, config: StageCConfig, topology: SparseTopology | None = None) -> None:
        super().__init__()
        config.validate()
        torch.manual_seed(config.seed)
        self.config = config
        self.stage_b_config = config.stage_b_config()
        self.topology = topology or SparseTopology.from_config(self.stage_b_config)
        if self.topology.n_vertices != config.n_vertices:
            raise ValueError("Stage C topology vertex count does not match the recurrence configuration.")
        self.geometry = MatrixFreeLaplacian(self.topology, self.stage_b_config)
        self.bands = nn.ModuleDict({
            name: MemoryBand(name, tau_range, config)
            for name, tau_range in zip(BAND_NAMES, config.band_ranges)
        })
        self.learned_initial_state = nn.Parameter(
            torch.zeros(len(BAND_NAMES), config.n_vertices, config.band_width, dtype=config.dtype, device=config.device)
        )
        self.register_buffer(
            "band_log_tau_lower",
            torch.tensor(
                [max(log(config.tau_min), log(lower)) for lower, _ in config.band_ranges],
                dtype=config.dtype,
                device=config.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "band_log_tau_upper",
            torch.tensor(
                [min(log(config.tau_max), log(upper)) for _, upper in config.band_ranges],
                dtype=config.dtype,
                device=config.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "vertex_contrast_basis",
            self._zero_sum_vertex_basis(config.n_vertices, dtype=config.dtype, device=config.device),
            persistent=True,
        )
        self.readout = nn.Linear(
            len(BAND_NAMES) * config.n_vertices * config.band_width,
            config.output_width,
            bias=True,
            dtype=config.dtype,
            device=config.device,
        )
        nn.init.xavier_uniform_(self.readout.weight, gain=0.5)
        nn.init.zeros_(self.readout.bias)
        # ``harmonic_ablation`` is the serialized CCT-G3.3 control. The mutable
        # alias remains solely for the isolated legacy Stage E diagnostic helper.
        self.disable_harmonic = bool(config.harmonic_ablation)
        self.register_parameter("unconstrained_cochain", None)
        self._last_parameters: Dict[str, GeneratorParameters] = {}

    @staticmethod
    def _zero_sum_vertex_basis(n_vertices: int, *, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Return a deterministic orthonormal basis for zero-sum vertex contrasts."""
        basis = torch.zeros(n_vertices, n_vertices - 1, dtype=dtype, device=device)
        for column in range(n_vertices - 1):
            scale = ((column + 1) * (column + 2)) ** -0.5
            basis[: column + 1, column] = scale
            basis[column + 1, column] = -(column + 1) * scale
        return basis

    def _readout_features(self, state: CohomodynamicState) -> torch.Tensor:
        """Concatenate per-band mean and fixed zero-sum vertex contrasts."""
        features = []
        for name in BAND_NAMES:
            band = state.by_name(name)
            mean = band.mean(dim=-2)
            contrast = torch.matmul(band.transpose(-2, -1), self.vertex_contrast_basis).transpose(-2, -1)
            contrast = contrast.reshape(*band.shape[:-2], -1)
            if self.config.contrast_readout_ablation:
                contrast = torch.zeros_like(contrast)
            features.extend((mean, contrast))
        return torch.cat(features, dim=-1)

    def initial_state(self, batch_shape: Sequence[int] = (), mode: Literal["zero", "learned"] = "zero") -> CohomodynamicState:
        if mode not in {"zero", "learned"}:
            raise ValueError("Initial state mode must be 'zero' or 'learned'.")
        tensors = []
        for index, name in enumerate(BAND_NAMES):
            if self.disable_harmonic and name == "harmonic":
                tensor = self.bands[name].reset(batch_shape, device=self.readout.weight.device, dtype=self.readout.weight.dtype)
            elif mode == "zero":
                tensor = self.bands[name].reset(batch_shape, device=self.readout.weight.device, dtype=self.readout.weight.dtype)
            else:
                base = self.learned_initial_state[index]
                tensor = base.expand(*batch_shape, -1, -1) if batch_shape else base
            tensors.append(tensor)
        return CohomodynamicState(*tensors)

    def fused_gate_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project all bands into stacked tensors for the sequence fast path."""
        gates = tuple(self.bands[name].gate for name in BAND_NAMES)
        forcing = torch.tanh(F.linear(
            x,
            torch.cat(tuple(gate.forcing_projection.weight for gate in gates), dim=0),
            torch.cat(tuple(gate.forcing_projection.bias for gate in gates), dim=0),
        )).reshape(*x.shape[:-1], len(BAND_NAMES), self.config.band_width)
        input_gate = torch.sigmoid(F.linear(
            x,
            torch.cat(tuple(gate.input_gate_projection.weight for gate in gates), dim=0),
            torch.cat(tuple(gate.input_gate_projection.bias for gate in gates), dim=0),
        )).reshape(*x.shape[:-1], len(BAND_NAMES), self.config.band_width // 2)
        transport_gate = torch.sigmoid(F.linear(
            x,
            torch.cat(tuple(gate.transport_projection.weight for gate in gates), dim=0),
            torch.cat(tuple(gate.transport_projection.bias for gate in gates), dim=0),
        )).reshape(*x.shape[:-1], len(BAND_NAMES), self.config.band_width // 2)
        offsets = (torch.tanh(F.linear(
            x,
            torch.cat(tuple(gate.timescale_projection.weight for gate in gates), dim=0),
            torch.cat(tuple(gate.timescale_projection.bias for gate in gates), dim=0),
        )) * self.config.max_log_timescale_offset).reshape(*x.shape[:-1], len(BAND_NAMES), self.config.band_width)
        geometry = torch.sigmoid(F.linear(
            x,
            torch.cat(tuple(gate.geometry_projection.weight for gate in gates), dim=0),
            torch.cat(tuple(gate.geometry_projection.bias for gate in gates), dim=0),
        )).reshape(*x.shape[:-1], len(BAND_NAMES))
        forcing = forcing * input_gate.repeat_interleave(2, dim=-1)
        return forcing, input_gate, transport_gate, offsets, geometry

    def fused_gate_values(self, x: torch.Tensor) -> Dict[str, GateValues]:
        """Project all three shared token gates in five batched linear calls."""
        forcing, input_gate, transport_gate, offsets, geometry = self.fused_gate_tensors(x)
        return {
            name: GateValues(
                forcing=forcing[..., index, :],
                input_gate=input_gate[..., index, :],
                transport_gate=transport_gate[..., index, :],
                log_timescale_offset=offsets[..., index, :],
                geometry_gate=geometry[..., index],
            )
            for index, name in enumerate(BAND_NAMES)
        }

    def step_fused_tensors(
        self,
        forcing: torch.Tensor,
        input_gate: torch.Tensor,
        transport_gate: torch.Tensor,
        log_timescale_offset: torch.Tensor,
        geometry_gate: torch.Tensor,
        state: CohomodynamicState,
        *,
        runtime_guard_mode: Literal["python", "tensor", "deferred", "disabled"] = "python",
        return_runtime_metrics: bool = False,
        store_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, CohomodynamicState] | Tuple[torch.Tensor, CohomodynamicState, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Exact all-band step using already-fused tensor gates.

        This production fast path is valid for the full three-band model. It
        intentionally falls back to the structured step for named ablations
        that disable a band or add an unconstrained cochain.
        """
        if self.disable_harmonic or self.unconstrained_cochain is not None:
            raise RuntimeError("step_fused_tensors is only valid for the full three-band production cell.")
        self._validate_state(forcing[..., 0, :], state)
        stacked_state = torch.stack(state.tensors(), dim=-3)
        generators = tuple(self.bands[name].generator for name in BAND_NAMES)
        log_tau_base = torch.stack(tuple(generator.log_tau_base for generator in generators), dim=0)
        rotation_bias = torch.stack(tuple(generator.rotation_bias for generator in generators), dim=0)
        input_injection = torch.stack(tuple(generator.input_injection for generator in generators), dim=0)
        log_tau = (log_tau_base + log_timescale_offset).clamp(
            min=self.band_log_tau_lower.view(1, -1, 1),
            max=self.band_log_tau_upper.view(1, -1, 1),
        )
        tau = torch.exp(log_tau)
        dissipation = (0.05 + input_gate) / tau[..., ::2]
        omega = (2.0 * transport_gate - 1.0 + rotation_bias) * self.config.rotation_limit
        pairs = stacked_state.reshape(*stacked_state.shape[:-1], self.config.band_width // 2, 2)
        angle = (omega * self.config.dt).unsqueeze(-2)
        decay = torch.exp((-dissipation * self.config.dt).unsqueeze(-2))
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotated_first = cosine * pairs[..., 0] - sine * pairs[..., 1]
        rotated_second = sine * pairs[..., 0] + cosine * pairs[..., 1]
        homogeneous = torch.stack((rotated_first, rotated_second), dim=-1) * decay.unsqueeze(-1)
        band_state = homogeneous.reshape_as(stacked_state) + self.config.dt * forcing.unsqueeze(-2) * input_injection
        correction = self.geometry.apply(band_state)
        alpha = (self.config.geometry_step_cap * geometry_gate).unsqueeze(-1).unsqueeze(-1)
        spectral_violation = (alpha * 2.0 * (self.config.n_vertices - 1) * self.config.max_geometry_edge_weight > 1.0).any()
        if runtime_guard_mode == "tensor":
            torch._assert(~spectral_violation, "Explicit geometry correction exceeded the configured spectral stability envelope.")
        elif runtime_guard_mode not in ("deferred", "disabled") and bool(spectral_violation.item()):
            raise FloatingPointError("Explicit geometry correction exceeded the configured spectral stability envelope.")
        value = band_state - alpha * correction
        geometry_energy = self.geometry.energy(value)
        energy_violation = (geometry_energy > self.stage_b_config.energy_limit).any()
        if runtime_guard_mode == "tensor":
            torch._assert(~energy_violation, "Geometry energy exceeded the configured runtime limit.")
        elif runtime_guard_mode not in ("deferred", "disabled") and bool(energy_violation.item()):
            raise FloatingPointError("Geometry energy exceeded the configured runtime limit.")
        state_norm = torch.linalg.vector_norm(value, dim=(-2, -1))
        norm_violation = (state_norm > self.config.state_norm_bound).any()
        if runtime_guard_mode == "tensor":
            torch._assert(~norm_violation, "Cohomodynamic state norm exceeded the configured runtime limit.")
        elif runtime_guard_mode not in ("deferred", "disabled") and bool(norm_violation.item()):
            raise FloatingPointError("Cohomodynamic state norm exceeded the configured runtime limit.")
        new_state = CohomodynamicState(value.select(-3, 0), value.select(-3, 1), value.select(-3, 2))
        if store_diagnostics:
            self._last_parameters = {
                name: GeneratorParameters(
                    forcing=forcing.select(-2, index),
                    dissipation=dissipation.select(-2, index),
                    omega=omega.select(-2, index),
                    geometry_gate=geometry_gate.select(-1, index),
                    tau=tau.select(-2, index),
                )
                for index, name in enumerate(BAND_NAMES)
            }
        features = self._readout_features(new_state)
        output = self.readout(features)
        if return_runtime_metrics:
            return output, new_state, (spectral_violation, geometry_energy, state_norm)
        return output, new_state

    def _validate_state(self, x: torch.Tensor, state: CohomodynamicState) -> None:
        expected = tuple(x.shape[:-1]) + (self.config.n_vertices, self.config.band_width)
        for name in BAND_NAMES:
            actual = tuple(state.by_name(name).shape)
            if actual != expected:
                raise ValueError(f"State band {name!r} has shape {actual}, expected {expected} for input {tuple(x.shape)}.")

    def step(
        self,
        x: torch.Tensor,
        state: CohomodynamicState,
        dissipation_scale: float = 1.0,
        fused_gates: Mapping[str, GateValues] | None = None,
        runtime_guard_mode: Literal["python", "tensor", "deferred", "disabled"] = "python",
        return_runtime_metrics: bool = False,
    ) -> Tuple[torch.Tensor, CohomodynamicState] | Tuple[torch.Tensor, CohomodynamicState, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not math.isfinite(dissipation_scale) or dissipation_scale < 0.0:
            raise ValueError("dissipation_scale must be finite and non-negative.")
        if x.ndim < 1 or x.shape[-1] != self.config.input_width:
            raise ValueError(f"Expected (..., {self.config.input_width}) token input, received {tuple(x.shape)}.")
        x = x.to(dtype=self.readout.weight.dtype, device=self.readout.weight.device)
        self._validate_state(x, state)
        active_names = tuple(name for name in BAND_NAMES if not (self.disable_harmonic and name == "harmonic"))
        step_gates = fused_gates or self.fused_gate_values(x)
        parameters: Dict[str, GeneratorParameters] = {
            name: self.bands[name].generator.parameters_from_gates(
                step_gates[name], dissipation_scale=dissipation_scale
            )
            for name in active_names
        }
        stacked_state = torch.stack([state.by_name(name) for name in active_names], dim=-3)
        stacked_forcing = torch.stack([parameters[name].forcing for name in active_names], dim=-2)
        stacked_injection = torch.stack([self.bands[name].generator.input_injection for name in active_names], dim=0)
        stacked_params = GeneratorParameters(
            forcing=stacked_forcing,
            dissipation=torch.stack([parameters[name].dissipation for name in active_names], dim=-2),
            omega=torch.stack([parameters[name].omega for name in active_names], dim=-2),
            geometry_gate=torch.stack([parameters[name].geometry_gate for name in active_names], dim=-1),
            tau=torch.stack([parameters[name].tau for name in active_names], dim=-2),
        )
        band_state = self.bands[active_names[0]].integrator.step(
            stacked_state,
            stacked_forcing.unsqueeze(-2) * stacked_injection,
            stacked_params,
            self.config.dt,
        )
        correction = self.geometry.apply(band_state)
        alpha = (self.config.geometry_step_cap * stacked_params.geometry_gate).unsqueeze(-1).unsqueeze(-1)
        spectral_violation = (alpha * 2.0 * (self.config.n_vertices - 1) * self.config.max_geometry_edge_weight > 1.0).any()
        if runtime_guard_mode == "tensor":
            torch._assert(~spectral_violation, "Explicit geometry correction exceeded the configured spectral stability envelope.")
        elif runtime_guard_mode not in ("deferred", "disabled") and bool(spectral_violation.item()):
            raise FloatingPointError("Explicit geometry correction exceeded the configured spectral stability envelope.")
        value = band_state - alpha * correction
        geometry_energy = self.geometry.energy(value)
        energy_violation = (geometry_energy > self.stage_b_config.energy_limit).any()
        if runtime_guard_mode == "tensor":
            torch._assert(~energy_violation, "Geometry energy exceeded the configured runtime limit.")
        elif runtime_guard_mode not in ("deferred", "disabled") and bool(energy_violation.item()):
            raise FloatingPointError("Geometry energy exceeded the configured runtime limit.")
        if self.unconstrained_cochain is not None:
            # Named Stage E C-ablation only: unconstrained vertex mixing.
            # It is absent from every full-production execution path.
            value = value + torch.einsum("ij,...jw->...iw", self.unconstrained_cochain, band_state)
        state_norm = torch.linalg.vector_norm(value, dim=(-2, -1))
        norm_violation = (state_norm > self.config.state_norm_bound).any()
        if runtime_guard_mode == "tensor":
            torch._assert(~norm_violation, "Cohomodynamic state norm exceeded the configured runtime limit.")
        elif runtime_guard_mode not in ("deferred", "disabled") and bool(norm_violation.item()):
            raise FloatingPointError("Cohomodynamic state norm exceeded the configured runtime limit.")
        updated = {name: value.select(-3, index) for index, name in enumerate(active_names)}
        if self.disable_harmonic:
            updated["harmonic"] = torch.zeros_like(state.by_name("harmonic"))
        new_state = CohomodynamicState(updated["fast"], updated["middle"], updated["harmonic"])
        features = self._readout_features(new_state)
        self._last_parameters = parameters
        output = self.readout(features)
        if return_runtime_metrics:
            return output, new_state, (spectral_violation, geometry_energy, state_norm)
        return output, new_state

    def parameter_inventory(self) -> Dict[str, Any]:
        entries = []
        for name, parameter in self.named_parameters():
            entries.append({"name": name, "shape": list(parameter.shape), "count": parameter.numel(), "requires_grad": parameter.requires_grad})
        return {
            "total_parameters": sum(entry["count"] for entry in entries),
            "entries": entries,
            "geometry": self.geometry.production_metadata(),
            "state_layout": "CohomodynamicState(fast, middle, harmonic), each (..., vertices, width)",
            "readout": {
                "feature_layout": (
                    "per-band mean with zeroed fixed contrast feature slots"
                    if self.config.contrast_readout_ablation
                    else "per-band mean plus fixed zero-sum vertex contrasts"
                ),
                "feature_dim": self.readout.in_features,
                "contrast_basis_shape": list(self.vertex_contrast_basis.shape),
                "harmonic_feature_values": "deterministically zeroed by harmonic_ablation"
                if self.config.harmonic_ablation
                else "active",
            },
        }

    def last_diagnostics(self) -> Dict[str, Any]:
        if not self._last_parameters:
            return {"available": False}
        return {
            "available": True,
            "spectral_estimates": DynamicsDiagnostics.spectral_estimates(self._last_parameters),
            "gate_stats": DynamicsDiagnostics.gate_stats(self._last_parameters),
        }


class SelectiveCohomodynamicSSM(nn.Module):
    """Causal step/chunk DCSS-CDI engine with constant structured streaming state."""

    def __init__(self, config: StageCConfig, topology: SparseTopology | None = None) -> None:
        super().__init__()
        self.config = config
        self.cell = CohomodynamicCell(config, topology=topology)

    def initial_state(self, batch_shape: Sequence[int] = (), mode: Literal["zero", "learned"] = "zero") -> CohomodynamicState:
        return self.cell.initial_state(batch_shape=batch_shape, mode=mode)

    def step(
        self,
        x: torch.Tensor,
        state: CohomodynamicState,
        fused_gates: Mapping[str, GateValues] | None = None,
        runtime_guard_mode: Literal["python", "tensor", "deferred", "disabled"] = "python",
        return_runtime_metrics: bool = False,
    ) -> Tuple[torch.Tensor, CohomodynamicState] | Tuple[torch.Tensor, CohomodynamicState, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.cell.step(
            x,
            state,
            fused_gates=fused_gates,
            runtime_guard_mode=runtime_guard_mode,
            return_runtime_metrics=return_runtime_metrics,
        )

    def forward_chunk(
        self,
        x: torch.Tensor,
        state: CohomodynamicState | None = None,
        state_mode: Literal["zero", "learned"] = "zero",
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, CohomodynamicState] | Tuple[torch.Tensor, CohomodynamicState, Tuple[CohomodynamicState, ...]]:
        if x.ndim < 2 or x.shape[-1] != self.config.input_width:
            raise ValueError(f"Expected (..., length, {self.config.input_width}) sequence input, received {tuple(x.shape)}.")
        batch_shape = x.shape[:-2]
        current = state if state is not None else self.initial_state(batch_shape=batch_shape, mode=state_mode)
        outputs = []
        intermediates = []
        for index in range(x.shape[-2]):
            output, current = self.step(x[..., index, :], current)
            outputs.append(output)
            if return_intermediates:
                intermediates.append(current)
        chunk_output = torch.stack(outputs, dim=-2)
        if return_intermediates:
            return chunk_output, current, tuple(intermediates)
        return chunk_output, current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_chunk(x)
        return output

    def parameter_inventory(self) -> Dict[str, Any]:
        return self.cell.parameter_inventory()

    def production_metadata(self) -> Dict[str, Any]:
        return {
            "engine": self.__class__.__name__,
            "integrator": "exact diagonal-plus-pairwise-skew block exponential",
            "geometry_order": "post-recurrence, pre-readout",
            "state_elements": self.config.total_state_dim,
            "forbidden_operations": ["torch.kron", "dense_per_token_state_matrix"],
            "geometry": self.cell.geometry.production_metadata(),
        }
