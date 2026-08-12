"""Named Stage E ablation primitives for the frozen DCSS recurrence."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .ssm import CayleyIntegrator, GateValues, GeneratorParameters, SelectiveCohomodynamicSSM, SelectiveGate


class UngatedSelectiveGate(SelectiveGate):
    """Stage E `U`: retains forcing projection but fixes all selective controls."""

    def forward(self, x: torch.Tensor) -> GateValues:
        forcing = torch.tanh(self.forcing_projection(x))
        pair_width = self.input_gate_projection.out_features
        ones = torch.ones(*x.shape[:-1], pair_width, dtype=x.dtype, device=x.device)
        zeros = torch.zeros(*x.shape[:-1], self.timescale_projection.out_features, dtype=x.dtype, device=x.device)
        geometry = torch.full(x.shape[:-1], 0.5, dtype=x.dtype, device=x.device)
        return GateValues(forcing=forcing, input_gate=ones, transport_gate=zeros[..., :pair_width], log_timescale_offset=zeros, geometry_gate=geometry)


class ExplicitEulerIntegrator(CayleyIntegrator):
    """Stage E `E`: intentionally less stable explicit discretization control."""

    def step(self, z: torch.Tensor, u: torch.Tensor, params: GeneratorParameters, dt: float) -> torch.Tensor:
        pairs = z.reshape(*z.shape[:-1], self.pair_width, 2)
        omega = params.omega.unsqueeze(-2)
        lam = params.dissipation.unsqueeze(-2)
        derivative = torch.stack(
            (-lam * pairs[..., 0] - omega * pairs[..., 1], omega * pairs[..., 0] - lam * pairs[..., 1]),
            dim=-1,
        ).reshape_as(z)
        return z + float(dt) * (derivative + u)


def apply_stage_e_ablation(model: SelectiveCohomodynamicSSM, variant: Literal["F", "U", "H", "E", "C", "G"]) -> SelectiveCohomodynamicSSM:
    """Apply exactly one named Stage E DCSS ablation in-place.

    `G` is supplied by constructing the model with ``geometry_ablation=True``.
    The remaining named diagnostics do not change the full-model default.
    """
    if variant == "F" or variant == "G":
        return model
    if variant == "U":
        for band in model.cell.bands.values():
            gate = UngatedSelectiveGate(model.config.input_width, model.config.band_width, model.config.max_log_timescale_offset).to(
                device=band.generator.log_tau_base.device, dtype=band.generator.log_tau_base.dtype
            )
            # Preserve the original forcing parameters for a closer budget match.
            gate.load_state_dict(band.gate.state_dict())
            for parameter_name, parameter in gate.named_parameters():
                if not parameter_name.startswith("forcing_projection"):
                    parameter.requires_grad_(False)
            band.gate = gate
            band.generator.gate = gate
        return model
    if variant == "H":
        model.cell.disable_harmonic = True
        return model
    if variant == "E":
        for band in model.cell.bands.values():
            band.integrator = ExplicitEulerIntegrator(model.config.band_width)
        return model
    if variant == "C":
        matrix = nn.Parameter(torch.eye(model.config.n_vertices, dtype=model.config.dtype, device=model.config.device) * 0.02)
        model.cell.unconstrained_cochain = matrix
        return model
    raise ValueError(f"Unsupported Stage E ablation variant: {variant}")
