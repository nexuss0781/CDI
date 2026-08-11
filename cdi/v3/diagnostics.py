"""Diagnostics for the DCSS-CDI Stage B sparse operator substrate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from .cochain import SparseCochainMap
from .laplacian import MatrixFreeLaplacian
from .reference import DenseReferenceOperators


@dataclass
class OperatorDiagnostics:
    """Out-of-hot-path mathematical checks for a declared PSD Laplacian."""

    laplacian: MatrixFreeLaplacian
    cochain: SparseCochainMap

    def check_symmetry(self, trials: int = 8) -> Dict[str, Any]:
        generator = torch.Generator(device=self.laplacian.config.device).manual_seed(self.laplacian.config.seed + 101)
        errors = []
        for _ in range(trials):
            x = self._state(generator)
            y = self._state(generator)
            left = (x * self.laplacian.apply(y)).sum()
            right = (self.laplacian.apply(x) * y).sum()
            denominator = (left.abs() + right.abs()).clamp_min(1e-12)
            errors.append(float(((left - right).abs() / denominator).detach().cpu()))
        maximum = max(errors, default=0.0)
        return {"declared_property": "symmetric", "max_relative_error": maximum, "passed": maximum <= 1e-5}

    def check_psd(self, trials: int = 8) -> Dict[str, Any]:
        generator = torch.Generator(device=self.laplacian.config.device).manual_seed(self.laplacian.config.seed + 102)
        energies = [float(self.laplacian.quadratic_form(self._state(generator)).detach().cpu()) for _ in range(trials)]
        dense = DenseReferenceOperators.build_small(self.laplacian).matrix().detach()
        eigenvalues = torch.linalg.eigvalsh(dense)
        minimum = float(eigenvalues.min().cpu())
        tolerance = 2e-6 if dense.dtype == torch.float32 else 1e-10
        return {
            "declared_property": "positive_semidefinite",
            "minimum_quadratic_form": min(energies, default=0.0),
            "minimum_eigenvalue": minimum,
            "negative_energy_count": sum(value < -tolerance for value in energies),
            "passed": minimum >= -tolerance and all(value >= -tolerance for value in energies),
        }

    def check_cochain(self, trials: int = 8) -> Dict[str, Any]:
        generator = torch.Generator(device=self.laplacian.config.device).manual_seed(self.laplacian.config.seed + 103)
        residuals = []
        basis_residuals = []
        for _ in range(trials):
            state = self._state(generator)
            residuals.append(float(self.cochain.residual(0, state).detach().cpu()))
        for vertex in range(self.laplacian.topology.n_vertices):
            basis = torch.zeros_like(self._state(generator))
            basis[vertex, 0] = 1.0
            basis_residuals.append(float(self.cochain.residual(0, basis).detach().cpu()))
        maximum = max(residuals + basis_residuals, default=0.0)
        threshold = 1e-5 if self.laplacian.config.dtype == torch.float32 else 1e-10
        return {
            "structural": True,
            "max_relative_residual": maximum,
            "random_max_relative_residual": max(residuals, default=0.0),
            "basis_max_relative_residual": max(basis_residuals, default=0.0),
            "threshold": threshold,
            "passed": maximum <= threshold,
        }

    def estimate_spectrum(self) -> Dict[str, Any]:
        dense = DenseReferenceOperators.build_small(self.laplacian).matrix().detach()
        eigenvalues = torch.linalg.eigvalsh(dense)
        nonzero = eigenvalues[eigenvalues > (1e-6 if dense.dtype == torch.float32 else 1e-12)]
        spectral_gap = float(nonzero.min().cpu()) if nonzero.numel() else 0.0
        return {
            "eigenvalues": [float(value) for value in eigenvalues.cpu()],
            "spectral_gap": spectral_gap,
            "connected_zero_modes": int((eigenvalues.abs() <= 1e-6).sum().cpu()),
        }

    def cohomological_health_score(self) -> Dict[str, Any]:
        """Combine spectrum, structural cochain, and finite-energy checks in ``[0, 1]``.

        A critical diagnostic failure returns exactly zero.  Otherwise the score
        is the geometric mean of: spectral gap progress toward its target,
        cochain accuracy relative to threshold, and the bounded energy factor.
        Individual components are returned to prevent a single scalar from
        obscuring a failed mathematical condition.
        """
        cochain = self.check_cochain()
        spectrum = self.estimate_spectrum()
        generator = torch.Generator(device=self.laplacian.config.device).manual_seed(self.laplacian.config.seed + 104)
        energy = float(self.laplacian.energy(self._state(generator)).mean().detach().cpu())
        finite_energy = bool(torch.isfinite(torch.tensor(energy)).item()) and 0.0 <= energy <= self.laplacian.config.energy_limit
        if not cochain["passed"] or not finite_energy:
            score = 0.0
            components = {"spectral_gap": 0.0, "cochain": 0.0, "energy": 0.0}
        else:
            gap_component = min(1.0, spectrum["spectral_gap"] / self.laplacian.config.spectral_target)
            cochain_component = max(0.0, min(1.0, 1.0 - cochain["max_relative_residual"] / max(cochain["threshold"], 1e-30)))
            energy_component = max(0.0, min(1.0, 1.0 - energy / self.laplacian.config.energy_limit))
            components = {"spectral_gap": gap_component, "cochain": cochain_component, "energy": energy_component}
            score = float((gap_component * cochain_component * energy_component) ** (1.0 / 3.0))
        return {
            "score": float(max(0.0, min(1.0, score))),
            "components": components,
            "spectral_gap": spectrum["spectral_gap"],
            "cochain_residual": cochain["max_relative_residual"],
            "energy": energy,
            "critical_conditions_passed": bool(cochain["passed"] and finite_energy),
        }

    def full_report(self) -> Dict[str, Any]:
        return {
            "symmetry": self.check_symmetry(),
            "psd": self.check_psd(),
            "cochain": self.check_cochain(),
            "spectrum": self.estimate_spectrum(),
            "cohomological_health_score": self.cohomological_health_score(),
        }

    def _state(self, generator: torch.Generator) -> torch.Tensor:
        return torch.randn(
            self.laplacian.topology.n_vertices,
            self.laplacian.config.state_width,
            dtype=self.laplacian.config.dtype,
            device=self.laplacian.config.device,
            generator=generator,
        )
