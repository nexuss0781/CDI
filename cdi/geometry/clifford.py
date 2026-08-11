"""§4.1 Clifford Algebra — verified real negative-signature generators.

The CDI convention is ``{γⁱ, γʲ} = -2 gⁱʲ I``.  The original v2 recursive
construction used symmetric Pauli-like blocks that square to ``+I`` and could
not satisfy this convention in dimension four.  This implementation uses
explicit real Cl(0, d) templates.  Their real module dimensions are reflected
by :attr:`CDIConfig.spinor_dim` and are validated for all supported dimensions.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from cdi.config import CDIConfig


class CliffordAlgebra:
    """Real Clifford algebra representation for the CDI negative convention.

    Each word in ``_NEGATIVE_REAL_WORDS`` is a Kronecker product of the real
    matrices ``I``, ``J``, ``X``, and ``Z``.  ``J²=-I`` while ``X²=Z²=I``;
    the selected words have odd ``J`` parity and pairwise anticommute.  Thus
    every generator squares to ``-I`` and the complete anti-commutation law is
    exact up to floating-point arithmetic.
    """

    _NEGATIVE_REAL_WORDS: Dict[int, Tuple[str, ...]] = {
        1: ("J",),
        2: ("IJ", "JX"),
        3: ("IJ", "JX", "JZ"),
        4: ("IIJ", "IJX", "JIZ", "JXX"),
        5: ("IIJ", "IJX", "JIZ", "JXX", "JZX"),
        6: ("IIJ", "IJX", "JIZ", "JXX", "JZX", "XJZ"),
        7: ("IIJ", "IJX", "JIZ", "JXX", "JZX", "XJZ", "ZJZ"),
        8: ("IIIJ", "IIJX", "IJIZ", "IJXX", "IJZX", "IXJZ", "XZJZ", "ZZJZ"),
    }

    def __init__(self, config: CDIConfig) -> None:
        self.config = config
        self.d = config.manifold_dim
        self.s = config.spinor_dim
        if self.d not in self._NEGATIVE_REAL_WORDS:
            raise ValueError(
                f"No verified real negative-signature Clifford template is declared for d={self.d}. "
                "Stage A validates d=1 through d=8."
            )
        self.flat_gammas: List[torch.Tensor] = self._build_flat_gammas(config.dtype)
        if len(self.flat_gammas) != self.d or any(gamma.shape != (self.s, self.s) for gamma in self.flat_gammas):
            raise RuntimeError("Clifford template shape does not match CDIConfig.spinor_dim.")

    @staticmethod
    def _real_factors(dtype: torch.dtype, device: torch.device | str | None = None) -> Dict[str, torch.Tensor]:
        return {
            "I": torch.eye(2, dtype=dtype, device=device),
            "J": torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype, device=device),
            "X": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device),
            "Z": torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device),
        }

    def _build_flat_gammas(self, dtype: torch.dtype) -> List[torch.Tensor]:
        factors = self._real_factors(dtype)
        generators: List[torch.Tensor] = []
        for word in self._NEGATIVE_REAL_WORDS[self.d]:
            matrix = torch.ones((1, 1), dtype=dtype)
            for symbol in word:
                matrix = torch.kron(matrix, factors[symbol])
            generators.append(matrix)
        return generators

    def clifford_action(self, covector: torch.Tensor, spinor: torch.Tensor) -> torch.Tensor:
        """Apply ``c(ξ)ψ = Σᵢ ξᵢ γⁱψ`` while preserving dtype and device."""
        if covector.shape[-1] != self.d or spinor.shape[-1] != self.s:
            raise ValueError(f"Expected covector dimension {self.d} and spinor dimension {self.s}.")
        result = torch.zeros_like(spinor)
        for index, gamma in enumerate(self.flat_gammas):
            local_gamma = gamma.to(dtype=spinor.dtype, device=spinor.device)
            result = result + covector[..., index:index + 1] * (spinor @ local_gamma.T)
        return result

    def gamma_at_point(self, frame: torch.Tensor) -> List[torch.Tensor]:
        """Construct ``γⁱ(x) = Σₐ eⁱₐ(x) γᵃ`` from the contravariant frame.

        The returned gamma matrices consequently satisfy the negative Clifford
        relation against ``g⁻¹``.  This is the correct index placement for the
        frame constructed from the Cholesky factor of ``g⁻¹``.
        """
        if frame.shape != (self.d, self.d):
            raise ValueError(f"Expected frame shape {(self.d, self.d)}, received {tuple(frame.shape)}.")
        curved: List[torch.Tensor] = []
        for index in range(self.d):
            gamma = torch.zeros((self.s, self.s), dtype=frame.dtype, device=frame.device)
            for flat_index, flat_gamma in enumerate(self.flat_gammas):
                gamma = gamma + frame[index, flat_index] * flat_gamma.to(dtype=frame.dtype, device=frame.device)
            curved.append(gamma)
        return curved

    def chirality(self) -> torch.Tensor:
        """Return the ordered Clifford volume element used as the Z₂ grading."""
        chirality = torch.eye(self.s, dtype=self.flat_gammas[0].dtype, device=self.flat_gammas[0].device)
        for gamma in self.flat_gammas:
            chirality = chirality @ gamma
        return chirality

    def verify_relations(self, metric_at_point: torch.Tensor | None = None) -> torch.Tensor:
        """Return max error in ``{γⁱ,γʲ} = -2 gⁱʲ I``.

        When no metric is supplied the flat inverse metric is identity.  A
        supplied metric must be the contravariant metric matching the gamma
        index convention, for example ``manifold.inverse_metric()[point]``.
        """
        device = self.flat_gammas[0].device
        dtype = self.flat_gammas[0].dtype
        metric = torch.eye(self.d, dtype=dtype, device=device) if metric_at_point is None else metric_at_point.to(dtype=dtype, device=device)
        identity = torch.eye(self.s, dtype=dtype, device=device)
        maximum = torch.zeros((), dtype=dtype, device=device)
        for left in range(self.d):
            for right in range(self.d):
                anti_commutator = self.flat_gammas[left] @ self.flat_gammas[right] + self.flat_gammas[right] @ self.flat_gammas[left]
                expected = -2.0 * metric[left, right] * identity
                maximum = torch.maximum(maximum, (anti_commutator - expected).abs().max())
        return maximum
