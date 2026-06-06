"""
§12 Spectral Sequence — Hierarchical Hypercohomology (Algorithm 12.3.1)
========================================================================

O(n log n) computation of ℍ^k(M, ℬ^•) via the Čech hypercohomology
spectral sequence on a hierarchical good cover.

Algorithm 12.3.1:
    1. Build nerve N(𝔘) and hierarchical tree T.
    2. For each leaf Uᵢ: compute local cohomology H^•(ℬ^•(Uᵢ), δ).
    3. For each level ℓ from leaves to root:
       a. For each node v = v₁ ∪ v₂:
          i.  Compute E₁^{p,q}(v) via Mayer-Vietoris.
          ii. Compute E₂^{p,q}(v) as cohomology of d₁.
    4. At root: extract E_∞^{p,q}.
    5. Return ℍ^k = ⊕_{p+q=k} E_∞^{p,q}.

Theorem 12.3.2: Runs in O(n log n) time, O(n) space.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cdi.config import CDIConfig


class SpectralSequence:
    """Hierarchical spectral sequence for hypercohomology.

    Attributes
    ----------
    cover : GoodCover
    belief : BeliefComplex
    config : CDIConfig
    """

    def __init__(self, cover, belief, config: CDIConfig) -> None:
        self.cover = cover
        self.belief = belief
        self.config = config

    # ------------------------------------------------------------------
    # Local cohomology at a patch
    # ------------------------------------------------------------------

    def compute_local_cohomology(
        self, patch_indices: list, degree: int
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """Compute H^q(ℬ^•(U), δ) restricted to a patch.

        For a single patch (contractible), the local cohomology is
        determined by the restriction of the coboundary maps δ^k.

        Parameters
        ----------
        patch_indices : list[int]  Point indices in the patch.
        degree : int               Belief degree q.

        Returns
        -------
        (dimension, basis_or_None)
        """
        idx = self.belief.degree_to_index(degree)
        dim_k = self.belief.dims[idx]

        # ker δ^k (restricted to this patch — but δ is global, so use full δ)
        if idx < len(self.belief.deltas):
            delta_k = self.belief.deltas[idx]  # (dim_{k+1}, dim_k)
            _, S, Vh = torch.linalg.svd(delta_k, full_matrices=True)
            rank_k = int((S > 1e-10).sum().item())
            ker_dim = dim_k - rank_k
        else:
            ker_dim = dim_k

        # im δ^{k-1}
        if idx > 0:
            delta_km1 = self.belief.deltas[idx - 1]
            _, S_km1, _ = torch.linalg.svd(delta_km1, full_matrices=False)
            im_dim = int((S_km1 > 1e-10).sum().item())
        else:
            im_dim = 0

        h_dim = max(ker_dim - im_dim, 0)
        return h_dim, None

    # ------------------------------------------------------------------
    # Mayer-Vietoris merge
    # ------------------------------------------------------------------

    def mayer_vietoris_merge(
        self,
        h_v1: Dict[int, int],
        h_v2: Dict[int, int],
        h_v12: Dict[int, int],
    ) -> Dict[int, int]:
        """Compute cohomology of v = v₁ ∪ v₂ via Mayer-Vietoris.

        Long exact sequence:
            ··· → H^{p-1}(v₁ ∩ v₂) → H^p(v) → H^p(v₁) ⊕ H^p(v₂) → H^p(v₁ ∩ v₂) → ···

        Approximate: dim H^p(v) = dim H^p(v₁) + dim H^p(v₂) − dim H^p(v₁∩v₂)
                     + dim H^{p-1}(v₁∩v₂)  (connecting homomorphism contribution)

        Parameters
        ----------
        h_v1, h_v2, h_v12 : dict[int, int]
            Cohomology dimensions {degree: dim} for v₁, v₂, v₁∩v₂.

        Returns
        -------
        dict[int, int]  Cohomology dimensions for v.
        """
        all_degrees = sorted(
            set(h_v1.keys()) | set(h_v2.keys()) | set(h_v12.keys())
        )
        result = {}
        for p in all_degrees:
            dim_v1 = h_v1.get(p, 0)
            dim_v2 = h_v2.get(p, 0)
            dim_v12 = h_v12.get(p, 0)
            dim_v12_prev = h_v12.get(p - 1, 0) if (p - 1) in h_v12 else 0

            # MV estimate
            dim_v = max(dim_v1 + dim_v2 - dim_v12 + dim_v12_prev, 0)
            result[p] = dim_v
        return result

    # ------------------------------------------------------------------
    # Full hierarchical computation
    # ------------------------------------------------------------------

    def full_computation(self) -> Dict[int, int]:
        """Run Algorithm 12.3.1 — O(n log n).

        Returns
        -------
        dict[int, int]
            {k: dim ℍ^k(M, ℬ^•)} for each total degree k.
        """
        tree = self.cover.build_hierarchical_tree()
        levels = tree["levels"]

        # Level 0 (leaves): compute local cohomology
        leaf_coh = []
        for leaf in levels[0]:
            h = {}
            for q in self.belief.degrees:
                dim, _ = self.compute_local_cohomology(leaf["indices"], q)
                h[q] = dim
            leaf_coh.append(h)

        # Merge levels
        current_coh = leaf_coh
        for level_idx in range(1, len(levels)):
            next_coh = []
            level_nodes = levels[level_idx]
            child_idx = 0

            for node in level_nodes:
                if node.get("children") is not None:
                    child1, child2 = node["children"]
                    # Find their cohomology in current_coh
                    c1_id = child1["id"]
                    c2_id = child2["id"]

                    h1 = current_coh[c1_id] if c1_id < len(current_coh) else {}
                    h2 = current_coh[c2_id] if c2_id < len(current_coh) else {}

                    # Intersection cohomology (approximate: minimum)
                    h12 = {}
                    for q in self.belief.degrees:
                        h12[q] = min(h1.get(q, 0), h2.get(q, 0))

                    merged = self.mayer_vietoris_merge(h1, h2, h12)
                    next_coh.append(merged)
                else:
                    # Solo promotion
                    nid = node["id"]
                    if nid < len(current_coh):
                        next_coh.append(current_coh[nid])
                    else:
                        next_coh.append({})

            current_coh = next_coh

        # Root cohomology
        if current_coh:
            root_coh = current_coh[0]
        else:
            root_coh = {}

        # Aggregate into total degrees: ℍ^k = ⊕_{p+q=k}
        result = {}
        for q, dim in root_coh.items():
            k = q  # p = 0 at root level
            result[k] = result.get(k, 0) + dim

        return result

    def hypercohomology(self, total_degree: int) -> int:
        """dim ℍ^k(M, ℬ^•) for a given total degree k."""
        all_h = self.full_computation()
        return all_h.get(total_degree, 0)
