"""
§2 Good Cover — Overlapping Patches and Nerve Complex
=====================================================

Implements the good cover {Uᵢ}ᵢ₌₁ⁿ from CDI Specification §2.

A good cover is a collection of open sets whose non-empty finite
intersections are all contractible.  Here the cover is built from
k-nearest-neighbour balls on the discretised manifold.

The *nerve* N(𝔘) of the cover is the simplicial complex whose
k-simplices are (k+1)-fold intersections of patches.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cdi.config import CDIConfig


class GoodCover:
    """k-NN based good cover with nerve complex.

    Each patch Uᵢ is the set of cover_k nearest neighbours of point i.
    Patches overlap by construction, satisfying the good-cover condition
    (star-shaped ⟹ contractible intersections).

    Attributes
    ----------
    config : CDIConfig
    patches : list[torch.Tensor]
        ``patches[i]`` — 1-D LongTensor of point indices in patch Uᵢ.
    edges : list[tuple[int, int]]
        1-simplices of the nerve (pairs with non-empty intersection).
    triangles : list[tuple[int, int, int]]
        2-simplices of the nerve.
    adjacency : torch.Tensor
        Dense (n, n) adjacency matrix of the nerve graph.
    """

    def __init__(self, manifold, config: CDIConfig) -> None:
        self.config = config
        self.n = config.n_points
        self.k = config.cover_k

        self.patches: List[torch.Tensor] = []
        self.edges: List[Tuple[int, int]] = []
        self.triangles: List[Tuple[int, int, int]] = []
        self.adjacency: Optional[torch.Tensor] = None

        self.build(manifold.points)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, points: torch.Tensor) -> None:
        """Build the cover from point positions via k-NN.

        Parameters
        ----------
        points : torch.Tensor
            Shape ``(n, d)`` — manifold discretisation.
        """
        dists = torch.cdist(points.detach(), points.detach())  # (n, n)
        _, knn_idx = dists.topk(self.k, dim=1, largest=False)  # (n, k)

        # Patches ----------------------------------------------------------
        self.patches = [knn_idx[i] for i in range(self.n)]

        # Adjacency / edges ------------------------------------------------
        adj = torch.zeros(self.n, self.n, dtype=torch.bool)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pi = set(self.patches[i].tolist())
                pj = set(self.patches[j].tolist())
                if pi & pj:  # non-empty intersection
                    adj[i, j] = True
                    adj[j, i] = True
        self.adjacency = adj.to(torch.float64)

        # 1-simplices (edges) of the nerve
        self.edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if adj[i, j]:
                    self.edges.append((i, j))

        # 2-simplices (triangles) of the nerve — O(n) for bounded geometry
        self.triangles = []
        adj_sets = [set() for _ in range(self.n)]
        for i, j in self.edges:
            adj_sets[i].add(j)
            adj_sets[j].add(i)
        for i, j in self.edges:
            common = adj_sets[i] & adj_sets[j]
            for k_vert in common:
                if k_vert > j:
                    self.triangles.append((i, j, k_vert))

    # ------------------------------------------------------------------
    # Intersection helpers
    # ------------------------------------------------------------------

    def intersection(self, i: int, j: int) -> torch.Tensor:
        """Point indices in Uᵢ ∩ Uⱼ."""
        pi = set(self.patches[i].tolist())
        pj = set(self.patches[j].tolist())
        inter = sorted(pi & pj)
        return torch.tensor(inter, dtype=torch.long)

    def triple_intersection(self, i: int, j: int, k: int) -> torch.Tensor:
        """Point indices in Uᵢ ∩ Uⱼ ∩ Uₖ."""
        pi = set(self.patches[i].tolist())
        pj = set(self.patches[j].tolist())
        pk = set(self.patches[k].tolist())
        inter = sorted(pi & pj & pk)
        return torch.tensor(inter, dtype=torch.long)

    # ------------------------------------------------------------------
    # Nerve simplices
    # ------------------------------------------------------------------

    def nerve_simplices(self, dim: int) -> list:
        """Return simplices of the nerve of given dimension.

        dim=0: patches (vertices), dim=1: edges, dim=2: triangles.
        """
        if dim == 0:
            return list(range(self.n))
        elif dim == 1:
            return list(self.edges)
        elif dim == 2:
            return list(self.triangles)
        else:
            return []  # higher simplices not tracked

    def adjacency_matrix(self) -> torch.Tensor:
        """Dense adjacency matrix of the nerve graph."""
        return self.adjacency

    # ------------------------------------------------------------------
    # Hierarchical tree (for Algorithm 12.3.1)
    # ------------------------------------------------------------------

    def build_hierarchical_tree(self) -> Dict:
        """Build a binary merge tree over the patches.

        At each level, adjacent patches are paired and merged.
        Returns a tree structure for the spectral-sequence algorithm.

        Returns
        -------
        dict
            ``{'levels': [...], 'root': node}``
            Each level is a list of nodes; each node is a dict with
            keys ``'indices'``, ``'children'``, ``'level'``.
        """
        # Level 0: each patch is a leaf
        leaves = [
            {"indices": self.patches[i].tolist(), "children": None,
             "level": 0, "id": i}
            for i in range(self.n)
        ]
        levels = [leaves]
        current_level = leaves

        level_num = 1
        while len(current_level) > 1:
            next_level = []
            used = set()
            for i in range(len(current_level)):
                if i in used:
                    continue
                # Find an unused neighbour to merge with
                merged = False
                for j in range(i + 1, len(current_level)):
                    if j in used:
                        continue
                    # Check overlap
                    si = set(current_level[i]["indices"])
                    sj = set(current_level[j]["indices"])
                    if si & sj:
                        node = {
                            "indices": sorted(si | sj),
                            "children": (current_level[i], current_level[j]),
                            "level": level_num,
                            "id": len(next_level),
                        }
                        next_level.append(node)
                        used.add(i)
                        used.add(j)
                        merged = True
                        break
                if not merged and i not in used:
                    # No partner — promote solo
                    node = dict(current_level[i])
                    node["level"] = level_num
                    node["id"] = len(next_level)
                    next_level.append(node)
                    used.add(i)

            levels.append(next_level)
            current_level = next_level
            level_num += 1

        return {"levels": levels, "root": current_level[0] if current_level else None}
