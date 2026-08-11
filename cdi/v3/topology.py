"""Deterministic simplicial topology for the DCSS-CDI Stage B substrate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, Tuple

import torch

from .config import DCSSConfig


@dataclass(frozen=True)
class SparseTopology:
    """Immutable, oriented fan-triangulated topology.

    Vertices are ordered ``0..n-1``.  Edges are stored exactly once as
    ``(u, v)`` with ``u < v``.  Faces are oriented as ``(0, i, i + 1)``.
    This deterministic construction is connected for every valid configuration
    and supplies a non-vacuous boundary-of-boundary identity.
    """

    vertices: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    faces: Tuple[Tuple[int, int, int], ...]
    seed: int
    config_name: str
    _device: str = "cpu"

    @classmethod
    def from_config(cls, config: DCSSConfig) -> "SparseTopology":
        config.validate()
        vertices = tuple(range(config.n_vertices))
        edge_set = {(i, (i + 1) % config.n_vertices) for i in range(config.n_vertices)}
        normalized_edges = {tuple(sorted(edge)) for edge in edge_set}
        faces = []
        for i in range(1, config.n_vertices - 1):
            face = (0, i, i + 1)
            faces.append(face)
            normalized_edges.update(
                (
                    tuple(sorted((face[0], face[1]))),
                    tuple(sorted((face[1], face[2]))),
                    tuple(sorted((face[0], face[2]))),
                )
            )
        edges = tuple(sorted(normalized_edges))
        topology = cls(vertices, edges, tuple(faces), config.seed, config.name, config.device)
        topology.validate()
        return topology

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def edge_index(self) -> torch.Tensor:
        """Return the canonical ``(2, n_edges)`` edge index on the topology device."""
        return torch.tensor(self.edges, dtype=torch.long, device=self._device).t().contiguous()

    @property
    def incidence(self) -> torch.Tensor:
        """Return signed vertex-edge incidence in sparse COO form."""
        row, col, values = [], [], []
        for edge_id, (source, target) in enumerate(self.edges):
            row.extend((source, target))
            col.extend((edge_id, edge_id))
            values.extend((-1.0, 1.0))
        indices = torch.tensor([row, col], dtype=torch.long, device=self._device)
        data = torch.tensor(values, dtype=torch.float32, device=self._device)
        return torch.sparse_coo_tensor(
            indices, data, (self.n_vertices, self.n_edges), device=self._device, check_invariants=True
        ).coalesce()

    @property
    def boundary_one(self) -> torch.Tensor:
        """Return sparse boundary ``∂₁: C₁ -> C₀`` with shape (vertices, edges)."""
        return self.incidence

    @property
    def boundary_two(self) -> torch.Tensor:
        """Return sparse oriented face boundary ``∂₂: C₂ -> C₁``."""
        if not self.faces:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self._device),
                torch.empty((0,), dtype=torch.float32, device=self._device),
                (self.n_edges, 0), device=self._device, check_invariants=True,
            ).coalesce()
        edge_to_id = {edge: index for index, edge in enumerate(self.edges)}
        row, col, values = [], [], []
        for face_id, (a, b, c) in enumerate(self.faces):
            # ∂[a,b,c] = [b,c] - [a,c] + [a,b]
            for left, right, sign in ((b, c, 1.0), (a, c, -1.0), (a, b, 1.0)):
                ordered = tuple(sorted((left, right)))
                orientation = 1.0 if (left, right) == ordered else -1.0
                row.append(edge_to_id[ordered])
                col.append(face_id)
                values.append(sign * orientation)
        indices = torch.tensor([row, col], dtype=torch.long, device=self._device)
        data = torch.tensor(values, dtype=torch.float32, device=self._device)
        return torch.sparse_coo_tensor(
            indices, data, (self.n_edges, self.n_faces), device=self._device, check_invariants=True
        ).coalesce()

    def to(self, device: str | torch.device) -> "SparseTopology":
        """Return a structurally identical topology whose lazily materialized tensors use device."""
        return SparseTopology(self.vertices, self.edges, self.faces, self.seed, self.config_name, str(device))

    def serialize(self) -> Dict[str, Any]:
        return {
            "vertices": list(self.vertices),
            "edges": [list(edge) for edge in self.edges],
            "faces": [list(face) for face in self.faces],
            "seed": self.seed,
            "config_name": self.config_name,
            "device": self._device,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def deserialize(cls, payload: Dict[str, Any]) -> "SparseTopology":
        topology = cls(
            tuple(int(vertex) for vertex in payload["vertices"]),
            tuple(tuple(int(value) for value in edge) for edge in payload["edges"]),
            tuple(tuple(int(value) for value in face) for face in payload["faces"]),
            int(payload["seed"]),
            str(payload["config_name"]),
            str(payload.get("device", "cpu")),
        )
        topology.validate()
        if "fingerprint" in payload and topology.fingerprint() != payload["fingerprint"]:
            raise ValueError("Topology fingerprint does not match serialized structural data.")
        return topology

    def fingerprint(self) -> str:
        canonical = {
            "vertices": self.vertices,
            "edges": self.edges,
            "faces": self.faces,
            "seed": self.seed,
            "config_name": self.config_name,
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        if self.vertices != tuple(range(self.n_vertices)):
            raise ValueError("Vertices must be contiguous and canonically ordered.")
        if any(left == right for left, right in self.edges):
            raise ValueError("Self-loops are not valid sparse topology edges.")
        if any(left > right for left, right in self.edges):
            raise ValueError("Edges must use canonical ascending orientation.")
        if len(set(self.edges)) != self.n_edges:
            raise ValueError("Duplicate directed edges are not valid.")
        if any(vertex not in self.vertices for edge in self.edges for vertex in edge):
            raise ValueError("Edge references an invalid vertex.")
        if len(set(self.faces)) != self.n_faces:
            raise ValueError("Duplicate faces are not valid.")
        if any(tuple(sorted(face)) != face or len(set(face)) != 3 for face in self.faces):
            raise ValueError("Faces must be ordered, non-degenerate triples.")
        if any(vertex not in self.vertices for face in self.faces for vertex in face):
            raise ValueError("Face references an invalid vertex.")
        if self.n_vertices and not self._is_connected():
            raise ValueError("Stage B production topology must be connected.")
        residual = torch.sparse.mm(self.boundary_one, self.boundary_two.to_dense())
        if residual.numel() and not bool(torch.allclose(residual, torch.zeros_like(residual), atol=0.0, rtol=0.0)):
            raise ValueError("Topology violates the structural boundary-of-boundary identity.")

    def _is_connected(self) -> bool:
        if not self.vertices:
            return True
        neighbours = {vertex: set() for vertex in self.vertices}
        for source, target in self.edges:
            neighbours[source].add(target)
            neighbours[target].add(source)
        visited = {self.vertices[0]}
        frontier = list(visited)
        while frontier:
            vertex = frontier.pop()
            additions = neighbours[vertex] - visited
            visited.update(additions)
            frontier.extend(additions)
        return len(visited) == self.n_vertices
