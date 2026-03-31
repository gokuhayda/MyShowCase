# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Hyperbolic Corpus Index
=======================

Maintains two parallel index structures for hybrid Euclidean + hyperbolic
retrieval:

1. **Euclidean ANN index** (FAISS ``IndexFlatIP`` or numpy fallback):
   Stores L2-normalised teacher embeddings for fast approximate nearest
   neighbour candidate selection in Euclidean space.

2. **Hyperbolic corpus buffer** (``torch.Tensor``):
   Stores student embeddings on the Lorentz manifold for geodesic reranking.

Retrieval uses the Euclidean index for recall and the hyperbolic buffer for
precision, following the decomposition:

    fast approximate search in Euclidean space
    →  precise semantic ranking in hyperbolic space

Design notes
------------
* FAISS is used if available; a pure-numpy ``O(N)`` cosine-similarity backend
  is the fallback for environments without native FAISS (e.g., Android/mobile).
* ``build()`` replaces all existing embeddings; ``add()`` appends.
* Embedding alignment is enforced: the i-th row of the Euclidean index must
  correspond to the i-th row of the hyperbolic buffer.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class _VectorBackend(ABC):
    """Minimal interface for swappable ANN backends."""

    @abstractmethod
    def add(self, embs: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self, query_vec: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


class _FAISSBackend(_VectorBackend):
    """FAISS ``IndexFlatIP`` backend (inner product = cosine after L2 norm)."""

    def __init__(self, dim: int) -> None:
        import faiss  # deferred to allow numpy fallback

        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)

    def add(self, embs: np.ndarray) -> None:
        self._index.add(embs.astype(np.float32))

    def search(
        self, query_vec: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores [k], indices [k]) for a single query vector."""
        scores, idxs = self._index.search(
            query_vec.astype(np.float32).reshape(1, -1), k
        )
        return scores[0], idxs[0]

    def size(self) -> int:
        return self._index.ntotal

    def reset(self) -> None:
        self._index.reset()


class _NumpyBackend(_VectorBackend):
    """Pure-numpy O(N) cosine-similarity fallback.

    No native dependencies; suitable for mobile / edge environments.
    """

    def __init__(self) -> None:
        self._matrix: Optional[np.ndarray] = None

    def add(self, embs: np.ndarray) -> None:
        embs_f32 = embs.astype(np.float32)
        if self._matrix is None:
            self._matrix = embs_f32
        else:
            self._matrix = np.concatenate([self._matrix, embs_f32], axis=0)

    def search(
        self, query_vec: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._matrix is not None, "Index is empty."
        sims: np.ndarray = self._matrix @ query_vec.astype(np.float32).reshape(-1)
        k_actual = min(k, len(self._matrix))
        idxs: np.ndarray = np.argsort(sims)[::-1][:k_actual]
        return sims[idxs], idxs

    def size(self) -> int:
        return len(self._matrix) if self._matrix is not None else 0

    def reset(self) -> None:
        self._matrix = None


def _make_backend(teacher_dim: int, use_faiss: bool) -> _VectorBackend:
    """Construct the appropriate backend, falling back to numpy if needed."""
    if use_faiss:
        try:
            return _FAISSBackend(teacher_dim)
        except ImportError:
            warnings.warn(
                "faiss is not installed. Falling back to the pure-numpy backend. "
                "Install faiss-cpu for faster ANN search.",
                ImportWarning,
                stacklevel=3,
            )
    return _NumpyBackend()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HyperbolicIndex:
    """Dual-space corpus index: Euclidean ANN + Lorentz manifold buffer.

    Maintains aligned Euclidean (teacher) and hyperbolic (student) embeddings
    for every indexed document.

    Args:
        teacher_dim: Dimension of the Euclidean teacher embeddings.
        use_faiss:   If ``True`` (default), use FAISS for ANN search.
                     Automatically falls back to numpy if FAISS is unavailable.

    Attributes:
        teacher_dim:     Euclidean embedding dimension.
        backend_name:    ``'faiss'`` or ``'numpy'`` — the active backend.

    Example::

        index = HyperbolicIndex(teacher_dim=384)
        index.build(teacher_embs_np, hyp_embs_tensor)

        scores, idxs = index.search(query_vec_np, k=20)
        hyp_embs = index.get_embeddings()[idxs]
    """

    def __init__(
        self,
        teacher_dim: int,
        use_faiss: bool = True,
    ) -> None:
        self.teacher_dim = teacher_dim
        self._backend: _VectorBackend = _make_backend(teacher_dim, use_faiss)
        self.backend_name: str = (
            "faiss" if isinstance(self._backend, _FAISSBackend) else "numpy"
        )
        self._hyp_embs: Optional[torch.Tensor] = None  # [N, student_dim + 1]

    # ------------------------------------------------------------------
    # Build / add
    # ------------------------------------------------------------------

    def build(
        self,
        teacher_embs: np.ndarray,
        hyp_embs: torch.Tensor,
    ) -> None:
        """Build the index from scratch, replacing any existing embeddings.

        Args:
            teacher_embs: L2-normalised teacher embeddings, shape
                          ``[N, teacher_dim]``, dtype float32.
            hyp_embs:     Lorentz manifold embeddings, shape
                          ``[N, student_dim + 1]``.  Device-agnostic; stored
                          on CPU.

        Raises:
            ValueError: If ``teacher_embs`` and ``hyp_embs`` have different
                        first dimensions, or if the teacher dim is wrong.
        """
        self._validate_inputs(teacher_embs, hyp_embs)
        self._backend.reset()
        self._hyp_embs = None

        self._backend.add(teacher_embs)
        self._hyp_embs = hyp_embs.cpu()

    def add(
        self,
        teacher_embs: np.ndarray,
        hyp_embs: torch.Tensor,
    ) -> None:
        """Append new embeddings to an existing index.

        Args:
            teacher_embs: L2-normalised teacher embeddings, shape
                          ``[M, teacher_dim]``, dtype float32.
            hyp_embs:     Lorentz manifold embeddings, shape
                          ``[M, student_dim + 1]``.

        Raises:
            ValueError: If the index is empty (call ``build`` first), or if
                        dimensions are inconsistent with existing embeddings.
        """
        if self._hyp_embs is None:
            raise ValueError(
                "Index is empty. Call build() before add()."
            )
        self._validate_inputs(teacher_embs, hyp_embs)

        # Verify student_dim consistency.
        existing_dim = self._hyp_embs.shape[1]
        if hyp_embs.shape[1] != existing_dim:
            raise ValueError(
                f"Hyperbolic dim mismatch: existing={existing_dim}, "
                f"new={hyp_embs.shape[1]}."
            )

        self._backend.add(teacher_embs)
        self._hyp_embs = torch.cat([self._hyp_embs, hyp_embs.cpu()], dim=0)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the Euclidean ANN index for the k nearest neighbours.

        Args:
            query_vec: L2-normalised query embedding, shape ``[teacher_dim]``
                       or ``[1, teacher_dim]``.
            k:         Number of candidates to retrieve.

        Returns:
            Tuple of:
              - scores: ``np.ndarray`` of shape ``[k]``, similarity scores.
              - indices: ``np.ndarray`` of shape ``[k]``, corpus indices.

        Raises:
            RuntimeError: If the index has not been built.
        """
        if self._hyp_embs is None:
            raise RuntimeError(
                "Index is empty. Call build() before search()."
            )
        k_actual = min(k, self._backend.size())
        qv = query_vec.reshape(-1).astype(np.float32)
        return self._backend.search(qv, k_actual)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_embeddings(self) -> torch.Tensor:
        """Return all stored hyperbolic (Lorentz) corpus embeddings.

        Returns:
            Tensor of shape ``[N, student_dim + 1]`` on CPU.

        Raises:
            RuntimeError: If the index has not been built.
        """
        if self._hyp_embs is None:
            raise RuntimeError(
                "Index is empty. Call build() before get_embeddings()."
            )
        return self._hyp_embs

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return self._backend.size()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the hyperbolic buffer and FAISS index to disk.

        Args:
            path: Directory path where artifacts will be written.
                  Creates ``hyp_embs.pt`` and, if FAISS, ``faiss.index``.
        """
        import os
        os.makedirs(path, exist_ok=True)

        if self._hyp_embs is not None:
            torch.save(self._hyp_embs, f"{path}/hyp_embs.pt")

        if isinstance(self._backend, _FAISSBackend):
            import faiss
            faiss.write_index(self._backend._index, f"{path}/faiss.index")

    def load(self, path: str) -> None:
        """Load persisted artifacts into this index.

        Args:
            path: Directory path previously written by ``save()``.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        import os

        hyp_path = f"{path}/hyp_embs.pt"
        if not os.path.exists(hyp_path):
            raise FileNotFoundError(f"Hyperbolic embeddings not found at {hyp_path}.")

        self._hyp_embs = torch.load(
            hyp_path, map_location="cpu", weights_only=False
        )

        faiss_path = f"{path}/faiss.index"
        if os.path.exists(faiss_path) and isinstance(self._backend, _FAISSBackend):
            import faiss
            self._backend._index = faiss.read_index(faiss_path)

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        teacher_embs: np.ndarray,
        hyp_embs: torch.Tensor,
    ) -> None:
        """Verify shape compatibility between teacher and hyperbolic embs."""
        if teacher_embs.shape[0] != hyp_embs.shape[0]:
            raise ValueError(
                f"Row count mismatch: teacher_embs has {teacher_embs.shape[0]} "
                f"rows, hyp_embs has {hyp_embs.shape[0]} rows."
            )
        if teacher_embs.shape[1] != self.teacher_dim:
            raise ValueError(
                f"teacher_embs dim {teacher_embs.shape[1]} does not match "
                f"index teacher_dim {self.teacher_dim}."
            )
