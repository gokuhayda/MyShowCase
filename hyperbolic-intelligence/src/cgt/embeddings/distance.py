# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Lorentz Distance Utilities for Hyperbolic Retrieval
====================================================

Standalone, inference-optimised batch distance utilities for the Lorentz
(hyperboloid) model.  These functions are **distinct from**
``LorentzSubstrateHardened`` (``cgt.geometry.lorentz_hardened``), which
provides training-time pairwise operations with full gradient support.

Design rationale
----------------
Retrieval requires an **asymmetric** access pattern: one query versus a
potentially large corpus. ``LorentzSubstrateHardened.dist_matrix`` computes
a full ``[N, N]`` pairwise matrix, which is unnecessary and memory-intensive
at inference. The utilities here operate on ``[M, N]`` cross-distance tensors
(typically ``M=1`` for a single query) and are decorated with
``@torch.no_grad()`` for zero-overhead inference.

Mathematical basis
------------------
All operations use the Lorentz (Minkowski) inner product:

    ⟨x, y⟩_L  = -x₀ y₀ + x₁ y₁ + … + xₙ yₙ

And the geodesic distance on the hyperboloid H^n with curvature K:

    d(x, y)  =  (1 / √K) · arccosh(−K · ⟨x, y⟩_L)

where K > 0 and points satisfy  −x₀² + ‖x_s‖² = −1/K.

Note
----
Do NOT duplicate or modify ``LorentzSubstrateHardened``.  Import it for
training-time geometry; use this module for inference-time distance queries.
"""

from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def lorentz_inner_product_batch(
    x: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """Compute the Minkowski inner product between x and every row of Y.

    Implements ⟨x_i, Y_j⟩_L = −x_i0 · Y_j0 + ⟨x_i_s, Y_j_s⟩₂  for all
    (i, j) pairs, where subscript 0 denotes the time coordinate and subscript
    s denotes the spatial coordinates.

    Args:
        x: Tensor of shape ``[M, D+1]``, points on the Lorentz manifold.
        Y: Tensor of shape ``[N, D+1]``, corpus points on the manifold.

    Returns:
        Tensor of shape ``[M, N]`` containing pairwise Minkowski inner
        products.

    Raises:
        ValueError: If the ambient dimensions of ``x`` and ``Y`` do not match.
    """
    if x.shape[-1] != Y.shape[-1]:
        raise ValueError(
            f"Ambient dimension mismatch: x has {x.shape[-1]}, Y has {Y.shape[-1]}."
        )

    # Time component: −x₀ · Y₀ᵀ  →  [M, N]
    t_prod: torch.Tensor = x[:, 0:1] * Y[:, 0:1].T

    # Spatial component: x_s · Y_sᵀ  →  [M, N]
    s_prod: torch.Tensor = x[:, 1:] @ Y[:, 1:].T

    return -t_prod + s_prod  # [M, N]


@torch.no_grad()
def lorentz_dist_batch(
    x: torch.Tensor,
    Y: torch.Tensor,
    K: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Lorentz geodesic distances from each point in x to each in Y.

    Implements the geodesic distance on H^n with curvature parameter K:

        d(xᵢ, Yⱼ) = (1 / √K) · arccosh(−K · ⟨xᵢ, Yⱼ⟩_L)

    This function is the inference counterpart of
    ``LorentzSubstrateHardened.dist_matrix``.  Key differences:

    * Decorated with ``@torch.no_grad()`` — no gradient overhead.
    * Computes an asymmetric ``[M, N]`` cross-distance matrix rather than
      a symmetric ``[N, N]`` pairwise matrix.
    * Safe ``arccosh`` via clamping (no Taylor expansion needed for retrieval
      because we only need distances, not smooth gradients near x=1).

    Mathematical correctness
    ------------------------
    The argument to ``arccosh`` must be ≥ 1.  For well-formed Lorentz points
    we have ``−K · ⟨x, y⟩_L ≥ 1``, which the clamp enforces numerically.

    Args:
        x:   Tensor of shape ``[M, D+1]``, query points on H^n.
        Y:   Tensor of shape ``[N, D+1]``, corpus points on H^n.
        K:   Curvature parameter (K > 0).  Matches
             ``student_model.substrate.K.item()``.
        eps: Stability floor for the ``arccosh`` argument; prevents log(0).

    Returns:
        Tensor of shape ``[M, N]`` containing geodesic distances.

    Raises:
        ValueError: If ``K`` ≤ 0 or if the ambient dimensions mismatch.
    """
    if K <= 0:
        raise ValueError(f"Curvature K must be positive, got K={K}.")
    if x.shape[-1] != Y.shape[-1]:
        raise ValueError(
            f"Ambient dimension mismatch: x has {x.shape[-1]}, Y has {Y.shape[-1]}."
        )

    inner: torch.Tensor = lorentz_inner_product_batch(x, Y)  # [M, N]

    # arg must satisfy arg ≥ 1 for arccosh to be defined.
    # For valid Lorentz points: −K · inner ≥ 1 by the manifold constraint.
    arg: torch.Tensor = torch.clamp(-K * inner, min=1.0 + eps, max=1e8)

    # d(x, y) = (1/√K) · arccosh(−K · ⟨x, y⟩_L)
    dists: torch.Tensor = torch.acosh(arg) / (K ** 0.5)  # [M, N]

    return dists


@torch.no_grad()
def lorentz_query_distances(
    query: torch.Tensor,
    corpus: torch.Tensor,
    K: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute geodesic distances from a single query point to all corpus points.

    Convenience wrapper around ``lorentz_dist_batch`` for the common
    retrieval case where M=1.

    Args:
        query:  Tensor of shape ``[1, D+1]`` or ``[D+1]``, the query on H^n.
        corpus: Tensor of shape ``[N, D+1]``, corpus points on H^n.
        K:      Curvature parameter (K > 0).
        eps:    Stability floor for the ``arccosh`` argument.

    Returns:
        Tensor of shape ``[N]`` containing distances from query to each
        corpus point.

    Raises:
        ValueError: If ``K`` ≤ 0 or shapes are incompatible.
    """
    q = query.unsqueeze(0) if query.dim() == 1 else query
    if q.shape[0] != 1:
        raise ValueError(
            f"query must be a single point (shape [1, D+1] or [D+1]), "
            f"got shape {query.shape}."
        )

    dists = lorentz_dist_batch(q, corpus, K=K, eps=eps)  # [1, N]
    return dists.squeeze(0)  # [N]
