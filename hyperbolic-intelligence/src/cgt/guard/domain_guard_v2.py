"""
cgt/guard/domain_guard_v2.py
=================================
DomainGuardV2 — strict domain separation enforcer for v2.

Rules
-----
- Euclidean tensors  : float32, NOT tagged on_manifold
- Hyperbolic tensors : float64, x₀ > 0, <x,x>_L ≈ -1/K
- No idempotent double-projection (assert_not_on_manifold)
- No linear op on manifold tensors (assert_linear_op_forbidden)

All violations raise RuntimeError immediately — zero silent fallback.
Zero imports from legacy cgt.*.
"""

from __future__ import annotations

import torch


class DomainGuardV2:
    """
    Validation layer for tensor domain separation (v2 isolated version).

    Usage
    -----
        from cgt.guard import GUARD_V2
        GUARD_V2.assert_hyperbolic(x, "hidden_states")
        GUARD_V2.assert_euclidean(e, "embeddings")
    """

    def __init__(self, manifold_tol: float = 1e-3, curvature: float = 1.0) -> None:
        self.manifold_tol = manifold_tol
        self.curvature = curvature
        self._registry: dict[int, bool] = {}

    # ── registry ──────────────────────────────────────────────────────────

    def tag_on_manifold(self, t: torch.Tensor) -> None:
        self._registry[id(t)] = True

    def untag(self, t: torch.Tensor) -> None:
        self._registry.pop(id(t), None)

    def is_on_manifold(self, t: torch.Tensor) -> bool:
        return self._registry.get(id(t), False)

    # ── assertions ────────────────────────────────────────────────────────

    def assert_euclidean(self, t: torch.Tensor, name: str = "tensor") -> None:
        """Euclidean: must be float32, must NOT be tagged on_manifold."""
        if t.dtype != torch.float32:
            raise RuntimeError(
                f"[DomainGuardV2] EUCLIDEAN VIOLATION: '{name}' dtype={t.dtype}. "
                f"Euclidean tensors MUST be torch.float32."
            )
        if self.is_on_manifold(t):
            raise RuntimeError(
                f"[DomainGuardV2] EUCLIDEAN VIOLATION: '{name}' is tagged on_manifold. "
                f"Call log_map before treating as Euclidean."
            )

    def assert_hyperbolic(
        self,
        t: torch.Tensor,
        name: str = "tensor",
        check_constraint: bool = True,
    ) -> None:
        """Hyperbolic: must be float64, x₀ > 0, Minkowski constraint satisfied."""
        if t.dtype != torch.float64:
            raise RuntimeError(
                f"[DomainGuardV2] HYPERBOLIC VIOLATION: '{name}' dtype={t.dtype}. "
                f"Hyperbolic tensors MUST be torch.float64."
            )
        if t.shape[-1] < 2:
            raise RuntimeError(
                f"[DomainGuardV2] HYPERBOLIC VIOLATION: '{name}' last dim {t.shape[-1]} < 2. "
                f"Lorentz points need at least (x₀, x_spatial)."
            )
        x0 = t[..., 0]
        if not (x0 > 0).all():
            raise RuntimeError(
                f"[DomainGuardV2] HYPERBOLIC VIOLATION: '{name}' x₀ ≤ 0 "
                f"(min={x0.min().item():.6f}). Must lie on UPPER sheet."
            )
        if check_constraint:
            mink = -t[..., 0:1] ** 2 + (t[..., 1:] ** 2).sum(dim=-1, keepdim=True)
            target = -1.0 / self.curvature
            err = (mink - target).abs().max().item()
            if err > self.manifold_tol:
                raise RuntimeError(
                    f"[DomainGuardV2] MANIFOLD CONSTRAINT VIOLATION: '{name}' "
                    f"Minkowski error={err:.6f} > tol={self.manifold_tol}. "
                    f"Run substrate.proj() before passing to hyperbolic ops."
                )

    def assert_not_on_manifold(self, t: torch.Tensor, name: str = "tensor") -> None:
        """Idempotency guard: raise if tensor is already tagged on_manifold."""
        if self.is_on_manifold(t):
            raise RuntimeError(
                f"[DomainGuardV2] IDEMPOTENCY VIOLATION: '{name}' is already on_manifold. "
                f"Applying exp_map twice causes spatial collapse. Call log_map first."
            )

    def assert_linear_op_forbidden(
        self, t: torch.Tensor, op: str = "op", name: str = "tensor"
    ) -> None:
        """Raise if manifold tensor is about to undergo a linear operation."""
        if self.is_on_manifold(t):
            raise RuntimeError(
                f"[DomainGuardV2] TANGENT ADDITION FALLACY: '{op}' on manifold tensor "
                f"'{name}'. Direct addition on Lorentz hyperboloid is GEOMETRICALLY INVALID. "
                f"Use exp_map(x, v) instead."
            )


# Module-level singleton
GUARD_V2 = DomainGuardV2(manifold_tol=1e-3, curvature=1.0)
