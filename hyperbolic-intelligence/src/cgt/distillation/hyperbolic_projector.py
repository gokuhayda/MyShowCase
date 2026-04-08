# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
hyperbolic_projector.py
=======================
Versioned teacher → student projectors for HyDRA distillation.

Three versions, co-located for comparison:

v1  HiddenProjector        — euclidean, Linear + LayerNorm.
                             Learns best linear approximation in R^128.
                             Blind to radius → feeds DegEq.
                             Moved here from distillation_v2.py for co-location.

v2  HyperbolicProjectorV2  — lifts to H^n via exp_map_zero.
                             Radius is implicit (norm of linear output).
                             May saturate at max_tangent_norm clamp,
                             killing gradient. Does not resolve DegEq.

v3  HyperbolicProjectorV3  — independent direction + radius branches.
                             dir_proj:  Linear → L2 normalize (unit direction)
                             rad_proj:  Linear → sigmoid → [r_min, r_max]
                             Radius gradient never saturates (sigmoid).
                             Intended for use with substrate.dist() loss
                             instead of cosine similarity.
                             DegEq becomes visible in rad_proj gradient
                             before RDC threshold is crossed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# v1 — Euclidean projection (original)
# ─────────────────────────────────────────────────────────────────────────────

class HiddenProjector(nn.Module):
    """
    v1 — Euclidean projection: Linear(d_teacher → d_student) + LayerNorm.

    Why this works:
        LayerNorm removes the L2 norm of teacher hiddens before comparison,
        equalizing tokens with large vs small norms. Without it, cosine
        similarity is dominated by high-norm tokens, not semantic proximity.

    Why this is geometrically weak:
        Learns W* = argmin_W L_hidden(W·h_T) where L_hidden is blind to
        radius. Result: correct angular alignment, zero radius guarantee.
        Feeds DegEq — the optimizer has no pressure to control depth.
    """

    def __init__(self, d_teacher: int = 768, d_student: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(d_teacher, d_student, bias=False)
        self.norm = nn.LayerNorm(d_student)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(d_teacher))

    def forward(self, h_teacher: torch.Tensor) -> torch.Tensor:
        """h_teacher: [..., d_teacher] → [..., d_student]"""
        return self.norm(self.proj(h_teacher))


# ─────────────────────────────────────────────────────────────────────────────
# v2 — Euclidean → H^n via exp_map_zero
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicProjectorV2(nn.Module):
    """
    v2 — Projects teacher hiddens to H^n via exp_map_zero.

    Why this is better than v1:
        Output lives on the manifold — substrate.dist() can be used as loss,
        giving gradient signal that sees both angle and radius.

    Why this still has problems:
        Radius = ‖W·h_T‖ (norm of linear output), which grows freely during
        training. If ‖W·h_T‖ > max_tangent_norm, the clamp saturates and
        kills the gradient for radius — same blind spot as v1.

    Use with:
        loss = substrate.dist(h_student, projector(h_teacher)).mean()
    """

    def __init__(
        self,
        d_teacher: int = 768,
        d_student: int = 128,
        substrate=None,
        max_tangent_norm: float = 1.5,
    ) -> None:
        super().__init__()
        self.substrate = substrate
        self.max_tangent_norm = max_tangent_norm
        self.linear = nn.Linear(d_teacher, d_student, bias=False)
        nn.init.normal_(self.linear.weight, std=1.0 / math.sqrt(d_teacher))

    def forward(self, h_teacher: torch.Tensor) -> torch.Tensor:
        """h_teacher: [..., d_teacher] → [..., n+1] on H^n"""
        v = self.linear(h_teacher)                     # [..., n]
        v0 = torch.zeros_like(v[..., :1])
        v_full = torch.cat([v0, v], dim=-1)            # [..., n+1], v0=0
        return self.substrate.exp_map_zero(
            v_full, max_tangent_norm=self.max_tangent_norm
        )


# ─────────────────────────────────────────────────────────────────────────────
# v3 — Independent direction + radius branches
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicProjectorV3(nn.Module):
    """
    v3 — Decoupled direction and radius for geometrically correct projection.

    Why this resolves v2's problem:
        v2 radius = ‖W·h‖ → can saturate at max_tangent_norm clamp.
        v3 radius = sigmoid(w·h) scaled to [r_min, r_max] → smooth everywhere,
        gradient always flows, value has direct semantic meaning (hierarchical
        depth of the token in context).

    Architecture:
        dir_proj:  Linear(d_teacher → d_student) → L2 normalize
                   Learns direction on unit sphere S^(n-1).
        rad_proj:  Linear(d_teacher → 1) → sigmoid → [r_min, r_max]
                   Learns geodesic depth from origin.
        v_spatial  = r * direction   (tangent vector with controlled radius)
        output     = exp_map_zero([0, v_spatial])

    Why ∂L_angle/∂r = 0 in this design:
        L_angle uses normalized direction only — radius cancels in L2 norm.
        L_radius uses the rad_proj output only — direction does not appear.
        The two gradients flow through separate parameters (dir_proj vs
        rad_proj) with no shared computation path.

    Known limitation:
        rad_proj needs supervision — without a geodesic loss (substrate.dist),
        it converges to r = (r_min + r_max) / 2 trivially (sigmoid fixed point).
        Always pair v3 with geodesic loss, never with cosine similarity.

    Use with:
        loss = substrate.dist(h_student, projector(h_teacher)).mean()
    """

    def __init__(
        self,
        d_teacher: int = 768,
        d_student: int = 128,
        substrate=None,
        r_min: float = 0.5,
        r_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.substrate = substrate
        self.r_min = r_min
        self.r_max = r_max

        # direction branch — learns semantic orientation
        self.dir_proj = nn.Linear(d_teacher, d_student, bias=False)
        nn.init.normal_(self.dir_proj.weight, std=1.0 / math.sqrt(d_teacher))

        # radius branch — learns hierarchical depth
        self.rad_proj = nn.Linear(d_teacher, 1, bias=True)
        nn.init.zeros_(self.rad_proj.weight)
        nn.init.zeros_(self.rad_proj.bias)   # init → sigmoid(0) = 0.5 → r_mid

    def forward(self, h_teacher: torch.Tensor) -> torch.Tensor:
        """
        h_teacher: [..., d_teacher]
        returns:   [..., n+1] on H^n
        """
        # direction: unit vector on S^(n-1)
        direction = F.normalize(self.dir_proj(h_teacher), p=2, dim=-1)  # [..., n]

        # radius: geodesic depth in [r_min, r_max]
        r = torch.sigmoid(self.rad_proj(h_teacher))         # [..., 1]
        r = self.r_min + r * (self.r_max - self.r_min)     # scale

        # tangent vector: r * direction, zero time component for exp_map_zero
        v_spatial = r * direction                           # [..., n]
        v0 = torch.zeros_like(v_spatial[..., :1])
        v_full = torch.cat([v0, v_spatial], dim=-1)         # [..., n+1]

        return self.substrate.exp_map_zero(v_full)          # [..., n+1] on H^n
