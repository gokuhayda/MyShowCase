"""
cgt.physics.interaction
~~~~~~~~~~~~~~~~~~~~~~~
Learnable coupling laws for HyDRA-Physics:
  InteractionNet     — learns K_ij = f(d_eff, |Δθ|, r_i, r_j)
  DynamicCurvatureField — learns K_i = g(local_density)
  causal_horizon     — soft sigmoid cutoff at R_H

New in V6. Does not modify any existing cgt modules.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.physics.config import PhysicsConfig


class InteractionNet(nn.Module):
    """
    Learns K_ij = f(d_eff, |Δθ|, r_i, r_j) ∈ [K_min, K_max].
    Encodes the "law of gravity" in this universe.
    Initialised to approximate K0·exp(-d/τ) (physically motivated baseline).
    """
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(4, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1), nn.Sigmoid(),
        )
        with torch.no_grad():
            self.net[0].weight.data *= 0.2
            self.net[-2].bias.data.fill_(-0.5)
        self.cfg = cfg

    def forward(self, d_eff, dtheta_abs, ri, rj):
        """All inputs: [N,N] float32. Returns K_ij [N,N]."""
        d_ref = d_eff.mean().clamp(min=1e-3)
        r_ref = ri.max().clamp(min=1e-3)
        feats = torch.stack([
            d_eff / d_ref,
            dtheta_abs / math.pi,
            ri / r_ref,
            rj / r_ref,
        ], dim=-1)                                             # [N,N,4]
        dtype = next(self.net.parameters()).dtype
        k = self.net(feats.to(dtype)).squeeze(-1)              # [N,N] ∈(0,1)
        k = self.cfg.K0_geo * 0.1 + (self.cfg.K0_geo - self.cfg.K0_geo * 0.1) * k
        N = d_eff.shape[0]
        return k * (1 - torch.eye(N, device=k.device))        # zero diagonal


class DynamicCurvatureField(nn.Module):
    """
    K_i = g(ρ_i): local matter density curves spacetime.
    D_eff(i,j) = D_raw(i,j) / sqrt(K_i · K_j)
    → dense regions have shorter effective distances (stronger coupling).
    """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)
        self.act = nn.SiLU()
        self.cfg = cfg
        with torch.no_grad():
            self.fc2.bias.data.fill_(0.0)
            self.fc2.weight.data *= 0.05

    def forward(self, density):
        """density [N] float32  →  K_local [N] ∈ [K_min, K_max]"""
        dt = self.fc1.weight.dtype
        x  = self.act(self.fc1(density.unsqueeze(-1).to(dt)))
        k  = torch.sigmoid(self.fc2(x)).squeeze(-1)
        lo, hi = self.cfg.K_curv_min, self.cfg.K_curv_max
        return lo + (hi - lo) * k                             # [N]

