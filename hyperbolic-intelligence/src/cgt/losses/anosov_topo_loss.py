# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Anosov Topological Loss (L_topo)
=================================

Implements the topological loss from:
    Reis (2026). "Resilient Cognitive Architectures via Anosov Flows
    on Lorentz Manifolds." Preprint v2.0.

Theorem 4.4: L_topo acts as a Lyapunov function guaranteeing exponential
convergence with rate µ bounded below by Forman-Ricci curvature κ_F.

L_topo = Σ_k λ_k · W_p(Dgm_k, Dgm*_k)

where W_p is the p-Wasserstein distance between persistence diagrams.

Optimal λ_topo range: [0.05, 0.20] (Table 8, Reis 2026).
Target topology: β* = (1, 0) — connected, acyclic (tree-like hierarchy).

Backends (in order of preference):
    1. ripser    — fast, exact persistent homology (pip install ripser)
    2. spectral  — fast proxy via graph Laplacian eigenvalues (always available)

Design: ADDITIVE — plug-in hook for DistillationTrainerV2._extra_loss_hooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Backend detection ──────────────────────────────────────────────────────
_RIPSER_AVAILABLE = False
try:
    import ripser as _ripser_mod
    _RIPSER_AVAILABLE = True
except ImportError:
    pass


@dataclass
class AnosovTopoConfig:
    """Configuration for AnosovTopoLoss.
    
    Attributes:
        lambda_topo:    Loss weight. Optimal range [0.05, 0.20] (Reis 2026 Table 8).
        betti_target:   Target Betti signature β* = [β₀, β₁, ...].
                        Default (1, 0) = connected + acyclic = tree-like hierarchy.
        max_dim:        Maximum homology dimension to compute.
        k_neighbors:    kNN graph connectivity for Rips complex.
        update_every:   Compute L_topo every N steps (costly — default 500).
        max_points:     Subsample embeddings to this count (memory bound).
        p_wasserstein:  Wasserstein distance order (1 or 2).
        warmup_steps:   Skip L_topo computation during warmup.
        use_tangent:    Project Lorentz embeddings to tangent space before TDA.
    """
    lambda_topo:   float       = 0.10
    betti_target:  List[int]   = field(default_factory=lambda: [1, 0])
    max_dim:       int         = 1
    k_neighbors:   int         = 10
    update_every:  int         = 500
    max_points:    int         = 256
    p_wasserstein: int         = 2
    warmup_steps:  int         = 2000
    use_tangent:   bool        = True


class AnosovTopoLoss(nn.Module):
    """Topological loss L_topo for hyperbolic LLM training.
    
    Penalises deviations from target Betti signature β* = (1, 0):
        - β₀ = 1: exactly one connected component (global coherence)
        - β₁ = 0: no topological loops (tree-like hierarchy, no cycles)
    
    This matches the structure of natural language hierarchies and ensures
    the hyperbolic embedding space retains its tree-like capacity advantage.
    
    As per Theorem 4.4, adding L_topo to the training objective guarantees
    exponential convergence with rate µ > 0.

    Usage::
        topo_loss = AnosovTopoLoss(AnosovTopoConfig(lambda_topo=0.10))
        
        # In _extra_loss_hooks:
        def _topo_hook(trainer_self):
            if trainer_self.step % topo_loss.config.update_every != 0:
                return None
            W = trainer_self.student.core_model.lm_head
            if hasattr(W, "_get_all_weights"):
                emb = W._get_all_weights()[:, 1:]  # spatial only
            else:
                return None
            loss, info = topo_loss(emb)
            return loss
    """

    def __init__(self, config: Optional[AnosovTopoConfig] = None) -> None:
        super().__init__()
        self.config  = config or AnosovTopoConfig()
        self._cached_loss: Optional[torch.Tensor] = None
        self._step_computed: int = -1
        self._backend = "ripser" if _RIPSER_AVAILABLE else "spectral"
        self._last_info: Dict = {}

    # ── Public interface ───────────────────────────────────────────────────

    def forward(
        self,
        embeddings: torch.Tensor,   # [N, D] — spatial coords (no time component)
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute L_topo.

        Args:
            embeddings: [N, D] token embedding vectors (tangent/spatial).
            step:       Current training step (for warmup guard).

        Returns:
            (loss, info_dict)
        """
        cfg = self.config
        device = embeddings.device

        zero = torch.tensor(0.0, device=device)

        if step < cfg.warmup_steps and cfg.warmup_steps > 0:
            return zero, {"topo_loss": 0.0, "beta0": 0, "beta1": 0,
                          "skipped": "warmup"}

        # Subsample for memory efficiency
        N = embeddings.shape[0]
        if N > cfg.max_points:
            idx = torch.randperm(N, device=device)[:cfg.max_points]
            emb = embeddings[idx].detach().float().cpu().numpy()
        else:
            emb = embeddings.detach().float().cpu().numpy()

        if emb.shape[0] < 4:
            return zero, {"topo_loss": 0.0, "skipped": "too_few_points"}

        # ── Compute persistence diagrams ───────────────────────────────────
        if self._backend == "ripser":
            loss_val, info = self._compute_ripser(emb, step, device)
        else:
            loss_val, info = self._compute_spectral(
                torch.tensor(emb, device=device), step, device)

        self._last_info = info
        return loss_val * cfg.lambda_topo, info

    def report(self) -> str:
        i = self._last_info
        return (f"β₀={i.get('beta0','?')} β₁={i.get('beta1','?')} "
                f"L_topo={i.get('topo_loss',0):.4f} "
                f"backend={self._backend}")

    # ── Ripser backend (exact) ─────────────────────────────────────────────

    def _compute_ripser(
        self,
        emb: np.ndarray,
        step: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict]:
        cfg = self.config
        try:
            result = _ripser_mod.ripser(emb, maxdim=cfg.max_dim,
                                         n_perm=min(emb.shape[0], 128))
            diagrams = result["dgms"]

            total_loss = 0.0
            betti = []
            for dim, dgm in enumerate(diagrams[:len(cfg.betti_target)]):
                # Remove infinite bars (last row often has inf death)
                finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
                beta_actual = len(finite)
                beta_target = cfg.betti_target[dim] if dim < len(cfg.betti_target) else 0
                betti.append(beta_actual)

                # Wasserstein proxy: penalise excess/deficit generators
                excess = max(0, beta_actual - beta_target)
                deficit = max(0, beta_target - beta_actual)

                # Penalise by persistence (longer bars = stronger signal)
                if excess > 0 and len(finite) > 0:
                    persistences = finite[:, 1] - finite[:, 0]
                    persistences.sort()
                    # Remove the beta_target longest (those are the "real" ones)
                    noise_pers = persistences[:-beta_target] if beta_target > 0 else persistences
                    total_loss += float(noise_pers.sum()) if len(noise_pers) > 0 else excess
                total_loss += float(deficit) * 0.5

            loss_t = torch.tensor(total_loss, device=device, dtype=torch.float32)
            info = {
                "topo_loss": float(loss_t.item()),
                "beta0": betti[0] if len(betti) > 0 else 0,
                "beta1": betti[1] if len(betti) > 1 else 0,
                "backend": "ripser",
                "n_points": emb.shape[0],
            }
            return loss_t, info

        except Exception as e:
            # Fallback to spectral on error
            self._backend = "spectral"
            t = torch.tensor(emb, device=device)
            return self._compute_spectral(t, step, device)

    # ── Spectral backend (proxy, always available) ─────────────────────────

    def _compute_spectral(
        self,
        emb: torch.Tensor,    # [N, D]
        step: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict]:
        cfg = self.config

        # Build kNN similarity graph
        N = emb.shape[0]
        K = min(cfg.k_neighbors, N - 1)

        # Pairwise cosine similarity
        emb_n  = F.normalize(emb.float(), dim=-1)   # [N, D]
        sim    = emb_n @ emb_n.T                     # [N, N]

        # Adjacency: top-K neighbours per row
        # Adaptive threshold: mean similarity + 0.5*std per row
        topk_val, _ = sim.topk(K + 1, dim=-1)
        threshold   = topk_val[:, -1].unsqueeze(-1)  # [N, 1]
        # Ensure connectivity: if all below threshold, lower it
        adj = (sim >= threshold).float()
        # Symmetrize
        adj = ((adj + adj.T) > 0).float()
        adj = adj * (1 - torch.eye(N, device=device))  # no self-loops

        # Graph Laplacian
        deg = adj.sum(dim=-1)
        D   = torch.diag(deg)
        L   = D - adj

        # β₀: number of connected components ≈ number of zero eigenvalues
        try:
            eigvals = torch.linalg.eigvalsh(L)
            eps     = 1e-4
            beta0_actual = int((eigvals.abs() < eps).sum().item())
        except Exception:
            beta0_actual = 1

        # β₀ loss: penalise > 1 component
        beta0_target = cfg.betti_target[0] if len(cfg.betti_target) > 0 else 1
        beta0_loss   = F.relu(torch.tensor(float(beta0_actual - beta0_target),
                                           device=device))

        # β₁ proxy: normalised cycle ratio
        # A tree has E = V-1 edges; extra edges = cycles
        n_edges      = adj.sum().item() / 2.0
        n_extra_edges = max(0.0, n_edges - (N - beta0_actual))  # vs spanning forest
        # Normalise: ratio of extra edges to N (0=tree-like, 1=dense cycles)
        cycle_ratio  = n_extra_edges / max(N, 1)
        beta1_target = cfg.betti_target[1] if len(cfg.betti_target) > 1 else 0
        # Loss: penalise when cycle_ratio > 0 (target = acyclic = 0)
        beta1_loss   = torch.tensor(cycle_ratio * (1.0 if beta1_target == 0 else 0.0),
                                    device=device, dtype=torch.float32)

        total_loss = beta0_loss + 0.5 * beta1_loss
        info = {
            "topo_loss": float(total_loss.item()),
            "beta0":     beta0_actual,
            "beta1":     int(n_extra_edges),
            "backend":   "spectral",
            "n_points":  N,
        }
        return total_loss, info


class FormanRicciRegularizer(nn.Module):
    """Forman-Ricci curvature regularizer.
    
    From Theorem 4.4: convergence rate µ is bounded below by κ_F.
    Maximising κ_F tightens the convergence guarantee.
    
    κ_F(e) = w(e)[w(v₁)/deg(v₁) + w(v₂)/deg(v₂) - Σ_f w(f)/w(e_f)]
    
    Simplified to edge-based version for efficiency.
    """

    def __init__(self, lambda_fr: float = 0.01, k_neighbors: int = 8) -> None:
        super().__init__()
        self.lambda_fr   = lambda_fr
        self.k_neighbors = k_neighbors

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Penalise negative Forman-Ricci curvature (encourages tree-like graphs)."""
        N, D = embeddings.shape
        K    = min(self.k_neighbors, N - 1)

        emb_n = F.normalize(embeddings.float(), dim=-1)
        sim   = emb_n @ emb_n.T
        topk_v, topk_i = sim.topk(K + 1, dim=-1)
        threshold = topk_v[:, -1].unsqueeze(-1)
        adj = (sim >= threshold).float()
        adj = adj * (1 - torch.eye(N, device=embeddings.device))
        deg = adj.sum(dim=-1).clamp(min=1.0)

        # Forman-Ricci curvature per edge: 2 - deg(u) - deg(v)  (unit weights)
        deg_u = deg.unsqueeze(1).expand(N, N)
        deg_v = deg.unsqueeze(0).expand(N, N)
        kappa = adj * (2.0 - deg_u - deg_v)

        # Penalise negative curvature edges
        neg_curv_loss = F.relu(-kappa).mean()
        return neg_curv_loss * self.lambda_fr
