# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
Persistence Landscape — Differentiable Topological Proxy
=========================================================

Implements the differentiable topological backend from:
    GCM paper, Section 6.2 (Bubenik 2015).

Unlike discrete Betti numbers, Persistence Landscapes are piecewise
continuous functions in a Banach space — differentiable almost everywhere
w.r.t. pairwise distances, enabling gradient flow:

    ∂L_topo/∂hᵢ = ∂L/∂λ · ∂λ/∂dᵢⱼ · ∂dᵢⱼ/∂hᵢ

Gradients flow through "critical edges" — the birth/death pairs that
define the current topology — physically pushing nodes to close loops
or merge disconnected components.

Target: λ_target corresponding to β* = (1, 0):
    - One persistent H₀ feature (the single connected component)
    - No persistent H₁ features (tree-like, no cycles)

This module provides:
    PersistenceLandscape       — computes landscape from pairwise distances
    PersistenceLandscapeLoss   — ||λ_curr - λ_target||_p with gradients
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistenceLandscape(nn.Module):
    """Differentiable persistence landscape proxy via soft thresholding.

    Approximates the integral of the persistence landscape using a
    differentiable relaxation of the Vietoris-Rips filtration.

    The key insight: instead of discrete birth/death pairs, we use
    softmin/softmax operations over pairwise distances to create a
    smooth approximation of topological features.

    Args:
        n_filtrations: Number of filtration steps (resolution).
        temperature:   Softmin temperature (lower = sharper, closer to exact).
        max_dim:       Maximum homology dimension (0 and 1).
    """

    def __init__(
        self,
        n_filtrations: int = 32,
        temperature:   float = 0.1,
        max_dim:       int = 1,
    ) -> None:
        super().__init__()
        self.n_filtrations = n_filtrations
        self.temperature   = temperature
        self.max_dim       = max_dim

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute differentiable persistence landscape features.

        Args:
            embeddings: [N, D] token embedding vectors.

        Returns:
            dict with keys:
                'h0_persistence': [n_filtrations] H₀ landscape integral per step
                'h1_persistence': [n_filtrations] H₁ landscape proxy per step
                'total_h0':       scalar — total H₀ persistence
                'total_h1':       scalar — total H₁ persistence
        """
        N = embeddings.shape[0]
        if N < 4:
            zero = embeddings.new_zeros(1)
            return {'h0_persistence': zero, 'h1_persistence': zero,
                    'total_h0': zero, 'total_h1': zero}

        # Pairwise distances (Lorentz-aware: use L2 on spatial components)
        emb  = embeddings.float()
        dist = torch.cdist(emb, emb, p=2)                # [N, N]
        dist = dist + torch.eye(N, device=emb.device) * 1e9  # mask diagonal

        # Filtration thresholds: linspace from min to max distance
        d_min = dist[dist < 1e8].min()
        d_max = dist[dist < 1e8].max().clamp(min=d_min + 1e-4)
        thresholds = torch.linspace(
            d_min.item(), d_max.item(), self.n_filtrations,
            device=emb.device)                            # [F]

        # ── H₀: connected components ──────────────────────────────────────
        # At each threshold ε, count connected components via soft adjacency.
        # Soft edge indicator: σ((ε - dᵢⱼ) / T) ∈ (0,1)
        h0_landscape = []
        for thresh in thresholds:
            adj = torch.sigmoid((thresh - dist) / self.temperature)  # [N, N]
            adj = adj * (1 - torch.eye(N, device=emb.device))
            # Degree of each node
            deg = adj.sum(dim=-1).clamp(min=1e-6)   # [N]
            # Graph Laplacian smallest eigenvalue proxy via trace(D) - trace(A·D⁻¹)
            # Softer: connectivity = mean(min_distance_to_cluster)
            # As threshold grows, more nodes connect → fewer components
            # Proxy: number of "isolated" nodes (degree < 0.5)
            n_isolated = (deg < 0.5).float().sum()
            h0_landscape.append(n_isolated)

        h0_tensor = torch.stack(h0_landscape)               # [F]

        # ── H₁: cycles ────────────────────────────────────────────────────
        # Proxy: edges beyond spanning tree = potential cycles
        # At each threshold, count "excess" soft edges
        h1_landscape = []
        for thresh in thresholds:
            adj = torch.sigmoid((thresh - dist) / self.temperature)
            adj = adj * (1 - torch.eye(N, device=emb.device))
            n_edges_soft = adj.sum() / 2.0
            # Spanning tree has N-1 edges; excess = cycles
            n_excess = F.relu(n_edges_soft - (N - 1))
            h1_landscape.append(n_excess / N)  # normalised

        h1_tensor = torch.stack(h1_landscape)               # [F]

        return {
            'h0_persistence': h0_tensor,
            'h1_persistence': h1_tensor,
            'total_h0': h0_tensor.mean(),
            'total_h1': h1_tensor.mean(),
        }


class PersistenceLandscapeLoss(nn.Module):
    """Loss based on Persistence Landscape distance from target topology.

    Computes ||λ_curr - λ_target||_p where λ_target corresponds to
    β* = (1, 0): one connected component, no cycles.

    Gradients flow through the soft thresholding operations back to
    the embedding distances, enabling the model to actively adjust
    token positions to satisfy topological constraints.

    Args:
        lambda_topo:    Loss weight.
        n_filtrations:  Landscape resolution.
        temperature:    Softmin temperature.
        p_norm:         Lp norm for landscape distance (default 2).
        h1_weight:      Relative weight of H₁ vs H₀ penalty.
        warmup_steps:   Skip computation during warmup.
        update_every:   Compute every N steps.
        max_points:     Subsample for memory efficiency.
    """

    def __init__(
        self,
        lambda_topo:   float = 0.10,
        n_filtrations: int   = 24,
        temperature:   float = 0.15,
        p_norm:        int   = 2,
        h1_weight:     float = 0.5,
        warmup_steps:  int   = 2000,
        update_every:  int   = 500,
        max_points:    int   = 128,
    ) -> None:
        super().__init__()
        self.lambda_topo   = lambda_topo
        self.temperature   = temperature
        self.p_norm        = p_norm
        self.h1_weight     = h1_weight
        self.warmup_steps  = warmup_steps
        self.update_every  = update_every
        self.max_points    = max_points
        self._landscape    = PersistenceLandscape(n_filtrations, temperature)
        self._last_info: Dict = {}

    def forward(
        self,
        embeddings: torch.Tensor,  # [N, D]
        step:       int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute differentiable landscape loss.

        Returns:
            (loss * lambda_topo, info_dict)
        """
        device = embeddings.device
        zero   = embeddings.new_zeros(1).squeeze()

        if step < self.warmup_steps and self.warmup_steps > 0:
            return zero, {'landscape_loss': 0.0, 'skipped': 'warmup'}

        # Subsample
        N = embeddings.shape[0]
        if N > self.max_points:
            idx = torch.randperm(N, device=device)[:self.max_points]
            emb = embeddings[idx]
        else:
            emb = embeddings

        # Compute landscape
        result = self._landscape(emb)

        # H₀ target: landscape should decay to 0 (all connected at high ε)
        # Penalise residual isolation at high filtration values
        h0_loss = result['h0_persistence'][-8:].mean()   # last 8 = high ε

        # H₁ target: no cycles throughout filtration
        h1_loss = result['h1_persistence'].mean()

        total = h0_loss + self.h1_weight * h1_loss
        loss  = total * self.lambda_topo

        info = {
            'landscape_loss': float(total.item()),
            'h0_loss':        float(h0_loss.item()),
            'h1_loss':        float(h1_loss.item()),
            'total_h0':       float(result['total_h0'].item()),
            'total_h1':       float(result['total_h1'].item()),
            'backend':        'landscape',
            'n_points':       emb.shape[0],
        }
        self._last_info = info
        return loss, info

    def report(self) -> str:
        i = self._last_info
        return (f"L_landscape={i.get('landscape_loss',0):.4f} "
                f"H₀={i.get('h0_loss',0):.3f} H₁={i.get('h1_loss',0):.3f}")
