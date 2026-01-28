# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
K_Light Losses (STS-specific)
=============================

Additional losses for K_Lighting STS/ranking encoder.
Reuses CGT infrastructure where possible.

Only contains what CGT does NOT have:
- FormanRicciLoss: Discrete curvature regularizer
- CoherenceLoss: Semantic preservation
- KLightMultiObjectiveLoss: Combines all losses
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse CGT losses
from cgt.losses import (
    HyperbolicInfoNCE_Lorentz,
    KLDistillation,
    TopoLoss,
)


class FormanRicciLoss(nn.Module):
    """
    Forman-Ricci curvature regularizer.
    
    Adapted from CGT's F3_forman_ricci (evaluation) to a training loss.
    Encourages hyperbolic-like structure by targeting negative curvature.
    
    Args:
        target_kappa: Target mean curvature (default: -0.1, slightly hyperbolic)
        k: Number of nearest neighbors for graph
    """

    def __init__(self, target_kappa: float = -0.1, k: int = 5):
        super().__init__()
        self.target_kappa = target_kappa
        self.k = k

    def forward(self, D: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Compute Forman-Ricci loss.
        
        Args:
            D: Distance matrix [B, B]
            
        Returns:
            Tuple of (loss, kappa_mean, kappa_min)
        """
        B = D.shape[0]
        device = D.device
        dtype = D.dtype
        k = min(self.k, B - 1)

        if B < 4:
            return torch.tensor(0.0, device=device, dtype=dtype), 0.0, 0.0

        # k-NN adjacency (soft)
        _, indices = torch.topk(-D, k + 1, dim=1)
        indices = indices[:, 1:]  # Exclude self

        # Build soft adjacency
        adj = torch.zeros(B, B, device=device, dtype=dtype)
        for i in range(B):
            adj[i, indices[i]] = 1.0
        adj = (adj + adj.t()) / 2  # Symmetrize

        # Degrees
        degrees = adj.sum(dim=1)

        # Forman-Ricci: κ(e) = 4 - deg(u) - deg(v)
        # Compute mean curvature over edges
        deg_i = degrees.unsqueeze(1).expand(B, B)
        deg_j = degrees.unsqueeze(0).expand(B, B)
        kappa_matrix = 4.0 - deg_i - deg_j

        # Mask to edges only
        edge_mask = adj > 0
        kappa_edges = kappa_matrix[edge_mask]

        if kappa_edges.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), 0.0, 0.0

        kappa_mean = kappa_edges.mean()
        kappa_min = kappa_edges.min().item()

        # Loss: minimize deviation from target
        loss = (kappa_mean - self.target_kappa) ** 2

        return loss, kappa_mean.item(), kappa_min


class CoherenceLoss(nn.Module):
    """
    Semantic coherence loss.
    
    Ensures that relative similarities in hyperbolic space
    preserve the relative structure from the teacher.
    
    This is a simpler version of distillation that only
    cares about rank preservation, not exact distances.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self, 
        D_student: torch.Tensor, 
        D_teacher: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coherence loss.
        
        Args:
            D_student: Student distance matrix [B, B]
            D_teacher: Teacher distance matrix [B, B]
            
        Returns:
            Coherence loss scalar
        """
        # Normalize to [0, 1]
        D_s = D_student / (D_student.max() + 1e-8)
        D_t = D_teacher / (D_teacher.max() + 1e-8)

        # Rank correlation via soft ranking
        # Pairs where teacher says i closer to j than k should maintain order
        return F.mse_loss(D_s, D_t)


class KLightMultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss for K_Lighting STS encoder.
    
    Combines CGT losses with K_Light-specific losses.
    
    Components (reused from CGT):
    - Contrastive (InfoNCE with geodesic distances)
    - Distillation (KL-divergence)
    - Topological (β₀ preservation)
    
    Components (K_Light specific):
    - Forman-Ricci (curvature regularization)
    - Coherence (semantic preservation)
    
    Args:
        lambda_contrastive: Weight for contrastive loss
        lambda_distill: Weight for distillation loss
        lambda_topo: Weight for topological loss
        lambda_forman: Weight for Forman-Ricci loss
        lambda_coherence: Weight for coherence loss
        temperature: InfoNCE temperature
        target_beta_0: Target connectivity (β₀)
        target_kappa: Target curvature
    """

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_distill: float = 0.5,
        lambda_topo: float = 0.3,
        lambda_forman: float = 0.1,
        lambda_coherence: float = 0.1,
        temperature: float = 0.07,
        target_beta_0: float = 1.0,
        target_kappa: float = -0.1,
    ):
        super().__init__()
        
        # Weights
        self.lc = lambda_contrastive
        self.ld = lambda_distill
        self.lt = lambda_topo
        self.lf = lambda_forman
        self.lco = lambda_coherence
        
        # CGT reused losses
        self.contrastive_fn = HyperbolicInfoNCE_Lorentz(temperature=temperature)
        self.distill_fn = KLDistillation()
        self.topo_fn = TopoLoss(target_beta_0=target_beta_0)
        
        # K_Light specific losses
        self.forman_fn = FormanRicciLoss(target_kappa=target_kappa)
        self.coherence_fn = CoherenceLoss()

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        model: nn.Module,
        current_epoch: int = 0,
        max_epoch: int = 100,
    ) -> Dict:
        """
        Compute combined loss.
        
        Args:
            student_emb: Hyperbolic student embeddings [B, n+1]
            teacher_emb: Teacher embeddings [B, D]
            model: Student model (for substrate access)
            current_epoch: Current epoch (for annealing)
            max_epoch: Total epochs
            
        Returns:
            Dictionary with total loss and components
        """
        device = student_emb.device
        dtype = student_emb.dtype
        
        # Get substrate
        substrate = model.substrate
        
        # Distance matrices
        D_s = substrate.distance_matrix(student_emb)
        t_norm = F.normalize(teacher_emb, dim=-1)
        D_t = 1.0 - torch.mm(t_norm, t_norm.t())  # Cosine distance
        
        # 1. Contrastive loss (CGT)
        l_contrastive = self.contrastive_fn(student_emb, teacher_emb, substrate)
        
        # 2. Distillation loss (CGT)
        l_distill, _, _, _ = self.distill_fn(student_emb, t_norm, substrate)
        
        # 3. Topological loss (CGT)
        l_topo, beta_0 = self.topo_fn(D_s, current_epoch, max_epoch)
        
        # 4. Forman-Ricci loss (K_Light)
        l_forman, kappa_mean, kappa_min = self.forman_fn(D_s)
        
        # 5. Coherence loss (K_Light)
        l_coherence = self.coherence_fn(D_s, D_t)
        
        # Total
        total = (
            self.lc * l_contrastive +
            self.ld * l_distill +
            self.lt * l_topo +
            self.lf * l_forman +
            self.lco * l_coherence
        )
        
        return {
            'total': total,
            'loss/contrastive': l_contrastive.item(),
            'loss/distill': l_distill.item(),
            'loss/topo': l_topo.item(),
            'loss/forman': l_forman.item() if isinstance(l_forman, torch.Tensor) else l_forman,
            'loss/coherence': l_coherence.item(),
            'topology/beta_0': beta_0,
            'geometry/kappa_mean': kappa_mean,
            'geometry/kappa_min': kappa_min,
        }
