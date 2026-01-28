# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Hyperbolic LLM Loss Functions
=============================

Loss functions for training Hyperbolic Transformers.

Components:
- HyperbolicLMLoss: Combined loss for language modeling
- RadiusRegularization: Soft radius constraint
- ManifoldFidelityLoss: Constraint violation penalty
- HyperbolicInfoNCE: Contrastive loss in hyperbolic space
- TeacherDistillationLoss: Teacher-student distillation

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened


class RadiusRegularization(nn.Module):
    """
    Soft radius regularization to prevent boundary drift.
    
    L_radius = (1/B) Σ max(0, d_H(h_i, o) - R_max)²
    """
    
    def __init__(
        self, 
        substrate: LorentzSubstrateHardened,
        radius_max: float = 10.0,
    ):
        super().__init__()
        self.substrate = substrate
        self.radius_max = radius_max
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        radii = self.substrate.lorentz_radius(hidden_states)
        violation = F.relu(radii - self.radius_max)
        return violation.pow(2).mean()


class ManifoldFidelityLoss(nn.Module):
    """
    Manifold constraint violation penalty.
    
    L_F1 = E[|⟨h,h⟩_L + 1/K|]
    """
    
    def __init__(self, substrate: LorentzSubstrateHardened):
        super().__init__()
        self.substrate = substrate
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.substrate.manifold_violation(hidden_states)


class HyperbolicInfoNCE(nn.Module):
    """
    Hyperbolic InfoNCE contrastive loss.
    
    Uses geodesic distance: sim(x, y) = exp(-d_H(x, y) / τ)
    """
    
    def __init__(
        self,
        substrate: LorentzSubstrateHardened,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.substrate = substrate
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = embeddings.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        D = self.substrate.distance_matrix(embeddings)
        logits = -D / self.temperature
        logits = logits.clamp(-50.0, 50.0)
        logits = logits - logits.max(dim=1, keepdim=True).values
        
        if labels is None:
            labels = torch.arange(B, device=embeddings.device)
        
        return F.cross_entropy(logits, labels)


class TeacherDistillationLoss(nn.Module):
    """
    Full distillation loss from Euclidean teacher (e.g., GPT-2).
    
    Combines output distribution alignment and hidden state alignment.
    """
    
    def __init__(
        self,
        substrate: LorentzSubstrateHardened,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.substrate = substrate
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = student_logits.device
        V = student_logits.shape[-1]
        
        s_logits = student_logits[..., :-1, :].contiguous()
        t_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Soft target loss
        s_log_probs = F.log_softmax(s_logits / self.temperature, dim=-1)
        t_probs = F.softmax(t_logits / self.temperature, dim=-1)
        
        l_soft = F.kl_div(
            s_log_probs.view(-1, V),
            t_probs.view(-1, V),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        l_hard = F.cross_entropy(
            s_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # Hidden state alignment
        l_hidden = torch.tensor(0.0, device=device)
        if student_hidden is not None and teacher_hidden is not None:
            B, L, _ = student_hidden.shape
            
            if L > 32:
                indices = torch.randperm(L, device=device)[:32]
                s_sample = student_hidden[:, indices, :]
                t_sample = teacher_hidden[:, indices, :]
            else:
                s_sample = student_hidden
                t_sample = teacher_hidden
            
            s_flat = s_sample.view(-1, s_sample.shape[-1])
            D_s = self.substrate.distance_matrix(s_flat)
            
            t_flat = t_sample.view(-1, t_sample.shape[-1])
            t_norm = F.normalize(t_flat, dim=-1)
            D_t = 1.0 - torch.mm(t_norm, t_norm.t())
            
            D_s_norm = D_s / (D_s.max() + 1e-8)
            D_t_norm = D_t / (D_t.max() + 1e-8)
            l_hidden = F.mse_loss(D_s_norm, D_t_norm.detach())
        
        l_total = self.alpha * l_soft + (1 - self.alpha) * l_hard + 0.1 * l_hidden
        
        return {
            'total': l_total,
            'loss/soft': l_soft.item(),
            'loss/hard': l_hard.item(),
            'loss/hidden': l_hidden.item() if isinstance(l_hidden, torch.Tensor) else l_hidden,
        }


class HyperbolicLMLoss(nn.Module):
    """
    Combined loss for Hyperbolic Language Model training.
    
    Total = λ_LM * L_LM + λ_InfoNCE * L_InfoNCE + λ_F1 * L_F1 + λ_radius * L_radius
    """
    
    def __init__(
        self,
        substrate: LorentzSubstrateHardened,
        vocab_size: int,
        lambda_lm: float = 1.0,
        lambda_infonce: float = 0.5,
        lambda_f1: float = 0.5,
        lambda_radius: float = 0.001,
        radius_max: float = 10.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.substrate = substrate
        self.vocab_size = vocab_size
        
        self.lambda_lm = lambda_lm
        self.lambda_infonce = lambda_infonce
        self.lambda_f1 = lambda_f1
        self.lambda_radius = lambda_radius
        
        self.radius_loss = RadiusRegularization(substrate, radius_max)
        self.manifold_loss = ManifoldFidelityLoss(substrate)
        self.infonce_loss = HyperbolicInfoNCE(substrate, temperature)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
        return_components: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = logits.device
        
        # LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        l_lm = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # InfoNCE
        B, L, D = hidden_states.shape
        if L > 1:
            mid = L // 2
            h_sample = hidden_states[:, mid, :]
            l_infonce = self.infonce_loss(h_sample)
        else:
            l_infonce = torch.tensor(0.0, device=device)
        
        # Manifold fidelity
        l_f1 = self.manifold_loss(hidden_states)
        
        # Radius
        l_radius = self.radius_loss(hidden_states)
        
        # Total
        l_total = (
            self.lambda_lm * l_lm +
            self.lambda_infonce * l_infonce +
            self.lambda_f1 * l_f1 +
            self.lambda_radius * l_radius
        )
        
        result = {'total': l_total}
        
        if return_components:
            result.update({
                'loss/lm': l_lm.item() if isinstance(l_lm, torch.Tensor) else l_lm,
                'loss/infonce': l_infonce.item() if isinstance(l_infonce, torch.Tensor) else l_infonce,
                'loss/manifold': l_f1.item() if isinstance(l_f1, torch.Tensor) else l_f1,
                'loss/radius': l_radius.item() if isinstance(l_radius, torch.Tensor) else l_radius,
            })
        
        return result


class HyperbolicLLMTrainingLoss(nn.Module):
    """
    Complete training loss for Hyperbolic LLM with teacher distillation.
    """
    
    def __init__(
        self,
        substrate: LorentzSubstrateHardened,
        vocab_size: int,
        lambda_lm: float = 1.0,
        lambda_infonce: float = 0.5,
        lambda_f1: float = 0.5,
        lambda_radius: float = 0.001,
        lambda_distill: float = 0.5,
        radius_max: float = 10.0,
        temperature: float = 0.07,
        distill_temperature: float = 2.0,
    ):
        super().__init__()
        self.substrate = substrate
        self.vocab_size = vocab_size
        
        self.lambda_lm = lambda_lm
        self.lambda_infonce = lambda_infonce
        self.lambda_f1 = lambda_f1
        self.lambda_radius = lambda_radius
        self.lambda_distill = lambda_distill
        
        self.lm_loss = HyperbolicLMLoss(
            substrate, vocab_size,
            lambda_lm=1.0,
            lambda_infonce=1.0,
            lambda_f1=1.0,
            lambda_radius=1.0,
            radius_max=radius_max,
            temperature=temperature,
        )
        
        self.distill_loss = TeacherDistillationLoss(
            substrate,
            temperature=distill_temperature,
        )
    
    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = student_logits.device
        
        base_result = self.lm_loss(
            student_logits, labels, hidden_states,
            return_components=True
        )
        
        l_lm = base_result.get('loss/lm', 0.0)
        l_infonce = base_result.get('loss/infonce', 0.0)
        l_f1 = base_result.get('loss/manifold', 0.0)
        l_radius = base_result.get('loss/radius', 0.0)
        
        l_distill = 0.0
        distill_components = {}
        if teacher_logits is not None:
            distill_result = self.distill_loss(
                student_logits, teacher_logits, labels,
                hidden_states, teacher_hidden
            )
            l_distill = distill_result['total'].item()
            distill_components = {
                'loss/distill_soft': distill_result.get('loss/soft', 0.0),
                'loss/distill_hard': distill_result.get('loss/hard', 0.0),
                'loss/distill_hidden': distill_result.get('loss/hidden', 0.0),
            }
            l_total_tensor = base_result['total'] + self.lambda_distill * distill_result['total']
        else:
            l_total_tensor = base_result['total']
        
        result = {
            'total': l_total_tensor,
            'loss/lm': l_lm,
            'loss/infonce': l_infonce,
            'loss/manifold': l_f1,
            'loss/radius': l_radius,
            'loss/distill': l_distill,
        }
        result.update(distill_components)
        
        return result
