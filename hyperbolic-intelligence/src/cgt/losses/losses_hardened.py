# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Loss Functions [HARDENED VERSION]
=====================================

EXACT implementation matching CGT_Paper_Ready_v6_1_HARDENED notebook Cell 26.
Multi-Objective Loss for CGT Training with Float64 precision.

V9.9.3: Float64 + Device consistency + Complete loss return

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default configurations (can be overridden)
NUM_EPOCHS = 100  # Used for topo annealing


class TopoLoss(nn.Module):
    """
    Topological Loss with temperature annealing.
    
    Uses spectral proxy for Betti-0 estimation via graph Laplacian eigenvalues.
    Temperature annealing: starts soft, becomes sharper over epochs.
    """
    
    def __init__(self, target_beta_0: float = 1.0, temp_init: float = 0.2, temp_min: float = 0.03):
        super().__init__()
        self.target = target_beta_0
        self.temp_init = temp_init
        self.temp_min = temp_min

    def temperature(self, epoch: int, max_epoch: int) -> float:
        """Compute annealed temperature for current epoch."""
        return max(self.temp_min, self.temp_init * (1 - epoch / max_epoch))

    def forward(self, D: torch.Tensor, epoch: int = 0, max_epoch: int = 100) -> Tuple[torch.Tensor, float]:
        """
        Compute topology loss.
        
        Args:
            D: Distance matrix [B, B]
            epoch: Current epoch
            max_epoch: Total epochs
            
        Returns:
            Tuple of (loss, beta_0_value)
        """
        tau = self.temperature(epoch, max_epoch)
        adj = torch.sigmoid((2.0 - D) / tau)
        laplacian = torch.diag(adj.sum(dim=1)) - adj
        eigs = torch.linalg.eigvalsh(laplacian)
        beta_0 = torch.sum(torch.exp(-eigs / tau))
        return (beta_0 - self.target) ** 2, beta_0.item()


class LipschitzRegularizer(nn.Module):
    """
    Lipschitz regularizer using RATIO formulation.
    
    Garante suavidade métrica via razão de expansão: 
    d_H(f(x), f(x+e)) / d_E(e) <= K
    
    IMPORTANT: Uses F.relu(ratio - 1.0), not F.relu(d_output - d_input)
    """

    def __init__(self, noise_scale: float = 0.05):
        super().__init__()
        self.noise_scale = noise_scale

    def forward(self, model: nn.Module, teacher_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute Lipschitz violation penalty.
        
        Args:
            model: CGT student model (must have substrate attribute)
            teacher_emb: Teacher embeddings [B, D]
            
        Returns:
            Mean Lipschitz violation (scalar)
        """
        dtype = teacher_emb.dtype
        teacher_emb = teacher_emb.to(dtype=dtype)
        noise = torch.randn_like(teacher_emb) * self.noise_scale

        emb_orig = model(teacher_emb, use_homeostatic=False)
        emb_pert = model(teacher_emb + noise, use_homeostatic=False)

        d_input = noise.norm(dim=-1)
        d_output = model.substrate.dist(emb_orig, emb_pert)

        # RATIO formulation: penalize when ratio > 1.0
        return F.relu(d_output / (d_input + 1e-8) - 1.0).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# F1/F3 CORRECTION LOSSES
# ═══════════════════════════════════════════════════════════════════════════════

class MinkowskiViolationLoss(nn.Module):
    """
    Penaliza desvios da restrição Lorentziana ⟨h,h⟩_L = -1/K.
    
    F1 CORRECTION: Esta loss força os embeddings a permanecerem
    no hiperboloide durante todo o treinamento, corrigindo o
    "Manifold Drift" identificado na auditoria.
    
    Mathematical Basis:
        Para pontos no hiperboloide H^n com curvatura K:
        ⟨h,h⟩_L = -h₀² + h₁² + ... + hₙ² = -1/K
        
        Violação = |⟨h,h⟩_L - (-1/K)|
        
    Args:
        weight_schedule: Estratégia de annealing do peso
            - 'constant': Peso fixo em max_weight
            - 'linear': Cresce linearmente de 0 a max_weight na primeira metade
            - 'warmup': Zero nos primeiros 10% de epochs, depois constante
        max_weight: Peso máximo da penalidade (default: 0.5)
    """
    
    def __init__(
        self, 
        weight_schedule: str = 'linear', 
        max_weight: float = 0.5
    ):
        super().__init__()
        self.weight_schedule = weight_schedule
        self.max_weight = max_weight
    
    def get_weight(self, epoch: int, max_epoch: int) -> float:
        """Retorna peso annealed para a epoch atual."""
        if self.weight_schedule == 'constant':
            return self.max_weight
        elif self.weight_schedule == 'linear':
            return self.max_weight * min(1.0, epoch / max(1, max_epoch * 0.5))
        elif self.weight_schedule == 'warmup':
            if epoch < max_epoch * 0.1:
                return 0.0
            return self.max_weight
        return self.max_weight
    
    def forward(
        self, 
        student_emb: torch.Tensor, 
        substrate,
        epoch: int = 0,
        max_epoch: int = 100
    ) -> torch.Tensor:
        """
        Computa penalidade de violação Minkowski.
        
        Args:
            student_emb: Embeddings hiperbólicos [B, D+1]
            substrate: LorentzSubstrate com curvatura K
            epoch: Epoch atual (para annealing)
            max_epoch: Total de epochs
            
        Returns:
            Penalidade escalar: weight * mean(|⟨h,h⟩_L + 1/K|)
        """
        if student_emb.shape[0] == 0:
            return torch.tensor(0.0, device=student_emb.device, requires_grad=True)
            
        K = substrate.K.to(student_emb.device, student_emb.dtype)
        
        # Produto interno de Minkowski: -h₀² + h₁² + ... + hₙ²
        inner = substrate.minkowski_inner(student_emb, student_emb).squeeze(-1)
        
        # Target: ⟨h,h⟩_L = -1/K para pontos no hiperboloide
        target = -1.0 / K
        
        # Violação: |⟨h,h⟩_L - (-1/K)|
        violation = torch.abs(inner - target)
        violation = torch.nan_to_num(violation, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Peso crescente ao longo do treino
        weight = self.get_weight(epoch, max_epoch)
        
        return weight * violation.mean()


class KNNConsistencyLoss(nn.Module):
    """
    Preserva estrutura de vizinhança local E global entre teacher e student.
    
    F3 CORRECTION (ENHANCED): Força alinhamento dos k vizinhos mais próximos
    (estrutura local) E consistência de distâncias em pares aleatórios
    (estrutura global). Isso corrige o problema onde KNN local pode
    preservar vizinhança mas quebrar topologia global (Betti-1).
    
    Mathematical Basis:
        LOCAL (k-NN):
        Para cada ponto i:
        1. Encontra os k vizinhos mais próximos no espaço do TEACHER
        2. Computa distribuição softmax de distâncias para estes vizinhos
        3. Loss_local = KL(P_teacher || P_student)
        
        GLOBAL (Random Sampling):
        1. Amostra pares aleatórios (i, j) do batch
        2. Impõe MSE entre distâncias: d_student[i,j] ≈ scale * d_teacher[i,j]
        3. Loss_global = MSE(d_student, d_teacher_scaled)
        
    Args:
        k: Número de vizinhos a considerar (default: 10)
        temperature: Temperatura do softmax (default: 0.1)
        global_weight: Peso do termo global (default: 0.5)
        global_sample_ratio: Fração de pares a amostrar (default: 0.1)
    """
    
    def __init__(
        self, 
        k: int = 10, 
        temperature: float = 0.1,
        global_weight: float = 0.5,
        global_sample_ratio: float = 0.1
    ):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.global_weight = global_weight
        self.global_sample_ratio = global_sample_ratio
    
    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        substrate
    ) -> torch.Tensor:
        """
        Computa loss de consistência k-NN local + global.
        
        Args:
            student_emb: Embeddings hiperbólicos [B, D+1]
            teacher_emb: Embeddings do teacher [B, D_teacher]
            substrate: LorentzSubstrate para distâncias geodésicas
            
        Returns:
            Combinação de KL divergence local + MSE global
        """
        B = student_emb.shape[0]
        device = student_emb.device
        dtype = student_emb.dtype
        
        if B <= self.k + 1:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        k = min(self.k, B - 1)
        
        # Teacher distances (cosine)
        t_norm = F.normalize(teacher_emb, dim=-1, eps=1e-8)
        D_teacher = 1.0 - torch.mm(t_norm, t_norm.t())
        
        # Student distances (geodesic)
        D_student = substrate.distance_matrix(student_emb)
        
        # ═══════════════════════════════════════════════════════════════════════
        # LOCAL LOSS: k-NN Consistency (preserves local neighborhood)
        # ═══════════════════════════════════════════════════════════════════════
        # k-NN definidos pelo teacher (ground truth)
        _, teacher_knn_idx = torch.topk(D_teacher, k + 1, dim=1, largest=False)
        teacher_knn_idx = teacher_knn_idx[:, 1:]  # Remove self
        
        # Coletar distâncias para os k vizinhos
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, k)
        d_teacher_knn = D_teacher[batch_idx, teacher_knn_idx]
        d_student_knn = D_student[batch_idx, teacher_knn_idx]
        
        # KL divergence entre distribuições de vizinhança
        P_teacher = F.softmax(-d_teacher_knn / self.temperature, dim=-1)
        P_student_log = F.log_softmax(-d_student_knn / self.temperature, dim=-1)
        local_loss = F.kl_div(P_student_log, P_teacher, reduction='batchmean')
        
        # ═══════════════════════════════════════════════════════════════════════
        # GLOBAL LOSS: Random Pair Consistency (preserves global structure)
        # F3 CORRECTION: Imposes long-range structure to prevent topological defects
        # ═══════════════════════════════════════════════════════════════════════
        global_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        if self.global_weight > 0 and B > 2:
            # Sample random pairs (forming a "global skeleton")
            num_global = max(int(B * B * self.global_sample_ratio), B)
            idx1 = torch.randint(0, B, (num_global,), device=device)
            idx2 = torch.randint(0, B, (num_global,), device=device)
            
            # Get distances for sampled pairs
            d_s_global = D_student[idx1, idx2]
            d_t_global = D_teacher[idx1, idx2]
            
            # Scale teacher distances to match student range (hyperbolic vs cosine)
            # This prevents scale mismatch from dominating the loss
            with torch.no_grad():
                d_s_mean = d_s_global.mean() + 1e-8
                d_t_mean = d_t_global.mean() + 1e-8
                scale = d_s_mean / d_t_mean
            
            # MSE between scaled distances
            global_loss = F.mse_loss(d_s_global, scale * d_t_global)
        
        # ═══════════════════════════════════════════════════════════════════════
        # COMBINED LOSS
        # ═══════════════════════════════════════════════════════════════════════
        total_loss = local_loss + self.global_weight * global_loss
        
        return torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)


class PowerLawDistillation(nn.Module):
    """
    Power-law distance distillation.
    
    Comprime distâncias euclidianas para a escala hiperbólica:
    d_target = d_teacher^α
    
    NOTE: Consider using KLDistillation instead for better gradient stability.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, 
        student_emb: torch.Tensor, 
        t_norm: torch.Tensor, 
        substrate
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Compute distillation loss.
        
        Returns:
            Tuple of (loss, D_s, D_t, d_t_mean)
        """
        dtype = student_emb.dtype
        student_emb = student_emb.to(dtype=dtype)
        t_norm = t_norm.to(dtype=dtype)

        # Power-law transformed teacher distances
        D_t = torch.pow(torch.clamp(1.0 - torch.mm(t_norm, t_norm.t()), min=1e-7), self.alpha)
        
        # Student geodesic distances
        D_s = substrate.distance_matrix(student_emb)

        # Normalized MSE loss
        loss = F.mse_loss(D_s / (D_s.max() + 1e-8), D_t.detach() / (D_t.max() + 1e-8))
        
        return loss, D_s, D_t.detach(), D_t.mean().item()


class KLDistillation(nn.Module):
    """
    KL-divergence distillation loss.
    
    AUDIT FIX v9.9.4: Replaces PowerLawDistillation for better gradient stability.
    
    The PowerLaw formulation d_target = d_teacher^0.5 has gradient ∂/∂x(√x) = 1/(2√x)
    which explodes as x → 0 (very similar pairs). This causes training instability
    for highly similar sentence pairs.
    
    KL-divergence instead aligns probability distributions derived from distance
    matrices, avoiding the gradient explosion issue.
    
    Args:
        teacher_temperature: Temperature for teacher softmax
        student_temperature: Temperature for student softmax
    """
    
    def __init__(
        self,
        teacher_temperature: float = 0.1,
        student_temperature: float = 0.07
    ):
        super().__init__()
        self.tau_t = teacher_temperature
        self.tau_s = student_temperature
    
    def forward(
        self,
        student_emb: torch.Tensor,
        t_norm: torch.Tensor,
        substrate
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Compute KL-divergence distillation loss.
        
        Args:
            student_emb: Hyperbolic student embeddings [B, D+1]
            t_norm: Normalized teacher embeddings [B, D_teacher]
            substrate: LorentzSubstrate for distance computation
            
        Returns:
            Tuple of (loss, D_s, D_t, d_t_mean) for compatibility with PowerLaw
        """
        # Teacher: cosine distance matrix
        D_t = 1.0 - torch.mm(t_norm, t_norm.t())  # [B, B]
        
        # Student: geodesic distance matrix
        D_s = substrate.distance_matrix(student_emb)
        
        # Convert to probability distributions (row-wise softmax)
        # Negative distance = higher similarity = higher probability
        P_t = F.softmax(-D_t / self.tau_t, dim=-1)
        P_s = F.log_softmax(-D_s / self.tau_s, dim=-1)
        
        # KL divergence: KL(P_t || P_s)
        loss = F.kl_div(P_s, P_t, reduction='batchmean')
        
        return loss, D_s, D_t.detach(), D_t.mean().item()


class SpectralManifoldAlignmentHardened(nn.Module):
    """
    Spectral alignment loss with hardened numerical stability.
    
    Aligns Laplacian spectra between student and teacher distance matrices.
    """

    def __init__(self, sigma: float = 1.0, min_batch: int = 16, max_batch: int = 128):
        super().__init__()
        self.sigma = sigma
        self.min_batch = min_batch
        self.max_batch = max_batch

    def forward(
        self, 
        D_s: torch.Tensor, 
        D_t: torch.Tensor, 
        eye: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute spectral alignment loss.
        
        Args:
            D_s: Student distance matrix [B, B]
            D_t: Teacher distance matrix [B, B]
            eye: Identity matrix [B, B]
            
        Returns:
            Tuple of (loss, spectral_gap)
        """
        batch_size = D_s.shape[0]
        dtype = D_s.dtype
        device = D_s.device

        if batch_size < self.min_batch or batch_size > self.max_batch:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True), 0.0

        # Gaussian adjacency matrices
        A_s = torch.exp(-D_s**2 / (2 * self.sigma**2)) * (1 - eye)
        A_t = torch.exp(-D_t**2 / (2 * self.sigma**2)) * (1 - eye)

        def get_laplacian(A, I):
            deg = A.sum(dim=1).clamp_min(1e-8)
            d_inv = torch.diag(torch.pow(deg, -0.5))
            return I - d_inv @ A @ d_inv

        L_s = get_laplacian(A_s, eye)
        L_t = get_laplacian(A_t, eye)

        try:
            eig_s = torch.linalg.eigvalsh(L_s + eye * 1e-6)
            eig_t = torch.linalg.eigvalsh(L_t + eye * 1e-6)
            
            # Skip first eigenvalue (always 0)
            s_v, t_v = eig_s[1:], eig_t[1:]
            
            # Normalized spectral alignment
            loss = F.mse_loss(
                s_v / (s_v.max().clamp_min(1e-8)), 
                t_v / (t_v.max().clamp_min(1e-8))
            )
            spectral_gap = (eig_s[1] - eig_s[0]).item()
            
            return loss, spectral_gap
        except RuntimeError:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True), 0.0


class HyperbolicInfoNCE_Lorentz(nn.Module):
    """
    InfoNCE contrastive loss using geodesic distances.
    
    Uses distance matrix from substrate directly, with logits = -D/τ.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_emb: torch.Tensor, teacher_emb: torch.Tensor, substrate) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            student_emb: Hyperbolic embeddings [B, n+1]
            teacher_emb: Teacher embeddings (unused, for API compatibility)
            substrate: Lorentz substrate
            
        Returns:
            Cross-entropy loss scalar
        """
        B = student_emb.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=student_emb.device, requires_grad=True)

        # Geodesic distance matrix [B, B]
        D = substrate.distance_matrix(student_emb)

        # Similarity = negative distance
        logits = -D / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values

        labels = torch.arange(B, device=student_emb.device)
        return F.cross_entropy(logits, labels)


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss combining all CGT training objectives.
    
    EXACT match to CGT_Paper_Ready_v6_1_HARDENED notebook Cell 26.
    
    Components:
    - Contrastive (InfoNCE with geodesic distances)
    - Radius stability
    - Distillation (Power-law geometric OR KL-divergence)
    - Spectral alignment
    - Topological (Betti-0 spectral proxy)
    - Lipschitz regularization
    - Minkowski violation (F1 CORRECTION)
    - k-NN consistency (F3 CORRECTION)

    V9.9.4: Added use_kl_distillation option for gradient stability.
    V9.9.5: Added F1/F3 correction losses (MinkowskiViolationLoss, KNNConsistencyLoss).
    """

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_distill: float = 0.7,
        lambda_spectral: float = 0.2,
        lambda_topo: float = 0.02,
        lambda_lipschitz: float = 0.01,
        radius_weight: float = 0.05,
        temperature: float = 0.07,
        target_beta_0: float = 1.0,
        power_law_alpha: float = 0.5,
        topo_center: float = 2.0,
        topo_temp: float = 0.15,
        spectral_max_batch: int = 128,
        num_epochs: int = 100,
        use_kl_distillation: bool = False,  # AUDIT FIX v9.9.4
        # F1/F3 CORRECTION parameters
        use_minkowski_penalty: bool = True,    # F1: Força geometria Lorentziana
        minkowski_max_weight: float = 0.5,     # F1: Peso máximo
        use_knn_consistency: bool = True,      # F3: Preserva topologia local
        lambda_knn: float = 0.3,               # F3: Peso da loss k-NN
        knn_k: int = 10,                       # F3: Número de vizinhos
    ):
        super().__init__()

        # Loss weights
        self.lc = lambda_contrastive
        self.ld = lambda_distill
        self.ls = lambda_spectral
        self.lt = lambda_topo
        self.ll = lambda_lipschitz
        self.lknn = lambda_knn if use_knn_consistency else 0.0  # F3 CORRECTION
        self.temp = temperature
        self.radius_weight = radius_weight
        self.topo_center = topo_center
        self.topo_temp = topo_temp
        self.num_epochs = num_epochs
        self.use_minkowski_penalty = use_minkowski_penalty  # F1 CORRECTION

        # Buffers for persistence in state_dict
        self.register_buffer("target_beta_0", torch.tensor(float(target_beta_0)))
        self.register_buffer("target_var", torch.tensor(1.0))
        self.register_buffer("target_mean", torch.tensor(1.0))

        # Sub-modules
        # AUDIT FIX v9.9.4: Option to use KL distillation for gradient stability
        if use_kl_distillation:
            self.distill_fn = KLDistillation()
        else:
            self.distill_fn = PowerLawDistillation(alpha=power_law_alpha)
        self.spectral_fn = SpectralManifoldAlignmentHardened(max_batch=spectral_max_batch)
        self.lipschitz_fn = LipschitzRegularizer()
        
        # F1/F3 CORRECTION: New loss components
        self.minkowski_fn = MinkowskiViolationLoss(max_weight=minkowski_max_weight) if use_minkowski_penalty else None
        self.knn_fn = KNNConsistencyLoss(k=knn_k) if use_knn_consistency else None

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        model: nn.Module,
        current_epoch: int = None
    ) -> Dict:
        """
        Compute multi-objective loss.

        Args:
            student_emb: Student embeddings [B, D+1] (hyperbolic)
            teacher_emb: Teacher embeddings [B, D_teacher]
            model: CGT model with substrate
            current_epoch: Current training epoch (optional)

        Returns:
            Dictionary with total loss and all components.
            Keys match exactly what CGTTrainer expects.
        """
        device = student_emb.device
        dtype = student_emb.dtype

        # Normalize teacher for distillation
        t_norm = F.normalize(teacher_emb, dim=-1, eps=1e-8)

        batch_size = student_emb.shape[0]
        eye = torch.eye(batch_size, device=device, dtype=dtype)
        substrate = model.substrate

        # Temperature annealing for topology
        if current_epoch is None:
            topo_tau = self.topo_temp
        else:
            topo_tau = max(
                0.03,
                self.topo_temp * (1.0 - current_epoch / max(1, self.num_epochs))
            )

        # ═══════════════════════════════════════════════════════════════════════
        # 1. InfoNCE Contrastive Loss
        # ═══════════════════════════════════════════════════════════════════════
        l_contr = torch.tensor(0.0, device=device, dtype=dtype)

        if self.lc > 0 and batch_size > 1:
            D = substrate.distance_matrix(student_emb)
            logits = (-D / self.temp).clamp(-50, 50)
            logits = logits - logits.max(dim=1, keepdim=True).values
            labels = torch.arange(batch_size, device=device)
            l_contr = F.cross_entropy(logits, labels)

        # ═══════════════════════════════════════════════════════════════════════
        # 2. Radius Stability Loss
        # ═══════════════════════════════════════════════════════════════════════
        radii = substrate.lorentz_radius(student_emb)

        t_var = self.target_var.to(device=device, dtype=dtype)
        t_mean = self.target_mean.to(device=device, dtype=dtype)

        r_var = radii.var() if batch_size > 1 else torch.tensor(0.0, device=device, dtype=dtype)

        l_radius = (F.mse_loss(r_var, t_var) +
                    F.mse_loss(radii.mean(), t_mean)) * 0.5 * self.radius_weight

        # ═══════════════════════════════════════════════════════════════════════
        # 3. Geometric Distillation Loss
        # ═══════════════════════════════════════════════════════════════════════
        l_distill, D_s, D_t, d_t_mean = self.distill_fn(student_emb, t_norm, substrate)

        # ═══════════════════════════════════════════════════════════════════════
        # 4. Spectral Alignment Loss
        # ═══════════════════════════════════════════════════════════════════════
        l_spec, spectral_gap = self.spectral_fn(D_s, D_t, eye)

        # ═══════════════════════════════════════════════════════════════════════
        # 5. Topological Loss (Annealed Betti-0 Spectral Proxy)
        # ═══════════════════════════════════════════════════════════════════════
        # AUDIT FIX: Original implementation depended on absolute scale of D_s,
        # violating topological invariance. Now uses dynamic τ relative to batch
        # statistics, ensuring the loss penalizes connectivity structure, not scale.
        l_topo = torch.tensor(0.0, device=device, dtype=dtype)
        beta_0_val = 1.0

        if self.lt > 0 and batch_size >= 8:
            # Dynamic tau based on batch mean distance (AUDIT CORRECTION)
            batch_mean_dist = D_s.detach().mean()
            tau_eff = topo_tau * (batch_mean_dist + 1e-6)
            
            # Scale-invariant adjacency: relative to batch statistics
            adj = torch.sigmoid((batch_mean_dist - D_s.detach()) / tau_eff) * (1 - eye)
            laplacian = torch.diag(adj.sum(dim=1)) - adj + eye * 1e-6

            try:
                eigs = torch.linalg.eigvalsh(laplacian)
                beta_0 = torch.sum(torch.exp(-eigs / tau_eff))
                l_topo = F.mse_loss(
                    beta_0,
                    self.target_beta_0.to(device=device, dtype=dtype)
                )
                beta_0_val = beta_0.item()
            except RuntimeError:
                pass

        # ═══════════════════════════════════════════════════════════════════════
        # 6. Lipschitz Regularization
        # ═══════════════════════════════════════════════════════════════════════
        l_lipschitz = torch.tensor(0.0, device=device, dtype=dtype)
        if self.ll > 0:
            l_lipschitz = self.lipschitz_fn(model, teacher_emb)

        # ═══════════════════════════════════════════════════════════════════════
        # 7. Minkowski Violation Loss (F1 CORRECTION)
        # ═══════════════════════════════════════════════════════════════════════
        l_minkowski = torch.tensor(0.0, device=device, dtype=dtype)
        minkowski_violation = 0.0
        if self.use_minkowski_penalty and self.minkowski_fn is not None:
            l_minkowski = self.minkowski_fn(
                student_emb, substrate, 
                epoch=current_epoch if current_epoch is not None else 0,
                max_epoch=self.num_epochs
            )
            # Compute raw violation for logging (without weight)
            K = substrate.K.to(device, dtype)
            inner = substrate.minkowski_inner(student_emb, student_emb).squeeze(-1)
            minkowski_violation = torch.abs(inner - (-1.0 / K)).mean().item()

        # ═══════════════════════════════════════════════════════════════════════
        # 8. k-NN Consistency Loss (F3 CORRECTION)
        # ═══════════════════════════════════════════════════════════════════════
        l_knn = torch.tensor(0.0, device=device, dtype=dtype)
        if self.lknn > 0 and self.knn_fn is not None and batch_size > 11:
            l_knn = self.knn_fn(student_emb, teacher_emb, substrate)

        # ═══════════════════════════════════════════════════════════════════════
        # TOTAL LOSS
        # ═══════════════════════════════════════════════════════════════════════
        l_total = (
            self.lc * (l_contr + l_radius) +
            self.ld * l_distill +
            self.ls * l_spec +
            self.lt * l_topo +
            self.ll * l_lipschitz +
            l_minkowski +           # F1 CORRECTION (weight is internal)
            self.lknn * l_knn       # F3 CORRECTION
        )

        # ═══════════════════════════════════════════════════════════════════════
        # RETURN - EXACT KEYS matching CGTTrainer._extract_loss expectations
        # ═══════════════════════════════════════════════════════════════════════
        return {
            'total': l_total,
            'loss/contrastive': l_contr.item() if isinstance(l_contr, torch.Tensor) else l_contr,
            'loss/radius': l_radius.item() if isinstance(l_radius, torch.Tensor) else l_radius,
            'loss/distill': l_distill.item() if isinstance(l_distill, torch.Tensor) else l_distill,
            'loss/spectral': l_spec.item() if isinstance(l_spec, torch.Tensor) else l_spec,
            'loss/topo': l_topo.item() if isinstance(l_topo, torch.Tensor) else l_topo,
            'loss/lipschitz': l_lipschitz.item() if isinstance(l_lipschitz, torch.Tensor) else l_lipschitz,
            'loss/minkowski': l_minkowski.item() if isinstance(l_minkowski, torch.Tensor) else l_minkowski,  # F1
            'loss/knn': l_knn.item() if isinstance(l_knn, torch.Tensor) else l_knn,  # F3
            'topology/beta_0': beta_0_val,
            'topology/spectral_gap': spectral_gap,
            'metric/d_t_mean': d_t_mean,
            'metric/minkowski_violation': minkowski_violation,  # F1 raw metric
        }


# Aliases for backward compatibility
MultiObjectiveLossHardened = MultiObjectiveLoss
SpectralManifoldAlignment = SpectralManifoldAlignmentHardened
