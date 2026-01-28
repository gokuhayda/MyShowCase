# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Student Model [HARDENED VERSION]
====================================

Exact implementation matching CGT_Paper_Ready_v6_1_HARDENED notebook.
Includes HomeostaticFieldHardened and CGTStudentHardened classes.

Key Features:
- Tangent-space mapping with exp_map
- F.normalize with scale buffer (0.7)
- use_homeostatic parameter
- Riemannian-aware anchor initialization

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from cgt.geometry.lorentz_hardened import (
    LorentzConfig,
    LorentzSubstrateHardened,
    safe_acosh,
)


def create_projector(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    use_spectral: bool = True,
    use_dropout: bool = False,  # REQUIRED FOR NUMERICAL PARITY WITH AGI_v2
    dropout_rate: float = 0.1   # REQUIRED FOR NUMERICAL PARITY WITH AGI_v2
) -> nn.Sequential:
    """
    Create MLP projector with spectral normalization and orthogonal init.
    
    Architecture:
        Linear → LayerNorm → GELU → [Dropout] → Linear → LayerNorm → GELU → [Dropout] → Linear
    
    Initialization:
        - Orthogonal init for hidden layers (ERank > 1.0)
        - Small normal init for output layer (prevents NaN in acosh)
    
    Args:
        input_dim: Input dimension (teacher embedding dim)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (student intrinsic dim)
        use_spectral: Whether to apply spectral normalization
        use_dropout: Whether to use Dropout (for AGI_v2 parity)
        dropout_rate: Dropout rate when use_dropout=True
        
    Returns:
        nn.Sequential projector module
    """
    def maybe_sn(layer):
        # REQUIRED FOR NUMERICAL PARITY WITH AGI_v2: disable spectral norm when using dropout
        if use_dropout:
            return layer
        return spectral_norm(layer) if use_spectral else layer

    l1 = nn.Linear(input_dim, hidden_dim)
    l2 = nn.Linear(hidden_dim, hidden_dim)
    l3 = nn.Linear(hidden_dim, output_dim)

    # Orthogonal initialization (vital for ERank > 1.0)
    nn.init.orthogonal_(l1.weight, gain=1.0)
    nn.init.orthogonal_(l2.weight, gain=1.0)

    # Fine initialization to avoid NaN in acosh
    nn.init.normal_(l3.weight, std=1e-4)
    if l3.bias is not None:
        nn.init.zeros_(l3.bias)

    # REQUIRED FOR NUMERICAL PARITY WITH AGI_v2
    if use_dropout:
        return nn.Sequential(
            l1, nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            l2, nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            l3
        )
    else:
        return nn.Sequential(
            maybe_sn(l1), nn.LayerNorm(hidden_dim), nn.GELU(),
            maybe_sn(l2), nn.LayerNorm(hidden_dim), nn.GELU(),
            maybe_sn(l3)
        )


class HomeostaticFieldHardened(nn.Module):
    """
    Homeostatic Field for density preservation [HARDENED].
    
    Uses Riemannian operations (log_map, exp_map) to move points
    toward learned anchors on the manifold.
    
    CRITICAL FIX: s_dim = substrate.n (not substrate.n - 1)
    This ensures dimension compatibility with CGTStudent output.
    
    AUDIT FIX v9.9.4: Removed train/eval discrepancy that caused covariate shift.
    Now behavior is consistent: always_apply controls whether field is active.
    
    Args:
        substrate: LorentzSubstrateHardened instance
        n_anchors: Number of anchor points
        alpha: Step size for homeostatic adjustment
        always_apply: If True, apply in both train and eval (default: True)
        
    Attributes:
        anchors: Learnable anchor points on manifold [n_anchors, n+1]
        
    Notes:
        - Classification: HEURISTIC (density preservation)
        - Anchors are initialized ON the hyperboloid
        - FIXED: Consistent behavior in train AND eval
    """
    
    def __init__(
        self,
        substrate: LorentzSubstrateHardened,
        n_anchors: int = 16,
        alpha: float = 0.1,
        always_apply: bool = True  # AUDIT FIX: configurable, default True
    ):
        super().__init__()
        self.substrate = substrate
        self.n_anchors = n_anchors
        self.alpha = alpha
        self.always_apply = always_apply  # AUDIT FIX

        # CRITICAL FIX: s_dim = substrate.n (not substrate.n - 1)
        # This ensures dimension match with CGTStudent output
        s_dim = substrate.n

        # Initialize spatial components
        init_s = torch.randn(n_anchors, s_dim) * 0.01

        # Compute time component: x₀ = √(1 + ||x_s||²)
        # This guarantees anchors are born ON the hyperboloid
        init_t = torch.sqrt(1.0 + (init_s**2).sum(dim=1, keepdim=True))
        self.anchors = nn.Parameter(torch.cat([init_t, init_s], dim=-1))

    def forward(self, hyp_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply homeostatic adjustment to embeddings.
        
        AUDIT FIX: Now applies consistently in train AND eval to avoid
        covariate shift that was degrading test performance.
        
        Steps:
        1. Find nearest anchor for each point
        2. Compute log_map direction to anchor
        3. Take small exp_map step toward anchor
        
        Args:
            hyp_emb: Hyperbolic embeddings [B, n+1]
            
        Returns:
            Adjusted embeddings [B, n+1]
        """
        # AUDIT FIX: Removed train-only check that caused distribution mismatch
        # Old code: if not self.training: return hyp_emb  # REMOVED
        
        if not self.always_apply:
            return hyp_emb

        dev = hyp_emb.device
        dtype = hyp_emb.dtype
        
        # Project anchors to ensure they're on manifold
        anchors_fixed = self.substrate.proj(self.anchors.to(dev, dtype))

        # Compute distances [Batch, Anchors]
        dists = self.substrate.distance_matrix_points(hyp_emb, anchors_fixed)

        # Find nearest anchor for each point
        nearest_idx = dists.argmin(dim=1)
        nearest_anchors = anchors_fixed[nearest_idx]

        # Compute direction in tangent space (log_map)
        direction = self.substrate.log_map(hyp_emb, nearest_anchors)
        
        # Take small step toward anchor (exp_map)
        return self.substrate.exp_map(hyp_emb, self.alpha * direction)

    def regularization_loss(self, device: torch.device) -> torch.Tensor:
        """
        Compute anchor spreading regularization.
        
        Encourages anchors to be spread apart, preventing collapse.
        Penalizes anchor pairs closer than 0.5 geodesic distance.
        
        Args:
            device: Target device for computation
            
        Returns:
            Regularization loss scalar
        """
        anchors_proj = self.substrate.proj(self.anchors.to(device))

        # Manual self-distance matrix for anchors
        inner = -torch.mm(anchors_proj[:, :1], anchors_proj[:, :1].t()) + \
                 torch.mm(anchors_proj[:, 1:], anchors_proj[:, 1:].t())
        d_matrix = safe_acosh(torch.clamp(-1.0 * inner, min=1.0 + 1e-7))

        # Exclude diagonal
        mask = ~torch.eye(self.n_anchors, dtype=torch.bool, device=device)
        if not mask.any():
            return torch.tensor(0.0, device=device)

        # Minimum distance to other anchors
        min_dists = d_matrix[mask].view(self.n_anchors, -1).min(dim=1)[0]
        
        # Penalize if minimum distance < 0.5
        return F.relu(0.5 - min_dists).mean()


class CGTStudentHardened(nn.Module):
    """
    Hyperbolic Student Encoder [HARDENED VERSION].
    
    Maps Euclidean teacher embeddings to Lorentz (hyperbolic) manifold
    via tangent-space projection and exponential map.
    
    Optimized for:
    - Riemannian stability
    - 24x compression ratio (384 → 16 effective dimensions)
    - Metric-consistent training
    
    Architecture:
        Teacher Emb → Projector → Normalize*Scale → Tangent Vector → Exp Map → Manifold
    
    AUDIT FIX v9.9.4: scale is now learnable parameter instead of fixed buffer.
    This allows the model to utilize the full hyperbolic capacity instead of
    being constrained to the quasi-Euclidean shell at radius 0.7.
    
    Args:
        teacher_dim: Teacher embedding dimension (e.g., 384 for MiniLM)
        student_dim: Student intrinsic dimension (hyperbolic dim)
        hidden_dim: Projector hidden dimension
        learnable_curvature: Whether curvature K is learnable
        initial_curvature: Initial K value
        curvature_min: Minimum K (prevents collapse)
        curvature_max: Maximum K (prevents explosion)
        learnable_scale: Whether scale is learnable (AUDIT FIX)
        initial_scale: Initial scale value
        scale_min: Minimum scale (prevents collapse)
        scale_max: Maximum scale (prevents explosion)
        
    Attributes:
        substrate: LorentzSubstrateHardened manifold
        projector: MLP encoder
        scale: Normalization scale (now learnable by default)
        homeostatic_layer: Optional HomeostaticFieldHardened
        anchors: Reference to homeostatic anchors (for Riemannian optimizer)
        
    Notes:
        - Classification: EXACT mapping via exp_map
        - AUDIT FIX: scale is learnable to access full hyperbolic capacity
        - use_homeostatic=False available for Lipschitz testing
    """
    
    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        hidden_dim: int = 256,
        learnable_curvature: bool = True,
        initial_curvature: float = 1.0,
        curvature_min: float = 0.1,
        curvature_max: float = 5.0,
        learnable_scale: bool = True,  # AUDIT FIX: now learnable by default
        initial_scale: float = 0.7,
        scale_min: float = 0.3,
        scale_max: float = 2.5,
        use_dropout: bool = False,  # REQUIRED FOR NUMERICAL PARITY WITH AGI_v2
        dropout_rate: float = 0.1,   # REQUIRED FOR NUMERICAL PARITY WITH AGI_v2
    ):
        """
        CGTStudentHardened - Hyperbolic Student Encoder.
        
        CORRECT USAGE (the only permitted pattern):
            student = CGTStudentHardened(
                teacher_dim=384,
                student_dim=32,
                hidden_dim=256,
                initial_curvature=1.0,
            )
            substrate = student.substrate  # ALWAYS get from student
        
        Args:
            teacher_dim: Teacher embedding dimension (e.g., 384 for MiniLM, 768 for MPNET)
            student_dim: Student intrinsic dimension (hyperbolic dim)
            hidden_dim: Projector hidden dimension
            initial_curvature: Initial curvature K (positive value)
        
        DEPRECATED PARAMETERS (will raise TypeError):
            - input_dim: Use teacher_dim instead
            - output_dim: Use student_dim instead  
            - substrate: Not allowed. Model creates its own substrate.
        """
        # BLINDAGEM: Validação obrigatória
        assert teacher_dim is not None, "teacher_dim is required"
        assert student_dim is not None, "student_dim is required"
        assert isinstance(teacher_dim, int) and teacher_dim > 0, "teacher_dim must be positive int"
        assert isinstance(student_dim, int) and student_dim > 0, "student_dim must be positive int"
        
        super().__init__()

        config = LorentzConfig(
            intrinsic_dim=student_dim,
            initial_curvature=initial_curvature,
            learnable_curvature=learnable_curvature,
            curvature_min=curvature_min,
            curvature_max=curvature_max
        )

        # Substrate and Projector as submodules
        self.substrate = LorentzSubstrateHardened(config)
        self.student_dim = student_dim
        
        # AUDIT FIX: Store scale bounds
        self.scale_min = scale_min
        self.scale_max = scale_max

        # REQUIRED FOR NUMERICAL PARITY WITH AGI_v2
        self.projector = create_projector(
            teacher_dim, hidden_dim, student_dim, 
            use_spectral=not use_dropout,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate
        )

        # AUDIT FIX: Learnable scale instead of fixed buffer
        # Use log-scale for unconstrained optimization
        import math
        if learnable_scale:
            self._log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))
        else:
            self.register_buffer('_log_scale', torch.tensor(math.log(initial_scale)))

        # Placeholders for trainer compatibility
        self.homeostatic_layer: Optional[HomeostaticFieldHardened] = None
        self.anchors: Optional[nn.Parameter] = None
    
    @property
    def scale(self) -> torch.Tensor:
        """Returns clamped scale value. AUDIT FIX: now learnable."""
        return torch.exp(self._log_scale).clamp(self.scale_min, self.scale_max)

    def init_homeostatic(
        self,
        n_anchors: int = 16,
        alpha: float = 0.1
    ) -> HomeostaticFieldHardened:
        """
        Initialize homeostatic field and expose anchors.
        
        Args:
            n_anchors: Number of anchor points
            alpha: Homeostatic step size
            
        Returns:
            Initialized HomeostaticFieldHardened
        """
        self.homeostatic_layer = HomeostaticFieldHardened(
            self.substrate, n_anchors, alpha
        )

        # Direct binding: expose anchors at parent model level
        # This is needed for RiemannianOptimizerWrapper
        self.anchors = self.homeostatic_layer.anchors
        self.anchors.on_manifold = True  # type: ignore

        return self.homeostatic_layer

    def forward(
        self,
        teacher_emb: torch.Tensor,
        use_homeostatic: bool = True
    ) -> torch.Tensor:
        """
        Forward pass: Euclidean → Tangent → Manifold.
        
        Args:
            teacher_emb: Teacher embeddings [B, teacher_dim]
            use_homeostatic: Whether to apply homeostatic refinement
            
        Returns:
            Hyperbolic embeddings on Lorentz manifold [B, student_dim+1]
        """
        dev = teacher_emb.device
        dtype = teacher_emb.dtype
        bs = teacher_emb.shape[0]

        # Fail-safe device synchronization
        if next(self.projector.parameters()).device != dev:
            self.to(dev)

        # 1. Euclidean projection and normalization
        projected = F.normalize(self.projector(teacher_emb), dim=-1) * self.scale

        # 2. Construct tangent vector at origin (T_0 M)
        # Lorentz tangent: [0, v₁, v₂, ..., vₙ]
        tangent = torch.zeros(bs, self.student_dim + 1, device=dev, dtype=dtype)
        tangent[:, 1:] = projected

        # 3. Exponential map (retraction to hyperboloid)
        origin = self.substrate.origin(bs).to(dev, dtype)
        hyp_emb = self.substrate.exp_map(origin, tangent)

        # 4. Homeostatic refinement
        if use_homeostatic and self.homeostatic_layer is not None:
            hyp_emb = self.homeostatic_layer(hyp_emb)

        # ═══════════════════════════════════════════════════════════════
        # F1 CORRECTION: Projeção explícita para garantir ⟨h,h⟩_L = -1/K
        # Isso corrige o "Manifold Drift" identificado na auditoria.
        # ═══════════════════════════════════════════════════════════════
        hyp_emb = self.substrate.proj(hyp_emb)

        return hyp_emb

    def get_curvature(self) -> float:
        """Get current curvature value as float."""
        return self.substrate.K.item()


class RiemannianOptimizerWrapper:
    """
    Wrapper for Riemannian optimization on Lorentz manifold.
    
    Converts Euclidean gradients to Riemannian gradients for
    parameters that live on the manifold (e.g., anchors).
    
    Args:
        base_optimizer: Standard PyTorch optimizer (e.g., AdamW)
        substrate: LorentzSubstrateHardened for gradient conversion
        
    Usage:
        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer = RiemannianOptimizerWrapper(base_opt, model.substrate)
        
        # Training loop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    Notes:
        - Only affects parameters with on_manifold=True attribute
        - Other parameters use standard Euclidean updates
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        substrate: LorentzSubstrateHardened
    ):
        self.base_optimizer = base_optimizer
        self.substrate = substrate
    
    def zero_grad(self):
        """Zero all gradients."""
        self.base_optimizer.zero_grad()
    
    def step(self):
        """
        Perform optimization step with Riemannian correction.
        
        For manifold parameters: convert Euclidean grad to Riemannian
        For other parameters: standard update
        """
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Check if parameter lives on manifold
                if hasattr(p, 'on_manifold') and p.on_manifold:
                    # Convert to Riemannian gradient
                    p.grad.data = self.substrate.riemannian_grad(
                        p.data, p.grad.data
                    )
        
        # Perform base optimizer step
        self.base_optimizer.step()
        
        # Project manifold parameters back to manifold
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if hasattr(p, 'on_manifold') and p.on_manifold:
                    p.data = self.substrate.proj(p.data)
    
    @property
    def param_groups(self):
        """Access base optimizer param_groups."""
        return self.base_optimizer.param_groups


class RiemannianAdam(torch.optim.Optimizer):
    """
    Full Riemannian Adam optimizer with parallel transport.
    
    AUDIT FIX v9.9.4: The RiemannianOptimizerWrapper does not perform
    parallel transport of momentum between tangent spaces. This causes
    momentum vectors from different tangent spaces to be incorrectly
    summed, introducing geometric error that grows with curvature.
    
    This implementation properly transports the momentum from T_{θ_{t-1}}
    to T_{θ_t} before combining with the current gradient.
    
    Args:
        params: Model parameters
        substrate: LorentzSubstrateHardened for Riemannian operations
        lr: Learning rate
        betas: Adam momentum coefficients
        eps: Numerical stability constant
        weight_decay: L2 regularization weight
        
    Usage:
        optimizer = RiemannianAdam(
            model.parameters(),
            substrate=model.substrate,
            lr=1e-4
        )
    """
    
    def __init__(
        self,
        params,
        substrate: LorentzSubstrateHardened,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        import math
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.substrate = substrate
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform Riemannian Adam step with parallel transport.
        
        For manifold parameters:
        1. Convert gradient to Riemannian
        2. Transport previous momentum to current tangent space
        3. Update momentum and second moment
        4. Compute search direction and retract via exp_map
        
        For Euclidean parameters:
        Standard Adam update.
        """
        import math
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                is_manifold = hasattr(p, 'on_manifold') and p.on_manifold
                
                if is_manifold:
                    # Convert to Riemannian gradient
                    grad = self.substrate.riemannian_grad(p.data, grad)
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if is_manifold:
                        state['prev_point'] = p.data.clone()
                
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # CRITICAL: Parallel transport of momentum for manifold params
                if is_manifold and state['step'] > 1:
                    prev_point = state['prev_point']
                    exp_avg = self._parallel_transport(prev_point, p.data, exp_avg)
                
                # Update first moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment
                if is_manifold:
                    grad_sq = self.substrate.minkowski_inner(grad, grad).abs()
                    exp_avg_sq.mul_(beta2).add_(
                        grad_sq.expand_as(exp_avg_sq), alpha=1 - beta2
                    )
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                
                if is_manifold:
                    # Riemannian update via exp_map
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    search_dir = exp_avg / denom.mean()
                    
                    # Geodesic step (retraction)
                    p_new = self.substrate.exp_map(p.data, -step_size * search_dir)
                    
                    # Store for next parallel transport
                    state['prev_point'] = p.data.clone()
                    p.data.copy_(p_new)
                else:
                    # Standard Euclidean Adam
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    if is_manifold:
                        decay_grad = self.substrate.riemannian_grad(p.data, p.data)
                        p_decayed = self.substrate.exp_map(
                            p.data, -group['lr'] * group['weight_decay'] * decay_grad
                        )
                        p.data.copy_(p_decayed)
                    else:
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
        
        return loss
    
    def _parallel_transport(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport vector v from T_x M to T_y M.
        
        Uses closed-form for the Lorentz (hyperboloid) model.
        """
        K = self.substrate.K.to(x.device, x.dtype)
        
        # Log map from x to y
        log_xy = self.substrate.log_map(x, y)
        log_xy_norm_sq = torch.abs(self.substrate.minkowski_inner(log_xy, log_xy)) + 1e-10
        log_xy_norm = torch.sqrt(log_xy_norm_sq)
        
        # Transport coefficients
        inner_v_log = self.substrate.minkowski_inner(v, log_xy)
        inner_x_y = self.substrate.minkowski_inner(x, y)
        
        coeff = inner_v_log / log_xy_norm_sq
        
        # Transported vector
        transported = v - coeff * (log_xy + K * inner_x_y * x)
        
        return transported
