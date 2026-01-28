# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Unit tests for CGT loss functions.

Tests cover:
- HyperbolicInfoNCE: metric consistency, gradient flow
- PowerLawDistillation: compression behavior
- SpectralManifoldAlignment: eigenvalue matching
- TopoLoss: connected component proxy
- MultiObjectiveLoss: aggregation and weighting
"""

import pytest
import torch
import numpy as np
from typing import Tuple

from cgt.geometry.lorentz import LorentzSubstrate, LorentzConfig
from cgt.losses.core import (
    HyperbolicInfoNCE,
    PowerLawDistillation,
    SpectralManifoldAlignment,
    TopoLoss,
    MultiObjectiveLoss,
    LossConfig,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device() -> torch.device:
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def lorentz(device: torch.device) -> LorentzSubstrate:
    """Create Lorentz substrate with default config."""
    config = LorentzConfig(dim=64, curvature=-1.0)
    return LorentzSubstrate(config)


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 32


@pytest.fixture
def embedding_dim() -> int:
    """Default embedding dimension."""
    return 64


@pytest.fixture
def teacher_embeddings(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate random teacher embeddings (Euclidean)."""
    return torch.randn(batch_size, 768, device=device)


@pytest.fixture
def student_embeddings(
    batch_size: int, embedding_dim: int, lorentz: LorentzSubstrate, device: torch.device
) -> torch.Tensor:
    """Generate valid student embeddings on Lorentz manifold."""
    # Create tangent vectors and project to manifold
    tangent = torch.randn(batch_size, embedding_dim, device=device) * 0.5
    # Prepend time component (will be computed by exp_map)
    origin = lorentz.origin(batch_size, device)
    return lorentz.exp_map(tangent, origin)


# =============================================================================
# HyperbolicInfoNCE Tests
# =============================================================================

class TestHyperbolicInfoNCE:
    """Tests for HyperbolicInfoNCE contrastive loss."""
    
    def test_initialization(self, lorentz: LorentzSubstrate):
        """Test loss initialization with different parameters."""
        loss_fn = HyperbolicInfoNCE(lorentz=lorentz, temperature=0.07)
        assert loss_fn.temperature == 0.07
        
        loss_fn = HyperbolicInfoNCE(lorentz=lorentz, temperature=0.1)
        assert loss_fn.temperature == 0.1
    
    def test_output_shape(
        self, lorentz: LorentzSubstrate, student_embeddings: torch.Tensor
    ):
        """Test that loss returns scalar."""
        loss_fn = HyperbolicInfoNCE(lorentz=lorentz)
        
        # Create anchor and positive pairs
        anchor = student_embeddings
        positive = student_embeddings + torch.randn_like(student_embeddings) * 0.01
        
        loss = loss_fn(anchor, positive)
        
        assert loss.shape == torch.Size([])
        assert loss.ndim == 0
    
    def test_loss_is_positive(
        self, lorentz: LorentzSubstrate, student_embeddings: torch.Tensor
    ):
        """Test that InfoNCE loss is non-negative."""
        loss_fn = HyperbolicInfoNCE(lorentz=lorentz)
        
        anchor = student_embeddings
        positive = student_embeddings + torch.randn_like(student_embeddings) * 0.01
        
        loss = loss_fn(anchor, positive)
        
        assert loss.item() >= 0
    
    def test_gradient_flow(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test that gradients flow through the loss."""
        loss_fn = HyperbolicInfoNCE(lorentz=lorentz)
        
        # Create embeddings that require grad
        anchor = torch.randn(16, 65, device=device, requires_grad=True)
        positive = torch.randn(16, 65, device=device, requires_grad=True)
        
        loss = loss_fn(anchor, positive)
        loss.backward()
        
        assert anchor.grad is not None
        assert positive.grad is not None
        assert not torch.isnan(anchor.grad).any()
        assert not torch.isnan(positive.grad).any()
    
    def test_temperature_effect(
        self, lorentz: LorentzSubstrate, student_embeddings: torch.Tensor
    ):
        """Test that temperature affects loss magnitude."""
        anchor = student_embeddings
        positive = student_embeddings + torch.randn_like(student_embeddings) * 0.1
        
        loss_low_temp = HyperbolicInfoNCE(lorentz=lorentz, temperature=0.01)
        loss_high_temp = HyperbolicInfoNCE(lorentz=lorentz, temperature=1.0)
        
        val_low = loss_low_temp(anchor, positive).item()
        val_high = loss_high_temp(anchor, positive).item()
        
        # Lower temperature should give higher loss for same inputs
        # (sharper distribution)
        assert val_low != val_high


# =============================================================================
# PowerLawDistillation Tests
# =============================================================================

class TestPowerLawDistillation:
    """Tests for PowerLawDistillation loss."""
    
    def test_initialization(self, lorentz: LorentzSubstrate):
        """Test initialization with different alpha values."""
        loss_fn = PowerLawDistillation(lorentz=lorentz, alpha=2.0)
        assert loss_fn.alpha == 2.0
        
        loss_fn = PowerLawDistillation(lorentz=lorentz, alpha=3.0)
        assert loss_fn.alpha == 3.0
    
    def test_output_shape(
        self, 
        lorentz: LorentzSubstrate, 
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor
    ):
        """Test that loss returns scalar."""
        loss_fn = PowerLawDistillation(lorentz=lorentz)
        
        loss = loss_fn(teacher_embeddings, student_embeddings)
        
        assert loss.shape == torch.Size([])
    
    def test_loss_is_finite(
        self,
        lorentz: LorentzSubstrate,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor
    ):
        """Test that loss is finite."""
        loss_fn = PowerLawDistillation(lorentz=lorentz)
        
        loss = loss_fn(teacher_embeddings, student_embeddings)
        
        assert torch.isfinite(loss)
    
    def test_gradient_flow(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test gradient flow through distillation loss."""
        loss_fn = PowerLawDistillation(lorentz=lorentz)
        
        teacher = torch.randn(16, 768, device=device)
        student = torch.randn(16, 65, device=device, requires_grad=True)
        
        loss = loss_fn(teacher, student)
        loss.backward()
        
        assert student.grad is not None
        assert not torch.isnan(student.grad).any()


# =============================================================================
# SpectralManifoldAlignment Tests
# =============================================================================

class TestSpectralManifoldAlignment:
    """Tests for SpectralManifoldAlignment loss."""
    
    def test_initialization(self, lorentz: LorentzSubstrate):
        """Test initialization with different k values."""
        loss_fn = SpectralManifoldAlignment(lorentz=lorentz, k=10)
        assert loss_fn.k == 10
    
    def test_output_shape(
        self,
        lorentz: LorentzSubstrate,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor
    ):
        """Test that loss returns scalar."""
        loss_fn = SpectralManifoldAlignment(lorentz=lorentz, k=5)
        
        loss = loss_fn(teacher_embeddings, student_embeddings)
        
        assert loss.shape == torch.Size([])
    
    def test_loss_is_non_negative(
        self,
        lorentz: LorentzSubstrate,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor
    ):
        """Test that spectral loss is non-negative."""
        loss_fn = SpectralManifoldAlignment(lorentz=lorentz, k=5)
        
        loss = loss_fn(teacher_embeddings, student_embeddings)
        
        assert loss.item() >= 0
    
    def test_identical_inputs(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test that loss is low for identical spectral structure."""
        loss_fn = SpectralManifoldAlignment(lorentz=lorentz, k=5)
        
        # Create embeddings with similar structure
        embeddings = torch.randn(32, 64, device=device)
        
        # Same embeddings should have low loss
        loss = loss_fn(embeddings, embeddings)
        
        assert loss.item() < 1e-5


# =============================================================================
# TopoLoss Tests
# =============================================================================

class TestTopoLoss:
    """Tests for TopoLoss (Betti-0 proxy)."""
    
    def test_initialization(self, lorentz: LorentzSubstrate):
        """Test initialization with different thresholds."""
        loss_fn = TopoLoss(lorentz=lorentz, threshold=0.5)
        assert loss_fn.threshold == 0.5
    
    def test_output_shape(
        self, lorentz: LorentzSubstrate, student_embeddings: torch.Tensor
    ):
        """Test that loss returns scalar."""
        loss_fn = TopoLoss(lorentz=lorentz)
        
        loss = loss_fn(student_embeddings)
        
        assert loss.shape == torch.Size([])
    
    def test_loss_is_non_negative(
        self, lorentz: LorentzSubstrate, student_embeddings: torch.Tensor
    ):
        """Test that topo loss is non-negative."""
        loss_fn = TopoLoss(lorentz=lorentz)
        
        loss = loss_fn(student_embeddings)
        
        assert loss.item() >= 0
    
    def test_gradient_flow(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test gradient flow through topology loss."""
        loss_fn = TopoLoss(lorentz=lorentz)
        
        embeddings = torch.randn(16, 65, device=device, requires_grad=True)
        
        loss = loss_fn(embeddings)
        loss.backward()
        
        assert embeddings.grad is not None


# =============================================================================
# MultiObjectiveLoss Tests
# =============================================================================

class TestMultiObjectiveLoss:
    """Tests for MultiObjectiveLoss aggregation."""
    
    def test_initialization(self, lorentz: LorentzSubstrate):
        """Test initialization with LossConfig."""
        config = LossConfig(
            lambda_contrastive=1.0,
            lambda_distillation=0.5,
            lambda_spectral=0.1,
            lambda_topo=0.01
        )
        
        loss_fn = MultiObjectiveLoss(lorentz=lorentz, config=config)
        
        assert loss_fn.config.lambda_contrastive == 1.0
        assert loss_fn.config.lambda_distillation == 0.5
    
    def test_output_shape(
        self,
        lorentz: LorentzSubstrate,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor
    ):
        """Test that aggregated loss returns scalar."""
        config = LossConfig()
        loss_fn = MultiObjectiveLoss(lorentz=lorentz, config=config)
        
        # Create required inputs
        anchor = student_embeddings
        positive = student_embeddings + torch.randn_like(student_embeddings) * 0.01
        
        loss, breakdown = loss_fn(
            anchor=anchor,
            positive=positive,
            teacher=teacher_embeddings,
            student=student_embeddings
        )
        
        assert loss.shape == torch.Size([])
        assert isinstance(breakdown, dict)
    
    def test_loss_breakdown(
        self,
        lorentz: LorentzSubstrate,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor
    ):
        """Test that breakdown contains all loss components."""
        config = LossConfig()
        loss_fn = MultiObjectiveLoss(lorentz=lorentz, config=config)
        
        anchor = student_embeddings
        positive = student_embeddings + torch.randn_like(student_embeddings) * 0.01
        
        loss, breakdown = loss_fn(
            anchor=anchor,
            positive=positive,
            teacher=teacher_embeddings,
            student=student_embeddings
        )
        
        expected_keys = ['contrastive', 'distillation', 'spectral', 'topo', 'total']
        for key in expected_keys:
            assert key in breakdown
    
    def test_lambda_weighting(
        self,
        lorentz: LorentzSubstrate,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor
    ):
        """Test that lambda weights affect total loss."""
        anchor = student_embeddings
        positive = student_embeddings + torch.randn_like(student_embeddings) * 0.01
        
        # Config with all lambdas = 1
        config1 = LossConfig(
            lambda_contrastive=1.0,
            lambda_distillation=1.0,
            lambda_spectral=1.0,
            lambda_topo=1.0
        )
        
        # Config with reduced lambdas
        config2 = LossConfig(
            lambda_contrastive=0.1,
            lambda_distillation=0.1,
            lambda_spectral=0.1,
            lambda_topo=0.1
        )
        
        loss_fn1 = MultiObjectiveLoss(lorentz=lorentz, config=config1)
        loss_fn2 = MultiObjectiveLoss(lorentz=lorentz, config=config2)
        
        loss1, _ = loss_fn1(
            anchor=anchor, positive=positive,
            teacher=teacher_embeddings, student=student_embeddings
        )
        
        loss2, _ = loss_fn2(
            anchor=anchor, positive=positive,
            teacher=teacher_embeddings, student=student_embeddings
        )
        
        # Loss with higher lambdas should be larger
        assert loss1.item() > loss2.item()
    
    def test_gradient_flow_all_components(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test gradient flow through all loss components."""
        config = LossConfig()
        loss_fn = MultiObjectiveLoss(lorentz=lorentz, config=config)
        
        teacher = torch.randn(16, 768, device=device)
        student = torch.randn(16, 65, device=device, requires_grad=True)
        anchor = torch.randn(16, 65, device=device, requires_grad=True)
        positive = torch.randn(16, 65, device=device, requires_grad=True)
        
        loss, _ = loss_fn(
            anchor=anchor, positive=positive,
            teacher=teacher, student=student
        )
        
        loss.backward()
        
        assert student.grad is not None
        assert anchor.grad is not None
        assert positive.grad is not None


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability edge cases."""
    
    def test_infonce_with_large_distances(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test InfoNCE with embeddings far apart."""
        loss_fn = HyperbolicInfoNCE(lorentz=lorentz)
        
        # Create embeddings with large tangent vectors
        anchor = torch.randn(16, 65, device=device) * 10
        positive = torch.randn(16, 65, device=device) * 10
        
        loss = loss_fn(anchor, positive)
        
        assert torch.isfinite(loss)
    
    def test_infonce_with_small_distances(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test InfoNCE with nearly identical embeddings."""
        loss_fn = HyperbolicInfoNCE(lorentz=lorentz)
        
        base = torch.randn(16, 65, device=device)
        anchor = base
        positive = base + torch.randn_like(base) * 1e-6
        
        loss = loss_fn(anchor, positive)
        
        assert torch.isfinite(loss)
    
    def test_distillation_with_zero_teacher(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test distillation loss with near-zero teacher embeddings."""
        loss_fn = PowerLawDistillation(lorentz=lorentz)
        
        teacher = torch.randn(16, 768, device=device) * 1e-8
        student = torch.randn(16, 65, device=device)
        
        loss = loss_fn(teacher, student)
        
        assert torch.isfinite(loss)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
