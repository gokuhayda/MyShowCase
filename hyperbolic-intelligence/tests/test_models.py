# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Unit tests for CGT models.

Tests cover:
- CGTStudent: forward pass, manifold projection, gradient flow
- HomeostaticField: density preservation behavior
- create_projector: MLP initialization and spectral normalization
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from cgt.geometry.lorentz import LorentzSubstrate, LorentzConfig
from cgt.models.student import (
    CGTStudent,
    StudentConfig,
    HomeostaticField,
    create_projector,
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
    """Create Lorentz substrate."""
    config = LorentzConfig(dim=64, curvature=-1.0)
    return LorentzSubstrate(config)


@pytest.fixture
def student_config() -> StudentConfig:
    """Default student configuration."""
    return StudentConfig(
        input_dim=768,
        hidden_dim=512,
        output_dim=64,
        num_layers=2,
        dropout=0.1,
        use_spectral_norm=True
    )


@pytest.fixture
def batch_size() -> int:
    """Default batch size."""
    return 32


# =============================================================================
# create_projector Tests
# =============================================================================

class TestCreateProjector:
    """Tests for projector MLP creation."""
    
    def test_basic_creation(self):
        """Test basic projector creation."""
        projector = create_projector(
            input_dim=768,
            hidden_dim=512,
            output_dim=64,
            num_layers=2
        )
        
        assert isinstance(projector, nn.Sequential)
    
    def test_output_dimension(self, device: torch.device):
        """Test projector output dimension."""
        projector = create_projector(
            input_dim=768,
            hidden_dim=512,
            output_dim=64,
            num_layers=2
        ).to(device)
        
        x = torch.randn(16, 768, device=device)
        out = projector(x)
        
        assert out.shape == (16, 64)
    
    def test_single_layer(self, device: torch.device):
        """Test single layer projector."""
        projector = create_projector(
            input_dim=768,
            hidden_dim=512,
            output_dim=64,
            num_layers=1
        ).to(device)
        
        x = torch.randn(16, 768, device=device)
        out = projector(x)
        
        assert out.shape == (16, 64)
    
    def test_deep_network(self, device: torch.device):
        """Test deeper projector network."""
        projector = create_projector(
            input_dim=768,
            hidden_dim=256,
            output_dim=64,
            num_layers=4
        ).to(device)
        
        x = torch.randn(16, 768, device=device)
        out = projector(x)
        
        assert out.shape == (16, 64)
    
    def test_spectral_norm_enabled(self):
        """Test that spectral normalization is applied."""
        projector = create_projector(
            input_dim=768,
            hidden_dim=512,
            output_dim=64,
            num_layers=2,
            use_spectral_norm=True
        )
        
        # Check if any layer has spectral norm
        has_spectral_norm = False
        for module in projector.modules():
            if hasattr(module, 'weight_orig'):
                has_spectral_norm = True
                break
        
        assert has_spectral_norm
    
    def test_gradient_flow(self, device: torch.device):
        """Test gradient flow through projector."""
        projector = create_projector(
            input_dim=768,
            hidden_dim=512,
            output_dim=64,
            num_layers=2
        ).to(device)
        
        x = torch.randn(16, 768, device=device, requires_grad=True)
        out = projector(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# HomeostaticField Tests
# =============================================================================

class TestHomeostaticField:
    """Tests for HomeostaticField density preservation."""
    
    def test_initialization(self, lorentz: LorentzSubstrate, device: torch.device):
        """Test field initialization."""
        field = HomeostaticField(
            lorentz=lorentz,
            num_anchors=100,
            dim=64
        ).to(device)
        
        assert field.anchors.shape == (100, 65)  # +1 for time component
    
    def test_forward_shape(self, lorentz: LorentzSubstrate, device: torch.device):
        """Test forward pass output shape."""
        field = HomeostaticField(
            lorentz=lorentz,
            num_anchors=50,
            dim=64
        ).to(device)
        
        x = torch.randn(16, 65, device=device)
        out = field(x)
        
        assert out.shape == x.shape
    
    def test_regularization_loss(
        self, lorentz: LorentzSubstrate, device: torch.device
    ):
        """Test regularization loss computation."""
        field = HomeostaticField(
            lorentz=lorentz,
            num_anchors=50,
            dim=64
        ).to(device)
        
        x = torch.randn(16, 65, device=device)
        _ = field(x)
        
        reg_loss = field.regularization_loss()
        
        assert reg_loss.shape == torch.Size([])
        assert reg_loss.item() >= 0
    
    def test_gradient_flow(self, lorentz: LorentzSubstrate, device: torch.device):
        """Test gradient flow through homeostatic field."""
        field = HomeostaticField(
            lorentz=lorentz,
            num_anchors=50,
            dim=64
        ).to(device)
        
        x = torch.randn(16, 65, device=device, requires_grad=True)
        out = field(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None


# =============================================================================
# CGTStudent Tests
# =============================================================================

class TestCGTStudent:
    """Tests for CGTStudent model."""
    
    def test_initialization(
        self, 
        lorentz: LorentzSubstrate, 
        student_config: StudentConfig,
        device: torch.device
    ):
        """Test model initialization."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        assert model.config == student_config
    
    def test_forward_shape(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device,
        batch_size: int
    ):
        """Test forward pass output shape."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        x = torch.randn(batch_size, 768, device=device)
        out = model(x)
        
        # Output should be on manifold: dim + 1 for time component
        assert out.shape == (batch_size, 65)
    
    def test_output_on_manifold(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device,
        batch_size: int
    ):
        """Test that output satisfies manifold constraint."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        x = torch.randn(batch_size, 768, device=device)
        out = model(x)
        
        # Check Lorentz constraint: -t^2 + x^2 = -1/c
        t = out[:, 0]
        space = out[:, 1:]
        
        constraint = -t**2 + (space**2).sum(dim=-1)
        expected = -1.0 / abs(lorentz.config.curvature)
        
        assert torch.allclose(constraint, torch.full_like(constraint, expected), atol=1e-5)
    
    def test_gradient_flow(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device,
        batch_size: int
    ):
        """Test gradient flow through student model."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        x = torch.randn(batch_size, 768, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_parameter_count(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device
    ):
        """Test that model has trainable parameters."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert num_params > 0
    
    def test_eval_mode(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device,
        batch_size: int
    ):
        """Test model behavior in eval mode."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        model.eval()
        
        x = torch.randn(batch_size, 768, device=device)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(out1, out2)
    
    def test_different_batch_sizes(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device
    ):
        """Test model with different batch sizes."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        for bs in [1, 16, 32, 64]:
            x = torch.randn(bs, 768, device=device)
            out = model(x)
            
            assert out.shape == (bs, 65)
    
    def test_save_load(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device,
        tmp_path
    ):
        """Test model save and load."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load into new model
        model2 = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        model2.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        x = torch.randn(16, 768, device=device)
        
        model.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for model components."""
    
    def test_full_pipeline(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device
    ):
        """Test full forward-backward pipeline."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for _ in range(3):
            x = torch.randn(16, 768, device=device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
        
        # Should complete without errors
        assert True
    
    def test_multiple_forward_passes(
        self,
        lorentz: LorentzSubstrate,
        student_config: StudentConfig,
        device: torch.device
    ):
        """Test multiple consecutive forward passes."""
        model = CGTStudent(
            config=student_config,
            lorentz=lorentz
        ).to(device)
        
        outputs = []
        for _ in range(5):
            x = torch.randn(16, 768, device=device)
            out = model(x)
            outputs.append(out)
        
        # All outputs should be valid (on manifold)
        for out in outputs:
            t = out[:, 0]
            space = out[:, 1:]
            constraint = -t**2 + (space**2).sum(dim=-1)
            expected = -1.0 / abs(lorentz.config.curvature)
            assert torch.allclose(constraint, torch.full_like(constraint, expected), atol=1e-5)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
