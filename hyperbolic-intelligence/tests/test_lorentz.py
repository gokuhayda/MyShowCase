# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Unit Tests for Lorentz Geometry Module
======================================

Tests for the Lorentz manifold substrate operations.

These tests verify:
1. Manifold constraint satisfaction (<x, x>_L = -1)
2. Distance metric consistency (non-negative, symmetric, triangle inequality)
3. Exponential/logarithmic map invertibility
4. Numerical stability under edge cases

Usage:
    pytest tests/test_lorentz.py -v

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

# Import CGT modules
from cgt.geometry import LorentzConfig, LorentzSubstrate


# ═══════════════════════════════════════════════════════════════════════════════
#                    TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def lorentz_config():
    """Default Lorentz configuration."""
    return LorentzConfig(
        ambient_dim=33,  # 32 + 1 for time coordinate
        initial_curvature=1.0,
        learnable_curvature=False,
    )


@pytest.fixture
def lorentz(lorentz_config):
    """Lorentz manifold substrate."""
    return LorentzSubstrate(lorentz_config)


@pytest.fixture
def sample_points(lorentz):
    """Generate sample points on the manifold."""
    batch_size = 10
    ambient_dim = lorentz.config.ambient_dim
    
    # Generate random tangent vectors at origin
    tangent = torch.randn(batch_size, ambient_dim - 1, dtype=torch.float64)
    
    # Map to manifold via exp_map from origin
    origin = lorentz.origin(batch_size)
    tangent_full = torch.cat([torch.zeros(batch_size, 1, dtype=torch.float64), tangent], dim=-1)
    
    points = lorentz.exp_map(origin, tangent_full)
    return points


# ═══════════════════════════════════════════════════════════════════════════════
#                    MANIFOLD CONSTRAINT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestManifoldConstraints:
    """Tests for Lorentz manifold constraint satisfaction."""
    
    def test_origin_on_manifold(self, lorentz):
        """Test that origin satisfies <x, x>_L = -1."""
        origin = lorentz.origin(1)
        inner = lorentz.inner_product(origin, origin)
        
        expected = -1.0
        assert torch.allclose(inner, torch.tensor([expected], dtype=torch.float64), atol=1e-10), \
            f"Origin inner product: {inner.item()}, expected: {expected}"
    
    def test_exp_map_stays_on_manifold(self, lorentz):
        """Test that exp_map output satisfies manifold constraint."""
        batch_size = 100
        origin = lorentz.origin(batch_size)
        
        # Random tangent vectors
        tangent = torch.randn(batch_size, lorentz.config.ambient_dim - 1, dtype=torch.float64)
        tangent_full = torch.cat([torch.zeros(batch_size, 1, dtype=torch.float64), tangent], dim=-1)
        
        # Map to manifold
        points = lorentz.exp_map(origin, tangent_full)
        
        # Check constraint
        inner = lorentz.inner_product(points, points)
        expected = torch.full((batch_size,), -1.0, dtype=torch.float64)
        
        assert torch.allclose(inner, expected, atol=1e-8), \
            f"Max constraint violation: {(inner + 1.0).abs().max().item()}"
    
    def test_projection_enforces_constraint(self, lorentz):
        """Test that projection enforces manifold constraint."""
        batch_size = 50
        
        # Generate points slightly off the manifold
        points = torch.randn(batch_size, lorentz.config.ambient_dim, dtype=torch.float64)
        points[:, 0] = torch.sqrt(1.0 + (points[:, 1:] ** 2).sum(dim=-1))
        
        # Add noise to violate constraint
        points = points + 0.1 * torch.randn_like(points)
        
        # Project back
        projected = lorentz.project(points)
        
        # Verify constraint
        inner = lorentz.inner_product(projected, projected)
        expected = torch.full((batch_size,), -1.0, dtype=torch.float64)
        
        assert torch.allclose(inner, expected, atol=1e-8), \
            f"Post-projection violation: {(inner + 1.0).abs().max().item()}"


# ═══════════════════════════════════════════════════════════════════════════════
#                    DISTANCE METRIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistanceMetric:
    """Tests for geodesic distance properties."""
    
    def test_distance_non_negative(self, lorentz, sample_points):
        """Test that distance is non-negative."""
        points = sample_points
        
        # Compute pairwise distances
        dist_matrix = lorentz.distance_matrix(points, points)
        
        assert (dist_matrix >= 0).all(), \
            f"Negative distances found: {dist_matrix.min().item()}"
    
    def test_distance_symmetric(self, lorentz, sample_points):
        """Test that distance is symmetric: d(x, y) = d(y, x)."""
        points = sample_points
        
        dist_matrix = lorentz.distance_matrix(points, points)
        
        # Check symmetry
        diff = (dist_matrix - dist_matrix.T).abs()
        assert diff.max() < 1e-10, \
            f"Asymmetry detected: max diff = {diff.max().item()}"
    
    def test_distance_zero_self(self, lorentz, sample_points):
        """Test that d(x, x) = 0."""
        points = sample_points
        
        dist_matrix = lorentz.distance_matrix(points, points)
        diagonal = torch.diag(dist_matrix)
        
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-10), \
            f"Non-zero self-distances: max = {diagonal.max().item()}"
    
    def test_triangle_inequality(self, lorentz, sample_points):
        """Test triangle inequality: d(x, z) <= d(x, y) + d(y, z)."""
        points = sample_points
        n = points.shape[0]
        
        dist = lorentz.distance_matrix(points, points)
        
        # Check triangle inequality for all triples
        violations = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if dist[i, k] > dist[i, j] + dist[j, k] + 1e-8:
                        violations += 1
        
        assert violations == 0, \
            f"Triangle inequality violations: {violations}/{n**3}"
    
    def test_distance_from_origin(self, lorentz):
        """Test distance from origin matches expected formula."""
        origin = lorentz.origin(1)
        
        # Create point at known distance
        t = 2.0  # Target distance
        
        # Point on manifold at distance t: (cosh(t), sinh(t), 0, ...)
        point = torch.zeros(1, lorentz.config.ambient_dim, dtype=torch.float64)
        point[0, 0] = torch.cosh(torch.tensor(t))
        point[0, 1] = torch.sinh(torch.tensor(t))
        
        computed_dist = lorentz.distance(origin, point)
        
        assert torch.allclose(computed_dist, torch.tensor([t], dtype=torch.float64), atol=1e-8), \
            f"Distance: {computed_dist.item()}, expected: {t}"


# ═══════════════════════════════════════════════════════════════════════════════
#                    EXPONENTIAL/LOGARITHMIC MAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpLogMaps:
    """Tests for exp_map and log_map invertibility."""
    
    def test_exp_log_inverse(self, lorentz):
        """Test that log_map(exp_map(v)) ≈ v."""
        batch_size = 50
        origin = lorentz.origin(batch_size)
        
        # Create tangent vectors at origin
        tangent_space = torch.randn(batch_size, lorentz.config.ambient_dim - 1, dtype=torch.float64)
        tangent_space = tangent_space * 0.5  # Keep small for numerical stability
        
        # Full tangent vector (zero time component at origin)
        tangent = torch.cat([torch.zeros(batch_size, 1, dtype=torch.float64), tangent_space], dim=-1)
        
        # Apply exp then log
        point = lorentz.exp_map(origin, tangent)
        recovered = lorentz.log_map(origin, point)
        
        # Compare (ignoring time component which should be ~0)
        diff = (recovered[:, 1:] - tangent[:, 1:]).abs()
        
        assert diff.max() < 1e-6, \
            f"Exp-Log inverse error: {diff.max().item()}"
    
    def test_log_exp_inverse(self, lorentz, sample_points):
        """Test that exp_map(log_map(y)) ≈ y."""
        origin = lorentz.origin(sample_points.shape[0])
        
        # Compute log then exp
        tangent = lorentz.log_map(origin, sample_points)
        recovered = lorentz.exp_map(origin, tangent)
        
        # Compare points
        diff = (recovered - sample_points).abs()
        
        assert diff.max() < 1e-6, \
            f"Log-Exp inverse error: {diff.max().item()}"
    
    def test_exp_map_at_zero_tangent(self, lorentz):
        """Test that exp_map with zero tangent returns base point."""
        batch_size = 10
        base = lorentz.origin(batch_size)
        
        zero_tangent = torch.zeros(batch_size, lorentz.config.ambient_dim, dtype=torch.float64)
        
        result = lorentz.exp_map(base, zero_tangent)
        
        assert torch.allclose(result, base, atol=1e-10), \
            f"exp_map(x, 0) ≠ x: max diff = {(result - base).abs().max().item()}"


# ═══════════════════════════════════════════════════════════════════════════════
#                    NUMERICAL STABILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumericalStability:
    """Tests for numerical stability under edge cases."""
    
    def test_distance_very_close_points(self, lorentz):
        """Test distance between very close points."""
        origin = lorentz.origin(1)
        
        # Create point very close to origin
        epsilon = 1e-8
        tangent = torch.zeros(1, lorentz.config.ambient_dim, dtype=torch.float64)
        tangent[0, 1] = epsilon
        
        nearby = lorentz.exp_map(origin, tangent)
        
        dist = lorentz.distance(origin, nearby)
        
        # Distance should be approximately epsilon
        assert not torch.isnan(dist), "NaN distance for close points"
        assert not torch.isinf(dist), "Inf distance for close points"
        assert dist >= 0, f"Negative distance: {dist.item()}"
    
    def test_distance_far_points(self, lorentz):
        """Test distance between far points."""
        origin = lorentz.origin(1)
        
        # Create point far from origin
        t = 10.0  # Large distance
        point = torch.zeros(1, lorentz.config.ambient_dim, dtype=torch.float64)
        point[0, 0] = torch.cosh(torch.tensor(t))
        point[0, 1] = torch.sinh(torch.tensor(t))
        
        dist = lorentz.distance(origin, point)
        
        assert not torch.isnan(dist), "NaN distance for far points"
        assert not torch.isinf(dist), "Inf distance for far points"
        assert torch.allclose(dist, torch.tensor([t], dtype=torch.float64), atol=1e-4), \
            f"Distance error: {dist.item()} vs {t}"
    
    def test_gradients_exist(self, lorentz):
        """Test that gradients exist for distance computation."""
        x = lorentz.origin(1).requires_grad_(True)
        
        # Create another point
        y = torch.zeros(1, lorentz.config.ambient_dim, dtype=torch.float64)
        y[0, 0] = 1.5
        y[0, 1] = np.sqrt(1.5**2 - 1)
        y = y.requires_grad_(True)
        
        dist = lorentz.distance(x, y)
        dist.backward()
        
        assert x.grad is not None, "No gradient for x"
        assert y.grad is not None, "No gradient for y"
        assert not torch.isnan(x.grad).any(), "NaN gradient for x"
        assert not torch.isnan(y.grad).any(), "NaN gradient for y"


# ═══════════════════════════════════════════════════════════════════════════════
#                    CURVATURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCurvature:
    """Tests for curvature functionality."""
    
    def test_curvature_scaling(self):
        """Test that curvature scales distances correctly."""
        config1 = LorentzConfig(ambient_dim=33, initial_curvature=1.0)
        config2 = LorentzConfig(ambient_dim=33, initial_curvature=0.5)
        
        lorentz1 = LorentzSubstrate(config1)
        lorentz2 = LorentzSubstrate(config2)
        
        # Create same tangent vector
        origin1 = lorentz1.origin(1)
        origin2 = lorentz2.origin(1)
        
        tangent = torch.zeros(1, 33, dtype=torch.float64)
        tangent[0, 1] = 1.0
        
        point1 = lorentz1.exp_map(origin1, tangent)
        point2 = lorentz2.exp_map(origin2, tangent)
        
        dist1 = lorentz1.distance(origin1, point1)
        dist2 = lorentz2.distance(origin2, point2)
        
        # Distances should be related by curvature ratio
        # d_K = d_1 / sqrt(K)
        ratio = dist2 / dist1
        expected_ratio = np.sqrt(1.0 / 0.5)
        
        assert torch.allclose(ratio, torch.tensor([expected_ratio], dtype=torch.float64), atol=0.1), \
            f"Curvature scaling: {ratio.item()} vs expected {expected_ratio}"


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
