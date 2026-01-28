# ==============================================================================
# H-AKORN SIMULATOR - Dual Mode: Poincaré 2D & Lorentz 3D
# ==============================================================================
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
#
# GPU-accelerated Kuramoto oscillator simulation on the Lorentz manifold.
# Supports both Poincaré disk (2D) and Lorentz hyperboloid (3D) visualization.
# ==============================================================================

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Literal

import numpy as np
import torch
import torch.nn.functional as F

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    HAS_DISPLAY = True
except ImportError:
    HAS_DISPLAY = False

try:
    from IPython.display import HTML, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


# ==============================================================================
# DEFAULT DATA
# ==============================================================================

DEFAULT_CONCEPTS = {
    'animals': ['dog', 'cat', 'wolf', 'lion', 'eagle', 'whale'],
    'tech': ['computer', 'algorithm', 'neural', 'quantum', 'robot', 'AI'],
    'nature': ['tree', 'river', 'mountain', 'ocean', 'forest', 'sky'],
    'abstract': ['love', 'freedom', 'justice', 'wisdom', 'truth', 'beauty'],
}

CLUSTER_COLORS = {
    'animals': '#ff6b6b',    # Coral red
    'tech': '#4ecdc4',       # Teal
    'nature': '#95e676',     # Light green
    'abstract': '#dda0dd',   # Plum
    'mammals': '#ff6b6b',
    'birds': '#ffd93d',
    'science': '#4ecdc4',
    'art': '#dda0dd',
}


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class HAKORNSimulatorConfig:
    """Configuration for H-AKORN simulation."""
    # Kuramoto parameters
    K: float = 3.0              # Coupling strength
    dt: float = 0.05            # Time step
    curvature: float = 1.0      # Hyperbolic curvature κ
    frequency_spread: float = 0.2  # Natural frequency variance
    
    # Dynamics
    drift_rate: float = 0.02           # Drift toward semantic targets
    topological_feedback: bool = True  # Enable R-based feedback
    feedback_strength: float = 0.15    # Feedback coefficient
    
    # Graph structure
    k_neighbors: int = 6        # kNN graph connectivity
    
    # Simulation
    total_steps: int = 300
    record_interval: int = 2
    
    # Visualization mode: 'lorentz' (3D) or 'poincare' (2D)
    mode: Literal['lorentz', 'poincare'] = 'lorentz'
    hyperboloid_resolution: int = 40  # Grid resolution for 3D surface
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


# ==============================================================================
# LORENTZ GEOMETRY (GPU-OPTIMIZED)
# ==============================================================================

class LorentzGeometryGPU:
    """
    GPU-accelerated Lorentz (hyperboloid) geometry operations.
    
    Uses the Lorentz model H^n: -x₀² + x₁² + ... + xₙ² = -1, x₀ > 0
    """
    
    def __init__(self, config: HAKORNSimulatorConfig):
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.curvature = config.curvature
    
    @torch.no_grad()
    def lorentz_inner(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Minkowski inner product: <u,v>_L = -u₀v₀ + u₁v₁ + ..."""
        return -u[..., 0] * v[..., 0] + (u[..., 1:] * v[..., 1:]).sum(dim=-1)
    
    @torch.no_grad()
    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance: d(u,v) = arccosh(-<u,v>_L)"""
        inner = self.lorentz_inner(u, v)
        return torch.acosh(torch.clamp(-inner, min=1.0 + 1e-7))
    
    @torch.no_grad()
    def distance_matrix(self, points: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix."""
        inner = -points[:, 0:1] @ points[:, 0:1].T + points[:, 1:] @ points[:, 1:].T
        return torch.acosh(torch.clamp(-inner, min=1.0 + 1e-7))
    
    def project_to_hyperboloid(self, spatial: torch.Tensor) -> torch.Tensor:
        """Project spatial coordinates to hyperboloid: t = sqrt(1 + ||x||²)"""
        t = torch.sqrt(1 + (spatial ** 2).sum(dim=-1, keepdim=True))
        return torch.cat([t, spatial], dim=-1)
    
    def to_poincare(self, h: torch.Tensor) -> torch.Tensor:
        """Convert hyperboloid to Poincaré disk: π(t,x) = x/(1+t)"""
        return h[..., 1:] / (1 + h[..., 0:1])
    
    def to_lorentz_3d(self, h: torch.Tensor) -> torch.Tensor:
        """
        Return 3D coordinates (t, x, y) on the Lorentz hyperboloid.
        
        The hyperboloid H² in ℝ³ is defined by: -t² + x² + y² = -1, t > 0
        """
        return h[..., :3]


# ==============================================================================
# H-AKORN GPU SIMULATOR
# ==============================================================================

class HAKORNSimulator:
    """
    GPU-accelerated H-AKORN simulator with dual visualization modes.
    
    Demonstrates "Synchronization as Clustering":
    Concepts like 'Dog' and 'Wolf' are close not because we placed them there.
    They are close because they RESONATE together. Structure emerges Bottom-Up.
    
    Supports:
    - mode='poincare': 2D Poincaré disk visualization
    - mode='lorentz': 3D Lorentz hyperboloid visualization
    """
    
    def __init__(
        self,
        config: Optional[HAKORNSimulatorConfig] = None,
        concepts: Optional[Dict[str, List[str]]] = None,
        embeddings: Optional[torch.Tensor] = None,
    ):
        self.config = config or HAKORNSimulatorConfig()
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype
        
        self.geometry = LorentzGeometryGPU(self.config)
        
        # Concept data
        self.concepts = concepts or DEFAULT_CONCEPTS
        self.words: List[str] = []
        self.categories: List[str] = []
        self.cluster_ids: List[int] = []
        
        # Build word list
        for cat_id, (cat_name, words) in enumerate(self.concepts.items()):
            for word in words:
                self.words.append(word)
                self.categories.append(cat_name)
                self.cluster_ids.append(cat_id)
        
        self.N = len(self.words)
        
        # Initialize embeddings
        if embeddings is not None:
            self._init_from_embeddings(embeddings)
        else:
            self._init_random_targets()
        
        # Initialize state
        self._init_state()
        
        # Build graph
        self._build_knn_graph()
        
        # History for animation (stores both formats)
        self.history = {
            'positions': [],      # Primary (depends on mode)
            'positions_3d': [],   # 3D Lorentz coordinates
            'positions_2d': [],   # 2D Poincaré coordinates
            'phases': [],
            'R': [],
            'time': [],
        }
    
    def _init_from_embeddings(self, embeddings: torch.Tensor):
        """Initialize target positions from provided embeddings."""
        emb = embeddings.to(self.device).to(self.dtype)
        
        # Project to 2D spatial (for 3D hyperboloid: t, x, y)
        if emb.shape[1] > 2:
            U, S, V = torch.linalg.svd(emb, full_matrices=False)
            emb_2d = U[:, :2] * S[:2]
        else:
            emb_2d = emb[:, :2]
        
        # Normalize and scale
        emb_2d = F.normalize(emb_2d, p=2, dim=-1) * 1.2
        self.target_spatial = emb_2d
        
        # Project to hyperboloid
        self.target_positions = self.geometry.project_to_hyperboloid(emb_2d)
    
    def _init_random_targets(self):
        """Initialize target positions based on cluster structure."""
        n_clusters = len(self.concepts)
        
        # Cluster centers on hyperboloid
        angles = torch.linspace(0, 2 * math.pi, n_clusters + 1)[:-1]
        radius = 1.0
        
        cluster_centers = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles)
        ], dim=-1).to(self.device).to(self.dtype)
        
        # Assign positions with spread
        positions = []
        for i, word in enumerate(self.words):
            cluster_id = self.cluster_ids[i]
            center = cluster_centers[cluster_id]
            offset = (torch.rand(2, device=self.device, dtype=self.dtype) - 0.5) * 0.3
            positions.append(center + offset)
        
        self.target_spatial = torch.stack(positions)
        self.target_positions = self.geometry.project_to_hyperboloid(self.target_spatial)
    
    def _init_state(self):
        """Initialize oscillator state with random chaos."""
        # Random initial positions (chaos state)
        init_spatial = (torch.rand(self.N, 2, device=self.device, dtype=self.dtype) - 0.5) * 2.5
        self.positions = self.geometry.project_to_hyperboloid(init_spatial)
        
        # Random phases
        self.phases = torch.rand(self.N, device=self.device, dtype=self.dtype) * 2 * math.pi
        
        # Natural frequencies (small spread)
        self.omega = torch.randn(self.N, device=self.device, dtype=self.dtype) * self.config.frequency_spread
        
        self.time = 0.0
    
    def _build_knn_graph(self):
        """Build k-nearest neighbors graph on target positions."""
        dist = self.geometry.distance_matrix(self.target_positions)
        _, indices = torch.topk(dist, self.config.k_neighbors + 1, largest=False, dim=1)
        self.neighbors = indices[:, 1:]  # Exclude self
        
        # Adjacency matrix
        self.adjacency = torch.zeros(self.N, self.N, device=self.device, dtype=self.dtype)
        for i in range(self.N):
            self.adjacency[i, self.neighbors[i]] = 1.0
        
        # Symmetrize
        self.adjacency = (self.adjacency + self.adjacency.T).clamp(max=1.0)
    
    def compute_order_parameter(self) -> float:
        """Compute Kuramoto order parameter Γ = |<e^{iφ}>|"""
        z = torch.exp(1j * self.phases.to(torch.complex64))
        return z.mean().abs().item()
    
    def get_phase_state(self) -> str:
        """Get current phase state name."""
        R = self.compute_order_parameter()
        if R < 0.3:
            return "CHAOS"
        elif R < 0.5:
            return "DRIFT"
        elif R < 0.7:
            return "METASTABLE"
        else:
            return "EMERGENCE"
    
    @torch.no_grad()
    def step(self):
        """Perform one simulation step."""
        K = self.config.K
        dt = self.config.dt
        kappa = self.config.curvature
        
        # Hyperbolic distance matrix
        dist = self.geometry.distance_matrix(self.positions)
        
        # Hyperbolic attention: exp(-κ * d)
        coupling = self.adjacency * torch.exp(-dist * kappa)
        
        # Phase differences
        phase_diff = self.phases.unsqueeze(0) - self.phases.unsqueeze(1)
        
        # Kuramoto interaction
        interaction = (coupling * torch.sin(phase_diff)).sum(dim=1)
        
        # Order parameter for feedback
        R = self.compute_order_parameter()
        
        # Phase dynamics: dφ/dt = ω + (K/N) * Σ A_ij * sin(φ_j - φ_i)
        dtheta = self.omega + (K / self.N) * interaction
        
        # Topological feedback modulation
        if self.config.topological_feedback:
            dtheta *= (1 + self.config.feedback_strength * (R - 0.5))
        
        # RK2 integration
        k1 = dtheta
        phases_mid = (self.phases + 0.5 * dt * k1) % (2 * math.pi)
        
        phase_diff_mid = phases_mid.unsqueeze(0) - phases_mid.unsqueeze(1)
        interaction_mid = (coupling * torch.sin(phase_diff_mid)).sum(dim=1)
        k2 = self.omega + (K / self.N) * interaction_mid
        
        if self.config.topological_feedback:
            k2 *= (1 + self.config.feedback_strength * (R - 0.5))
        
        # Update phases
        self.phases = (self.phases + dt * k2) % (2 * math.pi)
        
        # Drift spatial positions toward targets
        alpha = self.config.drift_rate * (0.5 + R * 0.5)
        spatial = self.positions[:, 1:]
        spatial = spatial + alpha * (self.target_spatial - spatial)
        
        # Re-project to hyperboloid
        self.positions = self.geometry.project_to_hyperboloid(spatial)
        
        self.time += dt
    
    def run(self, verbose: bool = True):
        """Run full simulation."""
        mode_name = "3D Lorentz Hyperboloid" if self.config.mode == 'lorentz' else "2D Poincaré Disk"
        
        if verbose:
            print(f"🌀 H-AKORN Simulation")
            print(f"   N={self.N} concepts, K={self.config.K}, κ={self.config.curvature}")
            print(f"   Device: {self.device}")
            print(f"   Visualization: {mode_name}")
            print()
        
        for step in range(self.config.total_steps):
            self.step()
            
            # Record history
            if step % self.config.record_interval == 0:
                # Store both formats
                lorentz_3d = self.geometry.to_lorentz_3d(self.positions)
                poincare = self.geometry.to_poincare(self.positions)
                
                self.history['positions_3d'].append(lorentz_3d.cpu().numpy().copy())
                self.history['positions_2d'].append(poincare.cpu().numpy().copy())
                
                # Primary depends on mode
                if self.config.mode == 'lorentz':
                    self.history['positions'].append(lorentz_3d.cpu().numpy().copy())
                else:
                    self.history['positions'].append(poincare.cpu().numpy().copy())
                
                self.history['phases'].append(self.phases.cpu().numpy().copy())
                self.history['R'].append(self.compute_order_parameter())
                self.history['time'].append(self.time)
            
            if verbose and step % 50 == 0:
                R = self.compute_order_parameter()
                state = self.get_phase_state()
                print(f"   t={self.time:.1f}: Γ={R:.3f} [{state}]")
        
        if verbose:
            final_R = self.compute_order_parameter()
            print(f"\n✓ Simulation complete. Final Γ = {final_R:.4f}")


# ==============================================================================
# 3D LORENTZ VISUALIZATION
# ==============================================================================

def create_hyperboloid_mesh(resolution: int = 40, t_max: float = 3.0):
    """
    Create mesh for Lorentz hyperboloid surface.
    
    Hyperboloid: t² - x² - y² = 1, t > 0
    Parametrization: t = cosh(r), x = sinh(r)*cos(θ), y = sinh(r)*sin(θ)
    """
    r = np.linspace(0, np.arccosh(t_max), resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    R, Theta = np.meshgrid(r, theta)
    
    T = np.cosh(R)
    X = np.sinh(R) * np.cos(Theta)
    Y = np.sinh(R) * np.sin(Theta)
    
    return T, X, Y


def plot_lorentz_3d(
    simulator: HAKORNSimulator,
    figsize: Tuple[int, int] = (12, 10),
    show_surface: bool = True,
    show_labels: bool = True,
    elev: float = 25,
    azim: float = 45,
) -> plt.Figure:
    """Plot final state on 3D Lorentz hyperboloid."""
    if not HAS_DISPLAY:
        print("Display libraries not available")
        return None
    
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#0a0a12')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0a0a12')
    
    # Draw hyperboloid surface
    if show_surface:
        T, X, Y = create_hyperboloid_mesh(resolution=30, t_max=2.5)
        ax.plot_surface(X, Y, T, alpha=0.15, color='cyan', 
                       edgecolor='none', antialiased=True)
        ax.plot_wireframe(X, Y, T, alpha=0.05, color='white', linewidth=0.3)
    
    # Get current positions in 3D Lorentz
    pos_3d = simulator.geometry.to_lorentz_3d(simulator.positions).cpu().numpy()
    phases = simulator.phases.cpu().numpy()
    
    # Category colors
    cat_colors = [CLUSTER_COLORS.get(cat, '#ffffff') for cat in simulator.categories]
    
    # Scatter points
    ax.scatter(pos_3d[:, 1], pos_3d[:, 2], pos_3d[:, 0],
               c=cat_colors, s=150, alpha=0.9,
               edgecolors='white', linewidth=1.5, depthshade=True)
    
    # Draw connections
    for i in range(simulator.N):
        for j in simulator.neighbors[i].cpu().numpy():
            if j > i:
                coh = (1 + np.cos(phases[i] - phases[j])) / 2
                ax.plot([pos_3d[i, 1], pos_3d[j, 1]],
                       [pos_3d[i, 2], pos_3d[j, 2]],
                       [pos_3d[i, 0], pos_3d[j, 0]],
                       color='cyan', alpha=0.1 + 0.4 * coh, linewidth=0.8)
    
    # Labels
    if show_labels:
        for i, word in enumerate(simulator.words):
            ax.text(pos_3d[i, 1], pos_3d[i, 2], pos_3d[i, 0] + 0.1,
                   word, fontsize=8, color='white', alpha=0.8,
                   ha='center', va='bottom')
    
    # Styling
    ax.set_xlabel('X', color='white', fontsize=12)
    ax.set_ylabel('Y', color='white', fontsize=12)
    ax.set_zlabel('t (time-like)', color='white', fontsize=12)
    
    R = simulator.compute_order_parameter()
    state = simulator.get_phase_state()
    ax.set_title(f'H-AKORN: Lorentz Hyperboloid\nΓ = {R:.3f} | {state}',
                color='white', fontsize=14, fontweight='bold')
    
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(colors='white')
    
    # Pane styling
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('white')
        pane.set_alpha(0.1)
    
    plt.tight_layout()
    return fig


# ==============================================================================
# 2D POINCARÉ VISUALIZATION
# ==============================================================================

def plot_poincare_2d(
    simulator: HAKORNSimulator,
    figsize: Tuple[int, int] = (10, 10),
    show_labels: bool = True,
) -> plt.Figure:
    """Plot final state on 2D Poincaré disk."""
    if not HAS_DISPLAY:
        print("Display libraries not available")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0a0a12')
    ax.set_facecolor('#0a0a12')
    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    
    # Draw Poincaré disk boundary
    circle = Circle((0, 0), 1, fill=False, color='white', linewidth=2, alpha=0.3)
    ax.add_patch(circle)
    
    # Inner glow
    for r, alpha in [(0.97, 0.1), (0.94, 0.05)]:
        ring = Circle((0, 0), r, fill=False, color='cyan', linewidth=1, alpha=alpha)
        ax.add_patch(ring)
    
    ax.axis('off')
    
    # Get Poincaré coordinates
    pos_2d = simulator.geometry.to_poincare(simulator.positions).cpu().numpy()
    phases = simulator.phases.cpu().numpy()
    
    # Category colors
    cat_colors = [CLUSTER_COLORS.get(cat, '#ffffff') for cat in simulator.categories]
    
    # Draw connections
    for i in range(simulator.N):
        for j in simulator.neighbors[i].cpu().numpy():
            if j > i:
                coh = (1 + np.cos(phases[i] - phases[j])) / 2
                ax.plot([pos_2d[i, 0], pos_2d[j, 0]],
                       [pos_2d[i, 1], pos_2d[j, 1]],
                       color='cyan', alpha=0.1 + 0.4 * coh, linewidth=0.8)
    
    # Scatter points
    ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
               c=cat_colors, s=150, alpha=0.9,
               edgecolors='white', linewidth=1.5, zorder=10)
    
    # Labels
    if show_labels:
        for i, word in enumerate(simulator.words):
            offset_x = 0.03 if pos_2d[i, 0] < 0 else -0.03
            ha = 'left' if pos_2d[i, 0] < 0 else 'right'
            ax.annotate(word, (pos_2d[i, 0], pos_2d[i, 1]),
                       xytext=(offset_x, 0.02), textcoords='offset fontsize',
                       fontsize=9, color='white', alpha=0.8, ha=ha, va='bottom')
    
    R = simulator.compute_order_parameter()
    state = simulator.get_phase_state()
    ax.set_title(f'H-AKORN: Poincaré Disk\nΓ = {R:.3f} | {state}',
                color='white', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ==============================================================================
# ANIMATION FUNCTIONS
# ==============================================================================

def create_animation_3d(
    simulator: HAKORNSimulator,
    figsize: Tuple[int, int] = (14, 6),
    interval: int = 50,
    show_surface: bool = True,
) -> FuncAnimation:
    """Create 3D animation on Lorentz hyperboloid with metrics panel."""
    if not HAS_DISPLAY:
        return None
    
    if not simulator.history['positions_3d']:
        print("Run simulator first!")
        return None
    
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#0a0a12')
    
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.set_facecolor('#0a0a12')
    
    ax_metrics = fig.add_subplot(122)
    ax_metrics.set_facecolor('#0a0a12')
    
    # Draw hyperboloid surface (static)
    if show_surface:
        T, X, Y = create_hyperboloid_mesh(resolution=25, t_max=2.5)
        ax_3d.plot_surface(X, Y, T, alpha=0.1, color='cyan', edgecolor='none')
    
    cat_colors = [CLUSTER_COLORS.get(cat, '#ffffff') for cat in simulator.categories]
    
    pos_init = simulator.history['positions_3d'][0]
    scatter = ax_3d.scatter(pos_init[:, 1], pos_init[:, 2], pos_init[:, 0],
                            c=cat_colors, s=100, alpha=0.9,
                            edgecolors='white', linewidth=1)
    
    ax_3d.set_xlabel('X', color='white')
    ax_3d.set_ylabel('Y', color='white')
    ax_3d.set_zlabel('t', color='white')
    ax_3d.tick_params(colors='white')
    
    title = ax_3d.set_title('', color='white', fontsize=12, fontweight='bold')
    
    # Metrics setup
    ax_metrics.set_xlim(0, len(simulator.history['R']))
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_xlabel('Time', color='white')
    ax_metrics.set_ylabel('Γ', color='white', fontsize=14)
    ax_metrics.set_title('Order Parameter', color='white', fontsize=12)
    ax_metrics.tick_params(colors='white')
    ax_metrics.axhspan(0, 0.3, alpha=0.1, color='red')
    ax_metrics.axhspan(0.3, 0.7, alpha=0.1, color='cyan')
    ax_metrics.axhspan(0.7, 1.0, alpha=0.1, color='green')
    ax_metrics.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
    ax_metrics.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
    
    line_R, = ax_metrics.plot([], [], color='cyan', linewidth=2)
    connection_lines = []
    
    def update(frame):
        nonlocal connection_lines
        
        pos = simulator.history['positions_3d'][frame]
        phases = simulator.history['phases'][frame]
        R = simulator.history['R'][frame]
        t = simulator.history['time'][frame]
        
        scatter._offsets3d = (pos[:, 1], pos[:, 2], pos[:, 0])
        
        if frame < len(simulator.history['positions_3d']) * 0.7:
            colors = plt.cm.hsv(phases / (2 * np.pi))
            scatter.set_facecolors(colors)
        else:
            scatter.set_facecolors(cat_colors)
        
        for line in connection_lines:
            line.remove()
        connection_lines = []
        
        for i in range(simulator.N):
            for j in simulator.neighbors[i].cpu().numpy():
                if j > i:
                    coh = (1 + np.cos(phases[i] - phases[j])) / 2
                    line, = ax_3d.plot([pos[i, 1], pos[j, 1]],
                                       [pos[i, 2], pos[j, 2]],
                                       [pos[i, 0], pos[j, 0]],
                                       color='cyan', alpha=0.05 + 0.25 * coh,
                                       linewidth=0.5)
                    connection_lines.append(line)
        
        line_R.set_data(range(frame+1), simulator.history['R'][:frame+1])
        
        state = "CHAOS" if R < 0.3 else "DRIFT" if R < 0.5 else "METASTABLE" if R < 0.7 else "EMERGENCE"
        title.set_text(f'Lorentz Hyperboloid | t={t:.1f} | Γ={R:.3f} | {state}')
        
        ax_3d.view_init(elev=25, azim=30 + frame * 0.5)
        
        return scatter, line_R, title
    
    anim = FuncAnimation(fig, update, frames=len(simulator.history['positions_3d']),
                         interval=interval, blit=False)
    plt.tight_layout()
    return anim


def create_animation_2d(
    simulator: HAKORNSimulator,
    figsize: Tuple[int, int] = (14, 6),
    interval: int = 50,
) -> FuncAnimation:
    """Create 2D animation on Poincaré disk with metrics panel."""
    if not HAS_DISPLAY:
        return None
    
    if not simulator.history['positions_2d']:
        print("Run simulator first!")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#0a0a12')
    
    ax_poincare = axes[0]
    ax_metrics = axes[1]
    
    # Poincaré disk setup
    ax_poincare.set_aspect('equal')
    ax_poincare.set_xlim(-1.15, 1.15)
    ax_poincare.set_ylim(-1.15, 1.15)
    ax_poincare.set_facecolor('#0a0a12')
    ax_poincare.set_title('Poincaré Disk', color='white', fontsize=14)
    
    circle = Circle((0, 0), 1, fill=False, color='white', linewidth=2, alpha=0.3)
    ax_poincare.add_patch(circle)
    ax_poincare.axis('off')
    
    cat_colors = [CLUSTER_COLORS.get(cat, '#ffffff') for cat in simulator.categories]
    
    scatter = ax_poincare.scatter([], [], c=[], s=100, alpha=0.9, edgecolors='white', linewidth=1)
    lines = LineCollection([], colors='gray', alpha=0.1, linewidths=0.5)
    ax_poincare.add_collection(lines)
    
    time_text = ax_poincare.text(0, 1.08, '', ha='center', va='bottom', 
                                  fontsize=12, color='white', fontweight='bold')
    gamma_text = ax_poincare.text(0, -1.08, '', ha='center', va='top',
                                   fontsize=14, color='cyan', fontfamily='monospace')
    
    # Metrics setup
    ax_metrics.set_facecolor('#0a0a12')
    ax_metrics.set_xlim(0, len(simulator.history['R']))
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_xlabel('Time', color='white')
    ax_metrics.set_ylabel('Γ', color='white', fontsize=14)
    ax_metrics.set_title('Synchronization Dynamics', color='white', fontsize=14)
    ax_metrics.tick_params(colors='white')
    ax_metrics.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
    ax_metrics.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
    ax_metrics.axhspan(0.3, 0.7, alpha=0.1, color='cyan')
    
    line_R, = ax_metrics.plot([], [], color='cyan', linewidth=2)
    
    def update(frame):
        pos = simulator.history['positions_2d'][frame]
        phases = simulator.history['phases'][frame]
        R = simulator.history['R'][frame]
        t = simulator.history['time'][frame]
        
        scatter.set_offsets(pos[:, :2])
        
        if frame < len(simulator.history['positions_2d']) * 0.7:
            colors = plt.cm.hsv(phases / (2 * np.pi))
        else:
            colors = cat_colors
        scatter.set_facecolors(colors)
        
        segments = []
        for i in range(simulator.N):
            for j in simulator.neighbors[i].cpu().numpy():
                segments.append([pos[i, :2], pos[j, :2]])
        lines.set_segments(segments)
        
        line_R.set_data(range(frame+1), simulator.history['R'][:frame+1])
        
        state = "CHAOS" if R < 0.3 else "DRIFT" if R < 0.5 else "METASTABLE" if R < 0.7 else "EMERGENCE"
        time_text.set_text(f'{state} | t={t:.1f}')
        gamma_text.set_text(f'Γ = {R:.3f}')
        
        return scatter, line_R, time_text, gamma_text, lines
    
    anim = FuncAnimation(fig, update, frames=len(simulator.history['positions_2d']),
                         interval=interval, blit=False)
    plt.tight_layout()
    return anim


def create_animation(
    simulator: HAKORNSimulator,
    mode: Optional[str] = None,
    **kwargs,
) -> FuncAnimation:
    """Create animation based on mode (auto-detect from config if not specified)."""
    mode = mode or simulator.config.mode
    
    if mode == 'lorentz':
        return create_animation_3d(simulator, **kwargs)
    else:
        return create_animation_2d(simulator, **kwargs)


# ==============================================================================
# TRIPTYCH VISUALIZATIONS
# ==============================================================================

def plot_evolution_triptych_3d(
    simulator: HAKORNSimulator,
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Plot 3 stages: Chaos → Drift → Emergence on 3D hyperboloid."""
    if not HAS_DISPLAY or not simulator.history['positions_3d']:
        return None
    
    n_frames = len(simulator.history['positions_3d'])
    frames = [0, n_frames // 2, n_frames - 1]
    titles = ['CHAOS (t=0)', 'DRIFT', 'EMERGENCE']
    
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#0a0a12')
    
    cat_colors = [CLUSTER_COLORS.get(cat, '#ffffff') for cat in simulator.categories]
    
    for idx, (frame, title_text) in enumerate(zip(frames, titles)):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_facecolor('#0a0a12')
        
        T, X, Y = create_hyperboloid_mesh(resolution=20, t_max=2.5)
        ax.plot_surface(X, Y, T, alpha=0.1, color='cyan', edgecolor='none')
        
        pos = simulator.history['positions_3d'][frame]
        phases = simulator.history['phases'][frame]
        R = simulator.history['R'][frame]
        
        if idx == 0:
            colors = plt.cm.hsv(phases / (2 * np.pi))
        else:
            colors = cat_colors
        
        ax.scatter(pos[:, 1], pos[:, 2], pos[:, 0],
                   c=colors, s=80, alpha=0.9,
                   edgecolors='white', linewidth=1)
        
        ax.set_title(f'{title_text}\nΓ = {R:.3f}', color='white', fontsize=11)
        ax.set_xlabel('X', color='white', fontsize=8)
        ax.set_ylabel('Y', color='white', fontsize=8)
        ax.set_zlabel('t', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        ax.view_init(elev=25, azim=45)
        
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('white')
            pane.set_alpha(0.1)
    
    plt.suptitle('H-AKORN: Semantic Resonance on Lorentz Hyperboloid',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_evolution_triptych_2d(
    simulator: HAKORNSimulator,
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Plot 3 stages: Chaos → Drift → Emergence on Poincaré disk."""
    if not HAS_DISPLAY or not simulator.history['positions_2d']:
        return None
    
    n_frames = len(simulator.history['positions_2d'])
    frames = [0, n_frames // 2, n_frames - 1]
    titles = ['CHAOS (t=0)', 'DRIFT', 'EMERGENCE']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('#0a0a12')
    
    cat_colors = [CLUSTER_COLORS.get(cat, '#ffffff') for cat in simulator.categories]
    
    for idx, (frame, title_text) in enumerate(zip(frames, titles)):
        ax = axes[idx]
        ax.set_facecolor('#0a0a12')
        ax.set_aspect('equal')
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        
        circle = Circle((0, 0), 1, fill=False, color='white', linewidth=2, alpha=0.3)
        ax.add_patch(circle)
        ax.axis('off')
        
        pos = simulator.history['positions_2d'][frame]
        phases = simulator.history['phases'][frame]
        R = simulator.history['R'][frame]
        
        if idx == 0:
            colors = plt.cm.hsv(phases / (2 * np.pi))
        else:
            colors = cat_colors
        
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=80, alpha=0.9,
                   edgecolors='white', linewidth=1)
        
        ax.set_title(f'{title_text}\nΓ = {R:.3f}', color='white', fontsize=11)
    
    plt.suptitle('H-AKORN: Semantic Resonance on Poincaré Disk',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_evolution_triptych(
    simulator: HAKORNSimulator,
    mode: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plot triptych based on mode."""
    mode = mode or simulator.config.mode
    
    if mode == 'lorentz':
        return plot_evolution_triptych_3d(simulator, **kwargs)
    else:
        return plot_evolution_triptych_2d(simulator, **kwargs)


def plot_final_state(
    simulator: HAKORNSimulator,
    mode: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plot final state based on mode."""
    mode = mode or simulator.config.mode
    
    if mode == 'lorentz':
        return plot_lorentz_3d(simulator, **kwargs)
    else:
        return plot_poincare_2d(simulator, **kwargs)


# ==============================================================================
# QUICK DEMO FUNCTION
# ==============================================================================

def run_hakorn_demo(
    K: float = 3.0,
    steps: int = 200,
    concepts: Optional[Dict[str, List[str]]] = None,
    animate: bool = True,
    mode: str = 'lorentz',  # 'lorentz' or 'poincare'
    device: str = "auto",
) -> Dict:
    """
    Run H-AKORN demo with selectable visualization mode.
    
    Parameters
    ----------
    K : float
        Coupling strength
    steps : int
        Number of simulation steps
    concepts : dict, optional
        Custom concept dictionary
    animate : bool
        Whether to create animation
    mode : str
        'lorentz' for 3D hyperboloid, 'poincare' for 2D disk
    device : str
        'auto', 'cuda', or 'cpu'
    
    Example
    -------
    ```python
    # 3D Lorentz (default)
    results = run_hakorn_demo(K=3.0, mode='lorentz')
    
    # 2D Poincaré
    results = run_hakorn_demo(K=3.0, mode='poincare')
    ```
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = HAKORNSimulatorConfig(
        K=K,
        total_steps=steps,
        device=device,
        mode=mode,
    )
    
    simulator = HAKORNSimulator(config=config, concepts=concepts)
    simulator.run(verbose=True)
    
    results = {
        'simulator': simulator,
        'final_R': simulator.compute_order_parameter(),
        'final_state': simulator.get_phase_state(),
        'history': simulator.history,
        'mode': mode,
    }
    
    if HAS_DISPLAY:
        mode_name = "3D Lorentz" if mode == 'lorentz' else "2D Poincaré"
        print(f"\n📊 Generating {mode_name} visualizations...")
        
        # Triptych
        fig_triptych = plot_evolution_triptych(simulator, mode=mode)
        if fig_triptych:
            plt.show()
        
        # Final state
        fig_final = plot_final_state(simulator, mode=mode)
        if fig_final:
            plt.show()
        
        # Animation
        if animate:
            print(f"🎬 Creating {mode_name} animation...")
            anim = create_animation(simulator, mode=mode, interval=50)
            results['animation'] = anim
            
            if HAS_IPYTHON:
                display(HTML(anim.to_jshtml()))
    
    return results


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Config & Core
    'HAKORNSimulatorConfig',
    'HAKORNSimulator',
    'LorentzGeometryGPU',
    # 3D Lorentz
    'create_animation_3d',
    'create_hyperboloid_mesh',
    'plot_lorentz_3d',
    'plot_evolution_triptych_3d',
    # 2D Poincaré
    'create_animation_2d',
    'plot_poincare_2d',
    'plot_evolution_triptych_2d',
    # Auto mode
    'create_animation',
    'plot_evolution_triptych',
    'plot_final_state',
    # Demo
    'run_hakorn_demo',
    # Data
    'DEFAULT_CONCEPTS',
    'CLUSTER_COLORS',
]
