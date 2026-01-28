# SPDX-License-Identifier: MIT
# Origin: PSI_SLM utils/metrics.py - Extracted visualization functions

"""
Visualization Utilities
=======================

Plotting functions for hyperbolic embeddings and training metrics.

Note: compute_distortion was NOT included due to conflict with
cgt.evaluation.metrics.compute_distortion
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_poincare_embedding(
    h_states: torch.Tensor,
    substrate,  # Any substrate with to_poincare method
    labels: Optional[torch.Tensor] = None,
    title: str = "Poincaré Disk Projection",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
):
    """
    Visualize hyperbolic embedding in Poincaré disk.
    
    Parameters
    ----------
    h_states : torch.Tensor
        Hyperbolic states in Lorentz model.
    substrate : object
        Geometric substrate with to_poincare() method.
    labels : torch.Tensor, optional
        Node labels for coloring.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure or None
        The created figure (None if matplotlib not available).
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return None
    
    # Project to Poincaré disk
    poincare_coords = substrate.to_poincare(h_states).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw unit disk boundary
    circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # Plot points (use first 2 dimensions for 2D visualization)
    if poincare_coords.shape[1] >= 2:
        x = poincare_coords[:, 0]
        y = poincare_coords[:, 1]
    else:
        x = poincare_coords[:, 0]
        y = np.zeros_like(x)
    
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        scatter = ax.scatter(x, y, c=labels_np, cmap='tab10', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(x, y, s=50, alpha=0.7, color='blue')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_phase_coherence_matrix(
    phases: torch.Tensor,
    title: str = "Phase Coherence Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """
    Visualize phase coherence as a heatmap.
    
    Parameters
    ----------
    phases : torch.Tensor
        Phase values, shape (N,).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure or None
        The created figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return None
    
    phases_np = phases.detach().cpu().numpy()
    
    # Compute coherence matrix: 1 + cos(θ_i - θ_j)
    phase_diff = phases_np[:, None] - phases_np[None, :]
    coherence = 1 + np.cos(phase_diff)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(coherence, cmap='viridis', vmin=0, vmax=2)
    plt.colorbar(im, ax=ax, label='Coherence (1 + cos(Δθ))')
    ax.set_title(title)
    ax.set_xlabel('Node j')
    ax.set_ylabel('Node i')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_divergence_test_results(
    epochs: List[int],
    gw_distances: List[float],
    title: str = "Divergence Test: GW Distance Over Time",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot GW divergence trajectory for the Divergence Test.
    
    Parameters
    ----------
    epochs : list of int
        Epoch numbers.
    gw_distances : list of float
        GW distances at each measurement.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure or None
        The created figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.semilogy(epochs, gw_distances, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('D_GW (log scale)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Mark potential phase transition
    if len(gw_distances) > 2:
        diffs = np.diff(gw_distances)
        if np.any(diffs > 0):
            max_diff_idx = np.argmax(diffs)
            ax.axvline(x=epochs[max_diff_idx + 1], color='r', linestyle='--',
                      label='Max divergence rate')
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Plot training loss curves.
    
    Parameters
    ----------
    history : dict
        Dictionary of loss names to lists of values.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure or None
        The created figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return None
    
    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.set_title(name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


class MetricsLogger:
    """
    Simple metrics logger for training.
    
    Attributes
    ----------
    history : dict
        Dictionary storing metric history.
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
    
    def log(self, metrics: Dict[str, float]):
        """Log a dictionary of metrics."""
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if name in self.history and self.history[name]:
            return self.history[name][-1]
        return None
    
    def get_average(self, name: str, window: int = 10) -> Optional[float]:
        """Get moving average for a metric."""
        if name in self.history and self.history[name]:
            values = self.history[name][-window:]
            return sum(values) / len(values)
        return None
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, values in self.history.items():
            if values:
                summary[name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'last': values[-1],
                }
        return summary
    
    def plot(self, save_path: Optional[str] = None):
        """Plot all metrics using plot_training_curves."""
        return plot_training_curves(self.history, save_path=save_path)


def plot_lorentz_embedding(
    h_states: torch.Tensor,
    substrate,
    labels: Optional[torch.Tensor] = None,
    title: str = "Lorentz Hyperboloid Embedding",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    elev: float = 20,
    azim: float = 45,
):
    """
    Visualize hyperbolic embedding on Lorentz hyperboloid (3D).
    
    Shows the first 2 spatial dimensions + time component as 3D surface.
    
    Parameters
    ----------
    h_states : torch.Tensor
        Hyperbolic states in Lorentz model, shape (N, ambient_dim).
        Uses x0 (time), x1, x2 for visualization.
    substrate : object
        Geometric substrate (for curvature info).
    labels : torch.Tensor, optional
        Node labels for coloring.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    elev : float
        Elevation angle for 3D view.
    azim : float
        Azimuth angle for 3D view.
    
    Returns
    -------
    matplotlib.Figure or None
        The created figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return None
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract coordinates
    coords = h_states.detach().cpu().numpy()
    x0 = coords[:, 0]  # Time component
    x1 = coords[:, 1]  # First spatial
    x2 = coords[:, 2] if coords.shape[1] > 2 else np.zeros_like(x1)  # Second spatial
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw hyperboloid surface (reference)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2, 30)
    U, V = np.meshgrid(u, v)
    
    # Hyperboloid: x0² - x1² - x2² = 1/K (upper sheet)
    K = substrate.K.item() if hasattr(substrate.K, 'item') else substrate.K
    R = 1.0 / np.sqrt(K)
    
    X_surf = R * np.sinh(V) * np.cos(U)
    Y_surf = R * np.sinh(V) * np.sin(U)
    Z_surf = R * np.cosh(V)
    
    # Plot surface with transparency
    ax.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.1, color='blue', linewidth=0)
    
    # Plot points
    if labels is not None:
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        scatter = ax.scatter(x1, x2, x0, c=labels_np, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label='Depth', shrink=0.6)
    else:
        ax.scatter(x1, x2, x0, c='red', s=50, alpha=0.8)
    
    ax.set_xlabel('$x_1$ (spatial)')
    ax.set_ylabel('$x_2$ (spatial)')
    ax.set_zlabel('$x_0$ (time)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Lorentz plot saved to: {save_path}")
    
    return fig


def plot_lorentz_2d_projection(
    h_states: torch.Tensor,
    substrate,
    labels: Optional[torch.Tensor] = None,
    title: str = "Lorentz Embedding (2D Projection)",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
):
    """
    Visualize Lorentz embedding as 2D projections.
    
    Creates two subplots:
    1. Spatial projection (x1 vs x2)
    2. Time-spatial projection (x1 vs x0)
    
    Parameters
    ----------
    h_states : torch.Tensor
        Hyperbolic states in Lorentz model.
    substrate : object
        Geometric substrate.
    labels : torch.Tensor, optional
        Node labels for coloring.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure or None
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return None
    
    coords = h_states.detach().cpu().numpy()
    x0 = coords[:, 0]  # Time
    x1 = coords[:, 1]  # Spatial 1
    x2 = coords[:, 2] if coords.shape[1] > 2 else np.zeros_like(x1)
    
    if labels is not None:
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    else:
        labels_np = None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Spatial projection (x1 vs x2)
    ax1 = axes[0]
    if labels_np is not None:
        sc1 = ax1.scatter(x1, x2, c=labels_np, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(sc1, ax=ax1, label='Depth')
    else:
        ax1.scatter(x1, x2, c='blue', s=50, alpha=0.8)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('Spatial Projection')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time-spatial (x1 vs x0)
    ax2 = axes[1]
    if labels_np is not None:
        sc2 = ax2.scatter(x1, x0, c=labels_np, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(sc2, ax=ax2, label='Depth')
    else:
        ax2.scatter(x1, x0, c='blue', s=50, alpha=0.8)
    ax2.set_xlabel('$x_1$ (spatial)')
    ax2.set_ylabel('$x_0$ (time)')
    ax2.set_title('Time-Spatial Projection')
    ax2.grid(True, alpha=0.3)
    
    # Add hyperboloid constraint line: x0 = sqrt(1/K + x1²)
    K = substrate.K.item() if hasattr(substrate.K, 'item') else substrate.K
    x1_line = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 100)
    x0_line = np.sqrt(1.0/K + x1_line**2)
    ax2.plot(x1_line, x0_line, 'r--', alpha=0.5, label='Hyperboloid')
    ax2.legend()
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Lorentz 2D plot saved to: {save_path}")
    
    return fig


__all__ = [
    "plot_poincare_embedding",
    "plot_lorentz_embedding",
    "plot_lorentz_2d_projection",
    "plot_phase_coherence_matrix",
    "plot_divergence_test_results",
    "plot_training_curves",
    "MetricsLogger",
    "HAS_MATPLOTLIB",
]
