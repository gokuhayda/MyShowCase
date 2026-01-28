# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Part IV.1c - Geometric Capacity Collapse Visualization
======================================================

AUDIT COMPLIANCE:
- Uses only matplotlib/numpy for visualization
- Tree embedding positions are for illustration only
- No geometry derivation - purely visual

PURPOSE:
Create publication-ready figure comparing tree embeddings in:
- Euclidean R² (2D plane) - showing capacity collapse
- Lorentz H² (3D hyperboloid) - showing exponential expansion

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#                    CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Tree parameters
DEFAULT_BRANCHING_FACTOR = 3
DEFAULT_TREE_DEPTH = 3

# Visual parameters
DEFAULT_FIGURE_SIZE = (16, 7)
DEFAULT_DPI = 300

# Colors
COLOR_ROOT = '#1D3557'
COLOR_INTERNAL = '#457B9D'
COLOR_LEAF = '#E63946'
COLOR_EDGE_EUC = '#6c757d'
COLOR_EDGE_HYP = '#2A9D8F'
COLOR_MANIFOLD = '#E8E8E8'


# ═══════════════════════════════════════════════════════════════════════════════
#                    TREE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_tree(branching_factor: int = 3, depth: int = 3) -> nx.DiGraph:
    """
    Create a balanced tree graph.
    
    Args:
        branching_factor: Number of children per node
        depth: Depth of the tree
    
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    node_id = 0
    level_nodes = [node_id]
    G.add_node(node_id, depth=0)
    node_id += 1
    
    for d in range(1, depth + 1):
        next_level = []
        for parent in level_nodes:
            for _ in range(branching_factor):
                G.add_node(node_id, depth=d)
                G.add_edge(parent, node_id)
                next_level.append(node_id)
                node_id += 1
        level_nodes = next_level
    
    return G


# ═══════════════════════════════════════════════════════════════════════════════
#                    EUCLIDEAN LAYOUT (2D)
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean_tree_layout(
    G: nx.DiGraph,
    depth: int
) -> Dict[int, Tuple[float, float]]:
    """
    Layout tree in Euclidean 2D plane.
    
    This demonstrates capacity collapse - nodes at deeper levels
    get crowded together.
    
    Args:
        G: Tree graph
        depth: Tree depth
    
    Returns:
        Dictionary mapping node ID to (x, y) coordinates
    """
    pos = {}
    
    # BFS to assign positions level by level
    root = 0
    pos[root] = (0.0, 0.0)
    
    queue = [(root, 0, -np.pi, np.pi)]  # node, depth, angle_start, angle_end
    
    while queue:
        node, d, angle_start, angle_end = queue.pop(0)
        
        children = list(G.successors(node))
        if not children:
            continue
        
        n_children = len(children)
        radius = (d + 1) * 1.5  # Linear radius growth
        
        angle_step = (angle_end - angle_start) / n_children
        
        for i, child in enumerate(children):
            angle = angle_start + (i + 0.5) * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[child] = (x, y)
            
            # Narrower angle range for children
            child_angle_start = angle - angle_step / 2
            child_angle_end = angle + angle_step / 2
            queue.append((child, d + 1, child_angle_start, child_angle_end))
    
    return pos


# ═══════════════════════════════════════════════════════════════════════════════
#                    HYPERBOLOID LAYOUT (3D)
# ═══════════════════════════════════════════════════════════════════════════════

def hyperboloid_tree_layout(
    G: nx.DiGraph,
    depth: int,
    curvature: float = 1.0
) -> Dict[int, Tuple[float, float, float]]:
    """
    Layout tree on Lorentz hyperboloid in 3D.
    
    This demonstrates exponential volume growth - nodes at deeper
    levels have exponentially more space.
    
    Args:
        G: Tree graph
        depth: Tree depth
        curvature: Hyperbolic curvature K
    
    Returns:
        Dictionary mapping node ID to (x, y, z) coordinates on hyperboloid
    """
    pos = {}
    
    # Root at apex of hyperboloid
    root = 0
    pos[root] = (0.0, 0.0, 1.0 / np.sqrt(curvature))
    
    queue = [(root, 0, 0.0, 2 * np.pi)]  # node, depth, angle_start, angle_end
    
    while queue:
        node, d, angle_start, angle_end = queue.pop(0)
        
        children = list(G.successors(node))
        if not children:
            continue
        
        n_children = len(children)
        
        # Hyperbolic distance from origin increases with depth
        # Using sinh for exponential-like growth
        hyp_dist = (d + 1) * 0.8
        
        # Convert hyperbolic distance to Euclidean radius on hyperboloid
        # sinh(d) for x,y coordinates, cosh(d) for z coordinate
        r_xy = np.sinh(hyp_dist) / np.sqrt(curvature)
        z = np.cosh(hyp_dist) / np.sqrt(curvature)
        
        angle_step = (angle_end - angle_start) / n_children
        
        for i, child in enumerate(children):
            angle = angle_start + (i + 0.5) * angle_step
            x = r_xy * np.cos(angle)
            y = r_xy * np.sin(angle)
            pos[child] = (x, y, z)
            
            # Children get a narrower but still adequate angle range
            child_angle_start = angle - angle_step / 2
            child_angle_end = angle + angle_step / 2
            queue.append((child, d + 1, child_angle_start, child_angle_end))
    
    return pos


# ═══════════════════════════════════════════════════════════════════════════════
#                    VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_geometric_comparison(
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    depth: int = DEFAULT_TREE_DEPTH,
    figsize: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
    dpi: int = DEFAULT_DPI,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create side-by-side comparison of tree embeddings in Euclidean vs Hyperbolic space.
    
    Args:
        branching_factor: Children per node
        depth: Tree depth
        figsize: Figure size
        dpi: Resolution
        output_path: Path to save figure (optional)
    
    Returns:
        Matplotlib figure
    """
    # Create tree
    G = create_tree(branching_factor, depth)
    
    # Get layouts
    euc_pos = euclidean_tree_layout(G, depth)
    hyp_pos = hyperboloid_tree_layout(G, depth)
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LEFT: Euclidean 2D
    # ═══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(121)
    ax1.set_title('Euclidean $\\mathbb{R}^2$\n(Capacity Collapse)', fontsize=14, fontweight='bold')
    
    # Draw edges
    for u, v in G.edges():
        x1, y1 = euc_pos[u]
        x2, y2 = euc_pos[v]
        ax1.plot([x1, x2], [y1, y2], color=COLOR_EDGE_EUC, linewidth=0.8, alpha=0.6)
    
    # Draw nodes
    for node in G.nodes():
        x, y = euc_pos[node]
        d = G.nodes[node]['depth']
        
        if d == 0:
            color = COLOR_ROOT
            size = 150
        elif d == depth:
            color = COLOR_LEAF
            size = 30
        else:
            color = COLOR_INTERNAL
            size = 60
        
        ax1.scatter(x, y, c=color, s=size, zorder=5, edgecolors='white', linewidths=0.5)
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Add annotation about crowding
    ax1.annotate(
        'Leaves crowd\ntogether',
        xy=(3.5, 0), xytext=(5, 2),
        fontsize=10, color='gray',
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7)
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # RIGHT: Lorentz Hyperboloid 3D
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Lorentz Hyperboloid $\\mathbb{H}^2$\n(Exponential Volume)', fontsize=14, fontweight='bold')
    
    # Draw hyperboloid surface (mesh)
    u_mesh = np.linspace(0, 2 * np.pi, 50)
    v_mesh = np.linspace(0, 2.5, 30)
    u_mesh, v_mesh = np.meshgrid(u_mesh, v_mesh)
    
    x_mesh = np.sinh(v_mesh) * np.cos(u_mesh)
    y_mesh = np.sinh(v_mesh) * np.sin(u_mesh)
    z_mesh = np.cosh(v_mesh)
    
    ax2.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.1, color=COLOR_MANIFOLD, 
                     linewidth=0, antialiased=True)
    ax2.plot_wireframe(x_mesh, y_mesh, z_mesh, alpha=0.1, color='gray', 
                       linewidth=0.2, rstride=5, cstride=5)
    
    # Draw edges
    for u, v in G.edges():
        x1, y1, z1 = hyp_pos[u]
        x2, y2, z2 = hyp_pos[v]
        ax2.plot([x1, x2], [y1, y2], [z1, z2], color=COLOR_EDGE_HYP, linewidth=0.8, alpha=0.7)
    
    # Draw nodes
    for node in G.nodes():
        x, y, z = hyp_pos[node]
        d = G.nodes[node]['depth']
        
        if d == 0:
            color = COLOR_ROOT
            size = 100
        elif d == depth:
            color = COLOR_LEAF
            size = 20
        else:
            color = COLOR_INTERNAL
            size = 40
        
        ax2.scatter(x, y, z, c=color, s=size, zorder=5, edgecolors='white', linewidths=0.3)
    
    # View angle
    ax2.view_init(elev=15, azim=45)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlabel('$x_0$ (time)')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_ROOT, label='Root'),
        mpatches.Patch(color=COLOR_INTERNAL, label='Internal'),
        mpatches.Patch(color=COLOR_LEAF, label='Leaves'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 Saved figure to: {output_path}")
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#                    RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_geometric_visualization(
    output_dir: Path,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    depth: int = DEFAULT_TREE_DEPTH,
) -> Dict:
    """
    Run geometric visualization and save results.
    
    Args:
        output_dir: Output directory
        branching_factor: Children per node
        depth: Tree depth
    
    Returns:
        Dictionary with metadata
    """
    print("\n" + "=" * 70)
    print("PART IV.1c - GEOMETRIC CAPACITY VISUALIZATION")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    fig = plot_geometric_comparison(
        branching_factor=branching_factor,
        depth=depth,
        output_path=output_dir / 'geometric_comparison.png',
    )
    
    # Also save as PDF for paper
    fig.savefig(output_dir / 'geometric_comparison.pdf', dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
    # Calculate tree statistics
    G = create_tree(branching_factor, depth)
    n_nodes = G.number_of_nodes()
    n_leaves = sum(1 for n in G.nodes() if G.out_degree(n) == 0)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'branching_factor': branching_factor,
        'depth': depth,
        'n_nodes': n_nodes,
        'n_leaves': n_leaves,
        'outputs': [
            str(output_dir / 'geometric_comparison.png'),
            str(output_dir / 'geometric_comparison.pdf'),
        ]
    }
    
    with open(output_dir / 'geometric_visualization_metadata.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Tree: {n_nodes} nodes, {n_leaves} leaves")
    print(f"📁 Results saved to: {output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PART IV.1c - GEOMETRIC CAPACITY VISUALIZATION")
    print("=" * 80)
    
    output_dir = Path("./results/part_iv_1c_visualization")
    run_geometric_visualization(output_dir)
