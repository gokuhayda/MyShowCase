"""
GraphConstructor: Embedding → Graph Structure
==============================================

Converts dense embeddings into explicit graph structure (adjacency + node states)
for use with CGTGW dynamics.

This module is the bridge between:
- Teacher embeddings (dense vectors)
- CGTGW (requires explicit relational structure)

Strategies
----------
- knn: k-nearest neighbors graph
- epsilon: ε-ball connectivity
- curvature_aware: adaptive connectivity based on local geometry

Author: Éric Reis
License: MIT
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMethod(Enum):
    """Graph construction methods."""
    KNN = "knn"
    EPSILON = "epsilon"
    CURVATURE_AWARE = "curvature_aware"
    FULL = "full"


@dataclass
class GraphConstructorConfig:
    """Configuration for graph construction."""
    
    method: str = "knn"
    k: int = 8
    epsilon: float = 0.5
    metric: str = "cosine"
    symmetric: bool = True
    self_loops: bool = False
    normalize_weights: bool = True


class GraphConstructor(nn.Module):
    """
    Constructs graph structure from dense embeddings.
    
    This is the relational induction operator that converts
    teacher embeddings into explicit structure for CGTGW.
    
    Parameters
    ----------
    config : GraphConstructorConfig
        Configuration for graph construction.
        
    Example
    -------
    >>> constructor = GraphConstructor(GraphConstructorConfig(method="knn", k=8))
    >>> adjacency, node_states = constructor(teacher_embeddings)
    >>> # adjacency: [N, N] - graph structure
    >>> # node_states: [N, D] - initial node features
    """
    
    def __init__(self, config: Optional[GraphConstructorConfig] = None):
        super().__init__()
        self.config = config or GraphConstructorConfig()
        
    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from embeddings.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings, shape [N, D]
            
        Returns
        -------
        adjacency : torch.Tensor
            Adjacency matrix, shape [N, N]
        node_states : torch.Tensor
            Initial node states (normalized embeddings), shape [N, D]
        """
        # Compute pairwise similarities/distances
        if self.config.metric == "cosine":
            sim_matrix = self._cosine_similarity_matrix(embeddings)
        elif self.config.metric == "euclidean":
            sim_matrix = self._euclidean_similarity_matrix(embeddings)
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}")
        
        # Build adjacency based on method
        method = GraphMethod(self.config.method)
        
        if method == GraphMethod.KNN:
            adjacency = self._knn_adjacency(sim_matrix)
        elif method == GraphMethod.EPSILON:
            adjacency = self._epsilon_adjacency(sim_matrix)
        elif method == GraphMethod.CURVATURE_AWARE:
            adjacency = self._curvature_aware_adjacency(sim_matrix, embeddings)
        elif method == GraphMethod.FULL:
            adjacency = self._full_adjacency(sim_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Post-processing
        if self.config.symmetric:
            adjacency = self._symmetrize(adjacency)
            
        if not self.config.self_loops:
            adjacency = adjacency * (1 - torch.eye(
                adjacency.shape[0], 
                device=adjacency.device,
                dtype=adjacency.dtype
            ))
            
        if self.config.normalize_weights:
            adjacency = self._normalize_adjacency(adjacency)
        
        # Node states are normalized input embeddings
        node_states = F.normalize(embeddings, p=2, dim=-1)
        
        return adjacency, node_states
    
    def _cosine_similarity_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity matrix."""
        x_norm = F.normalize(x, p=2, dim=-1)
        return torch.mm(x_norm, x_norm.t())
    
    def _euclidean_similarity_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute similarity from Euclidean distance (inverse)."""
        # Pairwise squared distances
        sq_norms = (x ** 2).sum(dim=-1, keepdim=True)
        distances_sq = sq_norms + sq_norms.t() - 2 * torch.mm(x, x.t())
        distances_sq = torch.clamp(distances_sq, min=0.0)
        distances = torch.sqrt(distances_sq + 1e-8)
        
        # Convert to similarity (inverse with softmax-like scaling)
        max_dist = distances.max()
        if max_dist > 1e-8:
            similarities = 1.0 - distances / max_dist
        else:
            similarities = torch.ones_like(distances)
            
        return similarities
    
    def _knn_adjacency(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """Build k-nearest neighbors adjacency."""
        N = sim_matrix.shape[0]
        k = min(self.config.k, N - 1)
        
        # Get top-k indices (excluding self)
        sim_no_self = sim_matrix.clone()
        sim_no_self.fill_diagonal_(-float('inf'))
        
        _, topk_indices = torch.topk(sim_no_self, k, dim=-1)
        
        # Build adjacency
        # ═══════════════════════════════════════════════════════════════════
        # DTYPE INHERITANCE: All tensors must inherit dtype from input
        # ═══════════════════════════════════════════════════════════════════
        dtype = sim_matrix.dtype
        device = sim_matrix.device
        
        adjacency = torch.zeros((N, N), device=device, dtype=dtype)
        row_indices = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)
        
        # Cast 1.0 to match dtype
        adjacency[row_indices.flatten(), topk_indices.flatten()] = torch.tensor(1.0, device=device, dtype=dtype)
        
        return adjacency
    
    def _epsilon_adjacency(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """Build ε-ball adjacency (connect if similarity > ε)."""
        # ═══════════════════════════════════════════════════════════════════
        # DTYPE INHERITANCE: Cast boolean result to input dtype
        # ═══════════════════════════════════════════════════════════════════
        return (sim_matrix > self.config.epsilon).to(dtype=sim_matrix.dtype)
    
    def _curvature_aware_adjacency(
        self, 
        sim_matrix: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build adjacency with curvature-aware connectivity.
        
        Nodes in high-curvature regions get more connections,
        nodes in flat regions get fewer.
        """
        N = sim_matrix.shape[0]
        
        # Estimate local curvature via neighbor variance
        k_estimate = min(self.config.k * 2, N - 1)
        
        sim_no_self = sim_matrix.clone()
        sim_no_self.fill_diagonal_(-float('inf'))
        _, neighbor_indices = torch.topk(sim_no_self, k_estimate, dim=-1)
        
        # Compute local variance as curvature proxy
        local_curvatures = []
        for i in range(N):
            neighbors = embeddings[neighbor_indices[i]]
            centroid = neighbors.mean(dim=0)
            variance = ((neighbors - centroid) ** 2).sum(dim=-1).mean()
            local_curvatures.append(variance)
        
        curvatures = torch.stack(local_curvatures)
        
        # Normalize curvatures to [0.5, 1.5] range for adaptive k
        curvatures_norm = curvatures / (curvatures.max() + 1e-8)
        adaptive_k = (0.5 + curvatures_norm) * self.config.k
        adaptive_k = adaptive_k.long().clamp(min=2, max=N-1)
        
        # Build adaptive adjacency
        # ═══════════════════════════════════════════════════════════════════
        # DTYPE INHERITANCE: All tensors must inherit dtype from input
        # ═══════════════════════════════════════════════════════════════════
        dtype = sim_matrix.dtype
        device = sim_matrix.device
        
        adjacency = torch.zeros((N, N), device=device, dtype=dtype)
        for i in range(N):
            k_i = adaptive_k[i].item()
            _, topk_i = torch.topk(sim_no_self[i], k_i)
            # Cast 1.0 to match dtype
            adjacency[i, topk_i] = torch.tensor(1.0, device=device, dtype=dtype)
        
        return adjacency
    
    def _full_adjacency(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """Fully connected graph with similarity weights."""
        return sim_matrix
    
    def _symmetrize(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Make adjacency symmetric."""
        return torch.max(adjacency, adjacency.t())
    
    def _normalize_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Row-normalize adjacency matrix."""
        row_sum = adjacency.sum(dim=-1, keepdim=True)
        row_sum = torch.clamp(row_sum, min=1e-8)
        return adjacency / row_sum


__all__ = [
    "GraphConstructor",
    "GraphConstructorConfig",
    "GraphMethod",
]
