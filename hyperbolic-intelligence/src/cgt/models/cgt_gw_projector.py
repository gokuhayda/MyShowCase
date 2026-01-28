"""
CGTGWProjector: Embedding Interface for Cartesian Pipeline
==========================================================

Wraps CGTGW (dynamic system) for use in embedding distillation pipelines.

This is NOT the CGTGW - it ORCHESTRATES the CGTGW.

Pipeline:
    Teacher embeddings
            ↓
    GraphConstructor (embeddings → adjacency)
            ↓
    CGTGW (dynamics on graph)
            ↓
    TemporalAggregator (trajectory → embedding)
            ↓
    Student embedding

Responsibilities:
- Adapt batch of embeddings to graph structure
- Invoke CGTGW as dynamic engine
- Export embeddings compatible with Cartesian pipeline

Author: Éric Reis
License: MIT
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.geometry.lorentz_hardened import LorentzSubstrateHardened, LorentzConfig
from cgt.models.cgt_gw import CGTGW, CGTGWConfig
from cgt.models.graph_constructor import GraphConstructor, GraphConstructorConfig


class AggregationMethod(Enum):
    """Temporal aggregation methods."""
    FINAL = "final"           # Last state only
    MEAN = "mean"             # Mean over trajectory
    ATTENTION = "attention"   # Learned attention over trajectory
    STABILITY = "stability"   # Stability-aware weighting


@dataclass
class CGTGWProjectorConfig:
    """Configuration for CGTGWProjector."""
    
    # Input/output dimensions
    input_dim: int = 384
    output_dim: int = 256
    
    # CGTGW config
    gw_embed_dim: int = 16
    gw_hidden_dim: int = 64
    gw_coupling_strength: float = 1.0
    gw_temperature: float = 0.5
    gw_curvature: float = 1.0
    gw_num_steps: int = 5
    
    # Graph construction
    graph_method: str = "knn"
    graph_k: int = 8
    graph_metric: str = "cosine"
    
    # Aggregation
    aggregation: str = "final"
    
    # Training
    lambda_topo: float = 0.1
    lambda_gw: float = 0.05
    lambda_coherence: float = 0.01


class TemporalAggregator(nn.Module):
    """
    Aggregates trajectory of states into single embedding.
    
    This is where dynamics → embedding happens.
    """
    
    def __init__(
        self,
        method: str = "final",
        embed_dim: int = 16,
        output_dim: int = 256,
    ):
        super().__init__()
        self.method = AggregationMethod(method)
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Projection from hyperbolic to output space
        # Input is embed_dim + 1 (Lorentz coordinates)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim + 1, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        
        if self.method == AggregationMethod.ATTENTION:
            self.attention = nn.Sequential(
                nn.Linear(embed_dim + 1, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
    
    def forward(
        self,
        trajectory: torch.Tensor,
        final_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate trajectory into embeddings.
        
        Parameters
        ----------
        trajectory : torch.Tensor
            Full trajectory [T, N, D+1] or None
        final_states : torch.Tensor
            Final hyperbolic states [N, D+1]
            
        Returns
        -------
        torch.Tensor
            Output embeddings [N, output_dim]
        """
        if self.method == AggregationMethod.FINAL:
            # Use only final states
            aggregated = final_states
            
        elif self.method == AggregationMethod.MEAN:
            # Mean over trajectory
            if trajectory is not None:
                aggregated = trajectory.mean(dim=0)
            else:
                aggregated = final_states
                
        elif self.method == AggregationMethod.ATTENTION:
            # Learned attention over trajectory
            if trajectory is not None:
                # trajectory: [T, N, D+1]
                attn_scores = self.attention(trajectory)  # [T, N, 1]
                attn_weights = F.softmax(attn_scores, dim=0)
                aggregated = (attn_weights * trajectory).sum(dim=0)  # [N, D+1]
            else:
                aggregated = final_states
                
        elif self.method == AggregationMethod.STABILITY:
            # Weight by stability (inverse variance)
            if trajectory is not None:
                # Compute per-node variance over time
                var = trajectory.var(dim=0, keepdim=True)  # [1, N, D+1]
                weights = 1.0 / (var.mean(dim=-1, keepdim=True) + 1e-8)  # [1, N, 1]
                weights = F.softmax(weights, dim=0)
                aggregated = (weights * trajectory).sum(dim=0)
            else:
                aggregated = final_states
        else:
            aggregated = final_states
        
        # Project to output dimension
        return self.projection(aggregated)


class CGTGWProjector(nn.Module):
    """
    CGT-GW Projector: Embedding interface for Cartesian pipeline.
    
    Wraps CGTGW (dynamic system) to accept teacher embeddings
    and produce student-compatible embeddings.
    
    This class does NOT contain dynamics - it ORCHESTRATES them.
    
    Architecture
    ------------
    1. GraphConstructor: embeddings → adjacency + node_states
    2. CGTGW: adjacency → evolved hyperbolic states
    3. TemporalAggregator: trajectory → output embeddings
    
    Parameters
    ----------
    config : CGTGWProjectorConfig
        Full configuration.
        
    Example
    -------
    >>> projector = CGTGWProjector(config)
    >>> z = projector(teacher_embeddings)  # [N, output_dim]
    >>> loss = contrastive_loss(z_i, z_j, scores)
    """
    
    def __init__(self, config: Optional[CGTGWProjectorConfig] = None):
        super().__init__()
        self.config = config or CGTGWProjectorConfig()
        
        # Input projection: teacher_dim → node_dim
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.gw_hidden_dim),
            nn.LayerNorm(self.config.gw_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.gw_hidden_dim, self.config.gw_embed_dim),
        )
        
        # Graph constructor
        self.graph_constructor = GraphConstructor(
            GraphConstructorConfig(
                method=self.config.graph_method,
                k=self.config.graph_k,
                metric=self.config.graph_metric,
                symmetric=True,
                self_loops=False,
                normalize_weights=True,
            )
        )
        
        # Lorentz substrate (shared reference, not owned by projector)
        self.substrate = LorentzSubstrateHardened(
            LorentzConfig(
                intrinsic_dim=self.config.gw_embed_dim,
                initial_curvature=self.config.gw_curvature,
                learnable_curvature=False,
            )
        )
        
        # Note: CGTGW requires fixed num_nodes, so we create it dynamically
        # or use a pooled version. For now, we'll handle variable batch sizes
        # by creating the model per-forward (not ideal but correct).
        # 
        # Alternative: Use a maximum batch size and pad.
        self._cgtgw_cache: Dict[int, CGTGW] = {}
        
        # Temporal aggregator
        self.aggregator = TemporalAggregator(
            method=self.config.aggregation,
            embed_dim=self.config.gw_embed_dim,
            output_dim=self.config.output_dim,
        )
        
        # Output projection (optional refinement)
        self.output_projection = nn.Linear(
            self.config.output_dim, 
            self.config.output_dim
        )
        
        self._init_weights()
        
        # Precision tracking for fallback diagnostics
        self._last_precision_used: Optional[str] = None
        
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Promote ALL module parameters to float64
        # CGT-GW operates in float64 for hyperbolic numerical stability.
        # nn.Linear weights default to float32, causing dtype mismatch.
        # This is NOT optional - it's mathematical consistency.
        # ═══════════════════════════════════════════════════════════════════
        self.double()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _get_or_create_cgtgw(self, num_nodes: int, device: torch.device) -> CGTGW:
        """Get or create CGTGW model for given batch size."""
        if num_nodes not in self._cgtgw_cache:
            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL: CGTGW must be float64 for hyperbolic geometry
            # The .double() call is NON-NEGOTIABLE - without it, internal
            # matmuls will fail with dtype mismatch (float32 weights × float64 input)
            # ═══════════════════════════════════════════════════════════════════
            cgtgw = CGTGW(
                num_nodes=num_nodes,
                embed_dim=self.config.gw_embed_dim,
                hidden_dim=self.config.gw_hidden_dim,
                coupling_strength=self.config.gw_coupling_strength,
                temperature=self.config.gw_temperature,
                curvature=self.config.gw_curvature,
            ).to(device).double()  # ← CRITICAL: Must be float64
            self._cgtgw_cache[num_nodes] = cgtgw
        return self._cgtgw_cache[num_nodes]
    
    def _forward_impl(
        self,
        teacher_embeddings: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Internal forward implementation.
        
        Parameters
        ----------
        teacher_embeddings : torch.Tensor
            Input teacher embeddings, shape [N, input_dim]
        return_trajectory : bool
            If True, also return full trajectory.
            
        Returns
        -------
        torch.Tensor
            Output embeddings, shape [N, output_dim]
            
        Notes
        -----
        CGT-GW operates in float64 for hyperbolic numerical stability.
        Module parameters are promoted to float64 in __init__ via self.double().
        """
        # ═══════════════════════════════════════════════════════════════════
        # HARD GUARDRAIL: Reject non-float64 input
        # CGTGWProjector is a geometric boundary. It must FAIL on invalid input.
        # DO NOT silently cast - this masks upstream boundary violations.
        # ═══════════════════════════════════════════════════════════════════
        if teacher_embeddings.dtype != torch.float64:
            import traceback
            traceback.print_stack(limit=15)
            raise RuntimeError(
                f"CGTGWProjector received {teacher_embeddings.dtype} input. "
                f"Expected torch.float64. This is a boundary violation. "
                f"Cast your input to .double() BEFORE calling CGTGWProjector."
            )
        
        N = teacher_embeddings.shape[0]
        device = teacher_embeddings.device
        
        # 1. Project input to node dimension
        node_features = self.input_projection(teacher_embeddings)  # [N, gw_embed_dim]
        
        # 2. Build graph structure
        adjacency, _ = self.graph_constructor(teacher_embeddings)
        adjacency = adjacency.to(device=device, dtype=torch.float64)
        
        # 3. Get CGTGW model for this batch size
        cgtgw = self._get_or_create_cgtgw(N, device)
        
        # Ensure CGTGW is on correct device/dtype
        cgtgw = cgtgw.to(device)
        cgtgw = cgtgw.double()
        
        # 4. Run dynamics
        outputs = cgtgw(
            adjacency=adjacency,
            num_steps=self.config.gw_num_steps,
        )
        
        final_states = outputs['states']  # [N, gw_embed_dim + 1]
        
        # 5. Aggregate trajectory → embedding
        z = self.aggregator(
            trajectory=None,
            final_states=final_states,
        )
        
        # 6. Final projection
        z = self.output_projection(z)
        
        # ═══════════════════════════════════════════════════════════════════
        # FINAL DTYPE VALIDATION: Guarantees geometric consistency
        # ═══════════════════════════════════════════════════════════════════
        assert z.dtype == torch.float64, \
            f"CGTGWProjector output dtype mismatch: {z.dtype} vs torch.float64"
        
        return z
    
    def forward(
        self,
        teacher_embeddings: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Project teacher embeddings through CGTGW dynamics.
        
        Attempts float64 execution first. Falls back to float32 on dtype mismatch.
        
        Parameters
        ----------
        teacher_embeddings : torch.Tensor
            Input teacher embeddings, shape [N, input_dim]
        return_trajectory : bool
            If True, also return full trajectory.
            
        Returns
        -------
        torch.Tensor
            Output embeddings, shape [N, output_dim]
        """
        try:
            # Attempt high-precision path (float64)
            out = self._forward_impl(teacher_embeddings, return_trajectory)
            self._last_precision_used = "float64"
            return out

        except RuntimeError as e:
            if "mat1 and mat2 must have the same dtype" not in str(e):
                raise e

            # === SAFE FALLBACK ===
            print(
                "[CGT-GW WARNING] Float64 execution failed due to dtype mismatch. "
                "Falling back to Float32 for this batch."
            )

            x32 = teacher_embeddings.float()
            out = self._forward_impl(x32, return_trajectory)
            self._last_precision_used = "float32"
            return out
    
    def compute_losses(
        self,
        teacher_embeddings: torch.Tensor,
        target_distances: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full CGT-GW losses.
        
        Parameters
        ----------
        teacher_embeddings : torch.Tensor
            Input teacher embeddings [N, input_dim]
        target_distances : torch.Tensor
            Target pairwise distances [N, N]
            
        Returns
        -------
        dict
            Loss components including 'total', 'task', 'topo', 'gw', 'coherence'
        """
        N = teacher_embeddings.shape[0]
        device = teacher_embeddings.device
        
        # Ensure inputs match module dtype (float64)
        teacher_embeddings = teacher_embeddings.to(dtype=torch.float64)
        target_distances = target_distances.to(dtype=torch.float64)
        
        # Build graph
        adjacency, _ = self.graph_constructor(teacher_embeddings)
        adjacency = adjacency.to(device=device, dtype=torch.float64)
        
        # Get CGTGW
        cgtgw = self._get_or_create_cgtgw(N, device)
        cgtgw = cgtgw.to(device)
        cgtgw = cgtgw.double()
        
        # Forward pass
        outputs = cgtgw(adjacency=adjacency, num_steps=self.config.gw_num_steps)
        
        # Compute losses using CGTGW's method
        losses = cgtgw.compute_losses(
            outputs=outputs,
            target_distances=target_distances,
            source_embeddings=teacher_embeddings,
            lambda_topo=self.config.lambda_topo,
            lambda_gw=self.config.lambda_gw,
            lambda_coherence=self.config.lambda_coherence,
        )
        
        return losses
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between projected embeddings.
        
        Uses Euclidean distance in output space.
        """
        return torch.norm(x - y, dim=-1)


__all__ = [
    "CGTGWProjector",
    "CGTGWProjectorConfig",
    "TemporalAggregator",
    "AggregationMethod",
]
