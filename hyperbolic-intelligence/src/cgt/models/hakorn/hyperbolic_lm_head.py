"""
hyperbolic_lm_head.py - OPTIMIZED BATCHED VERSION
Save to: /content/cgt_project_with_hakorn/src/cgt/models/hakorn/

Uses efficient batched operations with controlled memory usage
"""

import torch
import torch.nn as nn


class HyperbolicLMHead(nn.Module):
    """Optimized Hyperbolic LM Head with batched distance computation."""
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        substrate,
        tie_weights: bool = False,
        input_embeddings: nn.Module = None,
        chunk_size: int = 100,  # Vocab chunk - SMALL for memory
        seq_chunk_size: int = 64,  # Sequence chunk - NEW PARAMETER
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.substrate = substrate
        self.tie_weights = tie_weights
        self.chunk_size = chunk_size
        self.seq_chunk_size = seq_chunk_size
        
        if tie_weights:
            if input_embeddings is None:
                raise ValueError("tie_weights=True requires input_embeddings")
            self.vocab_embeddings = input_embeddings
        else:
            self.vocab_embeddings = nn.Parameter(torch.randn(vocab_size, d_model) * 0.01)
            with torch.no_grad():
                self.vocab_embeddings.data = self.substrate.proj(self.vocab_embeddings.data)
    
    def get_vocab_embeddings(self) -> torch.Tensor:
        if self.tie_weights:
            return self.vocab_embeddings.weight
        return self.substrate.proj(self.vocab_embeddings)
    
    def _compute_distances_batched(self, h_batch, v_batch):
        """
        Compute pairwise distances between h_batch and v_batch.
        
        Args:
            h_batch: [N, D] hidden states
            v_batch: [M, D] vocab embeddings
        
        Returns:
            distances: [N, M]
        """
        N, D = h_batch.shape
        M = v_batch.shape[0]
        
        # Compute distances using substrate.dist
        # We need to call dist(h_expanded, v_expanded) where both are [N*M, D]
        h_exp = h_batch.unsqueeze(1).expand(N, M, D).reshape(N * M, D)
        v_exp = v_batch.unsqueeze(0).expand(N, M, D).reshape(N * M, D)
        
        distances = self.substrate.dist(h_exp, v_exp)  # [N*M]
        return distances.reshape(N, M)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Chunk both sequence and vocabulary dimensions to control memory.
        
        Memory usage: O(seq_chunk_size * vocab_chunk_size * D)
        """
        B, L, D = hidden_states.shape
        h_flat = hidden_states.reshape(B * L, D)
        h_proj = self.substrate.proj(h_flat)  # [B*L, D]
        vocab_emb = self.get_vocab_embeddings()  # [V, D]
        V = vocab_emb.shape[0]
        
        # Allocate output
        logits = torch.zeros(B * L, V, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Double loop: chunk sequence AND vocabulary
        for s_start in range(0, B * L, self.seq_chunk_size):
            s_end = min(s_start + self.seq_chunk_size, B * L)
            h_chunk = h_proj[s_start:s_end]  # [seq_chunk, D]
            
            for v_start in range(0, V, self.chunk_size):
                v_end = min(v_start + self.chunk_size, V)
                v_chunk = vocab_emb[v_start:v_end]  # [vocab_chunk, D]
                
                # Compute distances for this seq_chunk x vocab_chunk block
                # Max memory: seq_chunk_size * vocab_chunk_size * D
                # Example: 64 * 100 * 512 * 4 bytes = 13 MB
                distances = self._compute_distances_batched(h_chunk, v_chunk)
                logits[s_start:s_end, v_start:v_end] = -distances
        
        return logits.reshape(B, L, V)
