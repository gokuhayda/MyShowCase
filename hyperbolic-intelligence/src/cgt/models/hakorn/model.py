# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
H-AKORN Transformer Model
=========================

Complete hyperbolic transformer with Kuramoto oscillator dynamics.

Architecture:
1. Token + Position embeddings (with optional hyperbolic embeddings)
2. H-AKORN Encoder stack
3. Language modeling head (or task-specific head)
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import HAKORNEncoder
from .hyperbolic_lm_head import HyperbolicLMHead


class HAKORNEmbedding(nn.Module):
    """
    Embedding layer for H-AKORN.
    
    Combines token embeddings and position embeddings.
    Optional: Project to hyperbolic space.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        max_position_embeddings: Maximum sequence length
        dropout: Dropout probability
        pad_token_id: Padding token ID
        use_hyperbolic_embedding: Whether to project to hyperbolic space
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        use_hyperbolic_embedding: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.use_hyperbolic_embedding = use_hyperbolic_embedding
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            d_model,
        )
        
        # Register position IDs buffer
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1))
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] token IDs
            position_ids: [B, L] position IDs (optional)
        
        Returns:
            embeddings: [B, L, D] embedded representations
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Get position embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, :L].to(device)
        
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Optional: Project to hyperbolic space (Poincaré ball)
        if self.use_hyperbolic_embedding:
            embeddings = self._project_to_hyperbolic(embeddings)
        
        return embeddings
    
    def _project_to_hyperbolic(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Project embeddings to Poincaré ball using exponential map.
        
        exp_0(x) = tanh(||x||) * x / ||x||
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=eps)
        return torch.tanh(norm) * (x / norm)


class HAKORNTransformer(nn.Module):
    """
    Complete H-AKORN Transformer for language modeling.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_position_embeddings: Maximum sequence length
        dropout: Dropout probability
        layer_norm_eps: Layer norm epsilon
        pad_token_id: Padding token ID
        substrate: Hyperbolic substrate (optional)
        curvature: Hyperbolic curvature
        coupling_strength: Kuramoto coupling strength
        use_phase_modulation: Whether to use phase modulation
        use_hyperbolic_embedding: Whether to use hyperbolic embeddings
        tie_word_embeddings: Whether to tie input/output embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        substrate = None,
        curvature: float = -1.0,
        coupling_strength: float = 1.0,
        use_phase_modulation: bool = True,
        use_hyperbolic_embedding: bool = False,
        tie_word_embeddings: bool = True,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.substrate = substrate  # Store substrate reference for distillation
        
        # Embeddings
        self.embeddings = HAKORNEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            pad_token_id=pad_token_id,
            use_hyperbolic_embedding=use_hyperbolic_embedding,
        )
        
        # Encoder
        self.encoder = HAKORNEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            substrate=substrate,
            curvature=curvature,
            coupling_strength=coupling_strength,
            use_phase_modulation=use_phase_modulation,
            activation=activation,
        )
        
        # Language modeling head - HYPERBOLIC REQUIRED
        if substrate is None:
            raise ValueError(
                "H-AKORN requires substrate for HyperbolicLMHead. "
                "Provide a LorentzSubstrate in __init__."
            )
        
        self.lm_head = HyperbolicLMHead(
            d_model=d_model,
            vocab_size=vocab_size,
            substrate=substrate,
            tie_weights=tie_word_embeddings,
            input_embeddings=self.embeddings.token_embeddings if tie_word_embeddings else None,
            chunk_size=500,  # Adjust if needed (250-1000)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_attention_mask(
        self,
        input_ids: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Create attention mask.
        
        Args:
            input_ids: [B, L] token IDs
            causal: Whether to use causal masking
        
        Returns:
            mask: [B, 1, L, L] attention mask
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Padding mask
        pad_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        
        if causal:
            # Causal mask
            causal_mask = torch.tril(
                torch.ones(L, L, device=device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
            
            # Combine masks
            mask = pad_mask & causal_mask
        else:
            mask = pad_mask.expand(B, 1, L, L)
        
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] or [B, 1, L, L] attention mask
            position_ids: [B, L] position IDs
            labels: [B, L] labels for language modeling (optional)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return dict output
        
        Returns:
            output: Dict with keys:
                - logits: [B, L, V] language modeling logits
                - loss: Scalar loss (if labels provided)
                - hidden_states: [B, L, D] final hidden states
                - all_hidden_states: List of hidden states per layer (if output_hidden_states)
                - all_attentions: List of attention weights per layer (if output_attentions)
                - all_phases: List of phases per layer
                - all_order_params: List of order parameters per layer
        """
        # Embeddings
        hidden_states = self.embeddings(input_ids, position_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids, causal=True)
        elif attention_mask.dim() == 2:
            # Convert [B, L] to [B, 1, L, L]
            B, L = attention_mask.shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(B, 1, L, L)
        
        # Encoder
        encoder_output, all_attentions, all_hidden_states, all_phases, all_order_params = self.encoder(
            hidden_states,
            attention_mask,
            output_attentions,
            output_hidden_states,
        )
        
        # Language modeling head
        logits = self.lm_head(encoder_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'hidden_states': encoder_output,
                'all_hidden_states': all_hidden_states,
                'all_attentions': all_attentions,
                'all_phases': all_phases,
                'all_order_params': all_order_params,
            }
        else:
            return (logits, loss, encoder_output)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: [B, L] initial token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (True) or use greedy decoding (False)
        
        Returns:
            generated: [B, max_length] generated token IDs
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                output = self.forward(generated, return_dict=True)
                logits = output['logits']
                
                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply nucleus (top-p) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or greedy decode
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences have generated EOS token (assuming EOS = 2)
                if (next_token == 2).all():
                    break
        
        return generated
    
    def reset_phases(self):
        """Reset phase dynamics for all layers."""
        self.encoder.reset_phases()
    
    def manifold_fidelity(self) -> Dict[str, float]:
        """
        Check manifold constraint satisfaction.
        
        Returns:
            Dictionary with manifold violation metrics
        """
        if self.substrate is None:
            return {'mean_violation': 0.0, 'max_violation': 0.0}
        
        with torch.no_grad():
            # Check a sample of embeddings
            device = self.embeddings.token_embeddings.weight.device
            sample_size = min(10, self.vocab_size)
            sample_ids = torch.randint(0, self.vocab_size, (sample_size,), device=device)
            emb = self.embeddings.token_embeddings(sample_ids)
            
            violation = self.substrate.manifold_violation(emb)
            violation_val = violation.item() if torch.isfinite(violation) else 0.0
            
            return {
                'mean_violation': violation_val,
                'max_violation': violation_val,
            }
    
    def radius_statistics(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        Compute radius statistics for hidden states.
        
        Args:
            hidden_states: [B, L, D] or [B, L, D+1] hidden states
        
        Returns:
            Dictionary with radius statistics
        """
        if self.substrate is None:
            return {
                'radius_mean': 0.0,
                'radius_std': 0.0,
                'radius_min': 0.0,
                'radius_max': 0.0,
            }
        
        with torch.no_grad():
            # Compute radii
            radii = self.substrate.lorentz_radius(hidden_states)
            
            # Sanitize (replace NaN/Inf)
            radii = torch.nan_to_num(radii, nan=0.0, posinf=1e4, neginf=-1e4)
            
            return {
                'radius_mean': radii.mean().item() if torch.isfinite(radii.mean()) else 0.0,
                'radius_std': radii.std().item() if torch.isfinite(radii.std()) else 0.0,
                'radius_min': radii.min().item() if torch.isfinite(radii.min()) else 0.0,
                'radius_max': radii.max().item() if torch.isfinite(radii.max()) else 0.0,
            }
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters.
        
        Args:
            non_embedding: Whether to exclude embedding parameters
        
        Returns:
            num_params: Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.embeddings.token_embeddings.weight.numel()
            n_params -= self.embeddings.position_embeddings.weight.numel()
        
        return n_params
