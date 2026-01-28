# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Extracted from contrastive_transfer.py

"""
Sentence-Transformer Teacher
============================

Unified wrapper for Sentence-Transformers models used as teachers
in knowledge distillation.

Example
-------
>>> teacher = SentenceTransformerTeacher("all-MiniLM-L6-v2")
>>> embeddings = teacher.encode(["Hello world", "Machine learning"])
>>> print(embeddings.shape)  # (2, 384)
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Optional import with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Using mock embeddings.")


class SentenceTransformerTeacher:
    """
    Teacher model using Sentence-Transformers for semantic embeddings.
    
    Supported models (lightweight → heavy):
    - all-MiniLM-L6-v2: 22M params, 384 dim (recommended)
    - all-MiniLM-L12-v2: 33M params, 384 dim
    - all-mpnet-base-v2: 110M params, 768 dim
    - paraphrase-MiniLM-L6-v2: 22M params, 384 dim (faster)
    
    Example
    -------
    >>> teacher = SentenceTransformerTeacher("all-MiniLM-L6-v2")
    >>> embeddings = teacher.encode(["Hello world", "Machine learning"])
    >>> print(embeddings.shape)  # (2, 384)
    """
    
    # Popular models with their dimensions
    MODEL_INFO = {
        "all-MiniLM-L6-v2": {"dim": 384, "params": "22M"},
        "all-MiniLM-L12-v2": {"dim": 384, "params": "33M"},
        "all-mpnet-base-v2": {"dim": 768, "params": "110M"},
        "paraphrase-MiniLM-L6-v2": {"dim": 384, "params": "22M"},
        "multi-qa-MiniLM-L6-cos-v1": {"dim": 384, "params": "22M"},
        "all-distilroberta-v1": {"dim": 768, "params": "82M"},
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize teacher model.
        
        Parameters
        ----------
        model_name : str
            Sentence-transformer model name
        device : str, optional
            Device to use (auto-detected if None)
        normalize : bool
            Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        
        if HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"Loading teacher model: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self._embed_dim = self.model.get_sentence_embedding_dimension()
        else:
            logger.warning("Using mock embeddings (install sentence-transformers)")
            self.model = None
            self._embed_dim = 384  # Default dimension
        
        # Cache for efficiency
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_enabled = True
    
    @property
    def embed_dim(self) -> int:
        """Return embedding dimension."""
        return self._embed_dim
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Parameters
        ----------
        texts : str or List[str]
            Input texts
        batch_size : int
            Batch size for encoding
        show_progress : bool
            Show progress bar
        use_cache : bool
            Use cache for repeated texts
        
        Returns
        -------
        torch.Tensor
            Embeddings of shape (N, D)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        if use_cache and self._cache_enabled:
            cached = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self._cache:
                    cached.append((i, self._cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if not uncached_texts:
                # All cached
                result = torch.zeros(len(texts), self._embed_dim)
                for i, emb in cached:
                    result[i] = emb
                return result.to(self.device)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached = []
        
        # Encode uncached
        if self.model is not None:
            embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
            )
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings)
        else:
            # Mock embeddings for testing without sentence-transformers
            embeddings = self._mock_encode(uncached_texts)
        
        # Update cache
        if use_cache and self._cache_enabled:
            for text, emb in zip(uncached_texts, embeddings):
                self._cache[text] = emb.cpu()
        
        # Combine cached and new
        if cached:
            result = torch.zeros(len(texts), self._embed_dim, device=embeddings.device)
            for i, emb in cached:
                result[i] = emb.to(embeddings.device)
            for i, emb in zip(uncached_indices, embeddings):
                result[i] = emb
            return result
        
        return embeddings.to(self.device).clone()
    
    def _mock_encode(self, texts: List[str]) -> torch.Tensor:
        """
        Generate mock embeddings for testing.
        
        Uses hash-based deterministic random vectors.
        """
        embeddings = []
        for text in texts:
            # Deterministic seed from text
            seed = hash(text) % (2**32)
            rng = np.random.RandomState(seed)
            emb = rng.randn(self._embed_dim).astype(np.float32)
            if self.normalize:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
        return torch.tensor(np.stack(embeddings), device=self.device)
    
    def get_similarity_matrix(
        self,
        texts: List[str],
        metric: Literal["cosine", "dot", "euclidean"] = "cosine",
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix.
        
        Parameters
        ----------
        texts : List[str]
            Input texts
        metric : str
            Similarity metric
        
        Returns
        -------
        torch.Tensor
            Similarity matrix of shape (N, N)
        """
        embeddings = self.encode(texts)
        
        if metric == "cosine":
            # Already normalized if self.normalize is True
            if not self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings @ embeddings.T
        elif metric == "dot":
            return embeddings @ embeddings.T
        elif metric == "euclidean":
            # Convert to distance, then to similarity
            diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
            dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
            return 1.0 / (1.0 + dist)  # Convert distance to similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_distance_matrix(
        self,
        texts: List[str],
        metric: Literal["cosine", "euclidean"] = "cosine",
    ) -> torch.Tensor:
        """
        Compute pairwise distance matrix.
        
        Parameters
        ----------
        texts : List[str]
            Input texts
        metric : str
            Distance metric
        
        Returns
        -------
        torch.Tensor
            Distance matrix of shape (N, N)
        """
        embeddings = self.encode(texts)
        
        if metric == "cosine":
            if not self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            sim = embeddings @ embeddings.T
            return 1.0 - sim  # Cosine distance
        elif metric == "euclidean":
            diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
            return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_similar(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query.
        
        Returns
        -------
        List of (text, similarity_score) tuples
        """
        query_emb = self.encode([query])
        corpus_emb = self.encode(corpus)
        
        if not self.normalize:
            query_emb = F.normalize(query_emb, p=2, dim=-1)
            corpus_emb = F.normalize(corpus_emb, p=2, dim=-1)
        
        similarities = (query_emb @ corpus_emb.T).squeeze(0)
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        return [(corpus[i], similarities[i].item()) for i in top_indices]
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


__all__ = ["SentenceTransformerTeacher", "HAS_SENTENCE_TRANSFORMERS"]
