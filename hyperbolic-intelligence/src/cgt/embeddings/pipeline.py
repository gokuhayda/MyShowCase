# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Hyperbolic Pipeline
===================

End-to-end semantic retrieval pipeline bridging Euclidean sentence embeddings,
hyperbolic projection, corpus indexing, and Lorentz geodesic reranking.

The pipeline integrates four modules:

  ``encoder.py``   — Euclidean teacher → Lorentz manifold projection
  ``index.py``     — Dual-space corpus index (FAISS + hyperbolic buffer)
  ``retrieval.py`` — Hybrid FAISS → Lorentz reranking

Full pipeline diagram
---------------------

  texts (List[str])
      │
      ▼  HyperbolicEncoder._encode_teacher()
  float32 numpy [N, teacher_dim]  (L2-normalised)
      │
      ├──────────────────────────────────────────►  HyperbolicIndex (FAISS)
      │                                              (Euclidean recall)
      ▼  HyperbolicEncoder._project_to_hyperbolic()
  torch.Tensor [N, student_dim + 1]  (on H^n, Lorentz)
      │
      └──────────────────────────────────────────►  HyperbolicIndex (buffer)
                                                     (hyperbolic reranking)

At query time:

  query text
      │
      ▼  teacher.encode()
  float32 numpy [1, teacher_dim]
      │
      ▼  FAISS search
  k_candidates indices
      │
      ▼  student(q_tensor, use_homeostatic=False)
  q_hyp [1, student_dim + 1]
      │
      ▼  hyperbolic_rerank()  ← Lorentz geodesic distances
  top_k RetrievedEvidence

Experimental status
-------------------
This pipeline is experimental and not optimised for production latency.
FAISS is used as a baseline ANN backend; scalability beyond medium-scale
datasets is not validated.  The hybrid Euclidean + hyperbolic retrieval
strategy is heuristic and not theoretically optimal.

Example::

    from cgt.embedding.pipeline import HyperbolicPipeline

    pipeline = HyperbolicPipeline(teacher_model, student_model)
    pipeline.index_corpus(texts)
    results = pipeline.query("What is hyperbolic embedding?")
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from cgt.embedding.encoder import HyperbolicEncoder
from cgt.embedding.index import HyperbolicIndex
from cgt.embedding.retrieval import HyperbolicRetriever, RetrievedEvidence


class HyperbolicPipeline:
    """End-to-end hyperbolic semantic retrieval pipeline.

    Composes ``HyperbolicEncoder``, ``HyperbolicIndex``, and
    ``HyperbolicRetriever`` into a single object.  Typical usage involves a
    one-time ``index_corpus()`` call followed by repeated ``query()`` calls.

    Args:
        teacher:     SentenceTransformer (or compatible) teacher model.
                     Must expose ``encode(texts, ...) -> np.ndarray``.
        student:     ``CGTStudentHardened`` (or compatible) student model.
                     Must be callable as ``student(tensor, use_homeostatic=bool)``.
        device:      Inference device for the student.  Auto-detected from
                     student parameters if ``None``.
        dtype:       Floating-point dtype for student inference.
                     Default: ``torch.float64`` (required for Lorentz stability).
        use_faiss:   Whether to use FAISS for ANN search.  Falls back to numpy
                     if FAISS is unavailable.

    Attributes:
        encoder:     Constructed ``HyperbolicEncoder`` instance.
        index:       Constructed ``HyperbolicIndex`` instance (``None`` until
                     ``index_corpus()`` is called).
        retriever:   Constructed ``HyperbolicRetriever`` instance (``None``
                     until ``index_corpus()`` is called).

    Raises:
        RuntimeError: If ``query()`` is called before ``index_corpus()``.

    Example::

        from sentence_transformers import SentenceTransformer
        from cgt.models import CGTStudentHardened
        from cgt.embedding.pipeline import HyperbolicPipeline

        teacher = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        student = CGTStudentHardened(teacher_dim=384, student_dim=32, hidden_dim=256)

        pipeline = HyperbolicPipeline(teacher, student)
        pipeline.index_corpus(corpus_texts)
        results = pipeline.query("What is hyperbolic geometry?")
        for ev in results:
            print(ev.rank, ev.text[:80])
    """

    def __init__(
        self,
        teacher,
        student: torch.nn.Module,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
        use_faiss: bool = True,
    ) -> None:
        self.encoder = HyperbolicEncoder(
            teacher=teacher,
            student=student,
            device=device,
            dtype=dtype,
        )
        self._use_faiss = use_faiss
        self.index: Optional[HyperbolicIndex] = None
        self.retriever: Optional[HyperbolicRetriever] = None
        self._corpus: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_corpus(
        self,
        texts: List[str],
        embed_batch: int = 128,
        hyp_batch: int = 256,
    ) -> None:
        """Build the dual-space index from a flat list of document strings.

        This is a potentially long-running operation for large corpora.  It:

        1. Encodes all texts with the teacher to produce Euclidean embeddings.
        2. Projects them to the Lorentz manifold via the student.
        3. Builds the FAISS index on Euclidean embeddings.
        4. Stores the hyperbolic embeddings for reranking.

        After this call, ``query()`` becomes available.

        Args:
            texts:       Corpus documents.  Must be non-empty.
            embed_batch: Batch size for teacher encoding.
            hyp_batch:   Batch size for student projection.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        self._corpus = texts

        # --- Determine teacher dimension from a probe encode ---
        probe: np.ndarray = self.encoder._encode_teacher([texts[0]])
        teacher_dim: int = probe.shape[1]

        # --- Build index object -----------------------------------------
        self.index = HyperbolicIndex(
            teacher_dim=teacher_dim,
            use_faiss=self._use_faiss,
        )

        # --- Encode and index in chunks to handle large corpora ----------
        N = len(texts)
        all_teacher: List[np.ndarray] = []
        all_hyp: List[torch.Tensor] = []

        for start in range(0, N, embed_batch):
            end = min(start + embed_batch, N)
            batch_texts = texts[start:end]

            teacher_embs, hyp_embs = self.encoder.encode_batch_with_teacher(
                batch_texts, batch_size=hyp_batch
            )
            all_teacher.append(teacher_embs)
            all_hyp.append(hyp_embs)

        teacher_all: np.ndarray = np.concatenate(all_teacher, axis=0)  # [N, D_t]
        hyp_all: torch.Tensor = torch.cat(all_hyp, dim=0)              # [N, D_s+1]

        self.index.build(teacher_all, hyp_all)

        # --- Construct retriever ----------------------------------------
        self.retriever = HyperbolicRetriever(
            encoder=self.encoder,
            index=self.index,
            corpus=self._corpus,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 5,
        k_candidates: int = 20,
        min_words: int = 4,
    ) -> List[RetrievedEvidence]:
        """Retrieve the most relevant documents for a query string.

        Delegates to ``HyperbolicRetriever.search`` which implements the full
        FAISS → Lorentz geodesic reranking pipeline.

        Args:
            text:         Query string.
            top_k:        Number of documents to return.
            k_candidates: FAISS candidates before Lorentz reranking.
                          A factor of 4× ``top_k`` is typical.
            min_words:    Minimum document length (word count) to include.

        Returns:
            List of ``RetrievedEvidence`` ordered by Lorentz relevance,
            length ≤ ``top_k``.

        Raises:
            RuntimeError: If ``index_corpus()`` has not been called.
        """
        if self.retriever is None:
            raise RuntimeError(
                "Pipeline has no index. Call index_corpus(texts) first."
            )
        return self.retriever.search(
            query=text,
            top_k=top_k,
            k_candidates=k_candidates,
            min_words=min_words,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the index artifacts to disk.

        Saves FAISS index and hyperbolic embeddings.  The encoder (teacher +
        student) weights are NOT saved here; manage those separately.

        Args:
            path: Directory path to write artifacts.

        Raises:
            RuntimeError: If no corpus has been indexed.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call index_corpus() first.")
        self.index.save(path)

    def load(self, path: str, corpus: List[str]) -> None:
        """Load persisted index artifacts and bind a corpus.

        After loading, ``query()`` becomes available without calling
        ``index_corpus()``.

        Args:
            path:   Directory path previously written by ``save()``.
            corpus: The original corpus list used when the index was built.
                    Must have the same length as the stored index.

        Raises:
            FileNotFoundError: If required artifact files are missing.
            ValueError: If ``len(corpus)`` does not match the stored index.
        """
        probe: np.ndarray = self.encoder._encode_teacher([corpus[0]])
        teacher_dim: int = probe.shape[1]

        self._corpus = corpus
        self.index = HyperbolicIndex(
            teacher_dim=teacher_dim,
            use_faiss=self._use_faiss,
        )
        self.index.load(path)

        if len(self.index) != len(corpus):
            raise ValueError(
                f"Stored index has {len(self.index)} entries but corpus has "
                f"{len(corpus)} documents."
            )

        self.retriever = HyperbolicRetriever(
            encoder=self.encoder,
            index=self.index,
            corpus=self._corpus,
        )

    # ------------------------------------------------------------------
    # Sanity checks (called after index_corpus for validation)
    # ------------------------------------------------------------------

    def sanity_check(self) -> dict:
        """Run basic sanity checks on the constructed index.

        Verifies embedding shapes, device consistency, dtype consistency, and
        the hyperboloid constraint (−x₀² + ‖x_s‖² ≈ −1/K).

        Returns:
            Dictionary of check results with keys:
              - ``n_docs``: Number of indexed documents.
              - ``teacher_dim``: Euclidean embedding dimension.
              - ``student_ambient_dim``: Ambient Lorentz dimension (D_s + 1).
              - ``hyp_shape``: Shape of the hyperbolic embedding buffer.
              - ``K``: Learned curvature value.
              - ``hyperboloid_error``: Mean absolute deviation from constraint.
              - ``dtype_ok``: Whether embeddings have the expected dtype.
              - ``device_ok``: Whether the student model is on the target device.

        Raises:
            RuntimeError: If no corpus has been indexed.
        """
        if self.index is None or self._corpus is None:
            raise RuntimeError("No index available. Call index_corpus() first.")

        hyp: torch.Tensor = self.index.get_embeddings()
        K_val: float = self.encoder.student.substrate.K.item()

        # Hyperboloid constraint: −x₀² + ‖x_s‖² = −1/K
        with torch.no_grad():
            t: torch.Tensor = hyp[:, 0]
            s: torch.Tensor = hyp[:, 1:]
            constraint_lhs: torch.Tensor = -t ** 2 + (s ** 2).sum(dim=1)
            target: float = -1.0 / K_val
            hyperboloid_error: float = (constraint_lhs - target).abs().mean().item()

        student_device = str(next(self.encoder.student.parameters()).device)

        return {
            "n_docs": len(self.index),
            "teacher_dim": self.index.teacher_dim,
            "student_ambient_dim": hyp.shape[1],
            "hyp_shape": tuple(hyp.shape),
            "K": K_val,
            "hyperboloid_error": hyperboloid_error,
            "dtype_ok": hyp.dtype == self.encoder.dtype,
            "device_ok": student_device.split(":")[0] == self.encoder.device.split(":")[0],
        }
