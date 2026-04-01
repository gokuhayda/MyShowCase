# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Hyperbolic Retrieval
====================

Hybrid retrieval pipeline: fast Euclidean ANN candidate selection followed
by precise Lorentz geodesic reranking.

Key design insight
------------------
Retrieval is decomposed into:

* **Fast approximate search in Euclidean space** — FAISS ``IndexFlatIP``
  over L2-normalised teacher embeddings provides high-recall candidates in
  sub-linear time.
* **Precise semantic ranking in hyperbolic space** — Lorentz geodesic
  distances reorder the FAISS candidates according to the curved geometry
  learned by ``CGTStudentHardened``.

This separation avoids the prohibitive cost of computing geodesic distances
over the entire corpus while preserving the geometric precision of the
hyperbolic student model.

Mathematical basis
------------------
The Lorentz inner product between query q and candidate c:

    ⟨q, c⟩_L = −q₀ c₀ + q_s · c_s

Geodesic distance with curvature K > 0:

    d(q, c) = (1/√K) · arccosh(−K · ⟨q, c⟩_L)

Candidates are ranked by ascending distance (smaller = more similar).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch

from cgt.embeddings.distance import lorentz_query_distances
from cgt.embeddings.encoder import HyperbolicEncoder
from cgt.embeddings.index import HyperbolicIndex


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedEvidence:
    """Single retrieved document with its retrieval metadata.

    Attributes:
        text:       The raw document text.
        score:      Retrieval score (reciprocal rank from Lorentz reranking).
        rank:       1-based rank in the final result list.
        corpus_idx: Index of this document in the original corpus list.
    """

    text: str
    score: float
    rank: int
    corpus_idx: int


@dataclass
class RetrievalResult:
    """Container for the output of a single retrieval query.

    Attributes:
        query:          Original query string.
        evidences:      Retrieved documents ordered by Lorentz reranking.
        top_k:          Number of documents requested.
        retrieval_mode: Always ``'lorentz'`` for ``HyperbolicRetriever``.
    """

    query: str
    evidences: List[RetrievedEvidence]
    top_k: int
    retrieval_mode: str = "lorentz"


# ---------------------------------------------------------------------------
# Standalone rerank utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def hyperbolic_rerank(
    query_hyp: torch.Tensor,
    candidate_hyp: torch.Tensor,
    K_val: float,
    top_k: int,
    eps: float = 1e-7,
) -> np.ndarray:
    """Reorder candidate embeddings by Lorentz geodesic distance to the query.

    Implements the reranking step of the hybrid retrieval pipeline.  Input
    candidates are the ``C`` hyperbolic embeddings selected by an upstream
    FAISS search.  The return value is an array of **relative** indices
    (values in ``[0, C-1]``) ordered by ascending geodesic distance (closest
    first).  To map back to absolute corpus indices, use::

        absolute_indices = faiss_indices[hyperbolic_rerank(...)]

    Mathematical detail
    -------------------
    Minkowski inner product between query ``q`` and candidate ``c``:

        ⟨q, c⟩_L = −q₀ c₀ + q_s · c_s

    where subscript 0 is the time coordinate and subscript s the spatial part.

    Geodesic distance:

        d(q, c) = (1/√K) · arccosh(−K · ⟨q, c⟩_L)

    Args:
        query_hyp:     Query embedding on H^n, shape ``[1, D+1]``.
        candidate_hyp: Candidate embeddings on H^n, shape ``[C, D+1]``.
        K_val:         Curvature parameter (K > 0).  Use
                       ``student_model.substrate.K.item()``.
        top_k:         Maximum number of candidates to return.
        eps:           Stability floor for ``arccosh``.

    Returns:
        ``np.ndarray`` of shape ``[min(top_k, C)]`` containing **relative**
        indices into ``candidate_hyp``, sorted by ascending geodesic distance.

    Raises:
        ValueError: If ``query_hyp`` is not shape ``[1, D+1]`` or dimensions
                    are incompatible.
    """
    C: int = len(candidate_hyp)
    if C == 0:
        return np.array([], dtype=np.int64)
    if query_hyp.shape[0] != 1:
        raise ValueError(
            f"query_hyp must have shape [1, D+1], got {query_hyp.shape}."
        )
    if query_hyp.shape[1] != candidate_hyp.shape[1]:
        raise ValueError(
            f"Ambient dim mismatch: query {query_hyp.shape[1]}, "
            f"candidates {candidate_hyp.shape[1]}."
        )

    # lorentz_query_distances returns [C] — distances from query to each candidate.
    dists: torch.Tensor = lorentz_query_distances(
        query_hyp, candidate_hyp, K=K_val, eps=eps
    )  # [C]

    k_actual: int = min(top_k, C)
    reranked: torch.Tensor = torch.argsort(dists)[:k_actual]  # ascending
    return reranked.cpu().numpy()  # [k_actual], values in [0, C-1]


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class HyperbolicRetriever:
    """Hybrid FAISS → Lorentz retriever over a pre-built ``HyperbolicIndex``.

    Executes the following pipeline for every query:

    1. Encode query text with the teacher (Euclidean, L2-normalised).
    2. FAISS candidate search: ``k_candidates`` approximate neighbours.
    3. Project query onto H^n via the student (``CGTStudentHardened``).
    4. Extract hyperbolic embeddings for the FAISS candidates.
    5. Rerank candidates by Lorentz geodesic distance.
    6. Return ``top_k`` evidence items.

    Args:
        encoder: ``HyperbolicEncoder`` wrapping teacher + student.
        index:   Pre-built ``HyperbolicIndex`` for the target corpus.
        corpus:  List of raw document strings indexed in ``index``.  The
                 ``i``-th string must correspond to the ``i``-th row in the
                 index.

    Raises:
        ValueError: If ``len(corpus)`` does not match ``len(index)``.

    Example::

        retriever = HyperbolicRetriever(encoder, index, corpus_texts)
        results = retriever.search("What is hyperbolic geometry?", top_k=5)
        for ev in results:
            print(ev.rank, ev.score, ev.text[:80])
    """

    def __init__(
        self,
        encoder: HyperbolicEncoder,
        index: HyperbolicIndex,
        corpus: List[str],
    ) -> None:
        if len(corpus) != len(index):
            raise ValueError(
                f"corpus has {len(corpus)} documents but index has "
                f"{len(index)} entries.  They must match."
            )
        self.encoder = encoder
        self.index = index
        self.corpus = corpus

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        k_candidates: int = 20,
        min_words: int = 4,
        rerank_hook: Optional[Callable] = None,
    ) -> List[RetrievedEvidence]:
        """Retrieve the top-k most relevant documents for a query.

        Args:
            query:        Input query string.
            top_k:        Final number of documents to return.
            k_candidates: Number of FAISS candidates before reranking.
                          Should be ≥ ``top_k``; a factor of 4× is typical.
            min_words:    Minimum word count; shorter documents are filtered
                          after reranking.
            rerank_hook:  Optional external reranker
                          ``(query: str, candidates: List[Tuple[str, float]])
                          → List[Tuple[str, float]]``.
                          If provided, it is applied after Lorentz reranking.

        Returns:
            List of ``RetrievedEvidence`` objects ordered by relevance
            (highest score first), length ≤ ``top_k``.

        Raises:
            RuntimeError: If the index is empty or the query encodes to NaN.
        """
        # --- Step 1: Euclidean teacher embedding of query -----------------
        teacher_q: np.ndarray = self.encoder._encode_teacher([query])  # [1, D_t]
        query_vec: np.ndarray = teacher_q[0]  # [D_t]

        # --- Step 2: FAISS candidate selection ----------------------------
        k_req: int = min(
            max(top_k * 4, k_candidates),
            len(self.index),
        )
        _, faiss_idxs = self.index.search(query_vec, k=k_req)
        faiss_idxs = np.array(faiss_idxs, dtype=np.int64)
        faiss_idxs = faiss_idxs[faiss_idxs >= 0]  # remove FAISS padding -1

        if len(faiss_idxs) == 0:
            return []

        # --- Step 3: Project query to hyperbolic space --------------------
        q_tensor = torch.tensor(
            teacher_q,
            dtype=self.encoder.dtype,
            device=self.encoder.device,
        )  # [1, D_t]
        self.encoder.student.eval()
        with torch.no_grad():
            try:
                q_hyp: torch.Tensor = self.encoder.student(
                    q_tensor, use_homeostatic=False
                )
            except TypeError:
                q_hyp = self.encoder.student(q_tensor)
        q_hyp = q_hyp.cpu()  # [1, D_s + 1]

        K_val: float = self.encoder.student.substrate.K.item()

        # --- Step 4: Hyperbolic embeddings of FAISS candidates ------------
        all_hyp: torch.Tensor = self.index.get_embeddings()
        candidate_hyp: torch.Tensor = all_hyp[faiss_idxs]  # [C, D_s + 1]
        # Ensure dtype consistency
        candidate_hyp = candidate_hyp.to(dtype=self.encoder.dtype)
        q_hyp = q_hyp.to(dtype=self.encoder.dtype)

        # --- Step 5: Lorentz geodesic reranking ---------------------------
        rel_idxs: np.ndarray = hyperbolic_rerank(
            q_hyp, candidate_hyp, K_val=K_val, top_k=k_req
        )
        abs_idxs: np.ndarray = faiss_idxs[rel_idxs]  # absolute corpus indices

        # --- Step 6: Filter and build evidence list -----------------------
        pairs: List[tuple] = [
            (int(idx), 1.0 / (rank + 1))
            for rank, idx in enumerate(abs_idxs)
            if len(self.corpus[int(idx)].split()) >= min_words
        ]

        # Optional external reranker
        if rerank_hook is not None:
            candidates_for_hook = [
                (self.corpus[idx], score) for idx, score in pairs[:k_candidates]
            ]
            reranked_ext = rerank_hook(query, candidates_for_hook)
            evidences: List[RetrievedEvidence] = [
                RetrievedEvidence(
                    text=text,
                    score=score,
                    rank=rank + 1,
                    corpus_idx=-1,  # index unknown after external rerank
                )
                for rank, (text, score) in enumerate(reranked_ext[:top_k])
            ]
        else:
            evidences = [
                RetrievedEvidence(
                    text=self.corpus[idx],
                    score=score,
                    rank=rank + 1,
                    corpus_idx=idx,
                )
                for rank, (idx, score) in enumerate(pairs[:top_k])
            ]

        return evidences
