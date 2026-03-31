# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
cgt.embedding — Hyperbolic Embedding and Retrieval Pipeline
============================================================

Inference pipeline for Lorentz-manifold semantic retrieval.  Bridges
Euclidean sentence embeddings (teacher), hyperbolic projection
(``CGTStudentHardened``), corpus indexing (FAISS + hyperbolic buffer), and
Lorentz geodesic reranking into a single composable API.

Submodules
----------
distance
    Standalone ``@torch.no_grad()`` Lorentz batch distance utilities.
    Optimised for asymmetric query-vs-corpus access at inference time.

encoder
    ``HyperbolicEncoder`` — wraps teacher + student for text-to-hyperbolic
    encoding.

index
    ``HyperbolicIndex`` — dual-space corpus index: FAISS (Euclidean recall)
    + torch.Tensor buffer (Lorentz reranking).

retrieval
    ``HyperbolicRetriever`` — hybrid FAISS → Lorentz rerank retriever.
    ``hyperbolic_rerank`` — standalone Lorentz reranking function.
    ``RetrievedEvidence``, ``RetrievalResult`` — result data classes.

pipeline
    ``HyperbolicPipeline`` — end-to-end semantic retrieval pipeline.

Quick start::

    from cgt.embedding.pipeline import HyperbolicPipeline

    pipeline = HyperbolicPipeline(teacher_model, student_model)
    pipeline.index_corpus(texts)
    results = pipeline.query("What is hyperbolic embedding?")

Status
------
**Experimental** — not optimised for production latency.  FAISS is used as
a baseline ANN backend; scalability beyond medium-scale datasets is not
validated.  The hybrid Euclidean + hyperbolic retrieval strategy is heuristic
and not theoretically optimal.
"""

from cgt.embedding.distance import (
    lorentz_dist_batch,
    lorentz_inner_product_batch,
    lorentz_query_distances,
)
from cgt.embedding.encoder import HyperbolicEncoder
from cgt.embedding.index import HyperbolicIndex
from cgt.embedding.retrieval import (
    HyperbolicRetriever,
    RetrievalResult,
    RetrievedEvidence,
    hyperbolic_rerank,
)
from cgt.embedding.pipeline import HyperbolicPipeline

__all__ = [
    # Distance utilities
    "lorentz_dist_batch",
    "lorentz_inner_product_batch",
    "lorentz_query_distances",
    # Encoder
    "HyperbolicEncoder",
    # Index
    "HyperbolicIndex",
    # Retrieval
    "HyperbolicRetriever",
    "RetrievedEvidence",
    "RetrievalResult",
    "hyperbolic_rerank",
    # Pipeline
    "HyperbolicPipeline",
]
