# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Hyperbolic Encoder
==================

Bridges a Euclidean sentence-embedding teacher (e.g., SentenceTransformer)
and a ``CGTStudentHardened`` student that projects embeddings onto the Lorentz
manifold.

Pipeline per call
-----------------
  text(s)
    │
    ▼  teacher.encode()
  float32 numpy [N, teacher_dim]  (L2-normalised in Euclidean space)
    │
    ▼  torch.tensor → student(x, use_homeostatic=False)
  float64 torch.Tensor [N, student_dim + 1]  (point on H^n, Lorentz model)

Design notes
------------
* The teacher is any object exposing ``encode(texts, ...)`` that returns a
  numpy array; ``sentence_transformers.SentenceTransformer`` is the canonical
  choice.
* The student is ``CGTStudentHardened`` from ``cgt.models``.  The encoder
  handles the ``use_homeostatic`` keyword gracefully and falls back to the
  positional-only call for other compatible models.
* All hyperbolic tensors are kept on the requested device with the requested
  dtype (default: float64 for geometric stability).
* Teacher encoding always runs on the SentenceTransformer's own device/dtype
  (usually CPU float32); conversion happens before the student forward pass.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


class HyperbolicEncoder:
    """Teacher → Hyperbolic projection encoder.

    Wraps a Euclidean sentence-embedding teacher and a ``CGTStudentHardened``
    student to produce Lorentz manifold embeddings for arbitrary text input.

    Args:
        teacher:    SentenceTransformer (or compatible) teacher model.
                    Must expose ``encode(texts, ...) -> np.ndarray``.
        student:    ``CGTStudentHardened`` model (or compatible).
                    Must be callable as ``student(tensor) -> torch.Tensor``.
        device:     Target device for student inference (e.g., ``'cuda'``).
                    If ``None``, auto-detects from the student's parameters.
        dtype:      Target dtype for student inference.
                    Default: ``torch.float64`` (required for Lorentz stability).

    Attributes:
        teacher:    The wrapped teacher model.
        student:    The wrapped student model.
        device:     Resolved inference device.
        dtype:      Resolved inference dtype.

    Example::

        from sentence_transformers import SentenceTransformer
        from cgt.models import CGTStudentHardened
        from cgt.embedding.encoder import HyperbolicEncoder

        teacher = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        student = CGTStudentHardened(teacher_dim=384, student_dim=32, hidden_dim=256)

        encoder = HyperbolicEncoder(teacher, student)
        hyp = encoder.encode_text("Hello, world!")  # [1, 33]
    """

    def __init__(
        self,
        teacher,
        student: torch.nn.Module,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.dtype = dtype

        # Auto-detect device from student parameters if not provided.
        if device is None:
            try:
                device = str(next(student.parameters()).device)
            except StopIteration:
                device = "cpu"
        self.device = device

        # Ensure student is in eval mode and on the correct device/dtype.
        self.student.to(device=self.device, dtype=self.dtype)
        self.student.eval()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a single text string to a hyperbolic embedding.

        Args:
            text: Input text.

        Returns:
            Tensor of shape ``[1, student_dim + 1]`` on the Lorentz manifold.
            The tensor is on CPU.
        """
        return self.encode_batch([text])

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Encode a list of texts to hyperbolic embeddings.

        Internally calls the teacher to obtain Euclidean embeddings, then
        projects them onto the Lorentz manifold via the student.

        Args:
            texts:      List of input strings.
            batch_size: Mini-batch size for the student forward pass.
                        Teacher batching is delegated to ``teacher.encode``.

        Returns:
            Tensor of shape ``[N, student_dim + 1]`` on CPU, where N is
            ``len(texts)``.  Points lie on the hyperboloid H^n.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        teacher_embs: np.ndarray = self._encode_teacher(texts)
        hyp_embs: torch.Tensor = self._project_to_hyperbolic(teacher_embs, batch_size)
        return hyp_embs

    def encode_batch_with_teacher(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Encode texts and return both teacher and student embeddings.

        Used internally by ``HyperbolicPipeline.index_corpus`` to obtain the
        Euclidean embeddings needed for the FAISS index alongside the
        hyperbolic embeddings needed for reranking.

        Args:
            texts:      List of input strings.
            batch_size: Mini-batch size for the student forward pass.

        Returns:
            Tuple of:
              - teacher_embs: ``np.ndarray`` of shape ``[N, teacher_dim]``,
                L2-normalised float32.
              - hyp_embs: ``torch.Tensor`` of shape ``[N, student_dim + 1]``
                on CPU, points on H^n.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        teacher_embs: np.ndarray = self._encode_teacher(texts)
        hyp_embs: torch.Tensor = self._project_to_hyperbolic(teacher_embs, batch_size)
        return teacher_embs, hyp_embs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_teacher(self, texts: List[str]) -> np.ndarray:
        """Run the teacher model and return L2-normalised float32 embeddings.

        Args:
            texts: Input strings.

        Returns:
            ``np.ndarray`` of shape ``[N, teacher_dim]``, dtype float32.
        """
        raw: np.ndarray = self.teacher.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return raw.astype(np.float32)

    def _project_to_hyperbolic(
        self,
        teacher_embs: np.ndarray,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Project Euclidean teacher embeddings to the Lorentz manifold.

        Processes embeddings in mini-batches to avoid OOM on large corpora.
        Supports models that accept ``use_homeostatic`` and falls back
        gracefully for models that do not.

        Args:
            teacher_embs: ``np.ndarray`` of shape ``[N, teacher_dim]``.
            batch_size:   Number of embeddings per forward pass.

        Returns:
            ``torch.Tensor`` of shape ``[N, student_dim + 1]`` on CPU.

        Raises:
            RuntimeError: If the student returns a non-Tensor output.
        """
        N: int = len(teacher_embs)
        all_hyp: List[torch.Tensor] = []

        self.student.eval()
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end: int = min(start + batch_size, N)
                batch_t = torch.tensor(
                    teacher_embs[start:end],
                    dtype=self.dtype,
                    device=self.device,
                )

                try:
                    hyp: torch.Tensor = self.student(batch_t, use_homeostatic=False)
                except TypeError:
                    # Fallback: student model does not accept use_homeostatic.
                    hyp = self.student(batch_t)

                if not isinstance(hyp, torch.Tensor):
                    raise RuntimeError(
                        f"Student model returned {type(hyp).__name__} instead of "
                        "torch.Tensor. Verify that the model is CGTStudentHardened."
                    )

                all_hyp.append(hyp.cpu())

        result: torch.Tensor = torch.cat(all_hyp, dim=0)  # [N, student_dim + 1]

        if result.ndim != 2:
            raise RuntimeError(
                f"Unexpected output shape from student: {result.shape}. "
                "Expected [N, student_dim + 1]."
            )

        return result
