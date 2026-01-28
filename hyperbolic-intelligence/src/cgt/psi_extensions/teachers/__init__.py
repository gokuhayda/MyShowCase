# SPDX-License-Identifier: MIT
# Origin: PSI_SLM - Integrated into CGT

"""
Teacher Model Wrappers
======================

Unified interfaces for teacher models used in knowledge distillation.

This module provides:
- SentenceTransformerTeacher: Wrapper for sentence-transformers models

Note: Requires `sentence-transformers` library
"""

from .sentence_transformer import SentenceTransformerTeacher

__all__ = [
    "SentenceTransformerTeacher",
]
