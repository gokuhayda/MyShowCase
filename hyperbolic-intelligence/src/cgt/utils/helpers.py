# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Utility Functions
=====================

General utilities for CGT experiments.

Functions
---------
- set_global_seed: Reproducibility configuration
- clear_memory: GPU memory management
- watermark_plot: Copyright watermark for figures
- academic_use_disclaimer: IP protection notice

Author: Éric Gustavo Reis de Sena
Date: January 2026
"""

from __future__ import annotations

import gc
import os
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> int:
    """
    Set all random seeds for complete reproducibility.

    This ensures deterministic behavior for:
    - Data shuffling
    - Model weight initialization
    - Dropout masks
    - Training order

    Args:
        seed: Random seed value.

    Returns:
        The seed value that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def academic_use_disclaimer() -> bool:
    """
    Display academic use disclaimer.

    This function MUST be called at runtime to ensure IP compliance visibility.

    Returns:
        True to indicate disclaimer was shown.
    """
    disclaimer = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              CONTRASTIVE GEOMETRIC TRANSFER (CGT) v1.0                   ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  Copyright © 2026 Éric Gustavo Reis de Sena                              ║
    ║  License: CC BY-NC-SA 4.0 (Academic/Non-Commercial Use Only)             ║
    ║                                                                          ║
    ║  ⚠️  COMMERCIAL USE IS STRICTLY PROHIBITED                               ║
    ║  📧  For commercial licensing: eirikreisena@gmail.com                    ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(disclaimer)
    return True


def watermark_plot(
    ax: Any,
    text: str = "© É.G.R. de Sena | CC BY-NC-SA 4.0 | Academic Use Only",
) -> Any:
    """
    Add copyright watermark to matplotlib axes.

    Args:
        ax: matplotlib axes object.
        text: Watermark text.

    Returns:
        The axes object with watermark added.
    """
    ax.text(
        0.99,
        0.01,
        text,
        transform=ax.transAxes,
        fontsize=7,
        color="gray",
        alpha=0.5,
        ha="right",
        va="bottom",
        style="italic",
    )
    return ax


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_summary(model: torch.nn.Module) -> dict:
    """
    Get model parameter summary.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }
