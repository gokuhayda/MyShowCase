# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
CGT Utilities Module
====================

General utilities for CGT experiments.

Exports
-------
set_global_seed
    Reproducibility configuration.
clear_memory
    GPU memory management.
academic_use_disclaimer
    IP protection notice.
watermark_plot
    Copyright watermark for figures.
get_device
    Device selection.
model_summary
    Parameter count summary.
"""

from cgt.utils.helpers import (
    academic_use_disclaimer,
    clear_memory,
    get_device,
    model_summary,
    set_global_seed,
    watermark_plot,
)

__all__ = [
    "set_global_seed",
    "clear_memory",
    "academic_use_disclaimer",
    "watermark_plot",
    "get_device",
    "model_summary",
]
