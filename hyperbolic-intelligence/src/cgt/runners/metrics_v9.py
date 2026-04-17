"""cgt.runners.metrics_v9 — v9-specific metrics callback.

Plugs into the generic training loop via `train(..., metrics_callback=collect_v9_gate_stats)`.

This module is the ONLY v9-specific code in the runner system. For other
architectures (v10, v11, Euclidean baseline, etc.), write a similar callback
that reads that architecture's diagnostics and return an empty dict for
layers it doesn't recognise.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch.nn as nn


def collect_v9_gate_stats(model: nn.Module) -> Dict[str, float]:
    """Mean across layers of the per-forward AngularPhysicsGate diagnostics.

    Returns an empty dict if the model has no angular_gate attached
    (so v7 baseline runs produce log rows with just train/val loss).
    """
    gms, gss, cas, crs = [], [], [], []

    # HyperbolicTransformerV2 stores its layers at model.encoder.layers
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return {}
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return {}

    for layer in layers:
        ga = getattr(layer, "angular_gate", None)
        if ga is None:
            continue
        # These are detached buffers populated on every forward by AngularPhysicsGate
        gms.append(ga.last_gate.mean().item())
        gss.append(ga.last_gate.std().item())
        cas.append(ga.last_cos_align.mean().item())
        crs.append(ga.last_corr_attn_K.item())

    if not gms:
        return {}    # v7 / non-v9 model: emit nothing

    return dict(
        gate_mean   = float(np.mean(gms)),
        gate_std    = float(np.mean(gss)),
        cos_align   = float(np.mean(cas)),
        corr_attn_K = float(np.mean(crs)),
    )
