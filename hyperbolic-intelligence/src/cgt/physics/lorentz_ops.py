"""
cgt.physics.lorentz_ops
~~~~~~~~~~~~~~~~~~~~~~~
Standalone float64 Lorentz manifold operations for HyDRA-Physics.
Wraps cgt.geometry.lorentz_v2 when available; falls back to inline
implementations so the physics module works without the full cgt tree.

New in V6. Does not modify any existing geometry modules.
"""
import math
import torch

EPS = 1e-7
K   = 1.0   # curvature constant

# ── Try to use cgt substrate ──────────────────────────────────────────────
try:
    from cgt.geometry.lorentz_v2 import LorentzSubstrateV2, LorentzConfigV2  # noqa
    _sub = LorentzSubstrateV2(LorentzConfigV2(curvature=1.0))

    def lorentz_inner(x, y):  return _sub.minkowski_inner(x, y)
    def lorentz_exp(x, v):    return _sub.exp_map(x, v)
    def lorentz_log(x, y):    return _sub.log_map(x, y)
    def lorentz_dist(x, y):   return _sub.dist(x, y)
    def lorentz_proj(x):      return _sub.proj(x)
    def safe_acosh(z):        return torch.acosh(z.clamp(min=1.0 + EPS))
    _BACKEND = "cgt"

except Exception:
    # ── Inline float64 implementations (no external dependency) ──────────
    def safe_acosh(z):
        return torch.acosh(z.clamp(min=1.0 + EPS))

    def lorentz_inner(x, y):
        x64, y64 = x.double(), y.double()
        out = (-x64[..., :1] * y64[..., :1]
               + (x64[..., 1:] * y64[..., 1:]).sum(dim=-1, keepdim=True))
        return out.to(x.dtype)

    def lorentz_proj(x):
        x64 = x.double()
        xs  = x64[..., 1:]
        x0  = (1.0 / K + (xs * xs).sum(-1, keepdim=True)).clamp(min=EPS).sqrt()
        return torch.cat([x0, xs], dim=-1).to(x.dtype)

    def lorentz_dist(x, y):
        inn = lorentz_inner(x, y)
        return safe_acosh((-K * inn).clamp(min=1.0 + EPS)) / math.sqrt(K)

    def lorentz_log(x, y):
        x64, y64 = x.double(), y.double()
        inn = lorentz_inner(x64, y64)
        d   = lorentz_dist(x64, y64).clamp(min=EPS)
        coeff = d / torch.sinh(d.clamp(max=10.0))
        return (coeff * (y64 + K * inn * x64)).to(x.dtype)

    def lorentz_exp(x, v):
        x64, v64 = x.double(), v.double()
        inn = lorentz_inner(v64, x64)
        v64 = v64 + K * inn * x64
        nv  = (-lorentz_inner(v64, v64).clamp(max=0)).sqrt().clamp(min=EPS)
        out = torch.cosh(nv) * x64 + torch.sinh(nv) / nv * v64
        return lorentz_proj(out).to(x.dtype)

    _BACKEND = "inline"

__all__ = [
    "lorentz_inner", "lorentz_exp", "lorentz_log",
    "lorentz_dist", "lorentz_proj", "safe_acosh",
    "K", "EPS", "_BACKEND",
]
