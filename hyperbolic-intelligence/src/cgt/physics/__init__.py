"""
cgt.physics
~~~~~~~~~~~
HyDRA-Physics: emergent universe simulator on the Lorentz manifold.
New subpackage added in V6. Does not affect any existing cgt modules.

Quick start:
    from cgt.physics import PhysicsConfig, HyDRAUniverse
    cfg = PhysicsConfig(N=128, dim=16, T=2000)
    uni = HyDRAUniverse(cfg)
"""
from cgt.physics.config import PhysicsConfig
from cgt.physics.interaction import InteractionNet, DynamicCurvatureField
from cgt.physics.universe import HyDRAUniverse
from cgt.physics.lorentz_ops import (
    lorentz_inner, lorentz_exp, lorentz_log,
    lorentz_proj, safe_acosh, K, EPS, _BACKEND,
)

__all__ = [
    "PhysicsConfig",
    "InteractionNet",
    "DynamicCurvatureField",
    "HyDRAUniverse",
    "lorentz_inner", "lorentz_exp", "lorentz_log",
    "lorentz_proj", "safe_acosh",
    "K", "EPS", "_BACKEND",
    "EuclideanUniverse",
]
from cgt.physics.euclidean_universe import EuclideanUniverse
