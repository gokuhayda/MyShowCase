"""
cgt.physics.config
~~~~~~~~~~~~~~~~~~
Configuration dataclass for HyDRA-Physics universe simulator.
New in V6. Does not affect any existing cgt modules.
"""
from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    # Universe
    N:   int   = 128
    dim: int   = 16
    dt:  float = 0.05      # Overdamped Langevin step
    T:   int   = 2000
    seed: int  = 42

    # Lennard-Jones potential: V(d) = V_scale*[exp(-d/sigma_a) - exp(-d/sigma_r)]
    sigma_a:  float = 1.8   # attraction scale
    sigma_r:  float = 0.6   # repulsion scale
    V_scale:  float = 0.08
    T_noise:  float = 0.008  # thermal noise kT

    # Coupling (separate for geometry and Kuramoto)
    K0_kur:  float = 0.003   # Kuramoto coupling
    K0_geo:  float = 0.05    # geodesic force baseline
    tau_kur: float = 1.5
    tau_geo: float = 2.0

    # Causal horizon
    R_max:       float = 8.0
    R_H:         float = 3.0
    tau_H:       float = 0.4
    use_horizon: bool  = True

    # Dynamic curvature
    use_dyn_curv: bool  = True
    K_curv_min:   float = 0.3
    K_curv_max:   float = 2.0

    # Learning
    learn_K:     bool  = True
    learn_start: int   = 300
    hidden_dim:  int   = 48
    lr:          float = 2e-4
    w_energy:    float = 0.08
    w_cluster:   float = 0.05
    w_topo:      float = 0.04
    w_entropy:   float = 0.03
    w_causal:    float = 0.02

    # Kuramoto phases
    freq_std: float = 0.12
    kdt:      float = 0.02

    log_every: int = 50
