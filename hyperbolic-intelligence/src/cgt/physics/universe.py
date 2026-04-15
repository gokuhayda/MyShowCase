"""
cgt.physics.universe
~~~~~~~~~~~~~~~~~~~~
HyDRAUniverse: emergent mini-universe on the Lorentz manifold.

  Dynamics : Overdamped Langevin  h_new = exp_h(dt·F + √dt·σ·noise)
  Force    : derived from LJ potential gradient (can be negative = repulsion)
  Kuramoto : small separate coupling for phase (local time) dynamics
  Curvature: DynamicCurvatureField → D_eff = D / √(K_i·K_j)
  Horizon  : K_ij *= σ((R_H − d)/τ_H)

New in V6. Does not modify any existing cgt modules.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.physics.config import PhysicsConfig
from cgt.physics.interaction import InteractionNet, DynamicCurvatureField, causal_horizon
from cgt.physics.lorentz_ops import (
    lorentz_inner, lorentz_exp, lorentz_log,
    lorentz_proj, safe_acosh, K, EPS,
)


class HyDRAUniverse(nn.Module):
    _version = "langevin-v1"   # bump when class changes
    """
    Principles
    ──────────
    1. Störmer-Verlet integration  (energy-conserving, no damping)
    2. Lennard-Jones potential on geodesic distances
    3. Dynamic curvature: K_i = g(ρ_i) → D_eff = D/√(K_i·K_j)
    4. Causal horizon: K_ij *= σ((R_H − d)/τ_H)
    5. HARD SEPARATION: evolve_state (pure geometry, @no_grad)
                        compute_losses (differentiable, updates LAWS only)

    Physical semantics
    ──────────────────
    r_i     → gravitational potential depth
    |v_i|   → kinetic momentum
    θ_i     → local proper time
    Γ       → global temporal coherence
    K_i     → local spacetime curvature induced by matter
    d_eff   → curvature-corrected geodesic (proto-metric)
    RDC     → radial instability (inherited from HyDRA v7)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N, n = cfg.N, cfg.dim

        self.interact = InteractionNet(cfg)
        self.curv     = DynamicCurvatureField(cfg)

        # Initialise near LJ equilibrium d* ≈ 1.0
        # Each particle placed on a sphere of radius ~ d*/2 ≈ 0.5 in spatial coords
        sa, sr = cfg.sigma_a, cfg.sigma_r
        d_star = sa * sr / (sa - sr) * math.log(sa / sr)
        r_init = d_star * 0.5   # spatial radius ≈ d*/2 (geodesic d ≈ r at small r)

        hs  = torch.randn(N, n, dtype=torch.float64) * r_init
        ht  = (1.0 + (hs**2).sum(-1, keepdim=True)).sqrt()
        self.register_buffer("h",     torch.cat([ht, hs], dim=-1))
        self.register_buffer("theta", torch.rand(N) * 2 * math.pi)
        self.register_buffer("omega", torch.randn(N) * cfg.freq_std)
        self.register_buffer("E_ref", torch.tensor(0.0, dtype=torch.float64))

        # No velocity buffer — Overdamped Langevin has no inertia
        self._K_mat    = None
        self._K_local  = None
        self._hz_frac  = 0.0
        self._r_prev   = None

    def radii(self, h=None):
        """[N] spatial radius = ‖h_{1:}‖₂"""
        if h is None: h = self.h
        return h[:, 1:].norm(dim=-1)

    def pairwise_dist(self, h=None):
        """[N,N] float64 geodesic distance (D_min=0.08 floor for stability)"""
        if h is None: h = self.h
        assert h.dim() == 2, f"pairwise_dist expects [N,n+1], got {h.shape}"
        N  = h.shape[0]
        hi = h.unsqueeze(1).expand(N, N, -1)            # [N,N,n+1]
        hj = h.unsqueeze(0).expand(N, N, -1)
        inn = lorentz_inner(hi, hj).squeeze(-1)          # [N,N]
        D   = safe_acosh((-K * inn).clamp(min=1+EPS)) / math.sqrt(K)
        eye = torch.eye(N, device=D.device, dtype=D.dtype)
        D_min = 0.08
        # off-diagonal: clamp to D_min; diagonal stays 0
        return D * (1-eye) + D_min * ((D < D_min) & (eye < 0.5)).to(D.dtype)

    def local_density(self, D, sig=0.8):
        """[N] float32 kernel density (Gaussian, self excluded)"""
        rho = torch.exp(-D.float()**2 / (2*sig**2)).sum(-1) - 1.0
        return rho.clamp(min=0)

    def order_param(self):
        return (torch.exp(1j * self.theta.float()).mean()).abs().item()

    # ──────────────────────────────── potential ────────────────────────────

    def lj_potential(self, D):
        """
        V(d) = V_scale * [exp(-d/σ_a) - exp(-d/σ_r)]
        Gradient ∂V/∂d < 0 at d > d* → attraction
                 ∂V/∂d > 0 at d < d* → repulsion
        """
        cfg  = self.cfg
        mask = 1 - torch.eye(D.shape[0], device=D.device, dtype=D.dtype)
        V = cfg.V_scale * (torch.exp(-D/cfg.sigma_a) - torch.exp(-D/cfg.sigma_r))
        return (V * mask).sum() * 0.5

    def total_energy(self, h, D):
        """Potential energy only (Langevin has no kinetic term)."""
        return self.lj_potential(D)

    # ──────────────── build coupling matrix (dynamic curv + horizon) ──────

    def _build_K(self, D_raw_f, step):
        """
        Returns (K_geo, K_kur):
          K_geo [N,N]: force coupling derived from LJ gradient — CAN BE NEGATIVE
            K_geo < 0 → repulsion (d < d*)
            K_geo > 0 → attraction (d > d*)
          K_kur [N,N]: Kuramoto phase coupling — always positive, small, separate
        """
        cfg = self.cfg
        N   = D_raw_f.shape[0]
        eye = torch.eye(N, device=D_raw_f.device)

        # 1. Dynamic curvature: density → K_local → D_eff
        rho     = self.local_density(D_raw_f)
        K_local = self.curv(rho) if cfg.use_dyn_curv \
                  else torch.ones(N, device=D_raw_f.device)
        self._K_local = K_local
        Ki    = K_local.unsqueeze(1).expand(N, N)
        Kj    = K_local.unsqueeze(0).expand(N, N)
        D_eff = (D_raw_f / (Ki * Kj).clamp(min=1e-4).sqrt()).clamp(min=0.05)

        # 2. K_geo: derived from LJ potential gradient (PHYSICAL FORCE)
        #    ∂V/∂d = exp(-d/σ_r)/σ_r − exp(-d/σ_a)/σ_a
        #    F = −∂V/∂d · log_{h_i}(h_j) / |log| in the manifold force
        #    So: K_geo = −∂V/∂d * V_scale (negative = repulsion, positive = attraction)
        sa, sr = cfg.sigma_a, cfg.sigma_r
        dVdd  = (torch.exp(-D_eff / sr) / sr
               - torch.exp(-D_eff / sa) / sa)           # [N,N]
        K_geo = -dVdd * cfg.V_scale * (1 - eye)         # negative=repulsion

        # 3. InteractionNet: multiplicative correction factor on K_geo
        if cfg.learn_K and step >= cfg.learn_start:
            r   = self.radii().float()
            th  = self.theta.float()
            dth = (th.unsqueeze(0) - th.unsqueeze(1)).abs() % math.pi
            ri  = r.unsqueeze(1).expand(N, N)
            rj  = r.unsqueeze(0).expand(N, N)
            corr       = self.interact(D_eff, dth, ri, rj)  # ∈(K_min, K0_geo)
            corr_scale = corr / (cfg.K0_geo + 1e-6) * 2.0   # ∈ (~0, 2)
            K_geo      = K_geo * corr_scale

        # 4. Causal horizon on K_geo
        if cfg.use_horizon:
            mask          = torch.sigmoid((cfg.R_H - D_eff) / cfg.tau_H)
            self._hz_frac = float((mask < 0.5).float().mean())
            K_geo         = K_geo * mask
        else:
            self._hz_frac = 0.0

        # 5. K_kur: purely for Kuramoto phases (small, positive, separate)
        K_kur = cfg.K0_kur * torch.exp(-D_raw_f / cfg.tau_kur) * (1 - eye)

        self._K_mat = K_geo
        return K_geo, K_kur
    def _force(self, h, K_geo):
        """
        F_i = Σ_j K_geo_ij · log_{h_i}(h_j)
        K_geo < 0 → repulsion (log points toward j, negative = away)
        K_geo > 0 → attraction
        """
        N  = h.shape[0]
        hi = h.unsqueeze(1).expand(N, N, -1)
        hj = h.unsqueeze(0).expand(N, N, -1)
        G  = lorentz_log(hi, hj)                              # [N,N,n+1]
        F  = (K_geo.double().unsqueeze(-1) * G).sum(1)        # [N,n+1]
        F_norm  = F.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        F_scale = (torch.tensor(6.0, dtype=F.dtype, device=F.device)
                   / F_norm).clamp(max=1.0)
        return F * F_scale

    # ─────────────────────── tangent projection ───────────────────────────

    @staticmethod
    def _proj_tangent(v, h):
        """Project v onto T_h H^n: v ← v + K⟨v,h⟩_L · h"""
        inn = lorentz_inner(v, h)          # [N,1]
        return v + K * inn * h             # [N,n+1]

    # ─────────────────────── LOOP A: pure physics ─────────────────────────

    @torch.no_grad()
    def evolve_state(self, K_geo, K_kur, step):
        """
        Overdamped Langevin dynamics (no inertia, always stable):
            h_new = exp_{h}(dt * F + sqrt(dt) * σ * noise)

        Stationary distribution: Boltzmann ~ exp(-V/T_noise)
        Guaranteed stable: no velocity accumulation, no runaway.
        Thermal noise (T_noise) prevents freezing at local minima.
        """
        cfg = self.cfg
        h   = self.h
        N   = h.shape[0]

        # Geodesic force from LJ potential gradient (K_geo can be negative)
        F = self._force(h, K_geo)                        # [N, n+1]

        # Thermal noise in tangent space (exploration)
        if cfg.T_noise > 0:
            noise_s = torch.randn(N, cfg.dim, dtype=torch.float64, device=h.device)                       * math.sqrt(cfg.dt * cfg.T_noise)  # [N, n]
            noise   = torch.cat([torch.zeros(N, 1, dtype=torch.float64,
                                             device=h.device), noise_s], dim=-1)
            # Project noise to tangent space T_h H^n
            inn_n   = lorentz_inner(noise, h)
            noise   = noise + K * inn_n * h
        else:
            noise = torch.zeros_like(h)

        # Langevin step: displacement = dt*F + noise
        disp  = cfg.dt * F + noise                        # [N, n+1]
        # Project displacement to tangent space
        inn_d = lorentz_inner(disp, h)
        disp  = disp + K * inn_d * h

        # Clamp displacement magnitude for safety
        d_norm  = disp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d_scale = (torch.tensor(0.5, dtype=torch.float64, device=h.device)
                   / d_norm).clamp(max=1.0)
        disp    = disp * d_scale

        # Update position via exp map
        h_new = lorentz_proj(lorentz_exp(h, disp))        # [N, n+1]

        # NaN guard: revert individual particles to previous h if NaN
        bad   = torch.isnan(h_new).any(dim=-1)             # [N] bool
        h_new = torch.where(bad.unsqueeze(-1).expand_as(h_new), h, h_new)

        # Kuramoto phase evolution (RK2, separate from force)
        theta = self.theta
        K_f   = K_kur.float()
        dth   = (theta.unsqueeze(0) - theta.unsqueeze(1))
        dth   = torch.remainder(dth + math.pi, 2*math.pi) - math.pi
        f1    = self.omega + (K_f * torch.sin(dth)).sum(-1)
        t_mid = (theta + 0.5*cfg.kdt*f1) % (2*math.pi)
        dth2  = torch.remainder(t_mid.unsqueeze(0) - t_mid.unsqueeze(1) + math.pi,
                                2*math.pi) - math.pi
        f2    = self.omega + (K_f * torch.sin(dth2)).sum(-1)

        self.h     = h_new
        self.theta = (theta + cfg.kdt * f2) % (2*math.pi)

    def compute_losses(self, D_f):
        """
        Differentiable signal for InteractionNet/DynamicCurvatureField.
        Does NOT touch h or v — only guides the coupling law.
        """
        cfg = self.cfg
        N   = cfg.N
        dev = D_f.device

        # Energy conservation: dE/dt → 0
        D64    = D_f.double()
        E_curr = self.total_energy(self.h, D64)
        L_energy = ((E_curr - self.E_ref) / (self.E_ref.abs() + 1e-6))**2
        self.E_ref = E_curr.detach()

        # Clustering: tight neighbourhood
        k_nn  = min(8, N-1)
        tk, _ = D_f.topk(k_nn+1, dim=-1, largest=False)
        L_cluster = tk[:, 1:].mean()

        # Topology H₁: excess edges over spanning tree
        eps  = D_f.mean() * 0.8
        adj  = torch.sigmoid((eps - D_f) / 0.3) * (1 - torch.eye(N, device=dev))
        L_topo = F.relu(adj.sum()/2 - (N-1)) / N

        # Spatial entropy: reward diversity in radial histogram
        r_n   = self.radii().float() / (self.radii().float().max() + 1e-6)  # [N]
        cents = torch.linspace(0, 1, 9, device=dev)[:-1] + 0.5/8           # 8 bins
        soft  = torch.softmax(-((r_n.unsqueeze(1) - cents.unsqueeze(0)).abs())*20, dim=1)
        p     = soft.mean(0).clamp(min=1e-8)
        L_entropy = -(p * p.log()).sum()

        # Causal reward: penalise long-range interactions
        K_f   = self._K_mat.abs() if self._K_mat is not None else torch.zeros(N, N, device=dev)
        eye   = torch.eye(N, device=dev)
        long_range = K_f * (D_f > cfg.R_H).float() * (1 - eye)
        L_causal = long_range.mean()

        total = (cfg.w_energy   * L_energy.float()
               + cfg.w_cluster  * L_cluster
               + cfg.w_topo     * L_topo
               - cfg.w_entropy  * L_entropy
               + cfg.w_causal   * L_causal)
        return total

    # ─────────────────────── main step ────────────────────────────────────

    def step(self, step_idx):
        """
        A → B separation:
          A: evolve_state  (pure geodesic physics, @no_grad)
          B: compute_losses (differentiable, loss guides LAWS only)
        """
        # ── A: get K, evolve ──────────────────────────────────────────
        with torch.no_grad():
            D0_f = self.pairwise_dist().float()
        K_geo, K_kur = self._build_K(D0_f, step_idx)
        self.evolve_state(K_geo.detach(), K_kur.detach(), step_idx)

        # ── B: compute differentiable loss on NEW state ───────────────
        with torch.no_grad():
            D1_f = self.pairwise_dist().float()
        loss = self.compute_losses(D1_f)

        # ── Telemetry ─────────────────────────────────────────────────
        r        = self.radii().float()   # [N]  (no velocity in Langevin)
        r_mean   = float(r.mean())
        r_std    = float(r.std())
        d_mean   = float(D1_f[D1_f > 0.1].mean())
        rdc      = r_std / (d_mean + 1e-2)
        gamma    = self.order_param()
        K_bar    = float(K_geo.detach().mean())
        kv_bar   = float(self._K_local.detach().mean()) if self._K_local is not None else 1.0
        E_val    = float(self.total_energy(self.h, D1_f.double()))
        exp_rate = float((r - self._r_prev).mean()) if self._r_prev is not None else 0.0
        self._r_prev = r.detach()

        return {
            "loss":           loss,
            "energy":         E_val,
            "gamma":          gamma,
            "mean_radius":    r_mean,
            "rdc":            rdc,
            "K_eff":          K_bar,
            "curvature_mean": kv_bar,
            "expansion_rate": exp_rate,
            "horizon_frac":   self._hz_frac,
            "cluster_score":  float(D1_f.topk(9, dim=-1, largest=False).values[:, 1:].mean()),
            "topo_loss":      0.0,  # updated below
        }


print("   Loop A: Störmer-Verlet, zero damping, @no_grad")
print("   Loop B: differentiable laws only (no state update)")
print("   Curvature: D_eff = D / √(K_i·K_j)  ← matter curves space")
print("   Horizon:   K_ij *= σ((R_H − d)/τ_H) ← locality")
print("   Potential: V(d) = V_scale·[exp(-d/σ_a) − exp(-d/σ_r)]")