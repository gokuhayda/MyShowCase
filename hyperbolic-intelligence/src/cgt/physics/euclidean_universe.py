"""
cgt.physics.euclidean_universe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EuclideanUniverse: control baseline for HyDRA-Physics.

Identical dynamics to HyDRAUniverse but operating in flat R^n
instead of the Lorentz manifold H^n.

    Positions  : h_i ∈ R^n  (no time component)
    Distances  : d_ij = ‖h_i − h_j‖₂  (Euclidean)
    Force      : F_i = −∂V/∂d · (h_j − h_i)/d_ij
    Update     : h_new = h + dt·F + √dt·σ·noise  (Langevin)
    Potential  : V(d) = V_scale·[exp(−d/σ_a) − exp(−d/σ_r)]
    Phases     : identical Kuramoto (same K_kur, same kdt)

Same PhysicsConfig, same InteractionNet architecture,
same hyperparameters → clean ablation of geometry only.

New in V6. Does not modify any existing cgt modules.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cgt.physics.config import PhysicsConfig
from cgt.physics.interaction import InteractionNet, DynamicCurvatureField


class EuclideanUniverse(nn.Module):
    """Control baseline: same dynamics as HyDRAUniverse but in R^n."""

    _version = "euclidean-v1"

    def __init__(self, cfg: PhysicsConfig):
        super().__init__()
        self.cfg = cfg
        N, n = cfg.N, cfg.dim

        self.interact = InteractionNet(cfg)
        self.curv     = DynamicCurvatureField(cfg)

        # Initialise near LJ equilibrium
        sa, sr = cfg.sigma_a, cfg.sigma_r
        r_init = sa * sr / (sa - sr) * math.log(sa / sr) * 0.5

        h0 = torch.randn(N, n, dtype=torch.float64) * r_init
        self.register_buffer("h",     h0)
        self.register_buffer("theta", torch.rand(N) * 2 * math.pi)
        self.register_buffer("omega", torch.randn(N) * cfg.freq_std)
        self.register_buffer("E_ref", torch.tensor(0.0, dtype=torch.float64))

        self._K_local  = None
        self._hz_frac  = 0.0
        self._r_prev   = None

    # ── helpers ───────────────────────────────────────────────────────────

    def radii(self, h=None):
        if h is None: h = self.h
        return h.norm(dim=-1)                           # [N]

    def pairwise_dist(self, h=None):
        """[N,N] Euclidean pairwise distances."""
        if h is None: h = self.h
        N = h.shape[0]
        diff = h.unsqueeze(1) - h.unsqueeze(0)         # [N,N,n]
        D    = diff.norm(dim=-1)                        # [N,N]
        D_min = 0.08
        eye  = torch.eye(N, device=D.device, dtype=D.dtype)
        return D * (1 - eye) + D_min * ((D < D_min) & (eye < 0.5)).to(D.dtype)

    def local_density(self, D, sig=0.8):
        return torch.exp(-D.float()**2 / (2*sig**2)).sum(-1) - 1.0

    def order_param(self):
        return (torch.exp(1j * self.theta.float()).mean()).abs().item()

    def lj_potential(self, D):
        cfg  = self.cfg
        mask = 1 - torch.eye(D.shape[0], device=D.device, dtype=D.dtype)
        V = cfg.V_scale * (torch.exp(-D/cfg.sigma_a) - torch.exp(-D/cfg.sigma_r))
        return (V * mask).sum() * 0.5

    def total_energy(self, h, D):
        return self.lj_potential(D)

    # ── force from LJ gradient (Euclidean) ───────────────────────────────

    def _build_K(self, D_raw_f, step):
        cfg = self.cfg
        N   = D_raw_f.shape[0]
        eye = torch.eye(N, device=D_raw_f.device)

        # Dynamic curvature (optional — same as hyperbolic version)
        rho     = self.local_density(D_raw_f)
        K_local = self.curv(rho) if cfg.use_dyn_curv \
                  else torch.ones(N, device=D_raw_f.device)
        self._K_local = K_local
        Ki    = K_local.unsqueeze(1).expand(N, N)
        Kj    = K_local.unsqueeze(0).expand(N, N)
        D_eff = (D_raw_f / (Ki * Kj).clamp(min=1e-4).sqrt()).clamp(min=0.05)

        # K_geo: from LJ gradient
        sa, sr  = cfg.sigma_a, cfg.sigma_r
        dVdd    = torch.exp(-D_eff/sr)/sr - torch.exp(-D_eff/sa)/sa
        K_geo   = -dVdd * cfg.V_scale * (1 - eye)

        if cfg.learn_K and step >= cfg.learn_start:
            r   = self.radii().float()
            th  = self.theta.float()
            dth = (th.unsqueeze(0) - th.unsqueeze(1)).abs() % math.pi
            ri  = r.unsqueeze(1).expand(N, N)
            rj  = r.unsqueeze(0).expand(N, N)
            corr       = self.interact(D_eff, dth, ri, rj)
            corr_scale = corr / (cfg.K0_geo + 1e-6) * 2.0
            K_geo      = K_geo * corr_scale

        if cfg.use_horizon:
            mask          = torch.sigmoid((cfg.R_H - D_eff) / cfg.tau_H)
            self._hz_frac = float((mask < 0.5).float().mean())
            K_geo         = K_geo * mask
        else:
            self._hz_frac = 0.0

        K_kur = cfg.K0_kur * torch.exp(-D_raw_f / cfg.tau_kur) * (1 - eye)
        self._K_mat = K_geo
        return K_geo, K_kur

    def _force(self, h, K_geo):
        """F_i = Σ_j K_geo_ij · (h_j − h_i) / d_ij  (Euclidean direction)."""
        N    = h.shape[0]
        hi   = h.unsqueeze(1).expand(N, N, -1)         # [N,N,n]
        hj   = h.unsqueeze(0).expand(N, N, -1)
        diff = (hj - hi).double()                       # [N,N,n] direction vectors
        d    = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit = diff / d                                 # [N,N,n] unit direction
        F    = (K_geo.double().unsqueeze(-1) * unit).sum(1)  # [N,n]
        # Clamp magnitude
        F_norm  = F.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        F_scale = (torch.tensor(6.0, dtype=F.dtype, device=F.device)
                   / F_norm).clamp(max=1.0)
        return F * F_scale

    # ── Langevin step (no manifold) ───────────────────────────────────────

    @torch.no_grad()
    def evolve_state(self, K_geo, K_kur, step):
        """Overdamped Langevin in R^n (no exp/log map needed)."""
        cfg = self.cfg
        h   = self.h

        F = self._force(h, K_geo)                       # [N, n]

        if cfg.T_noise > 0:
            noise = (torch.randn_like(h) *
                     math.sqrt(cfg.dt * cfg.T_noise))   # [N, n]
        else:
            noise = torch.zeros_like(h)

        disp   = cfg.dt * F + noise.double()
        d_norm = disp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d_scale= (torch.tensor(0.5, dtype=disp.dtype, device=disp.device)
                  / d_norm).clamp(max=1.0)
        disp   = disp * d_scale

        h_new  = h + disp
        bad    = torch.isnan(h_new).any(dim=-1)
        h_new  = torch.where(bad.unsqueeze(-1).expand_as(h_new), h, h_new)

        # Kuramoto RK2
        theta = self.theta
        K_f   = K_kur.float()
        dth   = torch.remainder(theta.unsqueeze(0) - theta.unsqueeze(1) + math.pi,
                                2*math.pi) - math.pi
        f1    = self.omega + (K_f * torch.sin(dth)).sum(-1)
        t_mid = (theta + 0.5*cfg.kdt*f1) % (2*math.pi)
        dth2  = torch.remainder(t_mid.unsqueeze(0) - t_mid.unsqueeze(1) + math.pi,
                                2*math.pi) - math.pi
        f2    = self.omega + (K_f * torch.sin(dth2)).sum(-1)

        self.h     = h_new
        self.theta = (theta + cfg.kdt * f2) % (2*math.pi)

    # ── compute_losses (identical to HyDRAUniverse) ───────────────────────

    def compute_losses(self, D_f):
        cfg = self.cfg
        N, dev = cfg.N, D_f.device

        E_curr   = self.total_energy(self.h, D_f.double())
        L_energy = ((E_curr - self.E_ref) / (self.E_ref.abs() + 1e-6))**2
        self.E_ref = E_curr.detach()

        k_nn  = min(8, N-1)
        tk, _ = D_f.topk(k_nn+1, dim=-1, largest=False)
        L_cluster = tk[:, 1:].mean()

        eps  = D_f.mean() * 0.8
        adj  = torch.sigmoid((eps - D_f) / 0.3) * (1 - torch.eye(N, device=dev))
        L_topo = F.relu(adj.sum()/2 - (N-1)) / N

        r_n   = self.radii().float() / (self.radii().float().max() + 1e-6)
        cents = torch.linspace(0, 1, 9, device=dev)[:-1] + 0.5/8
        soft  = torch.softmax(-((r_n.unsqueeze(1) - cents.unsqueeze(0)).abs())*20, dim=1)
        p     = soft.mean(0).clamp(min=1e-8)
        L_entropy = -(p * p.log()).sum()

        K_f      = self._K_mat.abs() if self._K_mat is not None \
                   else torch.zeros(N, N, device=dev)
        L_causal = (K_f * (D_f > cfg.R_H).float() *
                    (1 - torch.eye(N, device=dev))).mean()

        return (cfg.w_energy * L_energy.float()
              + cfg.w_cluster * L_cluster
              + cfg.w_topo    * L_topo
              - cfg.w_entropy * L_entropy
              + cfg.w_causal  * L_causal)

    # ── main step ─────────────────────────────────────────────────────────

    def step(self, step_idx):
        with torch.no_grad():
            D0_f = self.pairwise_dist().float()
        K_geo, K_kur = self._build_K(D0_f, step_idx)
        self.evolve_state(K_geo.detach(), K_kur.detach(), step_idx)

        with torch.no_grad():
            D1_f = self.pairwise_dist().float()
        loss = self.compute_losses(D1_f)

        r        = self.radii().float()
        E_val    = float(self.total_energy(self.h, D1_f.double()))
        exp_rate = float((r - self._r_prev).mean()) if self._r_prev is not None else 0.0
        self._r_prev = r.detach()

        return {
            "loss":           loss,
            "energy":         E_val,
            "gamma":          self.order_param(),
            "mean_radius":    float(r.mean()),
            "rdc":            float(r.std()) / (float(D1_f[D1_f > 0.1].mean()) + 1e-2),
            "K_eff":          float(K_geo.detach().mean()),
            "curvature_mean": float(self._K_local.detach().mean()) if self._K_local is not None else 1.0,
            "expansion_rate": exp_rate,
            "horizon_frac":   self._hz_frac,
            "cluster_score":  float(D1_f.topk(9, dim=-1, largest=False).values[:, 1:].mean()),
            "topo_loss":      0.0,
        }
