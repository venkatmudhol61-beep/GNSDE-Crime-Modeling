"""
GN-SDE: Graph Neural Stochastic Differential Equations for Crime Forecasting
Matches paper equations exactly:
  fc:      eq (15)  f = C(1-C)[alpha + Phi([AC,h])] - beta*L*C
  spatial: eq (18)  f = C(1-C)[alpha + Phi(phi_sp)] + rho*grad1 - beta*L*C
  attn:    eq (21)  f = C(1-C)[alpha + Phi(phi_sp)] + rho*grad1 - beta*m_beta*L*C
  latent:  eq (25)  f = C(1-C)[alpha + Psi(phi_C)] - beta*L_lat*C   (no rho term)
"""

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


def _sigmoid_inv(x: float) -> float:
    x = float(np.clip(x, 1e-6, 1 - 1e-6))
    return math.log(x / (1.0 - x))


def _safe_A(A) -> torch.Tensor:
    if isinstance(A, torch.Tensor):
        return A.clone().detach().float()
    return torch.tensor(A, dtype=torch.float32)


def _row_normalise(A: torch.Tensor) -> torch.Tensor:
    return A / A.sum(dim=1, keepdim=True).clamp(min=1.0)


def enrich_spatial_adjacency(A_border, coords, k=3, sigma=1.0):
    from scipy.spatial.distance import cdist
    N          = A_border.shape[0]
    D          = cdist(coords, coords)
    A_enriched = A_border.copy().astype(float)
    for i in range(N):
        for j in np.argsort(D[i])[1:k + 1]:
            if A_enriched[i, j] == 0:
                w = math.exp(-D[i, j] / sigma)
                A_enriched[i, j] = w
                A_enriched[j, i] = w
    return A_enriched


class _Memory(nn.Module):
    def __init__(self, N: int, mem_dim: int = 16):
        super().__init__()
        self.N       = N
        self.mem_dim = mem_dim
        self.gru     = nn.GRUCell(input_size=2, hidden_size=mem_dim)
        self.h: Optional[torch.Tensor] = None

    def reset(self, device) -> None:
        self.h = torch.zeros(self.N, self.mem_dim, device=device)

    def step(self, C: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        self.h = self.gru(torch.stack([C, L], dim=1), self.h)
        return self.h

    def detach(self) -> None:
        if self.h is not None:
            self.h = self.h.detach()

    def get(self, device) -> torch.Tensor:
        if self.h is None:
            return torch.zeros(self.N, self.mem_dim, device=device)
        return self.h.detach()


class _BaseSDE(nn.Module):
    noise_type = "diagonal"
    sde_type   = "ito"

    def smoothness_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)
    def L_supervision_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)
    def L_regularisation_loss(self, _):
        return torch.tensor(0.0, device=next(self.parameters()).device)
    def L_anchor_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)
    def set_observed_schedule(self, _): pass
    def update_L(self, C):
        return getattr(self, '_L', None)

    def pred_var(self, C_pred: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_sigma_pred * 2).clamp(min=1e-6)


# ─────────────────────────────────────────────────────────────────────
# 1. GNSDEfc — eq (15)
# ─────────────────────────────────────────────────────────────────────

class GNSDEfc(_BaseSDE):
    """
    f_fc_i = C_i(1-C_i)[alpha_i + Phi([(AC)_i, h_i])] - beta_i L_i(t) C_i
    Phi: single-hidden-layer Tanh MLP, input dim = 1 + mem_dim.
    """

    def __init__(self, A, alpha=0.3, beta=0.6,
                 hidden=32, mem_dim=16, dropout=0.1):
        super().__init__()
        A_t    = _safe_A(A)
        A_norm = _row_normalise(A_t)
        self.register_buffer("A", A_norm)
        N      = A_t.shape[0]
        self.N = N

        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))
        self.raw_sigma = nn.Parameter(torch.full((N,), _sigmoid_inv(0.10)))
        self.log_sigma_pred = nn.Parameter(torch.full((N,), math.log(0.03)))

        # Phi: single hidden layer, input [AC, h] in R^{1+mem_dim}
        self.f_net = nn.Sequential(
            nn.Linear(1 + mem_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.f_net:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        self.memory = _Memory(N, mem_dim)
        self._L     : Optional[torch.Tensor] = None

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)
    @property
    def sigma(self): return torch.sigmoid(self.raw_sigma) * 0.48 + 0.02

    def set_context(self, L, t_idx=0, T=1):
        self._L = L.view(-1)

    def f(self, t, C):
        C   = C.view(-1)
        AC  = torch.mv(self.A, C)
        mem = self.memory.get(C.device)
        phi = self.f_net(torch.cat([AC.unsqueeze(-1), mem], dim=1)).squeeze(-1)
        # eq (15): C(1-C)*[alpha + Phi] - beta*L*C
        dC = (C * (1 - C) * (self.alpha + phi)
              - self.beta * self._L * C)
        return dC.unsqueeze(0)

    def g(self, t, C):
        C = C.view(-1).clamp(0.0, 1.0)
        return (self.sigma * C * (1 - C)).unsqueeze(0)

    def step_memory(self, C, L=None):
        L_ = L if L is not None else self._L
        with torch.no_grad():
            self.memory.step(C.detach().view(-1), L_.view(-1))

    def reset_memory(self, device):
        self.memory.reset(device)

    def param_summary(self):
        a  = self.alpha.detach()
        b  = self.beta.detach()
        s  = self.sigma.detach()
        ps = torch.exp(self.log_sigma_pred).detach()
        print(f"[GNSDEfc]  alpha:{a.mean():.4f}  beta:{b.mean():.4f}  "
              f"sigma:{s.mean():.4f}  sigma_pred:{ps.mean():.4f}")
        print(f"[GNSDEfc]  drift: C(1-C)[alpha+Phi([AC,h])] - beta*L*C  (eq 15)")


# ─────────────────────────────────────────────────────────────────────
# 2. GNSDEspatial — eq (18)
# ─────────────────────────────────────────────────────────────────────

class GNSDEspatial(_BaseSDE):
    """
    f_sp_i = C_i(1-C_i)[alpha_i + Phi(phi_sp_i)] + rho*grad1_i - beta_i L_i(t) C_i
    phi_sp = [C, (Asp C), (A2 C), grad_feats(3, layernormed), h] in R^{6+mem_dim}
    Phi: two-hidden-layer Tanh MLP. rho = 0.1*sigmoid(raw_rho) in [0,0.1].
    """

    def __init__(self, A, alpha=0.3, beta=0.6,
                 hidden=32, mem_dim=16, dropout=0.1):
        super().__init__()
        A_t = _safe_A(A)
        A_t.fill_diagonal_(0.0)
        A_norm = _row_normalise(A_t)
        self.register_buffer("A", A_norm)

        A2_raw = A_norm @ A_norm
        A2_raw.fill_diagonal_(0.0)
        self.register_buffer("A2", _row_normalise(A2_raw))

        N      = A_t.shape[0]
        self.N = N

        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))
        self.raw_sigma = nn.Parameter(torch.full((N,), _sigmoid_inv(0.10)))
        self.log_sigma_pred = nn.Parameter(torch.full((N,), math.log(0.03)))
        self.raw_rho = nn.Parameter(torch.tensor(0.0))
        self.grad_norm = nn.LayerNorm(3, elementwise_affine=False)

        self.f_net = nn.Sequential(
            nn.Linear(6 + mem_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),       nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.f_net:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        self.memory = _Memory(N, mem_dim)
        self._L     : Optional[torch.Tensor] = None

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)
    @property
    def sigma(self): return torch.sigmoid(self.raw_sigma) * 0.48 + 0.02
    @property
    def rho(self):   return torch.sigmoid(self.raw_rho) * 0.1

    def set_context(self, L, t_idx=0, T=1):
        self._L = L.view(-1)

    def f(self, t, C):
        C   = C.view(-1)
        AC  = torch.mv(self.A,  C)
        A2C = torch.mv(self.A2, C)

        grad1     = AC - C
        grad2     = C  - A2C
        laplacian = (C - AC).pow(2)

        grad_feats = self.grad_norm(
            torch.stack([grad1, grad2, laplacian], dim=1))

        mem  = self.memory.get(C.device)
        feat = torch.cat([
            C.unsqueeze(-1), AC.unsqueeze(-1), A2C.unsqueeze(-1),
            grad_feats, mem
        ], dim=1)
        phi = self.f_net(feat).squeeze(-1)

        # eq (18): C(1-C)*[alpha + Phi] + rho*grad1 - beta*L*C
        dC = (C * (1 - C) * (self.alpha + phi)
              + self.rho * grad1
              - self.beta * self._L * C)
        return dC.unsqueeze(0)

    def g(self, t, C):
        C = C.view(-1).clamp(0.0, 1.0)
        return (self.sigma * C * (1 - C)).unsqueeze(0)

    def step_memory(self, C, L=None):
        L_ = L if L is not None else self._L
        with torch.no_grad():
            self.memory.step(C.detach().view(-1), L_.view(-1))

    def reset_memory(self, device):
        self.memory.reset(device)

    def param_summary(self):
        a  = self.alpha.detach()
        b  = self.beta.detach()
        s  = self.sigma.detach()
        ps = torch.exp(self.log_sigma_pred).detach()
        rh = self.rho.detach().item()
        print(f"[GNSDEspatial]  alpha:{a.mean():.4f}  beta:{b.mean():.4f}  "
              f"sigma:{s.mean():.4f}  sigma_pred:{ps.mean():.4f}  rho:{rh:.4f}")
        print(f"[GNSDEspatial]  drift: C(1-C)[alpha+Phi(phi_sp)] + rho*grad1 - beta*L*C  (eq 18)")


# ─────────────────────────────────────────────────────────────────────
# 3. GNSDEspatial_attention — eq (21)
# ─────────────────────────────────────────────────────────────────────

class GNSDEspatial_attention(_BaseSDE):
    """
    f_att_i = C_i(1-C_i)[alpha_i + Phi(phi_sp_i)] + rho*grad1_i
              - beta_i * m_beta,i * L_i(t) * C_i
    phi_sp identical to GNSDEspatial. m_beta via cross-region attention (eq 19-20).
    """

    def __init__(self, A, T=1, hidden=32, hidden_dim=64,
                 alpha=0.3, beta=0.6, n_heads=4,
                 mem_dim=16, dropout=0.1):
        super().__init__()
        A_t = _safe_A(A)

        A_sp = A_t.clone()
        A_sp.fill_diagonal_(0.0)
        A_sp_norm = _row_normalise(A_sp)
        self.register_buffer("A_spatial", A_sp_norm)
        A2_raw = A_sp_norm @ A_sp_norm
        A2_raw.fill_diagonal_(0.0)
        self.register_buffer("A2_spatial", _row_normalise(A2_raw))

        A_self = A_t.clone()
        A_self.fill_diagonal_(1.0)
        self.register_buffer("mask",  A_self > 0)
        self.register_buffer("mask_norm", _row_normalise(A_self.float()))
        self.register_buffer("scale", torch.sqrt(torch.tensor(float(hidden_dim))))

        N      = A_t.shape[0]
        self.N = N
        self.T = max(T, 1)

        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))
        self.raw_sigma = nn.Parameter(torch.full((N,), _sigmoid_inv(0.10)))
        self.log_sigma_pred = nn.Parameter(torch.full((N,), math.log(0.03)))
        self.raw_rho   = nn.Parameter(torch.tensor(0.0))

        self.f_net = nn.Sequential(
            nn.Linear(6 + mem_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),       nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.f_net:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        self.region_emb = nn.Embedding(N, 8)
        self.Wq = nn.Linear(13, hidden_dim)
        self.Wk = nn.Linear(13, hidden_dim)
        self.Wv = nn.Linear(13, hidden_dim)
        self.beta_mod_head = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
        for layer in [self.Wq, self.Wk, self.Wv]:
            layer.weight.data *= 0.5
            if layer.bias is not None:
                layer.bias.data.zero_()
        self.beta_mod_head[0].weight.data.zero_()
        self.beta_mod_head[0].bias.data.fill_(2.0)

        self.memory = _Memory(N, mem_dim)
        self._L     : Optional[torch.Tensor] = None
        self._t_idx : int = 0

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)
    @property
    def sigma(self): return torch.sigmoid(self.raw_sigma) * 0.48 + 0.02
    @property
    def rho(self):   return torch.sigmoid(self.raw_rho) * 0.1   # matches [0,0.1] as in spatial

    def set_context(self, L, t_idx=0, T=None):
        if T is not None:
            self.T = max(T, 1)
        self._L     = L.view(-1)
        self._t_idx = t_idx

    def _beta_modulator(self, L: torch.Tensor) -> torch.Tensor:
        device = L.device
        t_norm = self._t_idx / self.T
        agg_L  = torch.mv(self.mask_norm, L)
        emb    = self.region_emb(torch.arange(self.N, device=device))
        sin_t  = L.new_full((self.N, 1), math.sin(2 * math.pi * t_norm))
        cos_t  = L.new_full((self.N, 1), math.cos(2 * math.pi * t_norm))
        t_feat = L.new_full((self.N, 1), t_norm)
        feat   = torch.cat([L.unsqueeze(-1), agg_L.unsqueeze(-1),
                            emb, sin_t, cos_t, t_feat], dim=1)
        Q      = self.Wq(feat)
        K      = self.Wk(feat)
        V      = self.Wv(feat)
        scores = torch.matmul(Q, K.T) / self.scale
        scores = scores.masked_fill(~self.mask, -1e9)
        ctx    = torch.matmul(torch.softmax(scores, dim=1), V)
        return self.beta_mod_head(ctx).squeeze(-1)

    def f(self, t, C):
        C   = C.view(-1).clamp(0.0, 1.0)
        AC  = torch.mv(self.A_spatial, C)
        A2C = torch.mv(self.A2_spatial, C)

        grad1     = AC - C
        grad2     = C  - A2C
        laplacian = (C - AC).pow(2)

        mem  = self.memory.get(C.device)
        feat = torch.cat([C.unsqueeze(-1), AC.unsqueeze(-1),
                          A2C.unsqueeze(-1), grad1.unsqueeze(-1),
                          grad2.unsqueeze(-1), laplacian.unsqueeze(-1),
                          mem], dim=1)
        phi = self.f_net(feat).squeeze(-1)

        m_beta = self._beta_modulator(self._L)

        # eq (21): C(1-C)*[alpha + Phi] + rho*grad1 - beta*m_beta*L*C
        dC = (C * (1 - C) * (self.alpha + phi)
              + self.rho * grad1
              - self.beta * m_beta * self._L * C)
        return torch.nan_to_num(
            dC, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0)

    def g(self, t, C):
        C = C.view(-1).clamp(0.0, 1.0)
        return (self.sigma * C * (1 - C)).unsqueeze(0)

    def step_memory(self, C, L=None):
        L_ = L if L is not None else self._L
        with torch.no_grad():
            self.memory.step(C.detach().view(-1), L_.view(-1))

    def reset_memory(self, device):
        self.memory.reset(device)

    def param_summary(self):
        a  = self.alpha.detach()
        b  = self.beta.detach()
        s  = self.sigma.detach()
        ps = torch.exp(self.log_sigma_pred).detach()
        rh = self.rho.detach().item()
        print(f"[GNSDEattn]  alpha:{a.mean():.4f}  beta:{b.mean():.4f}  "
              f"sigma:{s.mean():.4f}  sigma_pred:{ps.mean():.4f}  rho:{rh:.4f}")
        print(f"[GNSDEattn]  drift: C(1-C)[alpha+Phi(phi_sp)] + rho*grad1 "
              f"- beta*m_beta*L*C  (eq 21)")


# ─────────────────────────────────────────────────────────────────────
# 4. GNSDElatent — eq (22)-(25)
# ─────────────────────────────────────────────────────────────────────

class GNSDElatent(_BaseSDE):
    """
    Latent enforcement (22)-(23):
      L_net_i = G(phi_L)*L_theta(phi_L) + (1-G(phi_L))*0.5
      L_lat_i = omega_i*L_i(t) + (1-omega_i)*L_net_i
      phi_L = [sin,cos,tbar, e_i(16), (Asp C)_i, h_L_i] in R^{20+dm}

    Crime drift (24)-(25):
      phi_C = [C,(AspC),(A2C),Llat,(AspLlat),dC,dL,sin,cos,h_C] in R^{9+dm}
      f_lat_i = C_i(1-C_i)[alpha_i + Psi(phi_C_i)] - beta_i*Llat_i*C_i
      (NOTE: no explicit rho term in latent — matches eq 25 exactly)

    L_supervision_loss() is an AUXILIARY TRAINING LOSS, not part of the
    drift f() or the theorem. It gives the enforcement network a direct,
    C-independent gradient path (Huber(L_lat, L_obs)) so it keeps
    learning on region-months where C(1-C) vanishes. It does not modify
    equations (22)-(25) in any way.
    """

    def __init__(self, A, T: int,
                 alpha=0.3, beta=0.6,
                 hidden_dim=32, mem_dim=16,
                 dropout=0.1):
        super().__init__()

        A_t = _safe_A(A)
        A_t.fill_diagonal_(0.0)
        A_norm = _row_normalise(A_t.clone())
        self.register_buffer("A", A_norm)
        A2_raw = A_norm @ A_norm
        A2_raw.fill_diagonal_(0.0)
        self.register_buffer("A2", _row_normalise(A2_raw))

        N      = A_t.shape[0]
        self.N = N
        self.T = max(T, 1)

        self.raw_alpha = nn.Parameter(
            torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(
            torch.full((N,), _sigmoid_inv(beta / 2.0)))
        self.raw_sigma = nn.Parameter(
            torch.full((N,), _sigmoid_inv(0.10)))
        self.log_sigma_pred = nn.Parameter(
            torch.full((N,), math.log(0.03)))

        self.node_emb = nn.Embedding(N, 16)   # e_i in R^16

        self.L_net = nn.Sequential(
            nn.Linear(20 + mem_dim, 64), nn.Tanh(),
            nn.Linear(64, 32),            nn.Tanh(),
            nn.Linear(32, 16),            nn.Tanh(),
            nn.Linear(16,  1),            nn.Sigmoid(),
        )
        self.L_gate = nn.Sequential(
            nn.Linear(20 + mem_dim, 16), nn.Tanh(),
            nn.Linear(16, 1),             nn.Sigmoid(),
        )

        self.raw_obs_gate = nn.Parameter(torch.zeros(N))

        self.crime_mlp = nn.Sequential(
            nn.Linear(9 + mem_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),   nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.crime_mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        self.mem_C = _Memory(N, mem_dim)
        self.mem_L = _Memory(N, mem_dim)
        self.memory   = self.mem_C
        self.memory_L = self.mem_L

        self._t_idx       : int                    = 0
        self._L_obs       : Optional[torch.Tensor] = None
        self._L_latent    : Optional[torch.Tensor] = None
        self._last_L_traj : Optional[torch.Tensor] = None

        self._huber_L = nn.HuberLoss(delta=0.1)   # for auxiliary loss only

    @property
    def alpha(self):    return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):     return 2.0 * torch.sigmoid(self.raw_beta)
    @property
    def sigma(self):    return torch.sigmoid(self.raw_sigma) * 0.48 + 0.02
    @property
    def obs_gate(self): return torch.sigmoid(self.raw_obs_gate)

    def pred_var(self, C_pred: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_sigma_pred * 2).clamp(min=1e-6)

    def set_context(self, L, t_idx=0, T=None):
        if T is not None:
            self.T = max(T, 1)
        self._t_idx = t_idx
        self._L_obs = L.view(-1)

    def set_observed_schedule(self, L_full): pass

    def _L_features(self, C: torch.Tensor) -> torch.Tensor:
        # phi_L = [sin, cos, tbar, e_i, (AC)_i, h_L_i]
        device = C.device
        t_norm = self._t_idx / self.T
        time_feat = torch.tensor(
            [math.sin(2 * math.pi * t_norm),
             math.cos(2 * math.pi * t_norm),
             t_norm],
            dtype=torch.float32, device=device
        ).unsqueeze(0).expand(self.N, -1)
        node_feat = self.node_emb(torch.arange(self.N, device=device))
        AC    = torch.mv(self.A, C).unsqueeze(-1)
        mem_L = self.mem_L.get(device)
        return torch.cat([time_feat, node_feat, AC, mem_L], dim=1)

    def _compute_L(self, C: torch.Tensor) -> torch.Tensor:
        feat   = self._L_features(C)
        L_base = self.L_net(feat).squeeze(-1)
        gate   = self.L_gate(feat).squeeze(-1)
        L_net  = gate * L_base + (1 - gate) * 0.5      # eq (22)
        if self._L_obs is not None:
            og = self.obs_gate
            return og * self._L_obs + (1 - og) * L_net # eq (23)
        return L_net

    def _crime_features(self, C: torch.Tensor,
                         L: torch.Tensor,
                         t_norm: float) -> torch.Tensor:
        # phi_C = [C, AC, A2C, Llat, (A Llat), dC, dL, sin, cos, h_C]
        AC    = torch.mv(self.A,  C)
        A2C   = torch.mv(self.A2, C)
        AL    = torch.mv(self.A,  L)
        sin_t = C.new_full((self.N,), math.sin(2 * math.pi * t_norm))
        cos_t = C.new_full((self.N,), math.cos(2 * math.pi * t_norm))
        mem   = self.mem_C.get(C.device)
        return torch.cat([
            C.unsqueeze(-1), AC.unsqueeze(-1), A2C.unsqueeze(-1),
            L.unsqueeze(-1), AL.unsqueeze(-1),
            (C - AC).unsqueeze(-1), (L - AL).unsqueeze(-1),
            sin_t.unsqueeze(-1), cos_t.unsqueeze(-1),
            mem,
        ], dim=1)

    def f(self, t, C):
        C = C.view(-1).clamp(0.0, 1.0)
        if torch.isnan(C).any() or torch.isinf(C).any():
            return torch.zeros(1, self.N, device=C.device)
        t_norm = self._t_idx / self.T
        L_t    = self._compute_L(C)
        self._L_latent    = L_t
        self._last_L_traj = L_t.detach().unsqueeze(0)
        c_feat  = self._crime_features(C, L_t, t_norm)
        psi = self.crime_mlp(c_feat).squeeze(-1)

        # eq (25): C(1-C)*[alpha + Psi] - beta*Llat*C   (NO rho term)
        dC = (C * (1 - C) * (self.alpha + psi)
              - self.beta * L_t * C)
        return torch.nan_to_num(
            dC, nan=0.0, posinf=0.0, neginf=0.0
        ).unsqueeze(0)

    def g(self, t, C):
        C = C.view(-1).clamp(0.0, 1.0)
        return torch.nan_to_num(
            (self.sigma * C * (1 - C)).unsqueeze(0), nan=0.0)

    def step_memory(self, C, L=None):
        C_   = C.detach().view(-1)
        Lobs = self._L_obs.detach().view(-1) \
               if self._L_obs is not None \
               else torch.zeros(self.N, device=C_.device)
        Llat = self._L_latent.detach().view(-1) \
               if self._L_latent is not None \
               else torch.zeros(self.N, device=C_.device)
        with torch.no_grad():
            self.mem_C.step(C_, Lobs)
            self.mem_L.step(C_, Llat)

    def reset_memory(self, device):
        self.mem_C.reset(device)
        self.mem_L.reset(device)

    def update_L(self, C):
        with torch.no_grad():
            return self._compute_L(C.view(-1))

    @property
    def L(self):
        device  = self.node_emb.weight.device
        old_idx = self._t_idx
        old_obs = self._L_obs
        self._L_obs = None
        rows = []
        dummy = torch.zeros(self.N, device=device)
        for t in range(self.T):
            self._t_idx = t
            rows.append(self._compute_L(dummy).detach())
        self._t_idx = old_idx
        self._L_obs = old_obs
        return torch.stack(rows, dim=0)

    def L_regularisation_loss(self, L_traj):
        if L_traj is None:
            return torch.tensor(0.0, device=self.raw_alpha.device)
        dev    = L_traj.device
        smooth = (L_traj[1:] - L_traj[:-1]).pow(2).mean() \
                 if L_traj.size(0) > 1 \
                 else torch.tensor(0.0, device=dev)
        lower   = torch.relu(0.05 - L_traj).pow(2).mean()
        upper   = torch.relu(L_traj - 0.95).pow(2).mean()
        AL      = torch.mv(self.A, L_traj[0])
        spatial = (L_traj[0] - AL).pow(2).mean()
        return smooth + 5.0 * (lower + upper) + 0.1 * spatial

    def L_supervision_loss(self):
        """Auxiliary training loss ONLY — not part of eq (22)-(25) or f().
        Gives node_emb/L_net/L_gate/obs_gate a gradient independent of C,
        since both paths inside f() (the gated Psi term and -beta*Llat*C)
        vanish when C=0."""
        if self._L_latent is None or self._L_obs is None:
            return torch.tensor(0.0, device=self.raw_alpha.device)
        return self._huber_L(self._L_latent, self._L_obs)

    def param_summary(self):
        a  = self.alpha.detach()
        b  = self.beta.detach()
        s  = self.sigma.detach()
        og = self.obs_gate.detach()
        ps = torch.exp(self.log_sigma_pred).detach()
        print(f"[GNSDElatent]  alpha:{a.mean():.4f}  beta:{b.mean():.4f}  "
              f"sigma:{s.mean():.4f}  sigma_pred:{ps.mean():.4f}  "
              f"obs_gate:{og.mean():.4f}")
        print(f"[GNSDElatent]  drift: C(1-C)[alpha+Psi(phi_C)] - beta*Llat*C  (eq 25)")
        if self._L_latent is not None:
            L = self._L_latent.detach()
            print(f"[GNSDElatent]  L_latent min:{L.min():.4f} "
                  f"max:{L.max():.4f} mean:{L.mean():.4f}")

    def get_learned_enforcement(self, nodes, months):
        L_np = self.L.cpu().numpy()
        rows = []
        for t, m in enumerate(months):
            for i, n in enumerate(nodes):
                rows.append({"month": m, "region_id": n,
                             "L_latent": float(L_np[t, i])})
        return pd.DataFrame(rows)


MODEL_MAP = {
    "fc"           : GNSDEfc,
    "spatial"      : GNSDEspatial,
    "spatial_attn" : GNSDEspatial_attention,
    "latent"       : GNSDElatent,
}