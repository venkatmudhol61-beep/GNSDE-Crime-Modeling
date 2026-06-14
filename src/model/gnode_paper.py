"""
GN-ODE Models — converted from GN-SDE Final
Hierarchy: latent >= spatial_attn >= spatial >= fc

Changes from GN-SDE:
  - Removed SDE noise: no sigma, no g(), no noise_type/sde_type
  - Removed GRU memory: no _Memory, no mem_dim, no step_memory/reset_memory
  - Reduced GNN input dims (no mem appended)
  - All variants use ODE solver only
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


# ─────────────────────────────────────────────────────────────────────
# Common base
# ─────────────────────────────────────────────────────────────────────

class _BaseODE(nn.Module):
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
    def ode_func(self, t, C):
        return self.f(t, C.unsqueeze(0)).squeeze(0)


# ─────────────────────────────────────────────────────────────────────
# 1. GNODEfc
# ─────────────────────────────────────────────────────────────────────

class GNODEfc(_BaseODE):
    """
    dC/dt = α C(1-C) + gnn(AC) - β L C

    gnn: Linear(1)->hidden->Tanh->Linear->1
    """

    def __init__(self, A, alpha=0.3, beta=0.6,
                 hidden=16, dropout=0.1):
        super().__init__()
        A_t    = _safe_A(A)
        A_norm = _row_normalise(A_t)
        self.register_buffer("A", A_norm)
        N      = A_t.shape[0]
        self.N = N

        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))

        self.gnn = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.gnn:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        self._L      : Optional[torch.Tensor] = None
        self._t_norm : float = 0.0

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)

    def set_context(self, L, t_idx=0, T=1):
        self._L      = L.view(-1)
        self._t_norm = t_idx / max(T, 1)

    def f(self, t, C):
        C    = C.view(-1)
        AC   = torch.mv(self.A, C)
        feat = AC.unsqueeze(-1)
        dC   = (self.alpha * C * (1 - C)
                + self.gnn(feat).squeeze(-1)
                - self.beta * self._L * C)
        return dC.unsqueeze(0)

    def param_summary(self):
        a, b = self.alpha.detach(), self.beta.detach()
        print(f"[GNODEfc]  alpha min:{a.min():.4f} max:{a.max():.4f} mean:{a.mean():.4f}")
        print(f"[GNODEfc]  beta  min:{b.min():.4f} max:{b.max():.4f} mean:{b.mean():.4f}")


# ─────────────────────────────────────────────────────────────────────
# 2. GNODEspatial
# ─────────────────────────────────────────────────────────────────────

class GNODEspatial(_BaseODE):
    """
    dC/dt = α C(1-C) + gnn(C, AC) - β L C

    gnn: Linear(2)->hidden->Tanh->Linear->1
    No self-loops in A.
    """

    def __init__(self, A, alpha=0.3, beta=0.6,
                 hidden=16, dropout=0.1):
        super().__init__()
        A_t = _safe_A(A)
        A_t.fill_diagonal_(0.0)
        A_norm = _row_normalise(A_t)
        self.register_buffer("A", A_norm)
        N      = A_t.shape[0]
        self.N = N

        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))

        self.gnn = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.gnn:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        self._L      : Optional[torch.Tensor] = None
        self._t_norm : float = 0.0

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)

    def set_context(self, L, t_idx=0, T=1):
        self._L      = L.view(-1)
        self._t_norm = t_idx / max(T, 1)

    def f(self, t, C):
        C    = C.view(-1)
        AC   = torch.mv(self.A, C)
        feat = torch.cat([C.unsqueeze(-1), AC.unsqueeze(-1)], dim=1)
        dC   = (self.alpha * C * (1 - C)
                + self.gnn(feat).squeeze(-1)
                - self.beta * self._L * C)
        return dC.unsqueeze(0)

    def param_summary(self):
        a, b = self.alpha.detach(), self.beta.detach()
        print(f"[GNODEspatial]  alpha min:{a.min():.4f} max:{a.max():.4f} mean:{a.mean():.4f}")
        print(f"[GNODEspatial]  beta  min:{b.min():.4f} max:{b.max():.4f} mean:{b.mean():.4f}")
        print(f"[GNODEspatial]  edges:{(self.A > 0).sum().item()}")


# ─────────────────────────────────────────────────────────────────────
# 3. GNODEspatial_attention
# ─────────────────────────────────────────────────────────────────────

class GNODEspatial_attention(_BaseODE):
    """
    Uses ODE solver (stable for attention weights).

    Branch 1: crime diffusion gnn([C, AC])
    Branch 2: attention on L -> alpha_mod, beta_mod per node
    dC/dt = α*alpha_mod*C(1-C) + gnn - β*beta_mod*L*C
    """

    def __init__(self, A, T=1, hidden=16, hidden_dim=32,
                 alpha=0.3, beta=0.6, n_heads=4, dropout=0.1):
        super().__init__()
        A_t = _safe_A(A)

        A_sp = A_t.clone()
        A_sp.fill_diagonal_(0.0)
        self.register_buffer("A_spatial", _row_normalise(A_sp))

        A_self = A_t.clone()
        A_self.fill_diagonal_(1.0)
        self.register_buffer("A_attn", A_self)
        self.register_buffer("mask",   A_self > 0)
        self.register_buffer("scale",
            torch.sqrt(torch.tensor(float(hidden_dim))))

        N      = A_t.shape[0]
        self.N = N
        self.T = max(T, 1)

        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))

        # Branch 1: spatial crime GNN — input [C, AC]
        self.spatial_gnn = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.spatial_gnn:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        # Branch 2: attention on L
        # input: [L(1), agg_L(1), emb(8), sin(1), cos(1), t(1)] = 13
        self.region_emb = nn.Embedding(N, 8)
        self.Wq         = nn.Linear(13, hidden_dim)
        self.Wk         = nn.Linear(13, hidden_dim)
        self.Wv         = nn.Linear(13, hidden_dim)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.beta_head  = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        for layer in [self.Wq, self.Wk, self.Wv]:
            layer.weight.data *= 0.5
            if layer.bias is not None:
                layer.bias.data.zero_()

        # bias=2.0 -> sigmoid(2)~0.88 at init
        for head in [self.alpha_head, self.beta_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.zero_()
                    if layer.bias is not None:
                        layer.bias.data.fill_(2.0)

        self._L      : Optional[torch.Tensor] = None
        self._t_norm : float = 0.0
        self._t_idx  : int   = 0

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)

    def set_context(self, L, t_idx=0, T=None):
        if T is not None:
            self.T = max(T, 1)
        self._L      = L.view(-1)
        self._t_norm = t_idx / self.T
        self._t_idx  = t_idx

    def _attn_features(self, L):
        device = L.device
        t_norm = self._t_idx / self.T
        agg_L  = torch.mv(self.A_attn, L)
        emb    = self.region_emb(torch.arange(self.N, device=device))
        sin_t  = L.new_full((self.N, 1), math.sin(2 * math.pi * t_norm))
        cos_t  = L.new_full((self.N, 1), math.cos(2 * math.pi * t_norm))
        t_feat = L.new_full((self.N, 1), t_norm)
        return torch.cat([L.unsqueeze(-1), agg_L.unsqueeze(-1),
                          emb, sin_t, cos_t, t_feat], dim=1)  # (N,13)

    def f(self, t, C):
        C = C.view(-1)
        L = self._L

        # Branch 1: crime diffusion
        AC = torch.mv(self.A_spatial, C)
        crime_term = self.spatial_gnn(
            torch.cat([C.unsqueeze(-1), AC.unsqueeze(-1)], dim=1)
        ).squeeze(-1)

        # Branch 2: attention on L
        feat   = self._attn_features(L)
        Q      = self.Wq(feat)
        K      = self.Wk(feat)
        V      = self.Wv(feat)
        scores = torch.matmul(Q, K.T) / self.scale
        scores = scores.masked_fill(~self.mask, -1e9)
        attn   = torch.softmax(scores, dim=1)
        ctx    = torch.matmul(attn, V)

        alpha_mod = self.alpha_head(ctx).squeeze(-1)
        beta_mod  = self.beta_head(ctx).squeeze(-1)

        dC = (self.alpha * alpha_mod * C * (1 - C)
              + crime_term
              - self.beta * beta_mod * L * C)
        return dC.unsqueeze(0)

    def param_summary(self):
        a, b = self.alpha.detach(), self.beta.detach()
        print(f"[GNODEattn]  alpha min:{a.min():.4f} max:{a.max():.4f} mean:{a.mean():.4f}")
        print(f"[GNODEattn]  beta  min:{b.min():.4f} max:{b.max():.4f} mean:{b.mean():.4f}")
        print(f"[GNODEattn]  edges={self.mask.sum().item()}")


# ─────────────────────────────────────────────────────────────────────
# 4. GNODElatent
# ─────────────────────────────────────────────────────────────────────

class GNODElatent(_BaseODE):
    """
    Joint crime-enforcement dynamics.

    L_t = obs_gate * L_obs + (1-obs_gate) * L_net(emb, t, AC)
    dC/dt = α C(1-C) + crime_mlp(C, AC, A2C, L_t, AL_t,
                                  C-AC, L_t-AL_t, sin, cos)
           - β L_t C
    """

    def __init__(self, A, T: int,
                 alpha=0.3, beta=0.6,
                 hidden_dim=32, dropout=0.1):
        super().__init__()

        A_t    = _safe_A(A)
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

        # node embeddings
        self.node_emb = nn.Embedding(N, 16)

        # L_net — input: [sin(1), cos(1), t_norm(1), emb(16), AC(1)] = 20
        self.L_net = nn.Sequential(
            nn.Linear(20, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16,  1), nn.Sigmoid(),
        )
        self.L_gate = nn.Sequential(
            nn.Linear(20, 16), nn.Tanh(),
            nn.Linear(16,  1), nn.Sigmoid(),
        )

        # observed L anchor — per node
        self.raw_obs_gate = nn.Parameter(torch.zeros(N))

        # crime drift MLP
        # input: [C(1), AC(1), A2C(1), L_t(1), AL_t(1),
        #         C-AC(1), L_t-AL_t(1), sin(1), cos(1)] = 9
        self.crime_mlp = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.crime_mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        self._t_idx       : int                    = 0
        self._L_obs       : Optional[torch.Tensor] = None
        self._L_latent    : Optional[torch.Tensor] = None
        self._last_L_traj : Optional[torch.Tensor] = None

    @property
    def alpha(self):    return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):     return 2.0 * torch.sigmoid(self.raw_beta)
    @property
    def obs_gate(self): return torch.sigmoid(self.raw_obs_gate)

    def set_context(self, L, t_idx=0, T=None):
        if T is not None:
            self.T = max(T, 1)
        self._t_idx = t_idx
        self._L_obs = L.view(-1)

    def set_observed_schedule(self, L_full): pass

    def _L_features(self, C: torch.Tensor) -> torch.Tensor:
        """(N, 20) features for L_net."""
        device = C.device
        t_norm = self._t_idx / self.T
        time_feat = torch.tensor(
            [math.sin(2 * math.pi * t_norm),
             math.cos(2 * math.pi * t_norm),
             t_norm],
            dtype=torch.float32, device=device
        ).unsqueeze(0).expand(self.N, -1)           # (N, 3)
        node_feat = self.node_emb(
            torch.arange(self.N, device=device))    # (N, 16)
        AC = torch.mv(self.A, C).unsqueeze(-1)      # (N, 1)
        return torch.cat([time_feat, node_feat, AC], dim=1)  # (N, 20)

    def _compute_L(self, C: torch.Tensor) -> torch.Tensor:
        feat   = self._L_features(C)
        L_base = self.L_net(feat).squeeze(-1)
        gate   = self.L_gate(feat).squeeze(-1)
        L_net  = gate * L_base + (1 - gate) * 0.5
        if self._L_obs is not None:
            og = self.obs_gate
            return og * self._L_obs + (1 - og) * L_net
        return L_net

    def _crime_features(self, C: torch.Tensor,
                         L: torch.Tensor,
                         t_norm: float) -> torch.Tensor:
        """(N, 9) features for crime_mlp."""
        AC    = torch.mv(self.A,  C)
        A2C   = torch.mv(self.A2, C)
        AL    = torch.mv(self.A,  L)
        sin_t = C.new_full((self.N,), math.sin(2 * math.pi * t_norm))
        cos_t = C.new_full((self.N,), math.cos(2 * math.pi * t_norm))
        return torch.cat([
            C.unsqueeze(-1),
            AC.unsqueeze(-1),
            A2C.unsqueeze(-1),
            L.unsqueeze(-1),
            AL.unsqueeze(-1),
            (C - AC).unsqueeze(-1),
            (L - AL).unsqueeze(-1),
            sin_t.unsqueeze(-1),
            cos_t.unsqueeze(-1),
        ], dim=1)  # (N, 9)

    def f(self, t, C):
        C = C.view(-1).clamp(0.0, 1.0)
        if torch.isnan(C).any() or torch.isinf(C).any():
            return torch.zeros(1, self.N, device=C.device)

        t_norm = self._t_idx / self.T
        L_t    = self._compute_L(C)

        self._L_latent    = L_t
        self._last_L_traj = L_t.detach().unsqueeze(0)

        c_feat  = self._crime_features(C, L_t, t_norm)
        gnn_out = self.crime_mlp(c_feat).squeeze(-1)

        dC = (self.alpha * C * (1 - C)
              + gnn_out
              - self.beta * L_t * C)
        return torch.nan_to_num(
            dC, nan=0.0, posinf=0.0, neginf=0.0
        ).unsqueeze(0)

    def update_L(self, C):
        with torch.no_grad():
            return self._compute_L(C.view(-1))

    @property
    def L(self):
        """Full schedule (T, N) for analysis."""
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
        return torch.stack(rows, dim=0)  # (T, N)

    def L_regularisation_loss(self, L_traj):
        if L_traj is None:
            return torch.tensor(0.0, device=self.raw_alpha.device)
        dev     = L_traj.device
        smooth  = (L_traj[1:] - L_traj[:-1]).pow(2).mean() \
                  if L_traj.size(0) > 1 \
                  else torch.tensor(0.0, device=dev)
        lower   = torch.relu(0.05 - L_traj).pow(2).mean()
        upper   = torch.relu(L_traj - 0.95).pow(2).mean()
        AL      = torch.mv(self.A, L_traj[0])
        spatial = (L_traj[0] - AL).pow(2).mean()
        return smooth + 5.0 * (lower + upper) + 0.1 * spatial

    def param_summary(self):
        a  = self.alpha.detach()
        b  = self.beta.detach()
        og = self.obs_gate.detach()
        print(f"[GNODElatent]  alpha    min:{a.min():.4f} max:{a.max():.4f} mean:{a.mean():.4f}")
        print(f"[GNODElatent]  beta     min:{b.min():.4f} max:{b.max():.4f} mean:{b.mean():.4f}")
        print(f"[GNODElatent]  obs_gate min:{og.min():.4f} max:{og.max():.4f} mean:{og.mean():.4f}")
        if self._L_latent is not None:
            L = self._L_latent.detach()
            print(f"[GNODElatent]  L_latent min:{L.min():.4f} max:{L.max():.4f} mean:{L.mean():.4f}")

    def get_learned_enforcement(self, nodes, months):
        L_np = self.L.cpu().numpy()
        rows = []
        for t, m in enumerate(months):
            for i, n in enumerate(nodes):
                rows.append({"month": m, "region_id": n,
                             "L_latent": float(L_np[t, i])})
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────
MODEL_MAP = {
    "fc"           : GNODEfc,
    "spatial"      : GNODEspatial,
    "spatial_attn" : GNODEspatial_attention,
    "latent"       : GNODElatent,
}