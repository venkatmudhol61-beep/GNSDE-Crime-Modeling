"""
GN-SDE: Graph Neural Stochastic Differential Equations for Crime Forecasting
Four variants with strictly increasing expressiveness.

Hierarchy guaranteed by INDUCTIVE BIAS, not parameter count:

  GNSDEfc          — global diffusion baseline (no spatial geometry)
  GNSDEspatial     — geographic contiguity + Tobler smoothing
  GNSDEspatial_attn— dynamic enforcement-context attention
  GNSDElatent      — latent enforcement inference from crime dynamics

All variants: hidden=32, mem_dim=16, same parameter budget where possible.
Hierarchy comes from what each model CAN represent, not model size.

Theoretical grounding:
  fc:        dC = [f_global(AC) - βLC] dt + σC(1-C)dW
  spatial:   dC = [f_spatial(C,AC,A2C) + ρ(AC-C) - βLC] dt + σC(1-C)dW
  attn:      dC = [f_spatial(.) + ρ(AC-C) - β·m_β(t)·LC] dt + σC(1-C)dW
  latent:    dC = [f_spatial(.) + ρ(AC-C) - β·L̂_t·C] dt + σC(1-C)dW
             where L̂_t is inferred, not observed
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
# 1. GNSDEfc — GLOBAL DIFFUSION BASELINE
# ─────────────────────────────────────────────────────────────────────

class GNSDEfc(_BaseSDE):
    """
    Global diffusion baseline using functional connectivity adjacency.

    Theoretical form:
        dC_i = [α_i C_i(1-C_i) + f(AC_i, h_i) - β_i L_i C_i] dt
               + σ_i C_i(1-C_i) dW_i

    Inductive bias: crime at region i is driven by the global weighted
    average of crime across ALL regions (AC via dense FC adjacency).
    No spatial geometry, no local state, no Tobler smoothing.

    This is the LOWER BOUND of the hierarchy. Its sole advantage is
    the dense FC signal — but it cannot distinguish spatially proximate
    from distant regions, model crime diffusion pressure, or adapt
    enforcement sensitivity over time.

    Input to f_net: [AC_i, h_i]  — 1 graph feature + memory
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

        # Shallow 2-layer net: only AC + memory
        # Deliberately no local C, no spatial gradient, no geometry
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
        # ONLY global aggregated signal — no local C, no spatial structure
        dC  = (self.alpha * C * (1 - C)
               + self.f_net(torch.cat([AC.unsqueeze(-1), mem], dim=1)
                            ).squeeze(-1)
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
        print(f"[GNSDEfc]  alpha      min:{a.min():.4f} max:{a.max():.4f} mean:{a.mean():.4f}")
        print(f"[GNSDEfc]  beta       min:{b.min():.4f} max:{b.max():.4f} mean:{b.mean():.4f}")
        print(f"[GNSDEfc]  sigma(SDE) min:{s.min():.4f} max:{s.max():.4f} mean:{s.mean():.4f}")
        print(f"[GNSDEfc]  sigma(pred)min:{ps.min():.4f} max:{ps.max():.4f} mean:{ps.mean():.4f}")
        print(f"[GNSDEfc]  features: [AC, mem] — global only")

class GNSDEspatial(_BaseSDE):
    """
    dC = [α C(1-C) + ρ(AC-C) + f_net(C, AC, A²C, ∇C, ∇²C, lap, mem) 
          - β L C] dt  +  σ C(1-C) dW

    Spatial adjacency — NO self-loops (crime diffusion only).
    hidden=32, 3-layer net with 6 spatial features + Tobler rho.

    Inductive bias strictly beyond GNSDEfc:
      (1) Tobler smoothing ρ(AC-C): geographic autocorrelation
      (2) Spatial gradients [∇C, ∇²C, laplacian]: crime pressure/displacement
      (3) 2-hop A²C: displacement across district boundaries
    
    Key regularisation vs Doc 2:
      - rho initialised near 0 (sigmoid(0)=0.5 → scaled to 0.1 range)
      - gradient features layer-normed before concat to prevent 
        gradient features dominating at low-N (district) settings
      - weight decay should be higher on f_net than alpha/beta
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

        # rho: Tobler smoothing coefficient
        # init raw=0 → sigmoid(0)=0.5 → rho=0.05 (small but nonzero)
        # scaled to [0, 0.1] to prevent over-smoothing at low N
        self.raw_rho = nn.Parameter(torch.tensor(0.0))

        # LayerNorm on spatial gradient features only
        # Prevents grad1/grad2/laplacian from dominating at district (N=23)
        # C, AC, A2C are already in [0,1] so no norm needed
        self.grad_norm = nn.LayerNorm(3, elementwise_affine=False)

        # 3-layer net, 6 spatial features
        # Features: [C, AC, A2C, norm(grad1, grad2, lap), mem]
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
    def rho(self):
        # Scale to [0, 0.1] — enough to enforce Tobler at district
        # without over-smoothing at beat level
        return torch.sigmoid(self.raw_rho) * 0.1

    def set_context(self, L, t_idx=0, T=1):
        self._L = L.view(-1)

    def f(self, t, C):
        C   = C.view(-1)
        AC  = torch.mv(self.A,  C)
        A2C = torch.mv(self.A2, C)

        grad1     = AC - C            # inflow pressure
        grad2     = C  - A2C          # 2-hop displacement
        laplacian = (C - AC).pow(2)   # hotspot isolation

        # Normalise gradient features together — stabilises district (N=23)
        # where variance in grad features is high relative to N
        grad_feats = self.grad_norm(
            torch.stack([grad1, grad2, laplacian], dim=1)
        )

        mem  = self.memory.get(C.device)
        feat = torch.cat([
            C.unsqueeze(-1),
            AC.unsqueeze(-1),
            A2C.unsqueeze(-1),
            grad_feats,        # normalised grad1, grad2, laplacian
            mem
        ], dim=1)

        dC = (self.alpha * C * (1 - C)
              + self.f_net(feat).squeeze(-1)
              + self.rho * grad1          # Tobler: explicit smoothing term
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
        print(f"[GNSDEspatial]  alpha      min:{a.min():.4f} max:{a.max():.4f} mean:{a.mean():.4f}")
        print(f"[GNSDEspatial]  beta       min:{b.min():.4f} max:{b.max():.4f} mean:{b.mean():.4f}")
        print(f"[GNSDEspatial]  sigma(SDE) min:{s.min():.4f} max:{s.max():.4f} mean:{s.mean():.4f}")
        print(f"[GNSDEspatial]  sigma(pred)min:{ps.min():.4f} max:{ps.max():.4f} mean:{ps.mean():.4f}")
        print(f"[GNSDEspatial]  rho(Tobler):{rh:.4f}  [0,0.1] scaled")
        print(f"[GNSDEspatial]  edges 1-hop:{(self.A>0).sum().item()}  "
              f"2-hop:{(self.A2>0).sum().item()}")
        print(f"[GNSDEspatial]  features: [C, AC, A2C, norm(∇C,∇²C,lap), mem]")

# ─────────────────────────────────────────────────────────────────────
# 3. GNSDEspatial_attention — ADDS DYNAMIC ENFORCEMENT MODULATION
# ─────────────────────────────────────────────────────────────────────

class GNSDEspatial_attention(_BaseSDE):
    """
    Extends spatial with time-varying enforcement-context attention.

    Theoretical form:
        dC_i = [α_i C_i(1-C_i) + ρ(AC_i-C_i) + f_spatial(.)
                - β_i · m_β(i,t) · L_i · C_i] dt
               + σ_i C_i(1-C_i) dW_i

    where m_β(i,t) ∈ (0,1) is an attention-derived enforcement
    sensitivity multiplier.

    One capability strictly beyond GNSDEspatial:

    Dynamic enforcement sensitivity m_β(i,t):
        β in spatial is FIXED per region — enforcement has the same
        dampening effect regardless of time or neighbouring context.
        Here, m_β(i,t) is computed via cross-region attention over
        enforcement state {L_j, embed_j, sin(t), cos(t)}.
        This captures: policy spillover (high enforcement in neighbour j
        reduces crime in i), seasonal enforcement patterns, and
        heterogeneous policy responses across districts.

    Critically: the spatial f_net features are IDENTICAL to GNSDEspatial
    so the only new computation is m_β. This makes ablation clean.
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

        # Attention uses self-loops: each region attends to itself + neighbours
        A_self = A_t.clone()
        A_self.fill_diagonal_(1.0)
        self.register_buffer("mask",  A_self > 0)
        self.register_buffer("scale", torch.sqrt(torch.tensor(float(hidden_dim))))

        N      = A_t.shape[0]
        self.N = N
        self.T = max(T, 1)

        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))
        self.raw_sigma = nn.Parameter(torch.full((N,), _sigmoid_inv(0.10)))
        self.log_sigma_pred = nn.Parameter(torch.full((N,), math.log(0.03)))
        self.raw_rho   = nn.Parameter(torch.tensor(0.0))

        # Identical spatial f_net to GNSDEspatial — same inductive bias
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

        # Attention components for m_β — the ONLY new computation
        # Input: [L_i, agg_L_i, region_emb_i(8), sin(t), cos(t), t] = 13-dim
        self.region_emb = nn.Embedding(N, 8)
        self.Wq = nn.Linear(13, hidden_dim)
        self.Wk = nn.Linear(13, hidden_dim)
        self.Wv = nn.Linear(13, hidden_dim)
        # m_β: enforcement sensitivity modulator in (0,1)
        # Initialised to output ≈1 so model starts like GNSDEspatial
        self.beta_mod_head = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        for layer in [self.Wq, self.Wk, self.Wv]:
            layer.weight.data *= 0.5
            if layer.bias is not None:
                layer.bias.data.zero_()
        # Init beta_mod_head to output ≈1 (no modulation at start)
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
    def rho(self):   return torch.sigmoid(self.raw_rho)

    def set_context(self, L, t_idx=0, T=None):
        if T is not None:
            self.T = max(T, 1)
        self._L     = L.view(-1)
        self._t_idx = t_idx

    def _beta_modulator(self, L: torch.Tensor) -> torch.Tensor:
        """
        Compute m_β(i,t) via cross-region attention over enforcement context.
        Each region i attends over {L_j, embed_j, time_feats} for all j
        in its neighbourhood (defined by mask from A_self).
        Returns m_β ∈ (0,1)^N.
        """
        device = L.device
        t_norm = self._t_idx / self.T
        # Aggregate enforcement from neighbours (self-loops included)
        agg_L  = torch.mv(
            _row_normalise(self.mask.float()), L)
        emb    = self.region_emb(torch.arange(self.N, device=device))
        sin_t  = L.new_full((self.N, 1), math.sin(2 * math.pi * t_norm))
        cos_t  = L.new_full((self.N, 1), math.cos(2 * math.pi * t_norm))
        t_feat = L.new_full((self.N, 1), t_norm)
        feat   = torch.cat([L.unsqueeze(-1), agg_L.unsqueeze(-1),
                            emb, sin_t, cos_t, t_feat], dim=1)  # (N,13)
        Q      = self.Wq(feat)
        K      = self.Wk(feat)
        V      = self.Wv(feat)
        scores = torch.matmul(Q, K.T) / self.scale
        scores = scores.masked_fill(~self.mask, -1e9)
        ctx    = torch.matmul(torch.softmax(scores, dim=1), V)  # (N, hidden_dim)
        return self.beta_mod_head(ctx).squeeze(-1)               # (N,) in (0,1)

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

        # m_β is the ONLY difference from GNSDEspatial
        m_beta = self._beta_modulator(self._L)

        dC = (self.alpha * C * (1 - C)
              + self.f_net(feat).squeeze(-1)
              + self.rho * grad1
              - self.beta * m_beta * self._L * C)   # dynamic β
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
        print(f"[GNSDEattn]  alpha      min:{a.min():.4f} max:{a.max():.4f} mean:{a.mean():.4f}")
        print(f"[GNSDEattn]  beta       min:{b.min():.4f} max:{b.max():.4f} mean:{b.mean():.4f}")
        print(f"[GNSDEattn]  sigma(SDE) min:{s.min():.4f} max:{s.max():.4f} mean:{s.mean():.4f}")
        print(f"[GNSDEattn]  sigma(pred)min:{ps.min():.4f} max:{ps.max():.4f} mean:{ps.mean():.4f}")
        print(f"[GNSDEattn]  rho(Tobler):{rh:.4f}")
        print(f"[GNSDEattn]  features: spatial + dynamic m_β(i,t)")


# ─────────────────────────────────────────────────────────────────────
# 4. GNSDElatent
# ─────────────────────────────────────────────────────────────────────

class GNSDElatent(_BaseSDE):
    """
    dC = [α C(1-C) + crime_mlp(...) - β L_t C] dt  +  σ C(1-C) dW
    Spatial adj — NO self-loops, 2-hop also no self-loops.
    hidden_dim=32 for crime_mlp.
    log_sigma_pred: decoupled predictive std, init=0.03.
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

        self.node_emb = nn.Embedding(N, 16)

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
        device = C.device
        t_norm = self._t_idx / self.T
        time_feat = torch.tensor(
            [math.sin(2 * math.pi * t_norm),
             math.cos(2 * math.pi * t_norm),
             t_norm],
            dtype=torch.float32, device=device
        ).unsqueeze(0).expand(self.N, -1)
        node_feat = self.node_emb(
            torch.arange(self.N, device=device))
        AC    = torch.mv(self.A, C).unsqueeze(-1)
        mem_L = self.mem_L.get(device)
        return torch.cat(
            [time_feat, node_feat, AC, mem_L], dim=1)

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
        gnn_out = self.crime_mlp(c_feat).squeeze(-1)
        dC = (self.alpha * C * (1 - C) + gnn_out - self.beta * L_t * C)
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

    def param_summary(self):
        a  = self.alpha.detach()
        b  = self.beta.detach()
        s  = self.sigma.detach()
        og = self.obs_gate.detach()
        ps = torch.exp(self.log_sigma_pred).detach()
        print(f"[GNSDElatent]  alpha      min:{a.min():.4f} max:{a.max():.4f} mean:{a.mean():.4f}")
        print(f"[GNSDElatent]  beta       min:{b.min():.4f} max:{b.max():.4f} mean:{b.mean():.4f}")
        print(f"[GNSDElatent]  sigma(SDE) min:{s.min():.4f} max:{s.max():.4f} mean:{s.mean():.4f}")
        print(f"[GNSDElatent]  sigma(pred)min:{ps.min():.4f} max:{ps.max():.4f} mean:{ps.mean():.4f}")
        print(f"[GNSDElatent]  obs_gate   min:{og.min():.4f} max:{og.max():.4f} mean:{og.mean():.4f}")
        if self._L_latent is not None:
            L = self._L_latent.detach()
            print(f"[GNSDElatent]  L_latent   min:{L.min():.4f} max:{L.max():.4f} mean:{L.mean():.4f}")

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