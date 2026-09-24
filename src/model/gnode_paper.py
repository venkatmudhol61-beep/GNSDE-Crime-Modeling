import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


def _sigmoid_inv(x):
    return math.log(x / (1.0 - x))


def _safe_A(A):
    if isinstance(A, torch.Tensor):
        return A.clone().detach().float()
    return torch.tensor(A, dtype=torch.float32)


# ── 1. Fully Connected ───────────────────────────────────────────────
class GNODEfc(nn.Module):
    def __init__(self, A, alpha=0.3, beta=0.6, hidden=16):
        super().__init__()
        A_t      = _safe_A(A)
        row_sums = A_t.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("A", A_t / row_sums)
        N = A_t.shape[0]
        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))
        self.gnn = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        for layer in self.gnn:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)

    def set_context(self, L, t_idx=0):
        self._L = L

    def forward(self, t, C):
        C = C.view(-1)
        L = self._L.view(-1)
        spatial = self.gnn(
            torch.mv(self.A, C).unsqueeze(-1)
        ).squeeze(-1)
        dC = self.alpha * C * (1 - C) + spatial - self.beta * L * C
        return dC

    def param_summary(self):
        a, b = self.alpha.detach(), self.beta.detach()
        print(f"[FC] alpha — min:{a.min():.4f}  max:{a.max():.4f}  mean:{a.mean():.4f}")
        print(f"[FC] beta  — min:{b.min():.4f}  max:{b.max():.4f}  mean:{b.mean():.4f}")


# ── 2. Spatial ───────────────────────────────────────────────────────
class GNODEspatial(nn.Module):
    def __init__(self, A, alpha=0.3, beta=0.6, hidden=16):
        super().__init__()
        A_t = _safe_A(A)
        A_t.fill_diagonal_(0.0)
        row_sums = A_t.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("A", A_t / row_sums)
        N = A_t.shape[0]
        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))
        self.gnn = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        for l in self.gnn:
            if isinstance(l, nn.Linear):
                l.weight.data *= 0.1
                if l.bias is not None:
                    l.bias.data.zero_()

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)

    def set_context(self, L, t_idx=0):
        self._L = L

    def forward(self, t, C):
        C   = C.view(-1)
        L   = self._L.view(-1)
        agg = torch.mv(self.A, C)
        spatial = self.gnn(torch.stack([C, agg], dim=1)).squeeze(-1)
        dC = self.alpha * C * (1 - C) + spatial - self.beta * L * C
        return dC

    def param_summary(self):
        a, b = self.alpha.detach(), self.beta.detach()
        print(f"[Spatial] alpha — min:{a.min():.4f}  max:{a.max():.4f}  mean:{a.mean():.4f}")
        print(f"[Spatial] beta  — min:{b.min():.4f}  max:{b.max():.4f}  mean:{b.mean():.4f}")
        print(f"[Spatial] nonzero edges: {(self.A > 0).sum().item()}")


# ── 3. Spatial + Attention ───────────────────────────────────────────
class GNODEspatial_attention(nn.Module):
    """
    Two-branch GN-ODE:
      Branch 1 (Spatial)  : crime diffusion  — MLP on [C, agg_C]
      Branch 2 (Attention): enforcement context — attention on L
                            outputs modulate alpha and beta per node

    Uses set_context(L, t_idx) pattern to match train_d.py.
    forward(t, C) takes 1-D C only — L is stored via set_context.
    """
    def __init__(self, A, T=1, hidden=16, hidden_dim=32,
                 alpha=0.3, beta=0.6):
        super().__init__()

        A_t = _safe_A(A)

        # Spatial: no self-loops, row-normalised
        A_sp = A_t.clone()
        A_sp.fill_diagonal_(0.0)
        row_sums = A_sp.sum(dim=1, keepdim=True).clamp(min=1.0)
        A_sp     = A_sp / row_sums

        # Attention: self-loops included
        A_self = A_t.clone()
        A_self.fill_diagonal_(1.0)

        self.register_buffer("A_spatial", A_sp)
        self.register_buffer("A_attn",    A_self)
        self.register_buffer("mask",      A_self > 0)
        self.register_buffer(
            "scale",
            torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        )

        N      = A_t.shape[0]
        self.N = N
        self.T = max(T, 1)

        self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(alpha)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(beta / 2.0)))

        # ── Branch 1: Spatial MLP on crime ───────────────────────────
        # input: [C_i, agg_C] = 2
        self.spatial_gnn = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

        # ── Branch 2: Attention on enforcement ───────────────────────
        # input: [L(1), agg_L(1), region_emb(8), sin(1), cos(1), t_norm(1)] = 13
        self.region_emb = nn.Embedding(N, 8)
        self.Wq         = nn.Linear(13, hidden_dim)
        self.Wk         = nn.Linear(13, hidden_dim)
        self.Wv         = nn.Linear(13, hidden_dim)

        # Two heads: modulate alpha and beta separately
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

        # ── Init ──────────────────────────────────────────────────────
        for layer in self.spatial_gnn:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

        for layer in [self.Wq, self.Wk, self.Wv]:
            layer.weight.data *= 0.5
            if layer.bias is not None:
                layer.bias.data.zero_()

        # Init heads to sigmoid(2) ≈ 0.88 for stable early training
        for head in [self.alpha_head, self.beta_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.zero_()
                    if layer.bias is not None:
                        layer.bias.data.fill_(2.0)

        # Context — set before each odeint call
        self._L     = None
        self._t_idx = 0

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):  return 2.0 * torch.sigmoid(self.raw_beta)

    def set_context(self, L: torch.Tensor, t_idx: int = 0):
        """Call before each odeint: model.set_context(L, t_idx=i)"""
        self._L     = L
        self._t_idx = t_idx

    def _attention_features(self, L):
        """
        Build (N, 13) enforcement feature matrix.
        [L(1), agg_L(1), region_emb(8), sin(1), cos(1), t_norm(1)] = 13
        """
        t_norm = self._t_idx / self.T
        device = L.device
        L      = L.view(-1)

        agg_L       = torch.mv(self.A_attn, L).unsqueeze(-1)      # (N, 1)
        sin_t       = torch.full((self.N, 1),
                                 float(np.sin(2 * np.pi * t_norm)),
                                 dtype=torch.float32, device=device)
        cos_t       = torch.full((self.N, 1),
                                 float(np.cos(2 * np.pi * t_norm)),
                                 dtype=torch.float32, device=device)
        t_feat      = torch.full((self.N, 1), float(t_norm),
                                 dtype=torch.float32, device=device)
        region_feat = self.region_emb(
            torch.arange(self.N, device=device)
        )                                                          # (N, 8)

        return torch.cat([
            L.unsqueeze(-1),   # (N, 1)  — 1
            agg_L,             # (N, 1)  — 2
            region_feat,       # (N, 8)  — 3-10
            sin_t,             # (N, 1)  — 11
            cos_t,             # (N, 1)  — 12
            t_feat,            # (N, 1)  — 13
        ], dim=1)              # (N, 13) ✓

    def forward(self, t, C):
        assert self._L is not None, "Call set_context(L, t_idx) before odeint"
        C = C.view(-1)
        L = self._L.view(-1)

        # ── Branch 1: Crime diffusion (spatial) ───────────────────────
        agg_C      = torch.mv(self.A_spatial, C)                  # (N,)
        crime_term = self.spatial_gnn(
            torch.stack([C, agg_C], dim=1)
        ).squeeze(-1)                                              # (N,)

        # ── Branch 2: Attention modulates alpha, beta ─────────────────
        feat   = self._attention_features(L)                       # (N, 13)
        Q      = self.Wq(feat)                                     # (N, D)
        K      = self.Wk(feat)
        V      = self.Wv(feat)
        scores = torch.matmul(Q, K.T) / self.scale                # (N, N)
        scores = scores.masked_fill(~self.mask, -1e9)
        attn   = torch.softmax(scores, dim=1)                      # (N, N)
        ctx    = torch.matmul(attn, V)                             # (N, D)

        alpha_mod = self.alpha_head(ctx).squeeze(-1)               # (N,)
        beta_mod  = self.beta_head(ctx).squeeze(-1)                # (N,)

        alpha_eff = self.alpha * alpha_mod                         # (N,)
        beta_eff  = self.beta  * beta_mod                          # (N,)

        # ── ODE ───────────────────────────────────────────────────────
        dC = (
            alpha_eff * C * (1 - C)
            + crime_term
            - beta_eff * L * C
        )
        return dC

    def param_summary(self):
        a = self.alpha.detach()
        b = self.beta.detach()
        print(f"[SpatialAttn] alpha — min:{a.min():.4f}  max:{a.max():.4f}  mean:{a.mean():.4f}")
        print(f"[SpatialAttn] beta  — min:{b.min():.4f}  max:{b.max():.4f}  mean:{b.mean():.4f}")
        print(f"[SpatialAttn] edges : {self.mask.sum().item()}")


# ── 4. Latent ────────────────────────────────────────────────────────
class GNODElatent(nn.Module):

    def __init__(self, A, T, alpha=0.3, beta=0.6, hidden_dim=32):
        super().__init__()

        A_t = _safe_A(A)
        self.register_buffer("A", A_t)

        N      = A_t.shape[0]
        self.N = N
        self.T = T

        self.raw_alpha = nn.Parameter(
            torch.full((N,), _sigmoid_inv(alpha), dtype=torch.float32)
        )
        self.raw_beta = nn.Parameter(
            torch.full((N,), _sigmoid_inv(beta / 2.0), dtype=torch.float32)
        )

        self.gnn = nn.Linear(N, N, bias=False)
        with torch.no_grad():
            self.gnn.weight.data *= 0.1

        self.node_emb = nn.Embedding(N, 16)

        self.L_net = nn.Sequential(
            nn.Linear(20, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16,  1), nn.Sigmoid(),
        )

        self.L_gate = nn.Sequential(
            nn.Linear(20, 1), nn.Sigmoid()
        )

    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self): return 2.0 * torch.sigmoid(self.raw_beta)

    def set_context(self, L, t_idx=0):
        self._t_idx = t_idx

    def _time_features(self, t_idx, C=None):
        device = self.node_emb.weight.device
        t_norm = t_idx / self.T

        time_feat = torch.tensor(
            [np.sin(2 * np.pi * t_norm),
             np.cos(2 * np.pi * t_norm),
             t_norm],
            dtype=torch.float32, device=device
        ).unsqueeze(0).expand(self.N, -1)                         # (N, 3)

        node_feat = self.node_emb(
            torch.arange(self.N, device=device)
        )                                                         # (N, 16)

        if C is not None:
            C_agg = (self.A @ C).unsqueeze(-1)                   # (N, 1)
        else:
            C_agg = torch.zeros(self.N, 1, device=device)

        return torch.cat([time_feat, node_feat, C_agg], dim=1)   # (N, 20)

    def _compute_L(self, feat):
        L_base = self.L_net(feat).squeeze(-1)
        gate   = self.L_gate(feat).squeeze(-1)
        return gate * L_base + (1 - gate) * 0.5

    @property
    def L(self):
        rows = [self._compute_L(self._time_features(t))
                for t in range(self.T)]
        return torch.stack(rows, dim=0)                           # (T, N)

    def forward(self, t, C):
        C    = C.view(-1)
        feat = self._time_features(self._t_idx, C=C)
        L_t  = self._compute_L(feat)

        dC = (
            self.alpha * C * (1 - C)
            + self.gnn(self.A @ C)
            - self.beta * L_t * C
        )
        return dC

    def smoothness_loss(self):
        L_mat = self.L
        diff  = L_mat[1:] - L_mat[:-1]
        return diff.pow(2).mean()

    def param_summary(self):
        a, b = self.alpha.detach(), self.beta.detach()
        print(f"[Latent] alpha — min:{a.min():.4f}  max:{a.max():.4f}  mean:{a.mean():.4f}")
        print(f"[Latent] beta  — min:{b.min():.4f}  max:{b.max():.4f}  mean:{b.mean():.4f}")
        L_mat = self.L.detach()
        print(f"[Latent] L     — min:{L_mat.min():.4f}  max:{L_mat.max():.4f}  mean:{L_mat.mean():.4f}  std:{L_mat.std():.4f}")

    def get_learned_enforcement(self, nodes, months):
        L_np = self.L.detach().cpu().numpy()
        rows = []
        for t, m in enumerate(months):
            for i, n in enumerate(nodes):
                rows.append({"month": m, "region_id": n, "L_latent": float(L_np[t, i])})
        return pd.DataFrame(rows)


# ── REGISTRY ─────────────────────────────────────────────────────────
MODEL_MAP = {
    "fc"           : GNODEfc,
    "spatial"      : GNODEspatial,
    "spatial_attn" : GNODEspatial_attention,
    "latent"       : GNODElatent,
}