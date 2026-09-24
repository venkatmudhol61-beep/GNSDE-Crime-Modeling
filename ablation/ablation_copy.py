"""
run_ablations.py

Single-file ablation study addressing Reviewer 2's request:
  (i)   sigma -> 0 (deterministic GN-ODE)
  (ii)  state-independent noise in place of g(C)
  (iii) removal of each architectural addition (Tobler rho, gradients,
        two-hop aggregation, attention, latent enforcement)
  (iv)  sensitivity to sigma bounds and the noise-collapse penalty
        lambda_sigma

USAGE
--------------------------------------------------------------------
    python run_ablations.py --dataset chicago_beats
    python run_ablations.py --dataset chicago_beats --group noise
    python run_ablations.py --dataset chicago_beats --group architecture
    python run_ablations.py --dataset chicago_beats --group sigma_sensitivity
    python run_ablations.py --dataset chicago_beats --calibrate_only
    python run_ablations.py --dataset chicago_beats --n_mc 20 --epochs 30 --patience 10   # fast dev pass
    python ablation/ablation.py --dataset chicago_beats --steps_per_update 150                #  once-per-epoch behaviour
    python run_ablations.py --dataset chicago_beats --group noise                         # run just one group

Produces:
    data/processed/{dataset}_ablation_summary.csv            (one row/config)
    data/processed/{dataset}_ablation_{name}_predictions.csv (per config)
"""

import argparse
import functools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchsde
from torchdiffeq import odeint
from scipy.stats import norm


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_CANDIDATE_MODULES = [
    "model.gnsde_new32_final_paper",
    "model.gnsde_new32_final_paper_review",
    "src.model.gnsde_new32_final_paper",
    "src.model.gnsde_new32_final_paper_review",
]

_model_mod = None
_import_errors = {}
for _candidate in _CANDIDATE_MODULES:
    try:
        _model_mod = __import__(_candidate, fromlist=[
            "GNSDEfc", "GNSDEspatial", "GNSDEspatial_attention", "GNSDElatent"])
        _MODEL_MODULE = _candidate
        break
    except ModuleNotFoundError as e:
        _import_errors[_candidate] = str(e)

if _model_mod is None:
    print(f"\nFAILED to import any of: {_CANDIDATE_MODULES}")
    print(f"  Script location      : {Path(__file__).resolve()}")
    print(f"  Assumed project root : {_PROJECT_ROOT}")
    print(f"  Assumed src root     : {_SRC_ROOT}  (exists={_SRC_ROOT.is_dir()})")
    if _SRC_ROOT.is_dir():
        print(f"  Contents of {_SRC_ROOT}:")
        for p in sorted(_SRC_ROOT.iterdir()):
            print(f"    {'[dir] ' if p.is_dir() else '      '}{p.name}")
        model_dir = _SRC_ROOT / "model"
        if model_dir.is_dir():
            print(f"  Contents of {model_dir}:")
            for p in sorted(model_dir.iterdir()):
                print(f"    {p.name}")
    for _candidate, _err in _import_errors.items():
        print(f"  tried '{_candidate}': {_err}")
    print(f"\nFix: none of the candidate paths above matched. Look at "
          f"the directory listing just printed, find your actual model "
          f".py file, and add its correct dotted import path to the "
          f"_CANDIDATE_MODULES list near the top of this file (or just "
          f"replace it with a single hardcoded string).\n")
    raise ModuleNotFoundError(
        f"Could not find the model file under any of: {_CANDIDATE_MODULES}")

print(f"[run_ablations] Using model module: {_MODEL_MODULE}")
GNSDEfc = _model_mod.GNSDEfc
GNSDEspatial = _model_mod.GNSDEspatial
GNSDEspatial_attention = _model_mod.GNSDEspatial_attention
GNSDElatent = _model_mod.GNSDElatent
# ───────────────────────────────────────────────────────────────────

# Single-threaded on purpose -- see PERFORMANCE NOTES above.
torch.set_num_threads(1)

CONFIGS = {
    "chicago_beats": {
        "crime"  : "data/processed/chicago_beats_crime_timeseries.csv",
        "arrest" : "data/processed/chicago_beats_arrest_timeseries.csv",
        "order"  : "data/raw/graph_beat_order.csv",
        "adj_fc" : "data/processed/chicago_beats_adjacencyfc.npy",
        "adj_sp" : "data/processed/chicago_beats_adjacency_spatial.npy",
    },
    "chicago_districts": {
        "crime"  : "data/processed/chicago_districts_crime_timeseries.csv",
        "arrest" : "data/processed/chicago_districts_arrest_timeseries.csv",
        "order"  : "data/raw/graph_district_order.csv",
        "adj_fc" : "data/processed/chicago_districts_adjacencyfc.npy",
        "adj_sp" : "data/processed/chicago_districts_adjacency_spatial.npy",
    },
    "nyc_precincts": {
        "crime"  : "data/processed/nyc_precincts_crime_timeseries.csv",
        "arrest" : "data/processed/nyc_precincts_arrest_timeseries.csv",
        "order"  : "data/raw/graph_precinct_order.csv",
        "adj_fc" : "data/processed/nyc_precincts_adjacencyfc.npy",
        "adj_sp" : "data/processed/nyc_precincts_adjacency_spatial.npy",
    },
}
# NOTE: adj_fc is only used if you later add "fc" to the ablation
# sweep (currently not part of Reviewer 2's ask, so build_ablation_
# configs() never touches it) -- added here purely for parity with
# your main training script's CONFIGS dict.

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
LR          = 3e-4
GRAD_CLIP   = 1.0
DT          = 0.1
DT_MC       = 0.02
N_MC_SDE    = 100
LAMBDA_L    = 1e-4
LAMBDA_LSUP = 0.3   # weight for latent's L_supervision_loss (matches main script)
SEED        = 42

EPOCHS   = 100     # reduced-budget default for the sweep, see docstring
PATIENCE = 20
NLL_WARMUP_FRAC = 0.3   # NLL weight ramps 0 -> NLL_MAX_WEIGHT over this
NLL_MAX_WEIGHT  = 0.3
STEPS_PER_UPDATE = 8    # optimizer steps this often (in timesteps) instead
                        # of once per full ~150-step epoch -- see
                        # PERFORMANCE NOTES above

# (iv) grid -- edit here, or override via --sigma_grid / --lambda_grid
SIGMA_BOUNDS_GRID = [
    (0.02, 0.50),   # default (matches base model classes)
    (0.01, 0.30),   # tighter -- less room for stochasticity
    (0.05, 0.80),   # wider -- more room for stochasticity
]
LAMBDA_SIGMA_GRID = [0.0, 0.01, 0.1]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# ABLATION-CAPABLE MODEL SUBCLASSES
# (built directly on top of your pasted base classes)
# ─────────────────────────────────────────────────────────────────────

class _SigmaBoundsMixin:
    """
    Shared sigma-bounds override, state-independent-noise g() override,
    and noise-collapse penalty. Mixed in before the base GN-SDE class
    in each ablation subclass below (see MRO order).
    """

    def _init_ablation_common(self, sigma_min, sigma_max, diffusion_type):
        assert sigma_max > sigma_min > 0.0, \
            f"require 0 < sigma_min < sigma_max, got ({sigma_min}, {sigma_max})"
        assert diffusion_type in ("boundary_vanishing", "constant"), \
            f"diffusion_type must be 'boundary_vanishing' or 'constant', got {diffusion_type!r}"
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.diffusion_type = diffusion_type

    @property
    def sigma(self):
        return (self.sigma_min +
                torch.sigmoid(self.raw_sigma) * (self.sigma_max - self.sigma_min))

    def g(self, t, C):
        """
        Overrides the base class's hardcoded boundary-vanishing g().
        diffusion_type="constant" implements ablation (ii): a
        state-independent sigma_i that does NOT vanish at C=0,1, so
        C(t) must be clipped post-step by the solver wrapper below
        (see make_step_fn) -- the base classes' own invariance-at-the-
        boundary property only holds for boundary-vanishing noise.
        """
        C = C.view(-1).clamp(0.0, 1.0)
        if self.diffusion_type == "constant":
            return self.sigma.unsqueeze(0).expand(1, self.N)
        return (self.sigma * C * (1 - C)).unsqueeze(0)

    def noise_collapse_penalty(self, margin: float = 0.05):
        """
        Hinge penalty discouraging sigma from collapsing to (near) its
        lower bound -- if sigma collapses to ~0 the SDE silently
        degenerates into the deterministic ODE baseline (ablation i)
        even when boundary_vanishing noise was requested. Weighted by
        lambda_sigma in the training loss; strength swept in (iv).
        """
        floor = self.sigma_min + margin
        return torch.relu(floor - self.sigma).pow(2).mean()

    def collapse_fraction(self, margin: float = 0.05) -> float:
        """Diagnostic only: fraction of nodes whose learned sigma sits
        within `margin` of sigma_min (i.e. how collapsed the model is)."""
        with torch.no_grad():
            floor = self.sigma_min + margin
            return float((self.sigma <= floor).float().mean().item())


class GNSDEspatialAblate(_SigmaBoundsMixin, GNSDEspatial):
    """
    use_tobler=False    : zeroes the explicit rho*(AC-C) Tobler term.
    use_gradients=False : zeroes the normalised [grad1, grad2,
                           laplacian] features before f_net (f_net's
                           input width is unchanged, it just sees
                           zeros in those three slots).
    use_two_hop=False   : A2C is replaced by AC everywhere -- the
                           model never sees two-hop information.
    diffusion_type       : "boundary_vanishing" (default, matches base
                           class) or "constant" (ablation ii).
    sigma_min/sigma_max  : override the base class's fixed [0.02,0.50]
                           diffusion bounds (ablation iv).

    Drift gating matches eq (18): C(1-C)*[alpha + Phi(.)] + rho*grad1
    - beta*L*C  -- growth and neural-correction terms share ONE gate,
    per the manuscript's drift decomposition (eq 9).
    """

    def __init__(self, *args,
                 use_tobler: bool = True,
                 use_gradients: bool = True,
                 use_two_hop: bool = True,
                 diffusion_type: str = "boundary_vanishing",
                 sigma_min: float = 0.02,
                 sigma_max: float = 0.50,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_tobler = use_tobler
        self.use_gradients = use_gradients
        self.use_two_hop = use_two_hop
        self._init_ablation_common(sigma_min, sigma_max, diffusion_type)

    def f(self, t, C):
        C   = C.view(-1)
        AC  = torch.mv(self.A,  C)
        A2C = torch.mv(self.A2, C) if self.use_two_hop else AC

        grad1     = AC - C
        grad2     = C  - A2C
        laplacian = (C - AC).pow(2)

        grad_feats = self.grad_norm(
            torch.stack([grad1, grad2, laplacian], dim=1)
        )
        if not self.use_gradients:
            grad_feats = torch.zeros_like(grad_feats)

        mem  = self.memory.get(C.device)
        feat = torch.cat([
            C.unsqueeze(-1), AC.unsqueeze(-1), A2C.unsqueeze(-1),
            grad_feats, mem
        ], dim=1)
        phi = self.f_net(feat).squeeze(-1)

        tobler_term = (self.rho * grad1 if self.use_tobler
                       else torch.zeros_like(grad1))

        # eq (18): C(1-C)*[alpha + Phi] + rho*grad1 - beta*L*C
        dC = (C * (1 - C) * (self.alpha + phi)
              + tobler_term
              - self.beta * self._L * C)
        return dC.unsqueeze(0)

    def param_summary(self):
        super().param_summary()
        print(f"[GNSDEspatialAblate]  use_tobler={self.use_tobler}  "
              f"use_gradients={self.use_gradients}  "
              f"use_two_hop={self.use_two_hop}  "
              f"diffusion_type={self.diffusion_type}  "
              f"sigma_bounds=[{self.sigma_min},{self.sigma_max}]")


class GNSDESpatialAttentionAblate(_SigmaBoundsMixin, GNSDEspatial_attention):
    """
    use_tobler / use_gradients / use_two_hop : same semantics as
        GNSDEspatialAblate, applied to the spatial_attn variant's
        (unnormalised) grad1/grad2/laplacian features.
    use_attention=False : m_beta(i,t) replaced by a constant tensor of
        ones (beta applied at fixed, time-invariant strength, exactly
        as in GNSDEspatial) -- the attention parameters (region_emb,
        Wq/Wk/Wv, beta_mod_head) still exist and are still updated,
        they simply never reach f().
    diffusion_type, sigma_min/sigma_max : as above.

    Drift gating matches eq (21): C(1-C)*[alpha + Phi(.)] + rho*grad1
    - beta*m_beta*L*C -- identical gating to GNSDEspatial's eq (18),
    per the manuscript's drift decomposition (eq 9).
    """

    def __init__(self, *args,
                 use_tobler: bool = True,
                 use_gradients: bool = True,
                 use_two_hop: bool = True,
                 use_attention: bool = True,
                 diffusion_type: str = "boundary_vanishing",
                 sigma_min: float = 0.02,
                 sigma_max: float = 0.50,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_tobler = use_tobler
        self.use_gradients = use_gradients
        self.use_two_hop = use_two_hop
        self.use_attention = use_attention
        self._init_ablation_common(sigma_min, sigma_max, diffusion_type)

    def f(self, t, C):
        C   = C.view(-1).clamp(0.0, 1.0)
        AC  = torch.mv(self.A_spatial, C)
        A2C = torch.mv(self.A2_spatial, C) if self.use_two_hop else AC

        grad1     = AC - C
        grad2     = C  - A2C
        laplacian = (C - AC).pow(2)
        if not self.use_gradients:
            grad1_f = torch.zeros_like(grad1)
            grad2_f = torch.zeros_like(grad2)
            laplacian_f = torch.zeros_like(laplacian)
        else:
            grad1_f, grad2_f, laplacian_f = grad1, grad2, laplacian

        mem  = self.memory.get(C.device)
        feat = torch.cat([C.unsqueeze(-1), AC.unsqueeze(-1),
                          A2C.unsqueeze(-1), grad1_f.unsqueeze(-1),
                          grad2_f.unsqueeze(-1), laplacian_f.unsqueeze(-1),
                          mem], dim=1)
        phi = self.f_net(feat).squeeze(-1)

        if self.use_attention:
            m_beta = self._beta_modulator(self._L)
        else:
            m_beta = torch.ones(self.N, device=C.device)

        tobler_term = (self.rho * grad1 if self.use_tobler
                       else torch.zeros_like(grad1))

        # eq (21): C(1-C)*[alpha + Phi] + rho*grad1 - beta*m_beta*L*C
        dC = (C * (1 - C) * (self.alpha + phi)
              + tobler_term
              - self.beta * m_beta * self._L * C)
        return torch.nan_to_num(
            dC, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0)

    def param_summary(self):
        super().param_summary()
        print(f"[GNSDESpatialAttentionAblate]  use_tobler={self.use_tobler}  "
              f"use_gradients={self.use_gradients}  "
              f"use_two_hop={self.use_two_hop}  "
              f"use_attention={self.use_attention}  "
              f"diffusion_type={self.diffusion_type}  "
              f"sigma_bounds=[{self.sigma_min},{self.sigma_max}]")


class GNSDElatentAblate(_SigmaBoundsMixin, GNSDElatent):
    """
    use_latent_inference=False : bypasses L_net/L_gate/obs_gate
        entirely -- _compute_L returns the observed enforcement L_obs
        directly when available (or the neutral 0.5 fallback the base
        class uses when it is not), regardless of what obs_gate has
        learned. A hard structural bypass (not merely biasing
        obs_gate towards 1 via initialisation), so it isolates the
        contribution of the inferred/gated latent signal cleanly:
        L_net, L_gate, node_emb, raw_obs_gate all still exist and are
        still updated, they simply never reach f() when this is False.
    diffusion_type, sigma_min/sigma_max : as above.

    NOTE: this class does NOT override f() -- it inherits eq (25)'s
    drift gating C(1-C)*[alpha + Psi(.)] - beta*L_lat*C directly from
    the base GNSDElatent class, so no separate fix is needed here as
    long as the imported model module has the corrected eq (25)
    gating (see gnsde_new32_final_paper_review.py).
    """

    def __init__(self, *args,
                 use_latent_inference: bool = True,
                 diffusion_type: str = "boundary_vanishing",
                 sigma_min: float = 0.02,
                 sigma_max: float = 0.50,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_latent_inference = use_latent_inference
        self._init_ablation_common(sigma_min, sigma_max, diffusion_type)

    def _compute_L(self, C: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_inference:
            if self._L_obs is not None:
                return self._L_obs
            return torch.full((self.N,), 0.5, device=C.device)
        return super()._compute_L(C)

    def param_summary(self):
        super().param_summary()
        print(f"[GNSDElatentAblate]  use_latent_inference="
              f"{self.use_latent_inference}  "
              f"diffusion_type={self.diffusion_type}  "
              f"sigma_bounds=[{self.sigma_min},{self.sigma_max}]")


# ─────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────

def load_data(dataset: str):
    cfg = CONFIGS[dataset]
    crime = pd.read_csv(cfg["crime"])
    arrest = pd.read_csv(cfg["arrest"])
    region_order = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
    regions = region_order
    months = sorted(crime["month"].unique())
    N, T = len(regions), len(months)

    C_all = torch.zeros(T, N)
    L_all = torch.zeros(T, N)
    for i, m in enumerate(months):
        c = (crime[crime["month"] == m]
             .set_index("region_id").reindex(regions).fillna(0)["C"].values)
        l = (arrest[arrest["month"] == m]
             .set_index("region_id").reindex(regions).fillna(0)["L"].values)
        C_all[i] = torch.tensor(c, dtype=torch.float32)
        L_all[i] = torch.tensor(l, dtype=torch.float32)

    A_matrix = torch.tensor(np.load(cfg["adj_sp"]), dtype=torch.float32)

    n_train = int(T * TRAIN_RATIO)
    n_val   = int(T * VAL_RATIO)
    train_idx = list(range(0, n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, T))

    return C_all, L_all, A_matrix, regions, months, train_idx, val_idx, test_idx, N, T


# ─────────────────────────────────────────────────────────────────────
# SHARED TRAIN + MC/ODE INFERENCE
# ─────────────────────────────────────────────────────────────────────

def make_step_fn(model, solver, dt, t_eval, is_latent):
    def step(C0):
        if solver == "sde":
            C_out = torchsde.sdeint(
                model, C0.unsqueeze(0), t_eval, method="euler", dt=dt)[-1, 0]
            if getattr(model, "diffusion_type", "boundary_vanishing") == "constant":
                # Constant noise does not vanish at the boundary --
                # clip post-step (see g() override docstring above).
                C_out = C_out.clamp(0.0, 1.0)
        else:  # "ode" -- deterministic GN-ODE ablation (i): drift only
            C_out = odeint(
                lambda t, C: model.f(t, C), C0.unsqueeze(0), t_eval,
                method="euler", options={"step_size": dt})[-1, 0]
        if is_latent and getattr(model, "_L_latent", None) is not None:
            model._last_L_traj = model._L_latent.detach().unsqueeze(0)
        return C_out
    return step


def train_model(model, C_all, L_all, T, train_idx, val_idx, solver,
                 epochs, patience, lr, lambda_sigma, is_latent,
                 steps_per_update=STEPS_PER_UPDATE):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    huber = nn.HuberLoss(delta=0.3)
    nll = nn.GaussianNLLLoss(full=False, eps=1e-6, reduction="mean")
    t_eval = torch.tensor([0.0, 1.0])
    step_fn = make_step_fn(model, solver, DT, t_eval, is_latent)
    device = next(model.parameters()).device

    best_val, patience_c, best_state = float("inf"), 0, None
    warmup_epochs = max(1, int(epochs * NLL_WARMUP_FRAC))

    for epoch in range(1, epochs + 1):
        model.train()
        model.reset_memory(device)
        nll_w = min(1.0, epoch / warmup_epochs) * NLL_MAX_WEIGHT

        total_huber, n_valid = 0.0, 0
        opt.zero_grad()
        pending = 0
        for k in range(len(train_idx) - 1):
            i, i_next = train_idx[k], train_idx[k + 1]
            C0, L, C_true = C_all[i], L_all[i], C_all[i_next]
            model.set_context(L, t_idx=i, T=T)
            C_pred = step_fn(C0)

            if torch.isnan(C_pred).any() or torch.isinf(C_pred).any():
                if hasattr(model, "memory"):
                    model.memory.detach()
                continue

            h_loss = huber(C_pred, C_true)
            var_now = model.pred_var(C_pred)
            n_loss = nll(C_pred, C_true, var_now)
            loss = h_loss + nll_w * n_loss

            if is_latent and getattr(model, "_last_L_traj", None) is not None:
                reg = model.L_regularisation_loss(model._last_L_traj)
                if math.isfinite(reg.item()):
                    loss = loss + LAMBDA_L * reg

                # Direct, C-independent supervision for L_lat -- matches
                # the main training script; gives node_emb/L_net/L_gate/
                # obs_gate a gradient on every region-month regardless of
                # C(1-C) vanishing at C=0. For use_latent_inference=False
                # configs this is a near-zero no-op since L_t := L_obs.
                if hasattr(model, "L_supervision_loss"):
                    l_sup = model.L_supervision_loss()
                    if math.isfinite(l_sup.item()):
                        loss = loss + LAMBDA_LSUP * l_sup

            if lambda_sigma > 0 and hasattr(model, "noise_collapse_penalty"):
                loss = loss + lambda_sigma * model.noise_collapse_penalty()

            loss.backward()
            model.step_memory(C_pred, L)
            if hasattr(model, "memory"):
                model.memory.detach()
            if hasattr(model, "memory_L"):
                model.memory_L.detach()

            total_huber += h_loss.item()
            n_valid += 1
            pending += 1

            # Take an optimizer step every `steps_per_update` timesteps
            # instead of once at the very end of the whole ~150-step
            # training trajectory. model.memory is already detach()'d
            # every step above, so the backward graph never extends
            # more than one timestep deep -- there is no correctness
            # reason to defer opt.step() until the full sequence
            # finishes.
            if pending >= steps_per_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad()
                pending = 0

        if pending > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            opt.zero_grad()

        train_huber = total_huber / max(n_valid, 1)

        model.eval()
        with torch.no_grad():
            model.reset_memory(device)
            v_total, v_n = 0.0, 0
            for k in range(len(val_idx) - 1):
                i, i_next = val_idx[k], val_idx[k + 1]
                C0, L, C_true = C_all[i], L_all[i], C_all[i_next]
                model.set_context(L, t_idx=i, T=T)
                C_pred = step_fn(C0)
                if torch.isnan(C_pred).any():
                    continue
                v_total += huber(C_pred, C_true).item()
                v_n += 1
                model.step_memory(C_pred, L)
            val_huber = v_total / v_n if v_n > 0 else float("nan")

        if not math.isfinite(val_huber):
            patience_c += 1
        elif val_huber < best_val:
            best_val, patience_c = val_huber, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
        if patience_c >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def mc_infer(model, solver, C_all, L_all, T, regions, months,
             train_idx, val_idx, n_mc, is_latent):
    t_eval_mc = torch.linspace(0.0, 1.0, 51)
    step_mc = make_step_fn(model, solver, DT_MC, t_eval_mc, is_latent)
    device = next(model.parameters()).device
    model.eval()

    months_set_train = {months[i] for i in train_idx}
    months_set_val   = {months[i] for i in val_idx}

    all_samples = []
    for s in range(n_mc):
        torch.manual_seed(1000 + s)
        model.reset_memory(device)
        preds = []
        with torch.no_grad():
            for i in range(T - 1):
                C0, L = C_all[i], L_all[i]
                model.set_context(L, t_idx=i, T=T)
                C_pred = step_mc(C0)
                model.step_memory(C_pred, L)
                preds.append(C_pred.cpu())
        all_samples.append(torch.stack(preds))

    samples_tensor = torch.stack(all_samples)          # (n_mc, T-1, N)
    pred_mean = samples_tensor.mean(0)
    mc_var = samples_tensor.var(0) if n_mc > 1 else torch.zeros_like(pred_mean)
    sigma_pred = torch.exp(model.log_sigma_pred).detach().cpu()
    total_var = mc_var + sigma_pred.unsqueeze(0).pow(2)
    total_std = total_var.clamp(min=1e-8).sqrt()

    val_start, val_end = val_idx[0], val_idx[-1]
    val_preds = pred_mean[val_start:val_end]
    val_stds  = total_std[val_start:val_end]
    val_true  = C_all[val_start + 1:val_end + 1][:val_preds.shape[0]]
    resid_abs = (val_true - val_preds).abs()
    z = float(np.clip(
        torch.quantile((resid_abs / val_stds.clamp(min=1e-6)).flatten(), 0.90).item(),
        0.5, 4.0))

    pred_p05 = (pred_mean - z * total_std).clamp(0.0, 1.0)
    pred_p95 = (pred_mean + z * total_std).clamp(0.0, 1.0)

    results = []
    for i in range(T - 1):
        m = months[i + 1]
        split = ("train" if m in months_set_train else
                 "val" if m in months_set_val else "test")
        for j, region in enumerate(regions):
            results.append({
                "month": m, "region_id": region,
                "C_gnsde": float(pred_mean[i, j]),
                "C_std": float(total_std[i, j]),
                "C_p05": float(pred_p05[i, j]),
                "C_p95": float(pred_p95[i, j]),
                "split": split,
            })
    return pd.DataFrame(results), z


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(pred_df, dataset):
    test_pred = pred_df[pred_df["split"] == "test"].copy()
    real_df = pd.read_csv(CONFIGS[dataset]["crime"])
    real_df["month"] = real_df["month"].astype(str)
    test_pred["month"] = test_pred["month"].astype(str)
    merged = pd.merge(real_df, test_pred, on=["month", "region_id"], how="inner")
    if merged.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"),
                "sMAPE": float("nan"), "CRPS": float("nan"),
                "PICP_90": float("nan"), "mean_interval_width": float("nan"),
                "NLL": float("nan"), "n_test": 0}

    y = merged["C"].values
    mu = merged["C_gnsde"].values
    sigma = np.clip(merged["C_std"].values, 1e-8, None)
    lo = merged["C_p05"].values
    hi = merged["C_p95"].values

    mae = float(np.mean(np.abs(y - mu)))
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    r2 = float(1 - np.sum((y - mu) ** 2) / np.sum((y - y.mean()) ** 2))
    smape = float(np.mean(2 * np.abs(y - mu) /
                           (np.abs(y) + np.abs(mu) + 1e-8)) * 100)
    z = (y - mu) / sigma
    crps = float((sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                            - 1.0 / np.sqrt(np.pi))).mean())
    nll = float((0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * z ** 2).mean())
    picp = float(((y >= lo) & (y <= hi)).mean() * 100)
    width = float((hi - lo).mean())

    return {"MAE": round(mae, 5), "RMSE": round(rmse, 5), "R2": round(r2, 5),
            "sMAPE": round(smape, 3), "CRPS": round(crps, 5),
            "PICP_90": round(picp, 2), "mean_interval_width": round(width, 5),
            "NLL": round(nll, 5), "n_test": len(merged)}


# ─────────────────────────────────────────────────────────────────────
# ABLATION CONFIGS
# ─────────────────────────────────────────────────────────────────────

SPATIAL_BASE_KW      = dict(alpha=0.3, beta=0.6, hidden=32, mem_dim=16)
SPATIAL_ATTN_BASE_KW = dict(alpha=0.3, beta=0.6, hidden=32,
                             hidden_dim=64, n_heads=4, mem_dim=16)
LATENT_BASE_KW       = dict(alpha=0.3, beta=0.6, hidden_dim=32, mem_dim=16)


def _kw(base, **extra):
    """Merge a base kwargs dict with per-config overrides -- returns a
    plain dict (picklable), never a closure. This is what makes configs
    safe to hand to worker processes: no lambdas, no functools.partial
    bound to a live tensor by reference-in-closure -- just a class
    reference plus a flat, picklable kwargs dict."""
    merged = dict(base)
    merged.update(extra)
    return merged


def build_ablation_configs(A, T, group, n_mc_sde):
    """
    Returns a list of config dicts. Each dict carries `model_cls` (a
    module-level class) and `model_args` / `model_kwargs` describing
    how to build that config's model, rather than a factory closure.
    """
    configs = []

    if group in ("noise", "all"):
        configs += [
            dict(name="spatial_full_sde_control", variant="spatial",
                 model_cls=GNSDEspatialAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_BASE_KW,
                                  diffusion_type="boundary_vanishing"),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="Control: full spatial model, boundary-vanishing "
                             "noise, standard SDE solver."),
            dict(name="i_deterministic_gnode", variant="spatial",
                 model_cls=GNSDEspatialAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_BASE_KW,
                                  diffusion_type="boundary_vanishing"),
                 solver="ode", n_mc=1, lambda_sigma=0.0,
                 description="(i) sigma -> 0: same drift network, integrated "
                             "with a plain ODE solver -- noise is never "
                             "sampled at train or eval time."),
            dict(name="ii_state_independent_noise", variant="spatial",
                 model_cls=GNSDEspatialAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_BASE_KW, diffusion_type="constant"),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="(ii) g(C) replaced by a constant, "
                             "state-independent sigma_i (does not vanish at "
                             "C=0,1); C(t) is clipped post-step."),
        ]

    if group in ("architecture", "all"):
        configs += [
            dict(name="spatial_full_control", variant="spatial",
                 model_cls=GNSDEspatialAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_BASE_KW),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="Control for the spatial-variant component "
                             "removals below (Tobler, gradients, two-hop)."),
            dict(name="iii_remove_tobler", variant="spatial",
                 model_cls=GNSDEspatialAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_BASE_KW, use_tobler=False),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="(iii) Tobler rho*(AC-C) term zeroed."),
            dict(name="iii_remove_gradients", variant="spatial",
                 model_cls=GNSDEspatialAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_BASE_KW, use_gradients=False),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="(iii) [grad1, grad2, laplacian] features "
                             "zeroed before f_net."),
            dict(name="iii_remove_two_hop", variant="spatial",
                 model_cls=GNSDEspatialAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_BASE_KW, use_two_hop=False),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="(iii) A2C replaced by AC everywhere -- model "
                             "never sees two-hop aggregation."),
            dict(name="spatial_attn_full_control", variant="spatial_attn",
                 model_cls=GNSDESpatialAttentionAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_ATTN_BASE_KW, T=T),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="Control for the attention removal below."),
            dict(name="iii_remove_attention", variant="spatial_attn",
                 model_cls=GNSDESpatialAttentionAblate, model_args=(A,),
                 model_kwargs=_kw(SPATIAL_ATTN_BASE_KW, T=T,
                                  use_attention=False),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="(iii) m_beta(i,t) replaced by a constant 1 "
                             "(beta applied at fixed, time-invariant "
                             "strength, as in GNSDEspatial)."),
            dict(name="latent_full_control", variant="latent",
                 model_cls=GNSDElatentAblate, model_args=(A,),
                 model_kwargs=_kw(LATENT_BASE_KW, T=T),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="Control for the latent-enforcement removal "
                             "below."),
            dict(name="iii_remove_latent_enforcement", variant="latent",
                 model_cls=GNSDElatentAblate, model_args=(A,),
                 model_kwargs=_kw(LATENT_BASE_KW, T=T,
                                  use_latent_inference=False),
                 solver="sde", n_mc=n_mc_sde, lambda_sigma=0.0,
                 description="(iii) L_net/L_gate/obs_gate bypassed -- L_t "
                             "falls back to the observed enforcement L_obs "
                             "directly (or the neutral 0.5 fallback when "
                             "L_obs is unavailable)."),
        ]

    if group in ("sigma_sensitivity", "all"):
        for sigma_min, sigma_max in SIGMA_BOUNDS_GRID:
            for lam in LAMBDA_SIGMA_GRID:
                name = f"iv_sigma[{sigma_min},{sigma_max}]_lambda{lam}"
                configs.append(dict(
                    name=name, variant="spatial",
                    model_cls=GNSDEspatialAblate, model_args=(A,),
                    model_kwargs=_kw(SPATIAL_BASE_KW,
                                     sigma_min=sigma_min, sigma_max=sigma_max),
                    solver="sde", n_mc=n_mc_sde, lambda_sigma=lam,
                    description=f"(iv) sigma bounds=[{sigma_min},{sigma_max}], "
                                f"lambda_sigma={lam}."))

    return configs


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def compute_latent_diagnostics(model, C_all, L_all, T):
    """
    Only meaningful for GNSDElatent-family models (has an `obs_gate`
    property). Walks the full timeline, computing L_t at every step
    under the model's ACTUAL learned obs_gate/L_net blend (not a
    synthetic rollout), and returns:
      - obs_gate mean/min/max : how much the model relies on raw L_obs
        (og -> 1) vs. the inferred/gated L_net (og -> 0).
      - corr_L_latent_L_obs   : Pearson correlation between the model's
        L_t and the raw observed L_obs across all nodes/months.

    Useful for explaining ablation results: for the full latent model,
    a corr well below 1.0 with a non-trivial obs_gate indicates L_net
    is doing real denoising work, not just mimicking L_obs. For the
    "remove_latent_enforcement" ablation (use_latent_inference=False),
    corr will be exactly 1.0 by construction (L_t := L_obs directly) --
    reported here too, as a sanity check that the bypass is working as
    intended, not as a meaningful "finding" for that config.
    """
    if not hasattr(model, "obs_gate"):
        return {}

    L_latent_all, L_obs_all = [], []
    with torch.no_grad():
        for i in range(T):
            model.set_context(L_all[i], t_idx=i, T=T)
            L_t = model.update_L(C_all[i])
            L_latent_all.append(L_t.cpu())
            L_obs_all.append(L_all[i].cpu())

    L_latent_flat = torch.cat(L_latent_all).numpy()
    L_obs_flat = torch.cat(L_obs_all).numpy()

    if L_latent_flat.std() < 1e-8 or L_obs_flat.std() < 1e-8:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(L_latent_flat, L_obs_flat)[0, 1])

    og = model.obs_gate.detach().cpu()
    return {
        "obs_gate_mean": round(float(og.mean()), 5),
        "obs_gate_min": round(float(og.min()), 5),
        "obs_gate_max": round(float(og.max()), 5),
        "corr_L_latent_L_obs": round(corr, 5) if math.isfinite(corr) else None,
    }


def run_one_config(cfg, C_all, L_all, T, regions, months,
                    train_idx, val_idx, dataset, epochs, patience,
                    steps_per_update):
    print("\n" + "=" * 78)
    print(f"ABLATION: {cfg['name']}")
    print(f"  {cfg['description']}")
    print("=" * 78)

    model = cfg["model_cls"](*cfg["model_args"], **cfg["model_kwargs"])
    is_latent = (cfg["variant"] == "latent")

    t0 = time.time()
    model, best_val_huber = train_model(
        model, C_all, L_all, T, train_idx, val_idx, cfg["solver"],
        epochs, patience, LR, cfg["lambda_sigma"], is_latent,
        steps_per_update=steps_per_update)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s, best val Huber = {best_val_huber:.5f}")

    t1 = time.time()
    pred_df, z_used = mc_infer(
        model, cfg["solver"], C_all, L_all, T, regions, months,
        train_idx, val_idx, cfg["n_mc"], is_latent)
    infer_time = time.time() - t1
    total_time = train_time + infer_time
    print(f"  inference in {infer_time:.1f}s (n_mc={cfg['n_mc']})  "
          f"-- total wall-clock: {total_time:.1f}s")

    out_pred = f"data/processed/{dataset}_ablation_{cfg['name']}_predictions.csv"
    Path(out_pred).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_pred, index=False)

    metrics = compute_metrics(pred_df, dataset)

    sigma_vals = model.sigma.detach()
    row = {
        "dataset": dataset, "name": cfg["name"], "variant": cfg["variant"],
        "description": cfg["description"], "solver": cfg["solver"],
        "n_mc": cfg["n_mc"], "lambda_sigma": cfg["lambda_sigma"],
        "steps_per_update": steps_per_update,
        "sigma_min": getattr(model, "sigma_min", None),
        "sigma_max": getattr(model, "sigma_max", None),
        "diffusion_type": getattr(model, "diffusion_type", "n/a"),
        "empirical_z90": round(z_used, 3),
        "train_time_sec": round(train_time, 1),
        "infer_time_sec": round(infer_time, 1),
        "total_time_sec": round(total_time, 1),
        "best_val_huber": round(best_val_huber, 5) if math.isfinite(best_val_huber) else None,
        "sigma_mean": round(float(sigma_vals.mean()), 5),
        "sigma_min_learned": round(float(sigma_vals.min()), 5),
        "sigma_max_learned": round(float(sigma_vals.max()), 5),
        "collapse_fraction": (round(model.collapse_fraction(), 4)
                               if hasattr(model, "collapse_fraction") else None),
    }
    row.update(metrics)

    if is_latent:
        latent_diag = compute_latent_diagnostics(model, C_all, L_all, T)
        row.update(latent_diag)
        if latent_diag:
            print(f"  obs_gate: mean={latent_diag['obs_gate_mean']} "
                  f"min={latent_diag['obs_gate_min']} "
                  f"max={latent_diag['obs_gate_max']}  "
                  f"corr(L_latent,L_obs)={latent_diag['corr_L_latent_L_obs']}")

    print(f"  MAE={row['MAE']}  RMSE={row['RMSE']}  R2={row['R2']}  "
          f"CRPS={row['CRPS']}  PICP_90={row['PICP_90']}  NLL={row['NLL']}")
    return row


def upsert_summary(rows, dataset):
    out_csv = f"data/processed/{dataset}_ablation_summary.csv"
    new_df = pd.DataFrame(rows)
    if Path(out_csv).exists():
        existing = pd.read_csv(out_csv)
        existing = existing[
            ~((existing.dataset == dataset) &
              (existing.name.isin(new_df.name)))
        ]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(out_csv, index=False)
    print(f"\nAblation summary -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     choices=list(CONFIGS.keys()))
    ap.add_argument("--group", default="all",
                     choices=["noise", "architecture", "sigma_sensitivity", "all"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--steps_per_update", type=int, default=STEPS_PER_UPDATE,
                     help="Optimizer step frequency in timesteps. Lower = "
                          "more frequent updates per epoch, at the cost of "
                          "noisier gradient estimates from fewer accumulated "
                          "steps. Set to a large number (e.g. 150) to "
                          "reproduce the old once-per-epoch behaviour.")
    ap.add_argument("--n_mc", type=int, default=N_MC_SDE,
                     help="Override N_MC_SDE for all SDE-solver configs -- "
                          "use a small value (e.g. 20) for fast dev passes, "
                          "then rerun the configs you're reporting at the "
                          "full default for final numbers.")
    ap.add_argument("--calibrate_only", action="store_true",
                     help="Skip training entirely; recompute metrics for "
                          "every ablation-config predictions CSV already on "
                          "disk for this dataset/group and rewrite the "
                          "summary CSV.")
    args = ap.parse_args()

    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx, N, T = \
        load_data(args.dataset)
    print(f"Dataset={args.dataset}  Regions={N}  Months={T}  "
          f"Split train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    configs = build_ablation_configs(A, T, args.group, args.n_mc)
    print(f"Group='{args.group}' -> {len(configs)} ablation configs queued "
          f"(n_mc={args.n_mc}, steps_per_update={args.steps_per_update}, "
          f"epochs={args.epochs}, patience={args.patience}).")

    rows = []
    t_start = time.time()

    if args.calibrate_only:
        for cfg in configs:
            out_pred = (f"data/processed/{args.dataset}_ablation_"
                        f"{cfg['name']}_predictions.csv")
            if not Path(out_pred).exists():
                print(f"  [skip] no predictions file for '{cfg['name']}' "
                      f"at {out_pred} -- run without --calibrate_only first.")
                continue
            pred_df = pd.read_csv(out_pred)
            metrics = compute_metrics(pred_df, args.dataset)
            row = {"dataset": args.dataset, "name": cfg["name"],
                   "variant": cfg["variant"], "description": cfg["description"],
                   "solver": cfg["solver"], "n_mc": cfg["n_mc"],
                   "lambda_sigma": cfg["lambda_sigma"]}
            row.update(metrics)
            rows.append(row)
            print(f"  {cfg['name']}: MAE={metrics['MAE']} R2={metrics['R2']} "
                  f"CRPS={metrics['CRPS']}")
    else:
        for cfg in configs:
            row = run_one_config(cfg, C_all, L_all, T, regions, months,
                                  train_idx, val_idx, args.dataset,
                                  args.epochs, args.patience,
                                  args.steps_per_update)
            rows.append(row)

    upsert_summary(rows, args.dataset)
    print(f"\nTotal wall-clock for this run: {time.time() - t_start:.1f}s")