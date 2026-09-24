"""
Classical Hawkes-process baselines for GN-SDE comparison.

Three variants, increasing kernel flexibility (mirrors the GNSDEfc/spatial/
latent hierarchy so results are directly comparable):

  hawkes_exp        — classical exponential-kernel multivariate Hawkes
  hawkes_powerlaw    — power-law (heavy-tailed) triggering kernel
  hawkes_rff         — nonparametric kernel estimated via Random Fourier
                        Features (RFF) regression, i.e. the kernel shape
                        is learned rather than assumed exponential/power-law
"""

import os
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)

# ─── CHANGE THESE 2 LINES (kept in sync with the GN-SDE script) ───────
DATASET = "chicago_beats"
VARIANTS = ["hawkes_exp", "hawkes_powerlaw", "hawkes_rff"]
# ─────────────────────────────────────────────────────────────────────

LAG_K       = 6          # history window (months) used for excitation sums
RFF_M       = 64         # number of random Fourier features
LR          = 3e-3
EPOCHS      = 300
PATIENCE    = 30
GRAD_CLIP   = 1.0
HUBER_DELTA = 0.3
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
RIDGE_L2    = 1e-4        # ridge penalty on RFF weights (kernel smoothness)

CONFIGS = {
    "chicago_beats": {
        "crime"  : "data/processed/chicago_beats_crime_timeseries.csv",
        "arrest" : "data/processed/chicago_beats_arrest_timeseries.csv",
        "order"  : "data/raw/graph_beat_order.csv",
        "adj_sp" : "data/processed/chicago_beats_adjacency_spatial.npy",
    },
    "chicago_districts": {
        "crime"  : "data/processed/chicago_districts_crime_timeseries.csv",
        "arrest" : "data/processed/chicago_districts_arrest_timeseries.csv",
        "order"  : "data/raw/graph_district_order.csv",
        "adj_sp" : "data/processed/chicago_districts_adjacency_spatial.npy",
    },
    "nyc_precincts": {
        "crime"  : "data/processed/nyc_precincts_crime_timeseries.csv",
        "arrest" : "data/processed/nyc_precincts_arrest_timeseries.csv",
        "order"  : "data/raw/graph_precinct_order.csv",
        "adj_sp" : "data/processed/nyc_precincts_adjacency_spatial.npy",
    },
}
cfg = CONFIGS[DATASET]

# ─────────────────────────────────────────────────────────────────────
# 1. DATA  (identical loading convention to the GN-SDE script)
# ─────────────────────────────────────────────────────────────────────
crime        = pd.read_csv(cfg["crime"])
arrest       = pd.read_csv(cfg["arrest"])
region_order = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
regions      = region_order
months       = sorted(crime["month"].unique())
N            = len(regions)
T            = len(months)

C_all = torch.zeros(T, N)
L_all = torch.zeros(T, N)
for i, m in enumerate(months):
    c = (crime[crime["month"] == m].set_index("region_id")
         .reindex(regions).fillna(0)["C"].values)
    l = (arrest[arrest["month"] == m].set_index("region_id")
         .reindex(regions).fillna(0)["L"].values)
    C_all[i] = torch.tensor(c, dtype=torch.float32)
    L_all[i] = torch.tensor(l, dtype=torch.float32)

A_raw = torch.tensor(np.load(cfg["adj_sp"]), dtype=torch.float32)
A_raw.fill_diagonal_(0.0)
A = A_raw / A_raw.sum(dim=1, keepdim=True).clamp(min=1.0)

n_train   = int(T * TRAIN_RATIO)
n_val     = int(T * VAL_RATIO)
train_idx = list(range(LAG_K, n_train))                 # need K history steps
val_idx   = list(range(n_train, n_train + n_val))
test_idx  = list(range(n_train + n_val, T))
print(f"Hawkes baselines | dataset={DATASET}  N={N}  T={T}  lag_K={LAG_K}")
print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

months_set_train = {months[i] for i in train_idx}
months_set_val   = {months[i] for i in val_idx}


# ─────────────────────────────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────────────────────────────
def _sigmoid_inv(x):
    x = float(np.clip(x, 1e-6, 1 - 1e-6))
    return math.log(x / (1.0 - x))


class _HawkesBase(nn.Module):
    """Shared parameterisation: enforcement-damped baseline + self/cross
    excitation, evaluated over a lag window. Subclasses supply kappa(k)."""

    def __init__(self, A, N, K):
        super().__init__()
        self.register_buffer("A", A)
        self.N, self.K = N, K

        self.raw_mu    = nn.Parameter(torch.full((N,), _sigmoid_inv(0.15)))
        self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(0.3)))
        self.raw_alpha_self  = nn.Parameter(torch.tensor(_sigmoid_inv(0.3)))
        self.raw_alpha_cross = nn.Parameter(torch.tensor(_sigmoid_inv(0.3)))
        self.log_sigma_pred  = nn.Parameter(torch.full((N,), math.log(0.05)))

        lags = torch.arange(1, K + 1, dtype=torch.float32)
        self.register_buffer("lags", lags)

    @property
    def mu(self):    return torch.sigmoid(self.raw_mu) * 0.5
    @property
    def beta(self):  return torch.sigmoid(self.raw_beta) * 2.0
    @property
    def alpha_self(self):  return torch.sigmoid(self.raw_alpha_self)
    @property
    def alpha_cross(self): return torch.sigmoid(self.raw_alpha_cross)

    def kappa(self):
        """Return kernel weights kappa(1..K), shape (K,). Override."""
        raise NotImplementedError

    def forward(self, C_hist, L_now):
        """
        C_hist : (K, N) most-recent-last window of past rates
        L_now  : (N,)   current enforcement level
        returns predicted C at next step, shape (N,)
        """
        k = self.kappa()                                   # (K,)
        self_excite  = torch.einsum("k,kn->n", k, C_hist)
        cross_hist   = torch.einsum("nm,km->kn", self.A, C_hist)  # (K,N) graph-lagged
        cross_excite = torch.einsum("k,kn->n", k, cross_hist)

        baseline = self.mu * torch.exp(-self.beta * L_now)
        pred = (baseline
                + self.alpha_self * self_excite
                + self.alpha_cross * cross_excite)
        return pred

    def pred_var(self):
        return torch.exp(self.log_sigma_pred * 2).clamp(min=1e-6)

    def kernel_penalty(self):
        """Optional smoothness/ridge penalty on the kernel (used by RFF)."""
        return torch.tensor(0.0, device=self.raw_mu.device)


class HawkesExp(_HawkesBase):
    """Classical exponential-decay triggering kernel: kappa(k) = exp(-delta k)."""

    def __init__(self, A, N, K):
        super().__init__(A, N, K)
        self.raw_delta = nn.Parameter(torch.tensor(0.0))  # softplus -> >0

    @property
    def delta(self):
        return torch.nn.functional.softplus(self.raw_delta) + 1e-3

    def kappa(self):
        k = torch.exp(-self.delta * self.lags)
        return k / k.sum().clamp(min=1e-6)  # normalise so alpha carries the magnitude

    def param_summary(self):
        print(f"[HawkesExp]  delta={self.delta.item():.4f}  "
              f"alpha_self={self.alpha_self.item():.4f}  "
              f"alpha_cross={self.alpha_cross.item():.4f}  "
              f"beta mean={self.beta.mean().item():.4f}")


class HawkesPowerLaw(_HawkesBase):
    """Heavy-tailed triggering kernel: kappa(k) = (k + eps)^(-p), p > 0."""

    def __init__(self, A, N, K, eps=1.0):
        super().__init__(A, N, K)
        self.eps = eps
        self.raw_p = nn.Parameter(torch.tensor(_sigmoid_inv(0.3)))  # p in (0,3)

    @property
    def p(self):
        return torch.sigmoid(self.raw_p) * 3.0 + 0.05

    def kappa(self):
        k = (self.lags + self.eps).pow(-self.p)
        return k / k.sum().clamp(min=1e-6)

    def param_summary(self):
        print(f"[HawkesPowerLaw]  p={self.p.item():.4f}  "
              f"alpha_self={self.alpha_self.item():.4f}  "
              f"alpha_cross={self.alpha_cross.item():.4f}  "
              f"beta mean={self.beta.mean().item():.4f}")


class HawkesRFF(_HawkesBase):
    """
    Nonparametric triggering kernel via Random Fourier Features:
        kappa(k) = (1/sqrt(M)) * sum_m w_m * cos(omega_m * k + b_m)

    omega_m ~ N(0, gamma^2) and b_m ~ Uniform(0, 2*pi) are drawn once and
    fixed (classical RFF construction, Rahimi & Recht 2007, adapted here
    to approximate the Hawkes kernel itself rather than a Gram matrix).
    Only the linear weights w_m are learned, giving a flexible, data-driven
    kernel shape without committing to exponential or power-law decay.
    """

    def __init__(self, A, N, K, M=RFF_M, gamma=0.5):
        super().__init__(A, N, K)
        self.M = M
        omega = torch.randn(M) * gamma
        b     = torch.rand(M) * 2 * math.pi
        self.register_buffer("omega", omega)
        self.register_buffer("b", b)
        self.w = nn.Parameter(torch.zeros(M))          # learned RFF weights
        # feature matrix over the fixed lag grid: (K, M)
        feats = torch.cos(torch.outer(self.lags, omega) + b)
        self.register_buffer("phi", feats)

    def kappa(self):
        k = (self.phi @ self.w) / math.sqrt(self.M)     # (K,)
        # keep kernel non-negative and normalised (Hawkes excitation is >=0)
        k = torch.nn.functional.softplus(k)
        return k / k.sum().clamp(min=1e-6)

    def kernel_penalty(self):
        return RIDGE_L2 * (self.w.pow(2).sum())

    def param_summary(self):
        k = self.kappa().detach()
        print(f"[HawkesRFF]  M={self.M}  "
              f"alpha_self={self.alpha_self.item():.4f}  "
              f"alpha_cross={self.alpha_cross.item():.4f}  "
              f"beta mean={self.beta.mean().item():.4f}  "
              f"kernel(k=1..{self.K})={np.round(k.numpy(), 3)}")


MODEL_MAP = {
    "hawkes_exp"      : HawkesExp,
    "hawkes_powerlaw" : HawkesPowerLaw,
    "hawkes_rff"       : HawkesRFF,
}

# ─────────────────────────────────────────────────────────────────────
# 3. TRAIN / EVAL HARNESS (shared across variants)
# ─────────────────────────────────────────────────────────────────────
huber_loss = nn.HuberLoss(delta=HUBER_DELTA)


def run_epoch(model, idx_list, train=True):
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    total_loss, valid = 0.0, 0
    with ctx:
        for i in idx_list:
            C_hist = C_all[i - model.K:i]        # (K, N)
            L_now  = L_all[i]
            C_true = C_all[i]
            C_pred = model(C_hist, L_now)
            if torch.isnan(C_pred).any() or torch.isinf(C_pred).any():
                continue
            loss = huber_loss(C_pred, C_true) + model.kernel_penalty()
            if not math.isfinite(loss.item()):
                continue
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            total_loss += loss.item()
            valid += 1
    return total_loss / valid if valid else float("nan")


def train_variant(variant):
    print("\n" + "=" * 70)
    print(f"TRAINING {variant}")
    print("=" * 70)
    global opt
    model = MODEL_MAP[variant](A, N, LAG_K)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=1e-6)

    best_val, patience_c = float("inf"), 0
    ckpt = f"data/processed/{DATASET}_{variant}_best.pt"
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_idx, train=True)
        val_loss   = run_epoch(model, val_idx, train=False)
        sched.step(epoch)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  time={time.time()-t0:.1f}s")
            t0 = time.time()

        if not math.isfinite(val_loss):
            patience_c += 1
        elif val_loss < best_val:
            best_val, patience_c = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_c += 1
        if patience_c >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    print(f"  best val loss: {best_val:.6f}")
    model.param_summary()
    return model


def calibrate_and_predict(model, variant):
    """One-step-ahead predictions for every t>=K, with a homoscedastic
    predictive std fit on the validation residuals (fast, classical
    analogue of GN-SDE's MC + empirical-z calibration)."""
    preds, trues = [], []
    with torch.no_grad():
        for i in range(LAG_K, T):
            C_hist = C_all[i - model.K:i]
            L_now  = L_all[i]
            preds.append(model(C_hist, L_now))
            trues.append(C_all[i])
    preds = torch.stack(preds)   # (T-K, N)
    trues = torch.stack(trues)

    idx_offset = LAG_K
    val_start  = val_idx[0]  - idx_offset
    val_end    = val_idx[-1] - idx_offset
    resid = (trues[val_start:val_end + 1] - preds[val_start:val_end + 1])
    sigma_resid = resid.std(dim=0).clamp(min=1e-4)          # per-region std
    sigma_pred  = torch.exp(model.log_sigma_pred).detach()
    total_std   = (sigma_resid.pow(2) + sigma_pred.pow(2)).sqrt()

    std_resid = (resid.abs() / total_std.unsqueeze(0).clamp(min=1e-6))
    z_emp = float(np.clip(
        torch.quantile(std_resid.flatten(), 0.90).item(), 0.5, 4.0))

    pred_p05 = preds - z_emp * total_std.unsqueeze(0)
    pred_p95 = preds + z_emp * total_std.unsqueeze(0)

    rows = []
    for row_i, i in enumerate(range(LAG_K, T)):
        m = months[i]
        split = ("train" if m in months_set_train else
                  "val"   if m in months_set_val   else "test")
        for j, region in enumerate(regions):
            rows.append({
                "month": m, "region_id": region,
                "C_pred": float(preds[row_i, j]),
                "C_std" : float(total_std[j]),
                "C_p05" : float(pred_p05[row_i, j]),
                "C_p95" : float(pred_p95[row_i, j]),
                "split" : split,
            })
    df = pd.DataFrame(rows)
    out_path = f"data/processed/{DATASET}_{variant}_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved predictions -> {out_path}")
    return df, z_emp


def gaussian_crps(y_true, mu, sigma):
    """Closed-form CRPS for a Normal(mu, sigma) predictive distribution
    evaluated against the observed value y_true (Gneiting & Raftery, 2007):

        CRPS = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]

    where z = (y_true - mu) / sigma, phi/Phi are the standard normal
    pdf/cdf. Uses the same per-region predictive std (C_std) already
    produced by calibrate_and_predict, so no extra sampling is needed.
    """
    sigma = np.clip(sigma, 1e-8, None)
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return crps


def gaussian_nll(y_true, mu, sigma):
    """Mean Gaussian negative log-likelihood of the observations under the
    predictive Normal(mu, sigma) distribution (same C_std used for CRPS
    and the 90% interval), i.e. -mean(log N(y_true; mu, sigma))."""
    sigma = np.clip(sigma, 1e-8, None)
    return float(np.mean(-norm.logpdf(y_true, loc=mu, scale=sigma)))


def report_test_metrics(pred_df, variant, z_emp):
    real_df = pd.read_csv(cfg["crime"])
    real_df["month"] = real_df["month"].astype(str)
    test_pred = pred_df[pred_df["split"] == "test"].copy()
    test_pred["month"] = test_pred["month"].astype(str)
    merged = pd.merge(real_df, test_pred, on=["month", "region_id"], how="inner")

    if merged.empty:
        print("  WARNING: merge empty, skipping metrics.")
        return

    y_true, y_pred = merged["C"].values, merged["C_pred"].values
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    smape = float(np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)

    in_interval = (merged["C"] >= merged["C_p05"]) & (merged["C"] <= merged["C_p95"])
    coverage_90 = in_interval.mean() * 100
    interval_w  = (merged["C_p95"] - merged["C_p05"]).mean()

    crps_vals = gaussian_crps(y_true, y_pred, merged["C_std"].values)
    crps = float(np.mean(crps_vals))
    nll = gaussian_nll(y_true, y_pred, merged["C_std"].values)

    print(f"\n===== {DATASET.upper()}  {variant.upper()} — TEST =====")
    print(f"  MAE            : {mae:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  R2             : {r2:.4f}")
    print(f"  sMAPE          : {smape:.2f}%")
    print(f"  CRPS           : {crps:.4f}")
    print(f"  NLL            : {nll:.4f}")
    print(f"  Coverage@90%   : {coverage_90:.1f}%  (target=90%)")
    print(f"  Interval width : {interval_w:.4f}")
    print(f"  Empirical z    : {z_emp:.3f}")
    print("=" * 48)
    return dict(variant=variant, mae=mae, rmse=rmse, r2=r2,
                smape=smape, crps=crps, nll=nll, coverage_90=coverage_90,
                interval_width=interval_w)


# ─────────────────────────────────────────────────────────────────────
# 4. RUN ALL VARIANTS
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary_rows = []
    for variant in VARIANTS:
        model = train_variant(variant)
        pred_df, z_emp = calibrate_and_predict(model, variant)
        result = report_test_metrics(pred_df, variant, z_emp)
        if result:
            summary_rows.append(result)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_summary = f"data/processed/{DATASET}_hawkes_baselines_summary.csv"
        summary_df.to_csv(out_summary, index=False)
        print(f"\nSaved baseline summary -> {out_summary}")
        print(summary_df.to_string(index=False))