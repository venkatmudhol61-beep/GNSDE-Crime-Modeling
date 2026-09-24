"""
ablation/correlated_noise_ablation.py
SCOPING NOTE
--------------------------------------------------------------------
Same reduced single-phase training budget as run_ablations.py, for
the same reason (this script trains 1 diagonal control + len(--ranks)
low-rank configs). See that file's docstring for the full rationale;
increase --epochs/--patience if you have budget for the full schedule.

USAGE
--------------------------------------------------------------------
    python ablation/correlated_noise_ablation.py --dataset chicago_beats
    python ablation/correlated_noise_ablation.py --dataset chicago_beats --ranks 4 8 16
    python ablation/correlated_noise_ablation.py --dataset chicago_beats --calibrate_only
    python ablation/correlated_noise_ablation.py --dataset chicago_beats --steps_per_update 8

Produces:
    data/processed/{dataset}_correlated_noise_summary.csv
    data/processed/{dataset}_correlated_noise_{name}_predictions.csv
"""

import argparse
import functools
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchsde
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────
# Import path resolution (same approach as run_ablations.py, verified
# working against this project's actual layout)
# ─────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # D:\gnsde crime modeling
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model.gnsde_new32_final_paper import (
    GNSDEfc, GNSDEspatial, GNSDEspatial_attention, GNSDElatent,
)

# Use all available cores
torch.set_num_threads(__import__("os").cpu_count())
# ───────────────────────────────────────────────────────────────────

# Use all available cores (was: torch.set_num_threads(1))
torch.set_num_threads(__import__("os").cpu_count())

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

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
LR          = 3e-4
GRAD_CLIP   = 1.0
DT          = 0.1
DT_MC       = 0.02
N_MC        = 100
SEED        = 42

EPOCHS   = 100     # reduced-budget default, see docstring
PATIENCE = 20
NLL_WARMUP_FRAC = 0.3
NLL_MAX_WEIGHT  = 0.3
STEPS_PER_UPDATE = 8   # optimizer steps this often (in timesteps) instead
                       # of once per full ~150-step epoch -- see FIX note

DEFAULT_RANKS = [4, 8, 16]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# CORRELATED (LOW-RANK FACTOR) NOISE MODEL
# ─────────────────────────────────────────────────────────────────────

class GNSDESpatialCorrelatedNoise(GNSDEspatial):
    """
    Same drift f() as the base GNSDEspatial (untouched -- this script
    isolates the noise structure only, not the drift). Diffusion is
    replaced with a rank-K factor model; see module docstring for the
    exact construction and why row-normalising the loadings keeps this
    a fair comparison against the diagonal baseline.

    Deliberately does NOT reparametrise self.sigma (e.g. via a
    sigma-bounds mixin) -- that is a separate ablation axis, already
    covered by run_ablations.py's sigma_sensitivity group. Keeping
    sigma exactly as the base class defines it means the
    "diagonal_noise_control" config here is numerically identical to
    your actual reported model's diffusion, not a reparametrised
    ablation-specific stand-in.
    """
    noise_type = "general"   # overrides base class's "diagonal"

    def __init__(self, *args, noise_rank: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_rank = noise_rank
        # Small random init -- gradient descent shapes the correlation
        # structure during training; nothing hand-crafted here.
        self.raw_loadings = nn.Parameter(torch.randn(self.N, noise_rank) * 0.1)

    @property
    def loadings(self):
        """Row-normalised loading matrix, shape (N, K). Row i has unit
        L2 norm, so node i's total noise variance is preserved exactly
        at sigma_i^2 * C_i^2(1-C_i)^2, matching the diagonal model."""
        L = self.raw_loadings
        norm = L.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return L / norm

    def g(self, t, C):
        C = C.view(-1).clamp(0.0, 1.0)
        envelope = (self.sigma * C * (1 - C)).unsqueeze(-1)   # (N, 1)
        G = envelope * self.loadings                           # (N, K)
        return G.unsqueeze(0)                                   # (1, N, K)

    def implied_correlation_matrix(self) -> torch.Tensor:
        """(N, N) matrix of implied shock correlations between nodes.
        Diagonal is exactly 1 (rows of `loadings` are unit-norm)."""
        with torch.no_grad():
            L = self.loadings
            return (L @ L.T).cpu()

    def factor_variance_explained(self) -> float:
        """Fraction of the loading matrix's total squared-norm ("energy")
        carried by its single largest singular value -- how close the
        noise is to having ONE dominant shared/citywide factor."""
        with torch.no_grad():
            s = torch.linalg.svdvals(self.loadings)
            return float((s[0] ** 2 / (s ** 2).sum()).item())

    def param_summary(self):
        super().param_summary()
        print(f"[GNSDESpatialCorrelatedNoise]  noise_rank={self.noise_rank}  "
              f"factor_variance_explained={self.factor_variance_explained():.4f}")


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
# TRAIN + MC INFERENCE
# (torchsde.sdeint handles "diagonal" and "general" noise identically
# from the caller's side -- no branching needed here on noise_type)
# ─────────────────────────────────────────────────────────────────────

def make_step_fn(model, dt, t_eval):
    def step(C0):
        return torchsde.sdeint(
            model, C0.unsqueeze(0), t_eval, method="euler", dt=dt)[-1, 0]
    return step


def train_model(model, C_all, L_all, T, train_idx, val_idx,
                 epochs, patience, lr, steps_per_update=STEPS_PER_UPDATE):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    huber = nn.HuberLoss(delta=0.3)
    nll = nn.GaussianNLLLoss(full=False, eps=1e-6, reduction="mean")
    t_eval = torch.tensor([0.0, 1.0])
    step_fn = make_step_fn(model, DT, t_eval)
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
                model.memory.detach()
                continue

            h_loss = huber(C_pred, C_true)
            var_now = model.pred_var(C_pred)
            n_loss = nll(C_pred, C_true, var_now)
            loss = h_loss + nll_w * n_loss
            loss.backward()

            model.step_memory(C_pred, L)
            model.memory.detach()

            total_huber += h_loss.item()
            n_valid += 1
            pending += 1

            # FIX: take an optimizer step every `steps_per_update`
            # timesteps instead of once at the end of the whole
            # ~150-step training trajectory. model.memory.detach() is
            # already called every step above, so the backward graph
            # never extends more than one timestep deep -- there was
            # no correctness reason to defer opt.step() until the full
            # sequence finished, only a large unnecessary reduction in
            # how many gradient updates the model got per epoch.
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


def mc_infer(model, C_all, L_all, T, regions, months, train_idx, val_idx, n_mc):
    t_eval_mc = torch.linspace(0.0, 1.0, 51)
    step_mc = make_step_fn(model, DT_MC, t_eval_mc)
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
    mc_var = samples_tensor.var(0)
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
    return pd.DataFrame(results), z, samples_tensor


# ─────────────────────────────────────────────────────────────────────
# METRICS + CORRELATION DIAGNOSTICS
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


def correlation_diagnostics(model, A_matrix):
    """
    Only meaningful for GNSDESpatialCorrelatedNoise (has
    implied_correlation_matrix). Reports:
      - factor_variance_explained : energy in the top singular value
      - mean_offdiag_correlation / mean_abs_offdiag_correlation
      - spatial_alignment_corr : corr(implied correlation, adjacency)
        across off-diagonal entries -- does the model's learned shock
        structure track geography?
    """
    if not hasattr(model, "implied_correlation_matrix"):
        return {}

    corr_mat = model.implied_correlation_matrix().numpy()
    N = corr_mat.shape[0]
    iu = np.triu_indices(N, k=1)
    offdiag = corr_mat[iu]

    adj = A_matrix.numpy()
    adj_offdiag = (adj[iu] > 0).astype(float)

    if offdiag.std() < 1e-8 or adj_offdiag.std() < 1e-8:
        spatial_align = float("nan")
    else:
        spatial_align = float(np.corrcoef(offdiag, adj_offdiag)[0, 1])

    return {
        "factor_variance_explained": round(model.factor_variance_explained(), 5),
        "mean_offdiag_correlation": round(float(offdiag.mean()), 5),
        "mean_abs_offdiag_correlation": round(float(np.abs(offdiag).mean()), 5),
        "spatial_alignment_corr": (round(spatial_align, 5)
                                    if math.isfinite(spatial_align) else None),
    }


def empirical_sample_correlation(samples_tensor, val_idx):
   
    val_start, val_end = val_idx[0], val_idx[-1]
    window = samples_tensor[:, val_start:val_end, :]     # (n_mc, t, N)
    n_mc, t_len, N = window.shape
    # Average each timestep's sample deviations from the MC mean, then
    # correlate node-pairs across (sample, time) as pseudo-observations.
    dev = window - window.mean(0, keepdim=True)           # (n_mc, t, N)
    dev = dev.reshape(n_mc * t_len, N).numpy()
    if dev.shape[0] < 2 or np.all(dev.std(axis=0) < 1e-10):
        return float("nan")
    corr = np.corrcoef(dev.T)
    iu = np.triu_indices(N, k=1)
    return float(np.nanmean(corr[iu]))


# ─────────────────────────────────────────────────────────────────────
# CONFIGS + MAIN
# ─────────────────────────────────────────────────────────────────────

def build_configs(A, ranks):
    diagonal_control = functools.partial(
        GNSDEspatial, A, alpha=0.3, beta=0.6, hidden=32, mem_dim=16)

    configs = [dict(
        name="diagonal_noise_control", kind="diagonal",
        factory=diagonal_control,
        description="Baseline: independent per-node Brownian motions "
                    "(the base model's original diagonal noise_type).")]

    for K in ranks:
        configs.append(dict(
            name=f"correlated_noise_rank{K}", kind="correlated", rank=K,
            factory=functools.partial(
                GNSDESpatialCorrelatedNoise, A, alpha=0.3, beta=0.6,
                hidden=32, mem_dim=16, noise_rank=K),
            description=f"Low-rank factor noise, K={K} shared factors "
                        f"(rank << N={A.shape[0]}), row-normalised "
                        f"loadings so per-node variance matches the "
                        f"diagonal control exactly -- only the "
                        f"cross-node correlation structure differs."))

    return configs


def run_one_config(cfg, C_all, L_all, A, T, regions, months,
                    train_idx, val_idx, dataset, epochs, patience,
                    steps_per_update):
    print("\n" + "=" * 78)
    print(f"CONFIG: {cfg['name']}")
    print(f"  {cfg['description']}")
    print("=" * 78)

    model = cfg["factory"]()

    t0 = time.time()
    model, best_val_huber = train_model(
        model, C_all, L_all, T, train_idx, val_idx, epochs, patience, LR,
        steps_per_update=steps_per_update)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s, best val Huber = {best_val_huber:.5f}")

    pred_df, z_used, samples_tensor = mc_infer(
        model, C_all, L_all, T, regions, months, train_idx, val_idx, N_MC)

    out_pred = (f"data/processed/{dataset}_correlated_noise_"
                f"{cfg['name']}_predictions.csv")
    Path(out_pred).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_pred, index=False)

    metrics = compute_metrics(pred_df, dataset)

    row = {
        "dataset": dataset, "name": cfg["name"], "kind": cfg["kind"],
        "rank": cfg.get("rank"), "description": cfg["description"],
        "steps_per_update": steps_per_update,
        "empirical_z90": round(z_used, 3),
        "train_time_sec": round(train_time, 1),
        "best_val_huber": round(best_val_huber, 5) if math.isfinite(best_val_huber) else None,
    }
    row.update(metrics)

    corr_diag = correlation_diagnostics(model, A)
    row.update(corr_diag)
    if corr_diag:
        empirical_corr = empirical_sample_correlation(samples_tensor, val_idx)
        row["empirical_sample_mean_offdiag_correlation"] = (
            round(empirical_corr, 5) if math.isfinite(empirical_corr) else None)
        print(f"  factor_variance_explained={corr_diag['factor_variance_explained']}  "
              f"mean_offdiag_corr={corr_diag['mean_offdiag_correlation']}  "
              f"spatial_alignment={corr_diag['spatial_alignment_corr']}  "
              f"empirical_sample_corr={row['empirical_sample_mean_offdiag_correlation']}")

    print(f"  MAE={row['MAE']}  RMSE={row['RMSE']}  R2={row['R2']}  "
          f"CRPS={row['CRPS']}  PICP_90={row['PICP_90']}  NLL={row['NLL']}")
    return row


def upsert_summary(rows, dataset):
    out_csv = f"data/processed/{dataset}_correlated_noise_summary.csv"
    new_df = pd.DataFrame(rows)
    if Path(out_csv).exists():
        existing = pd.read_csv(out_csv)
        existing = existing[
            ~((existing.dataset == dataset) & (existing.name.isin(new_df.name)))
        ]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(out_csv, index=False)
    print(f"\nSummary -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS,
                     help=f"Low-rank factor counts K to test "
                          f"(default {DEFAULT_RANKS}).")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--steps_per_update", type=int, default=STEPS_PER_UPDATE,
                     help="Optimizer step frequency in timesteps (see FIX "
                          "note in the module docstring).")
    ap.add_argument("--calibrate_only", action="store_true",
                     help="Skip training; recompute metrics from existing "
                          "predictions CSVs on disk and rewrite the summary. "
                          "NOTE: correlation diagnostics (factor_variance_"
                          "explained etc.) require the trained model object "
                          "and cannot be recovered this way -- this mode "
                          "only refreshes the standard accuracy/calibration "
                          "metrics.")
    args = ap.parse_args()

    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx, N, T = \
        load_data(args.dataset)
    print(f"Dataset={args.dataset}  Regions={N}  Months={T}  "
          f"Split train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    configs = build_configs(A, args.ranks)
    print(f"{len(configs)} configs queued: "
          f"{[c['name'] for c in configs]}")

    rows = []
    t_start = time.time()

    if args.calibrate_only:
        for cfg in configs:
            out_pred = (f"data/processed/{args.dataset}_correlated_noise_"
                        f"{cfg['name']}_predictions.csv")
            if not Path(out_pred).exists():
                print(f"  [skip] no predictions file for '{cfg['name']}' "
                      f"at {out_pred} -- run without --calibrate_only first.")
                continue
            pred_df = pd.read_csv(out_pred)
            metrics = compute_metrics(pred_df, args.dataset)
            row = {"dataset": args.dataset, "name": cfg["name"],
                   "kind": cfg["kind"], "rank": cfg.get("rank"),
                   "description": cfg["description"]}
            row.update(metrics)
            rows.append(row)
            print(f"  {cfg['name']}: MAE={metrics['MAE']} R2={metrics['R2']} "
                  f"CRPS={metrics['CRPS']}")
    else:
        for cfg in configs:
            row = run_one_config(cfg, C_all, L_all, A, T, regions, months,
                                  train_idx, val_idx, args.dataset,
                                  args.epochs, args.patience,
                                  args.steps_per_update)
            rows.append(row)

    upsert_summary(rows, args.dataset)
    print(f"\nTotal wall-clock: {time.time() - t_start:.1f}s")