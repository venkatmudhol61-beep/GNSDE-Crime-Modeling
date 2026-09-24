"""
st_probabilistic_baselines.py  (consolidated, MINI-BATCH FIX + TimeGrad ADDED)

All baseline comparators for Reviewer 2, point 4, in one file sharing
one data-loading / feature-construction / calibration pipeline:


USAGE
--------------------------------------------------------------------
    python baseline/baseline_reviewer.py --model persistgauss --dataset chicago_beats --calibrate
    python baseline/baseline_reviewer.py --model gp           --dataset chicago_beats --calibrate
    python baseline/baseline_reviewer.py --model stzinb       --dataset chicago_beats --calibrate
    python baseline/baseline_reviewer.py --model agcrn        --dataset chicago_beats --calibrate
    python baseline/baseline_reviewer.py --model diffstg      --dataset chicago_beats --calibrate
    python baseline/baseline_reviewer.py --model timegrad     --dataset chicago_beats --calibrate
    python baseline/baseline_reviewer.py --model ustd         --dataset chicago_beats --calibrate
    python baseline/baseline_reviewer.py --model stgncde      --dataset chicago_beats --calibrate

Produces: data/processed/{dataset}_{model}_final_predictions.csv
With --calibrate: appends one row to data/processed/calibration_summary.csv

RECOMMENDATION: before running diffstg/timegrad/ustd/stgncde at full
cost, do a cheap timing probe first, e.g.:
    python src/eval/st_probabilistic_baselines.py --model diffstg --dataset chicago_beats --epochs 2 --n_samples 5 --n_samples_train 2
and check the printed "Total wall-clock" to extrapolate the full-run
cost before committing to it.
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

try:
    import torchcde
    _HAS_TORCHCDE = True
except ImportError:
    _HAS_TORCHCDE = False

# Use all available cores (was: torch.set_num_threads(1))
torch.set_num_threads(os.cpu_count())

CONFIGS = {
    "chicago_beats": {
        "crime": "data/processed/chicago_beats_crime_timeseries.csv",
        "arrest": "data/processed/chicago_beats_arrest_timeseries.csv",
        "order": "data/raw/graph_beat_order.csv",
        "adj_sp": "data/processed/chicago_beats_adjacency_spatial.npy",
    },
    "chicago_districts": {
        "crime": "data/processed/chicago_districts_crime_timeseries.csv",
        "arrest": "data/processed/chicago_districts_arrest_timeseries.csv",
        "order": "data/raw/graph_district_order.csv",
        "adj_sp": "data/processed/chicago_districts_adjacency_spatial.npy",
    },
    "nyc_precincts": {
        "crime": "data/processed/nyc_precincts_crime_timeseries.csv",
        "arrest": "data/processed/nyc_precincts_arrest_timeseries.csv",
        "order": "data/raw/graph_precinct_order.csv",
        "adj_sp": "data/processed/nyc_precincts_adjacency_spatial.npy",
    },
}

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
LAG         = 6
HIDDEN      = 32
LR          = 3e-4
GRAD_CLIP   = 1.0
SEED        = 42
BATCH_SIZE  = 16   # mini-batch size, matches train_nn() in baseline.py

DIFFUSION_STEPS         = 50
N_SAMPLES_DIFFSTG        = 100
N_SAMPLES_DIFFSTG_TRAIN  = 5     # cheap pass for train-split rows
EPOCHS_DIFFSTG           = 80
PATIENCE_DIFFSTG         = 15

N_ENSEMBLE               = 5
EPOCHS_USTD              = 60
PATIENCE_USTD            = 12

EPOCHS_STGNCDE           = 80
PATIENCE_STGNCDE         = 15

AGCRN_HIDDEN   = 32
AGCRN_EMBED    = 8
AGCRN_EPOCHS   = 150
AGCRN_PATIENCE = 20

STZINB_SCALE     = 100.0   # pseudo-count rescaling -- see run_stzinb docstring
STZINB_HIDDEN    = 32
STZINB_EPOCHS    = 100
STZINB_PATIENCE  = 20
STZINB_N_MC      = 100

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# SHARED: data loading, feature/sequence construction
# ─────────────────────────────────────────────────────────────────────

def row_normalise_np(A: np.ndarray) -> np.ndarray:
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1.0, None)
    return A / row_sums


def load_data(dataset: str):
    cfg = CONFIGS[dataset]
    crime = pd.read_csv(cfg["crime"])
    arrest = pd.read_csv(cfg["arrest"])
    region_order = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
    regions = region_order
    months = sorted(crime["month"].unique())
    N, T = len(regions), len(months)

    C_all = np.zeros((T, N), dtype=np.float32)
    L_all = np.zeros((T, N), dtype=np.float32)
    for i, m in enumerate(months):
        c = (crime[crime["month"] == m]
             .set_index("region_id").reindex(regions).fillna(0)["C"].values)
        l = (arrest[arrest["month"] == m]
             .set_index("region_id").reindex(regions).fillna(0)["L"].values)
        C_all[i] = c
        L_all[i] = l

    A_raw = np.load(cfg["adj_sp"])
    A_np = row_normalise_np(A_raw.astype(np.float32))

    n_train = int(T * TRAIN_RATIO)
    n_val   = int(T * VAL_RATIO)
    train_idx = list(range(0, n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, T))

    return (torch.from_numpy(C_all), torch.from_numpy(L_all),
            torch.from_numpy(A_np), regions, months,
            train_idx, val_idx, test_idx)


def build_sequences(C_all, L_all, A, lag):
    """
    For every valid target timestep t in [lag, T), build:
      X_seq: (lag, N, 3) -- [C, AC, L] at each of the lag prior months
      y:     (N,)         -- C_all[t]  (the target)
    Returns a list of (t, X_seq, y) tuples.
    """
    T = C_all.shape[0]
    AC_all = torch.matmul(C_all, A.T)  # (T, N), neighbour-aggregated counts
    pairs = []
    for t in range(lag, T):
        window = slice(t - lag, t)
        X_seq = torch.stack(
            [C_all[window], AC_all[window], L_all[window]], dim=-1
        )  # (lag, N, 3)
        y = C_all[t]  # (N,)
        pairs.append((t, X_seq, y))
    return pairs


def split_label(t, months, months_set_train, months_set_val):
    m = months[t]
    if m in months_set_train:
        return "train"
    if m in months_set_val:
        return "val"
    return "test"


def empirical_z(residual_abs, std, target_q=0.90, lo=0.5, hi=4.0):
    """Empirical z-calibration, matching the convention used by the
    main GN-SDE training script rather than assuming a fixed Gaussian
    z=1.645."""
    ratio = (residual_abs / std.clamp(min=1e-6)).flatten()
    z = torch.quantile(ratio, target_q).item()
    return float(np.clip(z, lo, hi))


def assemble_and_save(pairs, months, regions, mu_all, std_all,
                       train_idx, val_idx, out_csv, z_used):
    months_set_train = {months[i] for i in train_idx}
    months_set_val   = {months[i] for i in val_idx}

    results = []
    for (t, _, _), mu_row, std_row in zip(pairs, mu_all, std_all):
        split = split_label(t, months, months_set_train, months_set_val)
        m = months[t]
        for j, region in enumerate(regions):
            mu = float(np.clip(mu_row[j], 0.0, 1.0))
            sd = float(std_row[j])
            results.append({
                "month": m,
                "region_id": region,
                "C_gnsde": mu,
                "C_std": sd,
                "C_p05": float(np.clip(mu - z_used * sd, 0.0, 1.0)),
                "C_p95": float(np.clip(mu + z_used * sd, 0.0, 1.0)),
                "split": split,
            })
    out_df = pd.DataFrame(results)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def iterate_minibatches(pairs, batch_size, generator=None):
    """Shuffles `pairs` and yields consecutive chunks of size
    `batch_size`. Call once per epoch (fresh shuffle each time), then
    take one optimizer step per yielded chunk -- this gives every
    model in this file the same "many updates per epoch" cadence
    that train_nn() in baseline.py already uses, instead of one
    full-batch update per epoch."""
    n = len(pairs)
    order = torch.randperm(n, generator=generator).tolist()
    for start in range(0, n, batch_size):
        idx_chunk = order[start:start + batch_size]
        yield [pairs[i] for i in idx_chunk]


# ─────────────────────────────────────────────────────────────────────
# SHARED: graph conv + sequence encoder building blocks
# ─────────────────────────────────────────────────────────────────────

class GraphConv(nn.Module):
    """y = A @ x @ W + b, applied over the last-but-one (node) dim."""

    def __init__(self, in_dim, out_dim, A):
        super().__init__()
        self.register_buffer("A", A)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        agg = torch.einsum("ij,...jf->...if", self.A, x)
        return self.lin(agg)


class STEncoder(nn.Module):
    """Graph-convolutional GRU: encodes a (lag, N, in_dim) window into
    a single (N, hidden) conditioning/context embedding."""

    def __init__(self, in_dim, hidden, A):
        super().__init__()
        self.hidden = hidden
        self.gcn = GraphConv(in_dim, hidden, A)
        self.gru = nn.GRUCell(hidden, hidden)

    def forward(self, X_seq):
        N = X_seq.shape[1]
        h = torch.zeros(N, self.hidden, device=X_seq.device)
        for k in range(X_seq.shape[0]):
            g = torch.tanh(self.gcn(X_seq[k]))
            h = self.gru(g, h)
        return h  # (N, hidden)


# ─────────────────────────────────────────────────────────────────────
# 0a. persistgauss -- persistence + Gaussian residual noise
# ─────────────────────────────────────────────────────────────────────

def run_persistgauss(dataset, out_csv):
    print(f"\n[persistgauss] Loading {dataset} ...")
    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx = load_data(dataset)
    T, N = C_all.shape
    C_np = C_all.numpy()

    train_pred_idx = [i for i in train_idx if i + 1 < T]
    resid = C_np[[i + 1 for i in train_pred_idx]] - C_np[train_pred_idx]
    node_std = np.clip(resid.std(axis=0), 1e-4, None)

    mu_all, std_all = [], []
    pairs_meta = []
    for t in range(1, T):
        mu_all.append(C_np[t - 1])
        std_all.append(node_std)
        pairs_meta.append((t, None, None))

    out_df = assemble_and_save(pairs_meta, months, regions, mu_all, std_all,
                                train_idx, val_idx, out_csv, z_used=1.645)
    print(f"[persistgauss] saved -> {out_csv}")
    return out_df


# ─────────────────────────────────────────────────────────────────────
# 0b. gp -- per-node Gaussian Process (Flaxman et al. 2015-style)
# ─────────────────────────────────────────────────────────────────────

def run_gp(dataset, out_csv):
    print(f"\n[gp] Loading {dataset} ...")
    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx = load_data(dataset)
    T, N = C_all.shape
    C_np = C_all.numpy()

    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              RBF(length_scale=12.0, length_scale_bounds=(1.0, 60.0)) +
              WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0)))

    X_all_t = np.arange(T).reshape(-1, 1).astype(float)
    X_train = X_all_t[train_idx]

    mean_mat = np.zeros((T, N))
    std_mat = np.zeros((T, N))
    t0 = time.time()
    for j, region in enumerate(regions):
        if j % 20 == 0:
            print(f"  region {j}/{N} ...")
        y_train = C_np[train_idx, j]
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       n_restarts_optimizer=1, alpha=1e-6)
        gp.fit(X_train, y_train)
        mean, std = gp.predict(X_all_t, return_std=True)
        mean_mat[:, j] = mean
        std_mat[:, j] = np.clip(std, 1e-6, None)
    print(f"[gp] fit+predict done in {time.time()-t0:.1f}s")

    mu_all, std_all, pairs_meta = [], [], []
    for t in range(1, T):
        mu_all.append(mean_mat[t])
        std_all.append(std_mat[t])
        pairs_meta.append((t, None, None))

    out_df = assemble_and_save(pairs_meta, months, regions, mu_all, std_all,
                                train_idx, val_idx, out_csv, z_used=1.645)
    print(f"[gp] saved -> {out_csv}")
    return out_df


# ─────────────────────────────────────────────────────────────────────
# 0c. stzinb -- STZINB-style Zero-Inflated Negative Binomial baseline
# ─────────────────────────────────────────────────────────────────────

class STZINBModel(nn.Module):
    """
    STZINB-style baseline: a graph-convolutional GRU encoder (same
    STEncoder used by diffstg/ustd/stgncde) feeding three per-node
    output heads that parameterise a Zero-Inflated Negative Binomial
    (ZINB) distribution -- the standard likelihood family for sparse,
    overdispersed spatio-temporal COUNT data (e.g. bike-share demand,
    crime counts), which is the core mechanism "STZINB"-style models
    are built around.

    SCOPING NOTE -- read before citing in the manuscript:
    A ZINB distribution is defined over non-negative INTEGERS. This
    pipeline's target C_i(t) is continuous, normalised to [0,1] (Eq. 3
    in the manuscript) -- raw complaint counts are not separately
    exposed by load_data(). To use a genuine count likelihood, this
    baseline rescales the target to an approximate pseudo-count via
    y_count = round(C_i(t) * STZINB_SCALE) (STZINB_SCALE=100 by
    default), trains the ZINB heads against that pseudo-count, and
    rescales predictions back to [0,1] by dividing by STZINB_SCALE.
    This is an approximation necessitated by the pipeline's data
    representation, not a recovery of the true original count scale.
    State this explicitly in the manuscript -- e.g. "an STZINB-style
    baseline trained against a fixed-scale pseudo-count reconstruction
    of the normalised target, since raw complaint counts are not part
    of the shared data pipeline used by the other baselines."

    Heads (all conditioned on the same (N, hidden) context embedding):
      pi  = sigmoid(W_pi h)      -- zero-inflation probability
      mu  = softplus(W_mu h)     -- Negative Binomial mean (> 0)
      r   = softplus(W_r h)      -- Negative Binomial dispersion (> 0)

    NLL (zero-inflated NB, in log-space for stability):
      y=0:  -log( pi + (1-pi) * NB(0; r, mu) )
      y>0:  -log( (1-pi) * NB(y; r, mu) )
    """

    def __init__(self, in_dim, hidden, A):
        super().__init__()
        self.encoder = STEncoder(in_dim, hidden, A)
        self.pi_head = nn.Linear(hidden, 1)
        self.mu_head = nn.Linear(hidden, 1)
        self.r_head = nn.Linear(hidden, 1)

    def forward(self, X_seq):
        h = self.encoder(X_seq)
        pi = torch.sigmoid(self.pi_head(h)).squeeze(-1)
        mu = nn.functional.softplus(self.mu_head(h)).squeeze(-1) + 1e-4
        r = nn.functional.softplus(self.r_head(h)).squeeze(-1) + 1e-4
        return pi, mu, r

    @staticmethod
    def nb_log_prob(y, mu, r):
        # log NB(y; r, mu), y a non-negative integer tensor (float dtype OK)
        return (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
                + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu)))

    def zinb_nll(self, y, pi, mu, r):
        log_nb0 = self.nb_log_prob(torch.zeros_like(y), mu, r)
        log_p_zero = torch.logaddexp(torch.log(pi + 1e-8),
                                      torch.log1p(-pi + 1e-8) + log_nb0)
        log_nb_y = self.nb_log_prob(y, mu, r)
        log_p_nonzero = torch.log1p(-pi + 1e-8) + log_nb_y
        is_zero = (y == 0).float()
        log_prob = is_zero * log_p_zero + (1 - is_zero) * log_p_nonzero
        return -log_prob.mean()

    @torch.no_grad()
    def sample(self, X_seq, n_samples):
        pi, mu, r = self.forward(X_seq)
        N = pi.shape[0]
        pi_rep = pi.unsqueeze(0).expand(n_samples, N)
        mu_rep = mu.unsqueeze(0).expand(n_samples, N)
        r_rep = r.unsqueeze(0).expand(n_samples, N)

        zero_mask = torch.bernoulli(pi_rep)
        # NB as a Gamma-Poisson mixture: lambda ~ Gamma(r, rate=r/mu), y ~ Poisson(lambda)
        gamma_dist = torch.distributions.Gamma(concentration=r_rep, rate=r_rep / mu_rep)
        lam = gamma_dist.sample()
        poisson_dist = torch.distributions.Poisson(lam)
        y_nb = poisson_dist.sample()
        y_sample = torch.where(zero_mask.bool(), torch.zeros_like(y_nb), y_nb)
        return y_sample  # (n_samples, N), in pseudo-count scale


def train_stzinb(train_pairs, val_pairs, in_dim, hidden, A, epochs, patience, lr, scale):
    model = STZINBModel(in_dim, hidden, A)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, patience_c, best_state = float("inf"), 0, None
    print(f"\n[STZINB] training: {len(train_pairs)} train pairs, "
          f"{len(val_pairs)} val pairs, {epochs} epochs max "
          f"(pseudo-count scale={scale}, batch_size={BATCH_SIZE})")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in iterate_minibatches(train_pairs, BATCH_SIZE):
            opt.zero_grad()
            batch_losses = []
            for _, X_seq, y0 in batch:
                y_count = torch.round(y0 * scale).clamp(min=0)
                pi, mu, r = model(X_seq)
                batch_losses.append(model.zinb_nll(y_count, pi, mu, r))
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            vlosses = []
            for _, X_seq, y0 in val_pairs:
                y_count = torch.round(y0 * scale).clamp(min=0)
                pi, mu, r = model(X_seq)
                vlosses.append(model.zinb_nll(y_count, pi, mu, r))
            val_loss = torch.stack(vlosses).mean().item() if vlosses else float("nan")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  train {train_loss:.5f}  val {val_loss:.5f}")

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val, patience_c = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
        if patience_c >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val NLL: {best_val:.5f}")
    return model


def run_stzinb(dataset, lag, epochs, patience, n_mc, out_csv, scale=STZINB_SCALE):
    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx = load_data(dataset)
    pairs = build_sequences(C_all, L_all, A, lag)
    months_set_train = {months[i] for i in train_idx}
    months_set_val   = {months[i] for i in val_idx}
    train_pairs = [p for p in pairs if months[p[0]] in months_set_train]
    val_pairs   = [p for p in pairs if months[p[0]] in months_set_val]

    model = train_stzinb(train_pairs, val_pairs, in_dim=3, hidden=STZINB_HIDDEN,
                          A=A, epochs=epochs, patience=patience, lr=LR, scale=scale)
    model.eval()

    print(f"  sampling ({n_mc} MC draws per pair, pseudo-count scale={scale}) ...")
    mu_all, std_all = [], []
    resid_abs_val, std_val = [], []
    t0 = time.time()
    for idx, (t, X_seq, y0) in enumerate(pairs):
        samples = model.sample(X_seq, n_mc)          # (n_mc, N), pseudo-count scale
        samples_rescaled = (samples / scale).clamp(0.0, 1.0)
        mu = samples_rescaled.mean(0)
        sd = samples_rescaled.std(0).clamp(min=1e-4)
        mu_all.append(mu.numpy())
        std_all.append(sd.numpy())
        if months[t] in months_set_val:
            resid_abs_val.append((y0 - mu).abs())
            std_val.append(sd)
        if idx % 40 == 0:
            print(f"    pair {idx}/{len(pairs)}  elapsed={time.time()-t0:.1f}s")

    z_used = 1.645
    if resid_abs_val:
        z_used = empirical_z(torch.stack(resid_abs_val), torch.stack(std_val))
    print(f"  empirical z@90%: {z_used:.3f}")

    out_df = assemble_and_save(pairs, months, regions, mu_all, std_all,
                                train_idx, val_idx, out_csv, z_used)
    print(f"[STZINB] saved -> {out_csv}")
    return out_df


# ─────────────────────────────────────────────────────────────────────
# 0d. agcrn -- Adaptive Graph Convolutional Recurrent Network
#     (Bai et al. 2020, already cited as bai2020adaptive)
# ─────────────────────────────────────────────────────────────────────

class AVWGCN(nn.Module):
    def __init__(self, N, in_dim, out_dim, embed_dim, cheb_k=2):
        super().__init__()
        self.cheb_k = cheb_k
        self.weight_pool = nn.Parameter(torch.randn(embed_dim, cheb_k, in_dim, out_dim) * 0.1)
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, out_dim))

    def forward(self, x, node_embeddings):
        N = node_embeddings.shape[0]
        supports = torch.softmax(torch.relu(node_embeddings @ node_embeddings.T), dim=1)
        support_set = [torch.eye(N, device=x.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(2 * supports @ support_set[-1] - support_set[-2])
        supports_stack = torch.stack(support_set, dim=0)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weight_pool)
        bias = node_embeddings @ self.bias_pool

        x_g = torch.einsum('knm,mi->kni', supports_stack, x)
        x_g = x_g.permute(1, 0, 2)
        out = torch.einsum('nki,nkio->no', x_g, weights) + bias
        return out


class AGCRNCell(nn.Module):
    def __init__(self, N, in_dim, hidden_dim, embed_dim, cheb_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = AVWGCN(N, in_dim + hidden_dim, 2 * hidden_dim, embed_dim, cheb_k)
        self.update = AVWGCN(N, in_dim + hidden_dim, hidden_dim, embed_dim, cheb_k)

    def forward(self, x, h, node_embeddings):
        combined = torch.cat([x, h], dim=-1)
        z_r = torch.sigmoid(self.gate(combined, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat([x, r * h], dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return z * h + (1 - z) * hc


class AGCRN(nn.Module):
    def __init__(self, N, in_dim, hidden_dim, embed_dim, cheb_k=2):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.node_embeddings = nn.Parameter(torch.randn(N, embed_dim) * 0.1)
        self.cell = AGCRNCell(N, in_dim, hidden_dim, embed_dim, cheb_k)
        self.out_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq):
        h = torch.zeros(self.N, self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.shape[0]):
            h = self.cell(x_seq[t], h, self.node_embeddings)
        return self.out_head(h).squeeze(-1)


def run_agcrn(dataset, out_csv, lag=LAG, hidden_dim=AGCRN_HIDDEN,
              embed_dim=AGCRN_EMBED, epochs=AGCRN_EPOCHS, patience=AGCRN_PATIENCE):
    print(f"\n[agcrn] Loading {dataset} ...")
    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx = load_data(dataset)
    pairs = build_sequences(C_all, L_all, A, lag)
    N = C_all.shape[1]
    months_set_train = {months[i] for i in train_idx}
    months_set_val   = {months[i] for i in val_idx}
    train_pairs = [p for p in pairs if months[p[0]] in months_set_train]
    val_pairs   = [p for p in pairs if months[p[0]] in months_set_val]

    model = AGCRN(N, in_dim=3, hidden_dim=hidden_dim, embed_dim=embed_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    huber = nn.HuberLoss(delta=0.3)

    best_val, patience_c, best_state = float("inf"), 0, None
    print(f"[agcrn] training: {len(train_pairs)} train pairs, "
          f"{len(val_pairs)} val pairs, {epochs} epochs max, batch_size={BATCH_SIZE}")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in iterate_minibatches(train_pairs, BATCH_SIZE):
            opt.zero_grad()
            batch_losses = [huber(model(X_seq), y0) for _, X_seq, y0 in batch]
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            vlosses = [huber(model(X_seq), y0) for _, X_seq, y0 in val_pairs]
            val_loss = torch.stack(vlosses).mean().item() if vlosses else float("nan")

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"elapsed={time.time()-t0:.1f}s")
            t0 = time.time()

        if not math.isfinite(val_loss):
            patience_c += 1
        elif val_loss < best_val:
            best_val, patience_c = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
        if patience_c >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[agcrn] best val loss: {best_val:.6f}")

    model.eval()
    mu_all = []
    with torch.no_grad():
        for t, X_seq, y0 in pairs:
            y_pred = model(X_seq).clamp(0.0, 1.0)
            mu_all.append(y_pred.numpy())
    train_resid = []
    with torch.no_grad():
        for _, X_seq, y0 in train_pairs:
            y_pred = model(X_seq).clamp(0.0, 1.0)
            train_resid.append((y0 - y_pred).numpy())
    resid_std = float(np.clip(np.concatenate(train_resid).std(), 1e-4, None))
    std_all = [np.full(N, resid_std) for _ in pairs]

    out_df = assemble_and_save(pairs, months, regions, mu_all, std_all,
                                train_idx, val_idx, out_csv, z_used=1.645)
    print(f"[agcrn] saved -> {out_csv}")
    return out_df


# ─────────────────────────────────────────────────────────────────────
# 1. DiffSTG-style: conditional denoising diffusion baseline
# ─────────────────────────────────────────────────────────────────────

def make_beta_schedule(n_steps, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, n_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


class DiffSTGDenoiser(nn.Module):
    def __init__(self, hidden, A, n_diff_steps):
        super().__init__()
        self.time_emb = nn.Embedding(n_diff_steps, hidden)
        self.gcn = GraphConv(1 + hidden + hidden, hidden, A)
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, y_noisy, h_cond, t_idx):
        N = y_noisy.shape[0]
        te = self.time_emb(
            torch.full((N,), t_idx, dtype=torch.long, device=y_noisy.device))
        feat = torch.cat([y_noisy.unsqueeze(-1), h_cond, te], dim=-1)
        g = torch.tanh(self.gcn(feat))
        return self.out(g).squeeze(-1)


class DiffSTGModel(nn.Module):
    def __init__(self, in_dim, hidden, A, n_diff_steps):
        super().__init__()
        self.encoder = STEncoder(in_dim, hidden, A)
        self.denoiser = DiffSTGDenoiser(hidden, A, n_diff_steps)
        self.n_diff_steps = n_diff_steps
        betas, alphas, alpha_bars = make_beta_schedule(n_diff_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, y0, t_idx, noise):
        ab = self.alpha_bars[t_idx]
        return torch.sqrt(ab) * y0 + torch.sqrt(1 - ab) * noise

    def training_loss(self, X_seq, y0):
        h_cond = self.encoder(X_seq)
        t_idx = int(torch.randint(0, self.n_diff_steps, (1,)).item())
        noise = torch.randn_like(y0)
        y_noisy = self.q_sample(y0, t_idx, noise)
        eps_pred = self.denoiser(y_noisy, h_cond, t_idx)
        return nn.functional.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, X_seq, n_samples):
        h_cond = self.encoder(X_seq)
        N = X_seq.shape[1]
        device = X_seq.device
        samples = []
        for _ in range(n_samples):
            y = torch.randn(N, device=device)
            for t_idx in reversed(range(self.n_diff_steps)):
                eps_pred = self.denoiser(y, h_cond, t_idx)
                alpha_t = self.alphas[t_idx]
                alpha_bar_t = self.alpha_bars[t_idx]
                beta_t = self.betas[t_idx]
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    y - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
                if t_idx > 0:
                    noise = torch.randn_like(y)
                    y = mean + torch.sqrt(beta_t) * noise
                else:
                    y = mean
            samples.append(y.clamp(0.0, 1.0))
        return torch.stack(samples)  # (n_samples, N)


def train_diffstg(train_pairs, val_pairs, in_dim, hidden, A, epochs, patience, lr):
    model = DiffSTGModel(in_dim, hidden, A, DIFFUSION_STEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, patience_c, best_state = float("inf"), 0, None
    print(f"\n[DiffSTG] training: {len(train_pairs)} train pairs, "
          f"{len(val_pairs)} val pairs, {epochs} epochs max, batch_size={BATCH_SIZE}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in iterate_minibatches(train_pairs, BATCH_SIZE):
            opt.zero_grad()
            batch_losses = [model.training_loss(X_seq, y0) for _, X_seq, y0 in batch]
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            vlosses = [model.training_loss(X_seq, y0) for _, X_seq, y0 in val_pairs]
            val_loss = torch.stack(vlosses).mean().item() if vlosses else float("nan")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  train {train_loss:.5f}  val {val_loss:.5f}")

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val, patience_c = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
        if patience_c >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val loss: {best_val:.5f}")
    return model


def run_diffstg(dataset, lag, epochs, patience, n_samples, n_samples_train, out_csv):
    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx = load_data(dataset)
    pairs = build_sequences(C_all, L_all, A, lag)
    months_set_train = {months[i] for i in train_idx}
    months_set_val   = {months[i] for i in val_idx}
    train_pairs = [p for p in pairs if months[p[0]] in months_set_train]
    val_pairs   = [p for p in pairs if months[p[0]] in months_set_val]

    model = train_diffstg(train_pairs, val_pairs, in_dim=3, hidden=HIDDEN,
                           A=A, epochs=epochs, patience=patience, lr=LR)

    n_train_pairs = len(train_pairs)
    n_eval_pairs = len(pairs) - n_train_pairs
    print(f"  sampling: {n_samples} MC draws for {n_eval_pairs} val/test pairs, "
          f"{n_samples_train} MC draws for {n_train_pairs} train pairs "
          f"(reduced train-split cost, calibration only uses test)")

    mu_all, std_all = [], []
    resid_abs_val, std_val = [], []
    t0 = time.time()
    for idx, (t, X_seq, y0) in enumerate(pairs):
        is_train = months[t] in months_set_train
        n = n_samples_train if is_train else n_samples
        samples = model.sample(X_seq, n)
        mu = samples.mean(0)
        sd = samples.std(0).clamp(min=1e-4)
        mu_all.append(mu.numpy())
        std_all.append(sd.numpy())
        if months[t] in months_set_val:
            resid_abs_val.append((y0 - mu).abs())
            std_val.append(sd)
        if idx % 20 == 0:
            print(f"    pair {idx}/{len(pairs)}  elapsed={time.time()-t0:.1f}s")

    z_used = 1.645
    if resid_abs_val:
        z_used = empirical_z(torch.stack(resid_abs_val), torch.stack(std_val))
    print(f"  empirical z@90%: {z_used:.3f}")

    out_df = assemble_and_save(pairs, months, regions, mu_all, std_all,
                                train_idx, val_idx, out_csv, z_used)
    print(f"[DiffSTG] saved -> {out_csv}")
    return out_df


# ─────────────────────────────────────────────────────────────────────
# 1b. TimeGrad-style: RNN-conditioned diffusion baseline (no graph term)
# ─────────────────────────────────────────────────────────────────────

class PerNodeEncoder(nn.Module):
    """
    TimeGrad-style encoder: a per-node GRU with NO cross-node (graph)
    mixing -- unlike STEncoder (used by diffstg/ustd/stgncde), each
    node's history is encoded independently, matching TimeGrad's
    original design (Rasul et al. 2021), which has no spatial term at
    all. This is the key architectural difference from the diffstg
    baseline above: same diffusion mechanism, no graph convolution.
    """

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)

    def forward(self, X_seq):
        # X_seq: (lag, N, in_dim) -- each node processed independently,
        # GRU weights shared across nodes (as in TimeGrad) but no
        # information crosses between nodes at any point.
        N = X_seq.shape[1]
        h = torch.zeros(N, self.hidden, device=X_seq.device)
        for k in range(X_seq.shape[0]):
            g = torch.tanh(self.input_proj(X_seq[k]))
            h = self.gru(g, h)
        return h  # (N, hidden)


class TimeGradDenoiser(nn.Module):
    """
    eps_theta(y_noisy, h_cond, t) for TimeGrad -- an MLP (NOT a graph
    convolution) predicting the injected noise, conditioned on the
    per-node RNN context embedding and a diffusion-timestep embedding.
    This is the direct TimeGrad analogue of DiffSTGDenoiser, with the
    GraphConv replaced by a plain per-node Linear layer.
    """

    def __init__(self, hidden, n_diff_steps):
        super().__init__()
        self.time_emb = nn.Embedding(n_diff_steps, hidden)
        self.net = nn.Sequential(
            nn.Linear(1 + hidden + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, y_noisy, h_cond, t_idx):
        N = y_noisy.shape[0]
        te = self.time_emb(
            torch.full((N,), t_idx, dtype=torch.long, device=y_noisy.device))
        feat = torch.cat([y_noisy.unsqueeze(-1), h_cond, te], dim=-1)
        return self.net(feat).squeeze(-1)


class TimeGradModel(nn.Module):
    """Same diffusion process (forward/reverse) as DiffSTGModel, but
    using the graph-free PerNodeEncoder + TimeGradDenoiser above."""

    def __init__(self, in_dim, hidden, n_diff_steps):
        super().__init__()
        self.encoder = PerNodeEncoder(in_dim, hidden)
        self.denoiser = TimeGradDenoiser(hidden, n_diff_steps)
        self.n_diff_steps = n_diff_steps
        betas, alphas, alpha_bars = make_beta_schedule(n_diff_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, y0, t_idx, noise):
        ab = self.alpha_bars[t_idx]
        return torch.sqrt(ab) * y0 + torch.sqrt(1 - ab) * noise

    def training_loss(self, X_seq, y0):
        h_cond = self.encoder(X_seq)
        t_idx = int(torch.randint(0, self.n_diff_steps, (1,)).item())
        noise = torch.randn_like(y0)
        y_noisy = self.q_sample(y0, t_idx, noise)
        eps_pred = self.denoiser(y_noisy, h_cond, t_idx)
        return nn.functional.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, X_seq, n_samples):
        h_cond = self.encoder(X_seq)
        N = X_seq.shape[1]
        device = X_seq.device
        samples = []
        for _ in range(n_samples):
            y = torch.randn(N, device=device)
            for t_idx in reversed(range(self.n_diff_steps)):
                eps_pred = self.denoiser(y, h_cond, t_idx)
                alpha_t = self.alphas[t_idx]
                alpha_bar_t = self.alpha_bars[t_idx]
                beta_t = self.betas[t_idx]
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    y - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
                if t_idx > 0:
                    noise = torch.randn_like(y)
                    y = mean + torch.sqrt(beta_t) * noise
                else:
                    y = mean
            samples.append(y.clamp(0.0, 1.0))
        return torch.stack(samples)


def train_timegrad(train_pairs, val_pairs, in_dim, hidden, epochs, patience, lr):
    model = TimeGradModel(in_dim, hidden, DIFFUSION_STEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, patience_c, best_state = float("inf"), 0, None
    print(f"\n[TimeGrad] training: {len(train_pairs)} train pairs, "
          f"{len(val_pairs)} val pairs, {epochs} epochs max "
          f"(no graph term, batch_size={BATCH_SIZE})")

    for epoch in range(1, epochs + 1):
        model.train()
        # mini-batch loop -- one optimizer step per batch, not per epoch
        # (same fix applied to diffstg/stzinb/ustd/stgncde/agcrn above,
        # kept consistent here so diffstg vs. timegrad differ ONLY in
        # the graph term, not in training regime)
        epoch_losses = []
        for batch in iterate_minibatches(train_pairs, BATCH_SIZE):
            opt.zero_grad()
            batch_losses = [model.training_loss(X_seq, y0) for _, X_seq, y0 in batch]
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            vlosses = [model.training_loss(X_seq, y0) for _, X_seq, y0 in val_pairs]
            val_loss = torch.stack(vlosses).mean().item() if vlosses else float("nan")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  train {train_loss:.5f}  val {val_loss:.5f}")

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val, patience_c = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
        if patience_c >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val loss: {best_val:.5f}")
    return model


def run_timegrad(dataset, lag, epochs, patience, n_samples, n_samples_train, out_csv):
    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx = load_data(dataset)
    pairs = build_sequences(C_all, L_all, A, lag)   # A unused by TimeGrad, kept
    months_set_train = {months[i] for i in train_idx}   # for pipeline consistency
    months_set_val   = {months[i] for i in val_idx}
    train_pairs = [p for p in pairs if months[p[0]] in months_set_train]
    val_pairs   = [p for p in pairs if months[p[0]] in months_set_val]

    model = train_timegrad(train_pairs, val_pairs, in_dim=3, hidden=HIDDEN,
                            epochs=epochs, patience=patience, lr=LR)

    n_train_pairs = len(train_pairs)
    n_eval_pairs = len(pairs) - n_train_pairs
    print(f"  sampling: {n_samples} MC draws for {n_eval_pairs} val/test pairs, "
          f"{n_samples_train} MC draws for {n_train_pairs} train pairs")

    mu_all, std_all = [], []
    resid_abs_val, std_val = [], []
    t0 = time.time()
    for idx, (t, X_seq, y0) in enumerate(pairs):
        is_train = months[t] in months_set_train
        n = n_samples_train if is_train else n_samples
        samples = model.sample(X_seq, n)
        mu = samples.mean(0)
        sd = samples.std(0).clamp(min=1e-4)
        mu_all.append(mu.numpy())
        std_all.append(sd.numpy())
        if months[t] in months_set_val:
            resid_abs_val.append((y0 - mu).abs())
            std_val.append(sd)
        if idx % 20 == 0:
            print(f"    pair {idx}/{len(pairs)}  elapsed={time.time()-t0:.1f}s")

    z_used = 1.645
    if resid_abs_val:
        z_used = empirical_z(torch.stack(resid_abs_val), torch.stack(std_val))
    print(f"  empirical z@90%: {z_used:.3f}")

    out_df = assemble_and_save(pairs, months, regions, mu_all, std_all,
                                train_idx, val_idx, out_csv, z_used)
    print(f"[TimeGrad] saved -> {out_csv}")
    return out_df


# ─────────────────────────────────────────────────────────────────────
# 2. USTD-style: aleatoric/epistemic uncertainty-decomposition ensemble
# ─────────────────────────────────────────────────────────────────────

class USTDMember(nn.Module):
    def __init__(self, in_dim, hidden, A):
        super().__init__()
        self.encoder = STEncoder(in_dim, hidden, A)
        self.mu_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, X_seq):
        h = self.encoder(X_seq)
        mu = torch.sigmoid(self.mu_head(h)).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1).clamp(-10.0, 4.0)
        return mu, logvar


def train_ustd_member(train_pairs, val_pairs, in_dim, hidden, A, epochs, patience, lr, seed):
    torch.manual_seed(seed)
    model = USTDMember(in_dim, hidden, A)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    nll = nn.GaussianNLLLoss(full=False, eps=1e-6, reduction="mean")

    best_val, patience_c, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        # mini-batch loop -- one optimizer step per batch, not per epoch
        for batch in iterate_minibatches(train_pairs, BATCH_SIZE):
            opt.zero_grad()
            mus, targets, logvars = [], [], []
            for _, X_seq, y0 in batch:
                mu, logvar = model(X_seq)
                mus.append(mu); targets.append(y0); logvars.append(logvar)
            mu_cat = torch.cat(mus)
            y_cat = torch.cat(targets)
            var_cat = torch.exp(torch.cat(logvars))
            loss = nll(mu_cat, y_cat, var_cat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        model.eval()
        with torch.no_grad():
            vmus, vtargets, vlogvars = [], [], []
            for _, X_seq, y0 in val_pairs:
                mu, logvar = model(X_seq)
                vmus.append(mu); vtargets.append(y0); vlogvars.append(logvar)
            val_loss = (nll(torch.cat(vmus), torch.cat(vtargets), torch.exp(torch.cat(vlogvars))).item()
                        if vmus else float("nan"))

        if not math.isfinite(val_loss):
            patience_c += 1
        elif val_loss < best_val:
            best_val, patience_c = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
        if patience_c >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def run_ustd(dataset, lag, epochs, patience, n_ensemble, out_csv):
    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx = load_data(dataset)
    pairs = build_sequences(C_all, L_all, A, lag)
    months_set_train = {months[i] for i in train_idx}
    months_set_val   = {months[i] for i in val_idx}
    train_pairs = [p for p in pairs if months[p[0]] in months_set_train]
    val_pairs   = [p for p in pairs if months[p[0]] in months_set_val]

    print(f"\n[USTD] training {n_ensemble} ensemble members "
          f"({len(train_pairs)} train / {len(val_pairs)} val pairs, batch_size={BATCH_SIZE})")
    members = []
    for m in range(n_ensemble):
        model, best_val = train_ustd_member(train_pairs, val_pairs, in_dim=3, hidden=HIDDEN,
                                             A=A, epochs=epochs, patience=patience, lr=LR, seed=SEED + m)
        print(f"  member {m + 1}/{n_ensemble}  best val NLL: {best_val:.5f}")
        members.append(model)
    for m in members:
        m.eval()

    mu_all, std_all = [], []
    resid_abs_val, std_val = [], []
    with torch.no_grad():
        for t, X_seq, y0 in pairs:
            mus, vars_ = [], []
            for model in members:
                mu, logvar = model(X_seq)
                mus.append(mu); vars_.append(torch.exp(logvar))
            mus = torch.stack(mus); vars_ = torch.stack(vars_)
            mean_pred = mus.mean(0)
            epistemic_var = mus.var(0, unbiased=False)
            aleatoric_var = vars_.mean(0)
            total_std = (epistemic_var + aleatoric_var).clamp(min=1e-8).sqrt()
            mu_all.append(mean_pred.numpy())
            std_all.append(total_std.numpy())
            if months[t] in months_set_val:
                resid_abs_val.append((y0 - mean_pred).abs())
                std_val.append(total_std)

    z_used = 1.645
    if resid_abs_val:
        z_used = empirical_z(torch.stack(resid_abs_val), torch.stack(std_val))
    print(f"  empirical z@90%: {z_used:.3f}")

    out_df = assemble_and_save(pairs, months, regions, mu_all, std_all,
                                train_idx, val_idx, out_csv, z_used)
    print(f"[USTD] saved -> {out_csv}")
    return out_df


# ─────────────────────────────────────────────────────────────────────
# 3. STG-NCDE-style: graph-convolutional neural CDE baseline
# ─────────────────────────────────────────────────────────────────────

class STGNCDEFunc(nn.Module):
    def __init__(self, hidden, input_dim, A):
        super().__init__()
        self.hidden = hidden
        self.input_dim = input_dim
        self.gcn = GraphConv(hidden, hidden, A)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden * input_dim), nn.Tanh(),
        )

    def forward(self, t, z):
        spatial = torch.tanh(self.gcn(z))
        combined = z + spatial
        out = self.net(combined)
        return out.view(-1, self.hidden, self.input_dim)


class STGNCDEModel(nn.Module):
    def __init__(self, input_dim, hidden, A):
        super().__init__()
        if not _HAS_TORCHCDE:
            raise ImportError(
                "STG-NCDE baseline requires the optional 'torchcde' package. "
                "Install it with `pip install torchcde` before running "
                "--model stgncde.")
        self.initial = nn.Linear(input_dim, hidden)
        self.func = STGNCDEFunc(hidden, input_dim, A)
        self.mu_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, X_seq):
        Xb = X_seq.permute(1, 0, 2)
        ts = torch.arange(Xb.shape[1], dtype=torch.float32)
        coeffs = torchcde.natural_cubic_coeffs(Xb, t=ts)
        X = torchcde.CubicSpline(coeffs, t=ts)
        z0 = self.initial(X.evaluate(ts[0]))
        z_T = torchcde.cdeint(X=X, func=self.func, z0=z0, t=ts[[0, -1]],
                               method="rk4", options={"step_size": 1.0})[:, -1]
        mu = torch.sigmoid(self.mu_head(z_T)).squeeze(-1)
        logvar = self.logvar_head(z_T).squeeze(-1).clamp(-10.0, 4.0)
        return mu, logvar


def train_stgncde(train_pairs, val_pairs, in_dim, hidden, A, epochs, patience, lr):
    model = STGNCDEModel(in_dim, hidden, A)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    huber = nn.HuberLoss(delta=0.3)
    nll = nn.GaussianNLLLoss(full=False, eps=1e-6, reduction="mean")
    nll_weight = 0.1

    best_val, patience_c, best_state = float("inf"), 0, None
    print(f"\n[STG-NCDE] training: {len(train_pairs)} train pairs, "
          f"{len(val_pairs)} val pairs, {epochs} epochs max, batch_size={BATCH_SIZE}")

    for epoch in range(1, epochs + 1):
        model.train()
        # mini-batch loop -- one optimizer step per batch, not per epoch
        epoch_losses = []
        for batch in iterate_minibatches(train_pairs, BATCH_SIZE):
            opt.zero_grad()
            mus, targets, logvars = [], [], []
            for _, X_seq, y0 in batch:
                mu, logvar = model(X_seq)
                mus.append(mu); targets.append(y0); logvars.append(logvar)
            mu_cat = torch.cat(mus)
            y_cat = torch.cat(targets)
            var_cat = torch.exp(torch.cat(logvars))
            loss = huber(mu_cat, y_cat) + nll_weight * nll(mu_cat, y_cat, var_cat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            vmus, vtargets = [], []
            for _, X_seq, y0 in val_pairs:
                mu, _ = model(X_seq)
                vmus.append(mu); vtargets.append(y0)
            val_loss = huber(torch.cat(vmus), torch.cat(vtargets)).item() if vmus else float("nan")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  train_huber {train_loss:.5f}  val_huber {val_loss:.5f}")

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val, patience_c = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
        if patience_c >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val Huber: {best_val:.5f}")
    return model


def run_stgncde(dataset, lag, epochs, patience, out_csv):
    C_all, L_all, A, regions, months, train_idx, val_idx, test_idx = load_data(dataset)
    pairs = build_sequences(C_all, L_all, A, lag)
    months_set_train = {months[i] for i in train_idx}
    months_set_val   = {months[i] for i in val_idx}
    train_pairs = [p for p in pairs if months[p[0]] in months_set_train]
    val_pairs   = [p for p in pairs if months[p[0]] in months_set_val]

    model = train_stgncde(train_pairs, val_pairs, in_dim=3, hidden=HIDDEN,
                           A=A, epochs=epochs, patience=patience, lr=LR)
    model.eval()

    mu_all, std_all = [], []
    resid_abs_val, std_val = [], []
    with torch.no_grad():
        for t, X_seq, y0 in pairs:
            mu, logvar = model(X_seq)
            sd = torch.exp(0.5 * logvar).clamp(min=1e-4)
            mu_all.append(mu.numpy())
            std_all.append(sd.numpy())
            if months[t] in months_set_val:
                resid_abs_val.append((y0 - mu).abs())
                std_val.append(sd)

    z_used = 1.645
    if resid_abs_val:
        z_used = empirical_z(torch.stack(resid_abs_val), torch.stack(std_val))
    print(f"  empirical z@90%: {z_used:.3f}")

    out_df = assemble_and_save(pairs, months, regions, mu_all, std_all,
                                train_idx, val_idx, out_csv, z_used)
    print(f"[STG-NCDE] saved -> {out_csv}")
    return out_df


# ─────────────────────────────────────────────────────────────────────
# SHARED: calibration (CRPS / PICP / NLL)
# ─────────────────────────────────────────────────────────────────────

def calibrate_and_append(out_df, dataset, variant, extra_fields=None):
    pred = out_df[out_df["split"] == "test"].copy()
    true_csv = Path("data/processed") / f"{dataset}_crime_timeseries.csv"
    true_df = pd.read_csv(true_csv)
    true_col = "C" if "C" in true_df.columns else "C_true"
    merged = pred.merge(true_df[["month", "region_id", true_col]],
                         on=["month", "region_id"], how="inner")
    if merged.empty:
        print("WARNING: merge empty, skipping calibration.")
        return

    y = merged[true_col].to_numpy(dtype=float)
    mu = merged["C_gnsde"].to_numpy(dtype=float)
    sigma = np.clip(merged["C_std"].to_numpy(dtype=float), 1e-8, None)
    lo = merged["C_p05"].to_numpy(dtype=float)
    hi = merged["C_p95"].to_numpy(dtype=float)

    mae = np.mean(np.abs(y - mu))
    rmse = np.sqrt(np.mean((y - mu) ** 2))
    r2 = 1 - np.sum((y - mu) ** 2) / np.sum((y - y.mean()) ** 2)
    smape = 100.0 * np.mean(2 * np.abs(y - mu) / (np.abs(y) + np.abs(mu) + 1e-12))
    z = (y - mu) / sigma
    crps = (sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))).mean()
    nll = (0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * z ** 2).mean()
    picp = float(((y >= lo) & (y <= hi)).mean())
    width = float((hi - lo).mean())

    row = {
        "dataset": dataset, "variant": variant, "split": "test", "n": len(merged),
        "MAE": round(float(mae), 5), "RMSE": round(float(rmse), 5),
        "R2": round(float(r2), 5), "sMAPE": round(float(smape), 3),
        "CRPS": round(float(crps), 5), "PICP_90": round(picp * 100, 2),
        "mean_interval_width": round(width, 5), "NLL": round(float(nll), 5),
    }
    if extra_fields:
        row.update(extra_fields)
    for k, v in row.items():
        print(f"  {k:22s}: {v}")

    summary_path = Path("data/processed/calibration_summary.csv")
    row_df = pd.DataFrame([row])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[~((existing.dataset == row["dataset"]) &
                               (existing.variant == row["variant"]) &
                               (existing.split == row["split"]))]
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_csv(summary_path, index=False)
    print(f"\nAppended to {summary_path}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     choices=["persistgauss", "gp", "stzinb", "agcrn",
                              "diffstg", "timegrad", "ustd", "stgncde"])
    ap.add_argument("--dataset", required=True,
                     choices=["chicago_beats", "chicago_districts", "nyc_precincts"])
    ap.add_argument("--lag", type=int, default=LAG)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                     help="Mini-batch size for all trainable models in this file "
                          "(matches train_nn()'s batch=16 default in baseline.py).")
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES_DIFFSTG,
                     help="[diffstg/timegrad only] MC reverse-diffusion samples for val/test pairs.")
    ap.add_argument("--n_samples_train", type=int, default=N_SAMPLES_DIFFSTG_TRAIN,
                     help="[diffstg/timegrad only] MC samples for train-split pairs (kept low; "
                          "calibration only uses test split).")
    ap.add_argument("--n_ensemble", type=int, default=N_ENSEMBLE,
                     help="[ustd only] number of ensemble members.")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibrate_only", action="store_true",
                     help="Skip training; load existing predictions CSV and just "
                          "recompute/append calibration metrics.")
    args = ap.parse_args()

    BATCH_SIZE = args.batch_size  # allow CLI override of the module-level default

    out_csv = f"data/processed/{args.dataset}_{args.model}_final_predictions.csv"

    if args.calibrate_only:
        if not Path(out_csv).exists():
            raise FileNotFoundError(f"No predictions file at {out_csv}; run without "
                                     f"--calibrate_only first.")
        print(f"[calibrate_only] Loading existing predictions -> {out_csv}")
        out_df = pd.read_csv(out_csv)
        calibrate_and_append(out_df, args.dataset, args.model)
        raise SystemExit(0)

    t0 = time.time()
    extra = {}
    # Reviewer 2, point 10: runtime/solver details. Most baselines here
    # have NO internal ODE/SDE/CDE solver at all (they are one-shot
    # forward passes) -- state that explicitly rather than leaving a
    # blank the reviewer will ask about again. Only stgncde (fixed-step
    # RK4 integrator) and diffstg/timegrad (fixed-length reverse
    # diffusion trajectory) have a step-based process worth reporting,
    # and neither uses an adaptive-tolerance solver, so there is no
    # rtol/atol to report for those either.
    NO_SOLVER_NOTE = "n/a (no internal ODE/SDE/CDE solver -- one-shot forward pass)"

    if args.model == "persistgauss":
        out_df = run_persistgauss(args.dataset, out_csv)
        extra = {"solver": NO_SOLVER_NOTE}

    elif args.model == "gp":
        out_df = run_gp(args.dataset, out_csv)
        extra = {"solver": NO_SOLVER_NOTE}

    elif args.model == "stzinb":
        epochs = args.epochs or STZINB_EPOCHS
        patience = args.patience or STZINB_PATIENCE
        out_df = run_stzinb(args.dataset, args.lag, epochs, patience,
                             STZINB_N_MC, out_csv)
        extra = {"pseudo_count_scale": STZINB_SCALE, "n_mc": STZINB_N_MC,
                  "solver": NO_SOLVER_NOTE}

    elif args.model == "agcrn":
        epochs = args.epochs or AGCRN_EPOCHS
        patience = args.patience or AGCRN_PATIENCE
        out_df = run_agcrn(args.dataset, out_csv, lag=args.lag,
                            epochs=epochs, patience=patience)
        extra = {"solver": NO_SOLVER_NOTE}

    elif args.model == "diffstg":
        epochs = args.epochs or EPOCHS_DIFFSTG
        patience = args.patience or PATIENCE_DIFFSTG
        out_df = run_diffstg(args.dataset, args.lag, epochs, patience,
                              args.n_samples, args.n_samples_train, out_csv)
        extra = {"n_samples_mc": args.n_samples, "diffusion_steps": DIFFUSION_STEPS,
                  "solver": "fixed-length reverse diffusion trajectory "
                            f"({DIFFUSION_STEPS} steps per sample); "
                            "no ODE/SDE integrator, no rtol/atol"}

    elif args.model == "timegrad":
        epochs = args.epochs or EPOCHS_DIFFSTG
        patience = args.patience or PATIENCE_DIFFSTG
        out_df = run_timegrad(args.dataset, args.lag, epochs, patience,
                               args.n_samples, args.n_samples_train, out_csv)
        extra = {"n_samples_mc": args.n_samples, "diffusion_steps": DIFFUSION_STEPS,
                  "solver": "fixed-length reverse diffusion trajectory "
                            f"({DIFFUSION_STEPS} steps per sample); "
                            "no ODE/SDE integrator, no rtol/atol"}

    elif args.model == "ustd":
        epochs = args.epochs or EPOCHS_USTD
        patience = args.patience or PATIENCE_USTD
        out_df = run_ustd(args.dataset, args.lag, epochs, patience,
                           args.n_ensemble, out_csv)
        extra = {"n_ensemble": args.n_ensemble, "solver": NO_SOLVER_NOTE}

    else:  # stgncde
        epochs = args.epochs or EPOCHS_STGNCDE
        patience = args.patience or PATIENCE_STGNCDE
        out_df = run_stgncde(args.dataset, args.lag, epochs, patience, out_csv)
        cde_steps = max(1, args.lag - 1)  # ts spans 0..lag-1, rk4 step_size=1.0
        extra = {"solver": "fixed-step RK4 (torchcde.cdeint), step_size=1.0, "
                            f"~{cde_steps} internal steps per {args.lag}-month "
                            f"input window; no adaptive rtol/atol"}

    total_wall_clock = time.time() - t0
    print(f"\nTotal wall-clock: {total_wall_clock:.1f}s")
    # Persisted for Reviewer 2, point 10 (computational overhead vs.
    # GN-SDE) -- previously only printed to console, never saved, so
    # there was nothing to join against run_ablations.py's timing
    # columns.
    extra["total_wall_clock_sec"] = round(total_wall_clock, 1)

    if args.calibrate:
        print("\n" + "=" * 60)
        print(f"Running calibration on {args.model} ...")
        print("=" * 60)
        calibrate_and_append(out_df, args.dataset, args.model, extra_fields=extra)