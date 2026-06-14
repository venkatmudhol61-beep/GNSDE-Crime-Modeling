import os
import math
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchsde
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)
from model.gnsde_new32_final_paper import MODEL_MAP
import time

torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision("high")
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
# ─── CHANGE THESE 2 LINES ────────────────────────────────────────────
DATASET = "chicago_beats"  
VARIANT = "latent"
# ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

N_SAMPLES   = 100
PERCENTILES = [5, 95]

HPARAMS = {
    "chicago_beats"     : dict(lr=3e-4, mem_dim=16, dt=0.1),
    "chicago_districts" : dict(lr=3e-4, mem_dim=16, dt=0.1),
    "nyc_precincts"     : dict(lr=3e-4, mem_dim=16, dt=0.1),
}

# All hidden=32. Hierarchy comes from inductive bias, not capacity.
# Epoch budgets reflect convergence time of each architecture.
VARIANT_HPARAMS = {
    "fc": dict(
        hidden=32, epochs_p1=120, epochs_p2=80,
        patience=30, huber_w=1.0, nll_w=1.0,
    ),
    "spatial": dict(
        hidden=32, epochs_p1=140, epochs_p2=100,
        patience=40, huber_w=1.0, nll_w=1.0,
    ),
    "spatial_attn": dict(
        hidden=32, epochs_p1=250, epochs_p2=120,
        patience=40, huber_w=1.0, nll_w=1.0,
    ),
    "latent": dict(
        hidden=32, epochs_p1=200, epochs_p2=100,
        patience=40, huber_w=1.0, nll_w=1.0,
    ),
}

hp            = HPARAMS[DATASET]
vhp           = VARIANT_HPARAMS[VARIANT]
LR            = hp["lr"]
MEM_DIM       = hp["mem_dim"]
DT            = hp["dt"]
HIDDEN        = vhp["hidden"]
EPOCHS_PHASE1 = vhp["epochs_p1"]
EPOCHS_PHASE2 = vhp["epochs_p2"]
PATIENCE      = vhp["patience"]
HUBER_W       = vhp["huber_w"]
NLL_W         = vhp["nll_w"]
LAMBDA_L      = 1e-4
GRAD_CLIP     = 1.0
TRAIN_RATIO   = 0.7
VAL_RATIO     = 0.1
SDE_METHOD    = "euler"
CHUNK_SIZE    = 16
CHECKPOINT    = f"data/processed/{DATASET}_{VARIANT}_final_best.pt"

DT_MC        = 0.02
T_EVAL_TRAIN = torch.tensor([0.0, 1.0])
T_EVAL_MC    = torch.linspace(0.0, 1.0, 51)

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
cfg = CONFIGS[DATASET]

# ─────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────
crime        = pd.read_csv(cfg["crime"])
arrest       = pd.read_csv(cfg["arrest"])
region_order = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
regions      = region_order
months       = sorted(crime["month"].unique())
N_regions    = len(regions)
T            = len(months)

print(f"GN-SDE  |  dataset={DATASET}  variant={VARIANT}")
print(f"Regions={N_regions}  Months={T}  Solver=SDE  MC_samples={N_SAMPLES}")

C_all = torch.zeros(T, N_regions)
L_all = torch.zeros(T, N_regions)
for i, m in enumerate(months):
    c = (crime[crime["month"] == m]
         .set_index("region_id").reindex(regions)
         .fillna(0)["C"].values)
    l = (arrest[arrest["month"] == m]
         .set_index("region_id").reindex(regions)
         .fillna(0)["L"].values)
    C_all[i] = torch.tensor(c, dtype=torch.float32)
    L_all[i] = torch.tensor(l, dtype=torch.float32)

print(f"C range: min={C_all.min():.4f} max={C_all.max():.4f} mean={C_all.mean():.4f}")
print(f"L range: min={L_all.min():.4f} max={L_all.max():.4f} mean={L_all.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────
# 2. ADJACENCY
# ─────────────────────────────────────────────────────────────────────
adj_file = cfg["adj_fc"] if VARIANT == "fc" else cfg["adj_sp"]
if not os.path.exists(adj_file):
    raise FileNotFoundError(f"Not found: {adj_file}")
A_matrix = torch.tensor(np.load(adj_file), dtype=torch.float32)
print(f"Adjacency {A_matrix.shape}  nonzero={(A_matrix > 0).sum().item()}")

# ─────────────────────────────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────────────────────────────
def build_model():
    if VARIANT == "latent":
        return MODEL_MAP[VARIANT](
            A_matrix, T=T, alpha=0.3, beta=0.6,
            hidden_dim=HIDDEN, mem_dim=MEM_DIM)
    elif VARIANT == "spatial_attn":
        return MODEL_MAP[VARIANT](
            A_matrix, T=T, alpha=0.3, beta=0.6,
            hidden=HIDDEN, hidden_dim=HIDDEN * 2,
            n_heads=4, mem_dim=MEM_DIM)
    else:
        return MODEL_MAP[VARIANT](
            A_matrix, alpha=0.3, beta=0.6,
            hidden=HIDDEN, mem_dim=MEM_DIM)

model    = build_model()
n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"Model={model.__class__.__name__}  params={n_params:,}")
model.param_summary()

# ─────────────────────────────────────────────────────────────────────
# 4. LOSSES
# ─────────────────────────────────────────────────────────────────────
huber_loss   = nn.HuberLoss(delta=0.3)
gaussian_nll = nn.GaussianNLLLoss(full=False, eps=1e-6, reduction="mean")

# ─────────────────────────────────────────────────────────────────────
# 5. SPLIT
# ─────────────────────────────────────────────────────────────────────
n_train   = int(T * TRAIN_RATIO)
n_val     = int(T * VAL_RATIO)
train_idx = list(range(0, n_train))
val_idx   = list(range(n_train, n_train + n_val))
test_idx  = list(range(n_train + n_val, T))
print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

# ─────────────────────────────────────────────────────────────────────
# 6. STEP FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def model_step(C0):
    C_out = torchsde.sdeint(
        model, C0.unsqueeze(0), T_EVAL_TRAIN,
        method=SDE_METHOD, dt=DT,
    )[-1, 0]
    if VARIANT == "latent" and model._L_latent is not None:
        model._last_L_traj = model._L_latent.detach().unsqueeze(0)
    return C_out

def model_step_mc(C0):
    C_out = torchsde.sdeint(
        model, C0.unsqueeze(0), T_EVAL_MC,
        method=SDE_METHOD, dt=DT_MC,
    )[-1, 0]
    if VARIANT == "latent" and model._L_latent is not None:
        model._last_L_traj = model._L_latent.detach().unsqueeze(0)
    return C_out

# ─────────────────────────────────────────────────────────────────────
# 7. EPOCH
# ─────────────────────────────────────────────────────────────────────
def run_epoch(idx_list, train=True, phase=1):
    model.train() if train else model.eval()
    if train:
        opt.zero_grad()

    device = next(model.parameters()).device
    if hasattr(model, "reset_memory"):
        model.reset_memory(device)

    total_loss       = 0.0
    valid_steps      = 0
    pairs            = len(idx_list) - 1
    chunk_tensor     = None
    chunk_step_count = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for k in range(pairs):
            i      = idx_list[k]
            i_next = idx_list[k + 1]
            C0     = C_all[i]
            L      = L_all[i]
            C_true = C_all[i_next]

            model.set_context(L, t_idx=i, T=T)
            C_pred = model_step(C0)

            if (torch.isnan(C_pred).any() or
                    torch.isinf(C_pred).any()):
                if hasattr(model, "memory"):
                    model.memory.detach()
                if hasattr(model, "step_memory"):
                    model.step_memory(C0, L)
                continue

            if phase == 1:
                step_loss = huber_loss(C_pred, C_true)
            else:
                h_loss    = huber_loss(C_pred, C_true)
                var_now   = model.pred_var(C_pred)
                n_loss    = gaussian_nll(C_pred, C_true, var_now)
                step_loss = HUBER_W * h_loss + NLL_W * n_loss

            if not math.isfinite(step_loss.item()):
                if hasattr(model, "memory"):
                    model.memory.detach()
                continue

            if VARIANT == "latent" and train:
                if (hasattr(model, "_last_L_traj") and
                        model._last_L_traj is not None):
                    reg = model.L_regularisation_loss(model._last_L_traj)
                    if math.isfinite(reg.item()):
                        step_loss = step_loss + LAMBDA_L * reg

            if hasattr(model, "step_memory"):
                model.step_memory(C_pred, L)
            if hasattr(model, "memory"):
                model.memory.detach()
            if hasattr(model, "memory_L"):
                model.memory_L.detach()

            chunk_tensor      = step_loss if chunk_tensor is None \
                                else chunk_tensor + step_loss
            chunk_step_count += 1
            is_last  = (k == pairs - 1)
            is_chunk = (chunk_step_count == CHUNK_SIZE)

            if train and (is_chunk or is_last) and \
                    chunk_tensor is not None:
                (chunk_tensor / chunk_step_count).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad()
                cv = chunk_tensor.item()
                if math.isfinite(cv):
                    total_loss  += cv
                    valid_steps += chunk_step_count
                chunk_tensor     = None
                chunk_step_count = 0
            elif not train:
                lv = step_loss.item()
                if math.isfinite(lv):
                    total_loss  += lv
                    valid_steps += 1

    if valid_steps == 0:
        return float("nan")
    return total_loss / valid_steps

# ─────────────────────────────────────────────────────────────────────
# 8. PHASE 1 — Huber only, log_sigma_pred frozen
# ─────────────────────────────────────────────────────────────────────
model.log_sigma_pred.requires_grad_(False)

opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=60, T_mult=2, eta_min=1e-6)

best_val, patience_c, history = float("inf"), 0, []

print("\n" + "=" * 75)
print(f"PHASE 1 — Huber drift warmup  "
      f"[{VARIANT}  hidden={HIDDEN}  epochs={EPOCHS_PHASE1}]")
print("=" * 75)
print(f"{'Epoch':>6}  {'Train':>12}  {'Val':>12}  {'LR':>12}  "
      f"{'sigma_pred':>12}  {'Time(s)':>8}")
print("=" * 75)

t0 = time.time()
for epoch in range(1, EPOCHS_PHASE1 + 1):
    train_loss = run_epoch(train_idx, train=True,  phase=1)
    val_loss   = run_epoch(val_idx,   train=False, phase=1)
    lr_now     = opt.param_groups[0]["lr"]
    sp         = torch.exp(model.log_sigma_pred).detach().mean().item()
    scheduler.step(epoch)
    history.append({"epoch": epoch, "phase": 1,
                    "train": train_loss, "val": val_loss,
                    "sigma_pred": sp})

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
              f"{lr_now:>12.2e}  {sp:>12.4f}  {elapsed:>8.1f}")
        t0 = time.time()

    if not math.isfinite(val_loss):
        patience_c += 1
    elif val_loss < best_val:
        best_val, patience_c = val_loss, 0
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"         >> best val {best_val:.6f} — saved (epoch {epoch})")
    else:
        patience_c += 1

    if patience_c >= PATIENCE:
        print(f"\nPhase 1 early stop at epoch {epoch}")
        break

print(f"\nPhase 1 best val: {best_val:.6f}")

# ─────────────────────────────────────────────────────────────────────
# 9. PHASE 2 — Huber + NLL, sigma_pred calibrates at full LR
# ─────────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
model.log_sigma_pred.requires_grad_(True)

sigma_pred_params = [model.log_sigma_pred]
other_params      = [p for n, p in model.named_parameters()
                     if n != "log_sigma_pred" and p.requires_grad]

opt2 = torch.optim.AdamW([
    {"params": sigma_pred_params, "lr": LR,       "weight_decay": 0.0},
    {"params": other_params,      "lr": LR * 0.1, "weight_decay": 1e-4},
])
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt2, T_0=40, T_mult=2, eta_min=1e-7)
opt = opt2

best_val2, patience_c2 = float("inf"), 0

print("\n" + "=" * 75)
print(f"PHASE 2 — Huber+NLL calibration  "
      f"[sigma_pred LR={LR:.0e}  drift LR={LR*0.1:.0e}]")
print("=" * 75)
print(f"{'Epoch':>6}  {'Train':>12}  {'Val':>12}  {'LR_sigma':>12}  "
      f"{'sigma_pred':>12}  {'Time(s)':>8}")
print("=" * 75)

t0 = time.time()
for epoch in range(1, EPOCHS_PHASE2 + 1):
    train_loss = run_epoch(train_idx, train=True,  phase=2)
    val_loss   = run_epoch(val_idx,   train=False, phase=2)
    lr_sigma   = opt.param_groups[0]["lr"]
    sp         = torch.exp(model.log_sigma_pred).detach().mean().item()
    scheduler2.step(epoch)
    history.append({"epoch": EPOCHS_PHASE1 + epoch, "phase": 2,
                    "train": train_loss, "val": val_loss,
                    "sigma_pred": sp})

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
              f"{lr_sigma:>12.2e}  {sp:>12.4f}  {elapsed:>8.1f}")
        t0 = time.time()

    if not math.isfinite(val_loss):
        patience_c2 += 1
    elif val_loss < best_val2:
        best_val2, patience_c2 = val_loss, 0
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"         >> best val {best_val2:.6f} — saved (epoch {epoch})")
    else:
        patience_c2 += 1

    if patience_c2 >= PATIENCE:
        print(f"\nPhase 2 early stop at epoch {epoch}")
        break

print(f"\nPhase 2 best val: {best_val2:.6f}")
model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
model.param_summary()

# ─────────────────────────────────────────────────────────────────────
# 10. MC INFERENCE
# ─────────────────────────────────────────────────────────────────────
print(f"\nRunning MC inference ({N_SAMPLES} samples) ...")
model.eval()
device = next(model.parameters()).device

months_set_train = {months[i] for i in train_idx}
months_set_val   = {months[i] for i in val_idx}

all_samples = []
for s in range(N_SAMPLES):
    if s % 20 == 0:
        print(f"  sample {s}/{N_SAMPLES} ...")
    torch.manual_seed(s)
    model.reset_memory(device)
    sample_preds = []
    with torch.no_grad():
        for i in range(T - 1):
            C0 = C_all[i]
            L  = L_all[i]
            model.set_context(L, t_idx=i, T=T)
            C_pred = model_step_mc(C0)
            if hasattr(model, "step_memory"):
                model.step_memory(C_pred, L)
            sample_preds.append(C_pred.cpu())
    all_samples.append(torch.stack(sample_preds))

samples_tensor = torch.stack(all_samples)
pred_mean      = samples_tensor.mean(0)
mc_var         = samples_tensor.var(0)
sigma_pred     = torch.exp(model.log_sigma_pred).detach().cpu()
total_var      = mc_var + sigma_pred.unsqueeze(0).pow(2)
total_std      = total_var.sqrt()

# Empirical z-calibration on validation set
val_start  = val_idx[0]
val_end    = val_idx[-1]
val_preds  = pred_mean[val_start : val_end]
val_stds   = total_std[val_start : val_end]
val_true   = C_all[val_start + 1 : val_end + 1][:val_preds.shape[0]]
std_resid  = ((val_true - val_preds).abs()
              / val_stds.clamp(min=1e-6))
z_empirical = float(np.clip(
    torch.quantile(std_resid.flatten(), 0.90).item(), 0.5, 4.0))
print(f"\n  Empirical z@90%: {z_empirical:.3f}  (Gaussian=1.282)")

pred_p05 = pred_mean - z_empirical * total_std
pred_p95 = pred_mean + z_empirical * total_std
pred_std  = total_std

# ─────────────────────────────────────────────────────────────────────
# 11. SAVE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────
results = []
for i in range(T - 1):
    m           = months[i + 1]
    split_label = ("train" if m in months_set_train else
                   "val"   if m in months_set_val   else "test")
    for j, region in enumerate(regions):
        results.append({
            "month"     : m,
            "region_id" : region,
            "C_gnsde"   : float(pred_mean[i, j]),
            "C_std"     : float(pred_std[i, j]),
            "C_p05"     : float(pred_p05[i, j]),
            "C_p95"     : float(pred_p95[i, j]),
            "split"     : split_label,
        })

pred_df  = pd.DataFrame(results)
out_pred = f"data/processed/{DATASET}_{VARIANT}_final_predictions.csv"
pred_df.to_csv(out_pred, index=False)
print(f"Saved predictions -> {out_pred}")

# ─────────────────────────────────────────────────────────────────────
# 12. TEST METRICS
# ─────────────────────────────────────────────────────────────────────
test_pred          = pred_df[pred_df["split"] == "test"].copy()
real_df            = pd.read_csv(cfg["crime"])
real_df["month"]   = real_df["month"].astype(str)
test_pred["month"] = test_pred["month"].astype(str)
merged             = pd.merge(real_df, test_pred,
                              on=["month", "region_id"], how="inner")

if merged.empty:
    print("WARNING: merge empty.")
else:
    y_true = merged["C"].values
    y_pred = merged["C_gnsde"].values
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    smape  = float(np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)

    in_interval    = ((merged["C"] >= merged["C_p05"]) &
                      (merged["C"] <= merged["C_p95"]))
    coverage_90    = in_interval.mean() * 100
    interval_width = (merged["C_p95"] - merged["C_p05"]).mean()

    print(f"\n===== {DATASET.upper()}  {VARIANT.upper()} — TEST =====")
    print(f"  MAE            : {mae:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  R2             : {r2:.4f}")
    print(f"  sMAPE          : {smape:.2f}%")
    print(f"  Coverage@90%   : {coverage_90:.1f}%  (target=90%)")
    print(f"  Interval width : {interval_width:.4f}")
    print(f"  Mean total std : {pred_std[len(train_idx):].mean():.4f}")
    print(f"  sigma_pred mean: {sigma_pred.mean():.4f}")
    print(f"  Empirical z    : {z_empirical:.3f}")
    print("=" * 48)

# ─────────────────────────────────────────────────────────────────────
# 13. LATENT ENFORCEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────
if VARIANT == "latent":
    print("\nCollecting latent enforcement schedule ...")
    model.eval()
    model.reset_memory(device)
    latent_rows = []
    with torch.no_grad():
        for i in range(T - 1):
            C0 = C_all[i]
            L  = L_all[i]
            model.set_context(L, t_idx=i, T=T)
            C_out = model_step(C0)
            L_new = model.update_L(C_out)
            model.step_memory(C_out, L_new)
            for j, region in enumerate(regions):
                latent_rows.append({
                    "month"    : months[i + 1],
                    "region_id": region,
                    "L_latent" : float(L_new[j].item()),
                })
    latent_df  = pd.DataFrame(latent_rows)
    out_latent = (f"data/processed/{DATASET}_{VARIANT}"
                  f"_final_enforcement.csv")
    latent_df.to_csv(out_latent, index=False)
    L_values = latent_df["L_latent"].values
    print(f"  L̂: mean={L_values.mean():.4f} std={L_values.std():.4f} "
          f"min={L_values.min():.4f} max={L_values.max():.4f}")
    real_df["month"]   = real_df["month"].astype(str)
    latent_df["month"] = latent_df["month"].astype(str)
    check = pd.merge(real_df, latent_df,
                     on=["region_id", "month"], how="inner")
    corr  = check[["C", "L_latent"]].corr().iloc[0, 1]
    print(f"  Corr(L̂,C)={corr:.4f}")
    print(f"  Saved -> {out_latent}")
# ─────────────────────────────────────────────────────────────────────
# 13b. HIERARCHY DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────
print("\n===== HIERARCHY DIAGNOSTICS =====")

if VARIANT == "spatial_attn":
    print("\n[spatial_attn] m_beta modulator analysis ...")
    model.eval()
    m_beta_all = []
    with torch.no_grad():
        for i in range(T - 1):
            L = L_all[i]
            model.set_context(L, t_idx=i, T=T)
            m_beta = model._beta_modulator(L)
            m_beta_all.append(m_beta.cpu())
    m_beta_tensor = torch.stack(m_beta_all)  # (T-1, N)

    print(f"  m_beta global: mean={m_beta_tensor.mean():.4f} "
          f"std={m_beta_tensor.std():.4f} "
          f"min={m_beta_tensor.min():.4f} "
          f"max={m_beta_tensor.max():.4f}")
    print(f"  m_beta per-region std (mean across regions): "
          f"{m_beta_tensor.std(0).mean():.4f}")
    print(f"  m_beta per-time std (mean across time): "
          f"{m_beta_tensor.std(1).mean():.4f}")

    if m_beta_tensor.std() < 0.01:
        print("  WARNING: m_beta is nearly constant "
              "-> attn collapsed to spatial on this dataset")
        print("  INTERPRETATION: enforcement sensitivity "
              "is uniform — spatial sufficient")
    else:
        print("  OK: m_beta shows variation "
              "-> attention mechanism is active")

    # Save for plotting
    import pandas as pd
    mb_rows = []
    for i in range(T - 1):
        for j, region in enumerate(regions):
            mb_rows.append({
                "month"    : months[i + 1],
                "region_id": region,
                "m_beta"   : float(m_beta_all[i][j].item()),
            })
    mb_df = pd.DataFrame(mb_rows)
    out_mb = f"data/processed/{DATASET}_{VARIANT}_mbeta.csv"
    mb_df.to_csv(out_mb, index=False)
    print(f"  Saved m_beta -> {out_mb}")

    # Rho
    print(f"  rho (Tobler): {model.rho.item():.4f}")
    if model.rho.item() < 0.01:
        print("  WARNING: rho near zero -> "
              "Tobler term inactive")


if VARIANT == "latent":
    print("\n[latent] obs_gate and L_latent analysis ...")
    og = model.obs_gate.detach()
    print(f"  obs_gate: mean={og.mean():.4f} "
          f"std={og.std():.4f} "
          f"min={og.min():.4f} "
          f"max={og.max():.4f}")

    if og.mean() > 0.85:
        print("  WARNING: obs_gate >> 0.5 "
              "-> latent mostly ignoring L_net inference")
        print("  INTERPRETATION: observed L sufficient "
              "-> latent collapses toward spatial_attn")
    elif og.mean() < 0.15:
        print("  WARNING: obs_gate << 0.5 "
              "-> model ignoring observed L entirely")
        print("  INTERPRETATION: L_net dominant "
              "-> check L_net plausibility")
    else:
        print("  OK: obs_gate balanced "
              "-> latent blending observed + inferred L")

    # L_latent vs L_observed correlation per split
    if os.path.exists(
            f"data/processed/{DATASET}_{VARIANT}_final_enforcement.csv"):
        enf_df = pd.read_csv(
            f"data/processed/{DATASET}_{VARIANT}_final_enforcement.csv")
        arr_df = pd.read_csv(cfg["arrest"])
        enf_df["month"] = enf_df["month"].astype(str)
        arr_df["month"] = arr_df["month"].astype(str)
        merged_L = pd.merge(enf_df, arr_df,
                            on=["month", "region_id"], how="inner")
        if not merged_L.empty:
            corr_all = merged_L[["L_latent", "L"]].corr().iloc[0, 1]
            print(f"  Corr(L_latent, L_observed) global: {corr_all:.4f}")
            if corr_all > 0.90:
                print("  WARNING: L_latent nearly identical to L_obs "
                      "-> latent adds no new enforcement information")
            elif corr_all < 0.30:
                print("  NOTE: L_latent diverges from L_obs "
                      "-> latent inferring different enforcement signal")
            else:
                print("  OK: partial correlation "
                      "-> latent blending and adjusting L_obs")

    # L_latent range check
    if model._L_latent is not None:
        Ll = model._L_latent.detach()
        print(f"  L_latent (last step): "
              f"mean={Ll.mean():.4f} std={Ll.std():.4f} "
              f"min={Ll.min():.4f} max={Ll.max():.4f}")
        if Ll.std() < 0.01:
            print("  WARNING: L_latent nearly uniform across regions "
                  "-> node_emb not differentiating regions")


if VARIANT == "spatial":
    print("\n[spatial] Tobler term analysis ...")
    rho = model.rho.item()
    print(f"  rho (Tobler coefficient): {rho:.4f}")
    if rho < 0.01:
        print("  WARNING: rho near zero -> "
              "Tobler term not contributing")
        print("  INTERPRETATION: spatial autocorrelation "
              "weak in this dataset")
    elif rho > 0.3:
        print("  NOTE: rho large -> strong Tobler smoothing "
              "-> likely explains performance on this dataset")
    else:
        print("  OK: moderate Tobler smoothing active")

    # Spatial autocorrelation of C (proxy for Moran's I)
    with torch.no_grad():
        C_last = C_all[-1]
        AC_last = torch.mv(model.A, C_last)
        moranish = float(
            torch.corrcoef(
                torch.stack([C_last, AC_last])
            )[0, 1].item()
        )
    print(f"  Pseudo-Moran's I (corr(C, AC)): {moranish:.4f}")
    if moranish > 0.5:
        print("  NOTE: high spatial autocorrelation "
              "-> explains why spatial wins on this dataset")


if VARIANT == "fc":
    print("\n[fc] baseline analysis ...")
    with torch.no_grad():
        C_last = C_all[-1]
        AC_last = torch.mv(model.A, C_last)
        corr_fc = float(
            torch.corrcoef(
                torch.stack([C_last, AC_last])
            )[0, 1].item()
        )
    print(f"  Corr(C, AC_fc): {corr_fc:.4f}")
    print(f"  FC adjacency edges: {(model.A > 0).sum().item()}")

print("\n===== END DIAGNOSTICS =====")
# ─────────────────────────────────────────────────────────────────────
# 14. HISTORY
# ─────────────────────────────────────────────────────────────────────
out_hist = f"data/processed/{DATASET}_{VARIANT}_final_history.csv"
pd.DataFrame(history).to_csv(out_hist, index=False)
print(f"History -> {out_hist}")
print("\nDone.")