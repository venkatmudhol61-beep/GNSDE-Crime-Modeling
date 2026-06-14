import os
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchdiffeq import odeint
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)
from model.gnode_paper import MODEL_MAP
import time

torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision("high")
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

# ─── CHANGE THESE 2 LINES ────────────────────────────────────────────
DATASET = "chicago_beats"
VARIANT = "spatial_attn"
# ─────────────────────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

ODE_VARIANTS = {"fc", "spatial", "spatial_attn", "latent"}
SDE_VARIANTS = set()

HPARAMS = {
    "chicago_beats"     : dict(lr=3e-4, epochs=200, patience=20,
                               hidden=16, mem_dim=16, dt=0.1),
    "chicago_districts" : dict(lr=3e-4, epochs=200, patience=20,
                               hidden=16, mem_dim=16, dt=0.1),
    "nyc_precincts"     : dict(lr=3e-4, epochs=200, patience=20,
                               hidden=16, mem_dim=16, dt=0.1),
}
hp          = HPARAMS[DATASET]
EPOCHS      = hp["epochs"]
LR          = hp["lr"]
PATIENCE    = hp["patience"]
HIDDEN      = hp["hidden"]
MEM_DIM     = hp["mem_dim"]
DT          = hp["dt"]
LAMBDA_L    = 1e-4
GRAD_CLIP   = 1.0
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
ODE_METHOD  = "euler"
CHUNK_SIZE  = 16
CHECKPOINT  = f"data/processed/{DATASET}_{VARIANT}_final_best.pt"

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

print(f"GN-ODE  |  dataset={DATASET}  variant={VARIANT}")
print(f"Regions={N_regions}  Months={T}  Solver=ODE")

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

print(f"C range: min={C_all.min():.4f} max={C_all.max():.4f} "
      f"mean={C_all.mean():.4f}")
print(f"L range: min={L_all.min():.4f} max={L_all.max():.4f} "
      f"mean={L_all.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────
# 2. ADJACENCY
# ─────────────────────────────────────────────────────────────────────
adj_file = cfg["adj_fc"] if VARIANT == "fc" else cfg["adj_sp"]
if not os.path.exists(adj_file):
    raise FileNotFoundError(f"Not found: {adj_file}")
A_matrix = torch.tensor(np.load(adj_file), dtype=torch.float32)
print(f"Adjacency {A_matrix.shape}  "
      f"nonzero={(A_matrix > 0).sum().item()}")

# ─────────────────────────────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────────────────────────────
def build_model():
    if VARIANT == "latent":
        return MODEL_MAP[VARIANT](
            A_matrix, T=T, alpha=0.3, beta=0.6,
            hidden_dim=HIDDEN)
    elif VARIANT == "spatial_attn":
        return MODEL_MAP[VARIANT](
            A_matrix, T=T, alpha=0.3, beta=0.6,
            hidden=HIDDEN, hidden_dim=32,
            n_heads=4)
    else:
        return MODEL_MAP[VARIANT](
            A_matrix, alpha=0.3, beta=0.6,
            hidden=HIDDEN)
model    = build_model()
n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"Model={model.__class__.__name__}  params={n_params:,}")
model.param_summary()

# ─────────────────────────────────────────────────────────────────────
# 4. OPTIMISER
# ─────────────────────────────────────────────────────────────────────
opt       = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=1e-4)
loss_fn   = nn.HuberLoss(delta=0.3)
t_eval    = torch.tensor([0.0, 1.0])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=60, T_mult=2, eta_min=1e-6)

# ─────────────────────────────────────────────────────────────────────
# 5. SPLIT
# ─────────────────────────────────────────────────────────────────────
n_train   = int(T * TRAIN_RATIO)
n_val     = int(T * VAL_RATIO)
train_idx = list(range(0, n_train))
val_idx   = list(range(n_train, n_train + n_val))
test_idx  = list(range(n_train + n_val, T))
print(f"Split: train={len(train_idx)} "
      f"val={len(val_idx)} test={len(test_idx)}")

# ─────────────────────────────────────────────────────────────────────
# 6. STEP
# ─────────────────────────────────────────────────────────────────────
def model_step(C0):
    func = lambda t, C: model.ode_func(t, C)
    C_traj = odeint(
        func, C0, t_eval,
        method=ODE_METHOD,
        options={"step_size": DT},
    )
    C_out = C_traj[-1]
    if VARIANT == "latent" and model._L_latent is not None:
        model._last_L_traj = model._L_latent.detach().unsqueeze(0)
    return C_out

# ─────────────────────────────────────────────────────────────────────
# 7. EPOCH
# ─────────────────────────────────────────────────────────────────────
def run_epoch(idx_list, train=True):
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

            # unified — fc/spatial ignore T safely
            model.set_context(L, t_idx=i, T=T)

            C_pred = model_step(C0)

            if (torch.isnan(C_pred).any() or
                    torch.isinf(C_pred).any()):
                if hasattr(model, "memory"):
                    model.memory.detach()
                if hasattr(model, "step_memory"):
                    model.step_memory(C0, L)
                continue

            step_loss = loss_fn(C_pred, C_true)

            if not math.isfinite(step_loss.item()):
                if hasattr(model, "memory"):
                    model.memory.detach()
                continue

            if VARIANT == "latent" and train:
                if (hasattr(model, "_last_L_traj") and
                        model._last_L_traj is not None):
                    reg = model.L_regularisation_loss(
                        model._last_L_traj)
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
# 8. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────
best_val, patience_c, history = float("inf"), 0, []

print("\n" + "=" * 70)
print(f"{'Epoch':>6}  {'Train':>12}  {'Val':>12}  "
      f"{'LR':>12}  {'Time(s)':>8}")
print("=" * 70)

t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_idx, train=True)
    val_loss   = run_epoch(val_idx,   train=False)
    lr_now     = opt.param_groups[0]["lr"]
    scheduler.step(epoch)
    history.append({"epoch": epoch,
                    "train": train_loss,
                    "val":   val_loss})

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>12.6f}  "
              f"{val_loss:>12.6f}  {lr_now:>12.2e}  "
              f"{elapsed:>8.1f}")
        t0 = time.time()

    if not math.isfinite(val_loss):
        patience_c += 1
    elif val_loss < best_val:
        best_val, patience_c = val_loss, 0
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"         >> best val {best_val:.6f} "
              f"— saved (epoch {epoch})")
    else:
        patience_c += 1

    if patience_c >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print(f"\nBest val loss: {best_val:.6f}")
model.param_summary()

# ─────────────────────────────────────────────────────────────────────
# 9. PREDICTIONS
# ─────────────────────────────────────────────────────────────────────
print("\nGenerating predictions ...")
model.load_state_dict(
    torch.load(CHECKPOINT, weights_only=True))
model.eval()

device = next(model.parameters()).device
if hasattr(model, "reset_memory"):
    model.reset_memory(device)

months_set_train = {months[i] for i in train_idx}
months_set_val   = {months[i] for i in val_idx}
results          = []

with torch.no_grad():
    for i in range(T - 1):
        C0 = C_all[i]
        L  = L_all[i]

        model.set_context(L, t_idx=i, T=T)
        C_pred = model_step(C0)

        if hasattr(model, "step_memory"):
            model.step_memory(C_pred, L)

        m           = months[i + 1]
        split_label = ("train" if m in months_set_train else
                       "val"   if m in months_set_val   else "test")
        for region, value in zip(regions, C_pred.numpy()):
            results.append({"month"    : m,
                            "region_id": region,
                            "C_gnode"  : float(value),
                            "split"    : split_label})

pred_df  = pd.DataFrame(results)
out_pred = (f"data/processed/{DATASET}_{VARIANT}"
            f"_final_predictions.csv")
pred_df.to_csv(out_pred, index=False)
print(f"Saved -> {out_pred}")

# ─────────────────────────────────────────────────────────────────────
# 10. TEST METRICS
# ─────────────────────────────────────────────────────────────────────
test_pred          = pred_df[pred_df["split"] == "test"].copy()
real_df            = pd.read_csv(cfg["crime"])
real_df["month"]   = real_df["month"].astype(str)
test_pred["month"] = test_pred["month"].astype(str)
merged             = pd.merge(real_df, test_pred,
                              on=["month", "region_id"],
                              how="inner")

if merged.empty:
    print("WARNING: merge empty.")
else:
    y_true = merged["C"].values
    y_pred = merged["C_gnode"].values
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    smape  = float(np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)
    print(f"\n===== {DATASET.upper()}  {VARIANT.upper()} "
          f"— TEST =====")
    print(f"  MAE   : {mae:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  R2    : {r2:.4f}")
    print(f"  sMAPE : {smape:.2f}%")
    print("=" * 48)

# ─────────────────────────────────────────────────────────────────────
# 11. LATENT ENFORCEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────
if VARIANT == "latent":
    print("\nCollecting latent enforcement schedule ...")
    model.eval()
    latent_rows = []

    with torch.no_grad():
        for i in range(T - 1):
            C0 = C_all[i]
            L  = L_all[i]
            model.set_context(L, t_idx=i, T=T)
            C_out = model_step(C0)
            L_new = model.update_L(C_out)
            for j, region in enumerate(regions):
                latent_rows.append({
                    "month"    : months[i + 1],
                    "region_id": region,
                    "L_latent" : float(L_new[j].item()),
                })
# ─────────────────────────────────────────────────────────────────────
# 12. HISTORY
# ─────────────────────────────────────────────────────────────────────
out_hist = (f"data/processed/{DATASET}_{VARIANT}"
            f"_final_history.csv")
pd.DataFrame(history).to_csv(out_hist, index=False)
print(f"History -> {out_hist}")