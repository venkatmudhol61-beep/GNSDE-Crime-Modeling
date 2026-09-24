import os  #table 1 currected file
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchdiffeq import odeint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model.gnode_paper import MODEL_MAP

# ─── CHANGE THESE 2 LINES ONLY ──────────────────────────────────────
DATASET = "nyc_precincts"   # "chicago_beats" | "chicago_districts" | "nyc_precincts"
VARIANT = "latent"       # "fc" | "spatial" | "spatial_attn" | "latent"
# ────────────────────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

print(f"Training GN-ODE  |  dataset={DATASET}  variant={VARIANT}")

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS — NYC gets gentler LR and more epochs
# ─────────────────────────────────────────────────────────────────────────────
HPARAMS = {
    "chicago_beats"     : {"lr": 5e-4, "epochs": 200, "patience": 20},
    "chicago_districts" : {"lr": 5e-4, "epochs": 200, "patience": 20},
    "nyc_precincts"     : {"lr": 5e-4, "epochs": 400, "patience": 40},
}
hp       = HPARAMS[DATASET]
EPOCHS   = hp["epochs"]
LR       = hp["lr"]
PATIENCE = hp["patience"]

LR_L        = 5e-3
LAMBDA_L    = 1e-3
GRAD_CLIP   = 0.5
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
ODE_METHOD  = "rk4"
CHECKPOINT  = f"data/processed/{DATASET}_{VARIANT}_best.pt"

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

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
crime        = pd.read_csv(cfg["crime"])
arrest       = pd.read_csv(cfg["arrest"])
region_order = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
regions      = region_order
months       = sorted(crime["month"].unique())
N            = len(regions)
T            = len(months)

print(f"Regions   : {N}")
print(f"Months    : {T}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD ADJACENCY
# ─────────────────────────────────────────────────────────────────────────────
adj_file = cfg["adj_fc"] if VARIANT == "fc" else cfg["adj_sp"]

if not os.path.exists(adj_file):
    raise FileNotFoundError(
        f"\nAdjacency file not found: {adj_file}"
        f"\nBuild it first with your adjacency builder script."
    )
A_matrix = torch.tensor(np.load(adj_file), dtype=torch.float32)
print(f"Adjacency : {A_matrix.shape}  nonzero={(A_matrix > 0).sum().item()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. INIT MODEL
# ─────────────────────────────────────────────────────────────────────────────
# ── Section 3: INIT MODEL — replace the existing block ──────────────
if VARIANT == "latent":
    model = MODEL_MAP[VARIANT](A_matrix, T=T, alpha=0.3, beta=0.6)
elif VARIANT == "spatial_atten":
    model = MODEL_MAP[VARIANT](A_matrix, T=T, alpha=0.3, beta=0.6)   # T now required
else:
    model = MODEL_MAP[VARIANT](A_matrix, alpha=0.3, beta=0.6)

print(f"Model     : {model.__class__.__name__}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
if VARIANT == "latent":
    base_params = [p for n, p in model.named_parameters()
                   if "L_net" not in n and "node_emb" not in n]
    lnet_params = [p for n, p in model.named_parameters()
                   if "L_net" in n or "node_emb" in n]
    opt = torch.optim.Adam([
        {"params": base_params, "lr": LR},
        {"params": lnet_params, "lr": LR_L},
    ])
else:
    opt = torch.optim.Adam(model.parameters(), lr=LR)

loss_fn   = nn.MSELoss()
t_span    = torch.tensor([0.0, 1.0])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=10, min_lr=1e-6, verbose=False
)

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
model.param_summary()

# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER
# ─────────────────────────────────────────────────────────────────────────────
def get_state(month):
    C = (crime[crime["month"] == month]
         .set_index("region_id").reindex(regions).fillna(0)["C"].values)
    L = (arrest[arrest["month"] == month]
         .set_index("region_id").reindex(regions).fillna(0)["L"].values)
    return torch.tensor(C, dtype=torch.float32), torch.tensor(L, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 6. SPLIT
# ─────────────────────────────────────────────────────────────────────────────
n_train      = int(T * TRAIN_RATIO)
n_val        = int(T * VAL_RATIO)
months_train = months[:n_train]
months_val   = months[n_train : n_train + n_val]
months_test  = months[n_train + n_val :]
print(f"Split     : train={len(months_train)}  val={len(months_val)}  test={len(months_test)}")

month_to_idx = {m: i for i, m in enumerate(months)}

# ─────────────────────────────────────────────────────────────────────────────
# 7. ODE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def make_ode_fn(month_idx):
    if VARIANT in ("latent", "attention"):          # both need t_idx
        return lambda t, y: model(t, y, month_idx)
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 8. EPOCH FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(month_list, train=True):
    if train:
        model.train()
        opt.zero_grad()
    else:
        model.eval()

    total_loss = 0.0
    pairs      = len(month_list) - 1

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i in range(pairs):
            C0, L     = get_state(month_list[i])
            C_true, _ = get_state(month_list[i + 1])

            # set context before odeint — L and month index
            model.set_context(L, t_idx=i)

            C_pred = odeint(
                model,
                C0,
                t_span,
                method=ODE_METHOD,
            )[-1]

            total_loss += loss_fn(C_pred, C_true)

        mean_loss = total_loss / pairs

        if train:
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

    return mean_loss.item()
# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
best_val, patience_c, history = float("inf"), 0, []

print("\n" + "=" * 58)
print(f"{'Epoch':>6} {'Train':>12} {'Val':>12} {'LR':>10}")
print("=" * 58)

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(months_train, train=True)
    val_loss   = run_epoch(months_val,   train=False)
    lr_now     = opt.param_groups[0]["lr"]
    scheduler.step(val_loss)
    history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
    print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {lr_now:>10.2e}")
    if val_loss < best_val:
        best_val, patience_c = val_loss, 0
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"         >> best val {best_val:.6f} — saved")
    else:
        patience_c += 1
        if patience_c >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nBest val loss : {best_val:.6f}")
model.param_summary()

# ─────────────────────────────────────────────────────────────────────────────
# 10. PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating predictions from best checkpoint ...")
model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
model.eval()

months_set_train = set(months_train)
months_set_val   = set(months_val)
results          = []

with torch.no_grad():
    for i in range(T - 1):
        C0, L  = get_state(months[i])
        t_idx  = month_to_idx[months[i]]
        model.set_context(L, t_idx=t_idx)
        C_pred = odeint(model, C0, t_span, method=ODE_METHOD)[-1]  # pass C0 not y0

        m = months[i + 1]
        split_label = (
            "train" if m in months_set_train else
            "val"   if m in months_set_val   else
            "test"
        )
        for region, value in zip(regions, C_pred.numpy()):
            results.append({
                "month"     : m,
                "region_id" : region,
                "C_gnode"   : float(value),
                "split"     : split_label,
            })

pred_df  = pd.DataFrame(results)
out_pred = f"data/processed/{DATASET}_{VARIANT}_predictions.csv"
pred_df.to_csv(out_pred, index=False)
print(f"Saved → {out_pred}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. TEST METRICS
# ─────────────────────────────────────────────────────────────────────────────
test_pred          = pred_df[pred_df["split"] == "test"].copy()
real_df            = pd.read_csv(cfg["crime"])
real_df["month"]   = real_df["month"].astype(str)
test_pred["month"] = test_pred["month"].astype(str)
merged             = pd.merge(real_df, test_pred, on=["month", "region_id"], how="inner")

if merged.empty:
    print("WARNING: merge empty — check region_id and month columns match.")
else:
    y_true = merged["C"].values
    y_pred = merged["C_gnode"].values
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    smape  = float(np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)
    print(f"\n===== {DATASET.upper()}  {VARIANT.upper()}  — TEST SET =====")
    print(f"  MAE   : {mae:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  R2    : {r2:.4f}")
    print(f"  sMAPE : {smape:.2f}%")
    print("=" * 45)

# ─────────────────────────────────────────────────────────────────────────────
# 12. LATENT ENFORCEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if VARIANT == "latent":
    print("\nGenerating learned enforcement analysis ...")
    latent_df  = model.get_learned_enforcement(regions, months)
    out_latent = f"data/processed/{DATASET}_{VARIANT}_enforcement.csv"
    latent_df.to_csv(out_latent, index=False)
    print(f"Saved → {out_latent}")
    L_values = latent_df["L_latent"].values
    print(f"\n===== LEARNED ENFORCEMENT ANALYSIS =====")
    print(f"  Mean : {L_values.mean():.4f}")
    print(f"  Std  : {L_values.std():.4f}")
    print(f"  Min  : {L_values.min():.4f}")
    print(f"  Max  : {L_values.max():.4f}")
    mean_by_region = (
        latent_df.groupby("region_id")["L_latent"]
        .mean().sort_values(ascending=False)
    )
    print("\n  Top 3 regions by enforcement intensity:")
    for region, val in mean_by_region.head(3).items():
        print(f"    Region {region}: L = {val:.4f}")
    real_df["month"]   = real_df["month"].astype(str)
    latent_df["month"] = latent_df["month"].astype(str)
    check = pd.merge(real_df, latent_df, on=["region_id", "month"], how="inner")
    corr  = check[["C", "L_latent"]].corr().iloc[0, 1]
    print(f"\n  Correlation of learned L with crime C : {corr:.4f}")
    if abs(corr) < 0.2:
        print("  Excellent — learned L is largely orthogonal to crime")
    elif abs(corr) < 0.4:
        print("  Acceptable — some correlation but model is disentangling signals")
    else:
        print("  High — consider increasing LAMBDA_L regularisation")
    print("=" * 45)

# ─────────────────────────────────────────────────────────────────────────────
# 13. SAVE TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────────────────
out_hist = f"data/processed/{DATASET}_{VARIANT}_training_history.csv"
pd.DataFrame(history).to_csv(out_hist, index=False)
print(f"History → {out_hist}")