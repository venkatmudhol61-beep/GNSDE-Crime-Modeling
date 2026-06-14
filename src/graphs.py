"""
visualize_fc_gnode.py
=====================

Visualization for Fully Connected GN-ODE predictions.

Fixes included:
  1. Uses region_id instead of District
  2. Safe merge on month
  3. Burn-in applied AFTER merge
  4. Timestamp normalization
  5. Debug prints added
  6. Handles empty merge safely

Run from project root:
    python src/visualize_fc_gnode.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Load real data
# ======================
real = pd.read_csv("data/processed/chicago_districts_crime_timeseries.csv")

# Normalize timestamps
real["month"] = (
    pd.to_datetime(real["month"])
      .dt.to_period("M")
      .dt.to_timestamp()
)

# ======================
# Load GNODE predictions
# ======================
gnode = pd.read_csv("data/processed/chicago_districts_latent_final_predictions.csv")

# Normalize timestamps
gnode["month"] = (
    pd.to_datetime(gnode["month"])
      .dt.to_period("M")
      .dt.to_timestamp()
)

# Ensure non-negative predictions
gnode["C_gnode"] = gnode["C_gnode"].clip(lower=0)

# ======================
# Debug info
# ======================
print("\n===== DEBUG INFO =====")

print("\nReal columns:")
print(real.columns.tolist())

print("\nPrediction columns:")
print(gnode.columns.tolist())

print("\nUnique region_ids in real:")
print(sorted(real["region_id"].unique())[:20])

print("\nUnique region_ids in predictions:")
print(sorted(gnode["region_id"].unique())[:20])

# ======================
# Select region
# ======================
region_id_val = 1
BURN_IN = 12

r = (
    real[real["region_id"] == region_id_val]
    .sort_values("month")
    .reset_index(drop=True)
)

g = (
    gnode[gnode["region_id"] == region_id_val]
    .sort_values("month")
    .reset_index(drop=True)
)

print(f"\nRows in real region {region_id_val}: {len(r)}")
print(f"Rows in pred region {region_id_val}: {len(g)}")

# ======================
# Merge safely on month
# ======================
merged = pd.merge(
    r[["month", "C"]],
    g[["month", "C_gnode", "split"]],
    on="month",
    how="inner"
).sort_values("month").reset_index(drop=True)

print(f"Merged rows BEFORE burn-in: {len(merged)}")

# ======================
# Apply burn-in
# ======================
if len(merged) <= BURN_IN:
    print("\nERROR: Burn-in removes all rows.")
    print("Set BURN_IN = 0 or choose another region.")
    exit()

merged = merged.iloc[BURN_IN:].reset_index(drop=True)

print(f"Merged rows AFTER burn-in: {len(merged)}")

if merged.empty:
    print("\nERROR: merged dataframe is empty.")
    print("Months or region_ids do not match.")
    exit()

print("\nMerged head:")
print(merged.head())

# ======================
# Compute split boundaries
# ======================
all_months = sorted(real["month"].unique())

T = len(all_months)

n_train = int(T * 0.7)
n_val   = int(T * 0.1)

val_start  = all_months[n_train]
test_start = all_months[n_train + n_val]

print("\n===== DATA SUMMARY =====")
print(f"Regions in real data : {real['region_id'].nunique()}")
print(f"Total months         : {T}")
print(f"Train / Val / Test   : {n_train} / {n_val} / {T - n_train - n_val}")
print(f"Val starts at        : {val_start.date()}")
print(f"Test starts at       : {test_start.date()}")

# ============================================================
# FIGURE 1 — Trajectory
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    merged["month"],
    merged["C"],
    label="Real Crime",
    linewidth=1.5,
    alpha=0.7,
    color="steelblue",
)

ax.plot(
    merged["month"],
    merged["C_gnode"],
    label="FC GN-SDE Prediction",
    linewidth=2,
    color="darkorange",
)

ax.axvline(
    val_start,
    linestyle="--",
    color="green",
    alpha=0.8,
    label=f"Val Start ({val_start.date()})",
)

ax.axvline(
    test_start,
    linestyle="--",
    color="red",
    alpha=0.8,
    label=f"Test Start ({test_start.date()})",
)

ax.set_title(
    f"Latent GN-SDE vs Real Crime — Region {region_id_val}",
    fontsize=14,
)

ax.set_ylabel("Crime Intensity (Normalized)")
ax.set_xlabel("Time")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "trajectory_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: trajectory_plot.png")

# ============================================================
# FIGURE 2 — Residuals
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))

residual = merged["C_gnode"].values - merged["C"].values

ax.plot(
    merged["month"],
    residual,
    color="black",
    linewidth=1,
    label="Residual"
)

ax.axhline(
    0,
    linestyle="--",
    color="grey",
    alpha=0.6
)

ax.axvline(
    val_start,
    linestyle="--",
    color="green",
    alpha=0.8,
    label=f"Val Start ({val_start.date()})",
)

ax.axvline(
    test_start,
    linestyle="--",
    color="red",
    alpha=0.8,
    label=f"Test Start ({test_start.date()})",
)

ax.set_title(
    f"Residuals (Pred − Real) — Region {region_id_val}",
    fontsize=14,
)

ax.set_ylabel("Residual")
ax.set_xlabel("Time")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: residual_plot.png")

# ============================================================
# FIGURE 3 — Scatter Plot
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(
    merged["C"],
    merged["C_gnode"],
    alpha=0.5,
    s=20,
    color="steelblue",
    label="Predictions",
)

# Perfect prediction line
lims = [
    min(merged["C"].min(), merged["C_gnode"].min()),
    max(merged["C"].max(), merged["C_gnode"].max()),
]

ax.plot(
    lims,
    lims,
    "r--",
    linewidth=1.5,
    label="Perfect Fit"
)

ax.set_title(
    f"Predicted vs Real — Region {region_id_val}",
    fontsize=13,
)

ax.set_xlabel("Real Crime Intensity")
ax.set_ylabel("Predicted Crime Intensity")

ax.legend()
ax.grid(alpha=0.3)

ax.set_aspect("equal", "box")

plt.tight_layout()

plt.savefig(
    "scatter_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: scatter_plot.png")

# ======================
# Finished
# ======================
print("\n✅ All plots saved:")
print("   trajectory_plot.png")
print("   residual_plot.png")
print("   scatter_plot.png")