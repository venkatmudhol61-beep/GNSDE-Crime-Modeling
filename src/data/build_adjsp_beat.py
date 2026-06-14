import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import os

# =========================
# SETTINGS
# =========================
CRIME_FILE = "data/raw/crimes.csv"
ORDER_FILE = "data/raw/graph_beat_order.csv"
OUT_SP     = "data/processed/chicago_beats_adjacency_spatial.npy"
OUT_FC     = "data/processed/chicago_beats_adjacencyfc.npy"
K          = 8   # increased from 4 — beats need more neighbors

# =========================
# 1. Load graph beat order
# =========================
graph_beats = (
    pd.read_csv(ORDER_FILE)
    .iloc[:, 0]
    .astype(int)
    .tolist()
)
N = len(graph_beats)
print(f"Number of beats: {N}")

# =========================
# 2. Load crime data — NO date filter
# =========================
df = pd.read_csv(CRIME_FILE)
df = df.dropna(subset=["Latitude", "Longitude", "Beat"])
df["Beat"] = df["Beat"].astype(int)
df = df[df["Beat"].isin(graph_beats)]
print(f"Unique beats in data: {df['Beat'].nunique()}/{N}")

# =========================
# 3. Compute centroids in strict graph order
# =========================
centroids = (
    df.groupby("Beat")[["Latitude", "Longitude"]]
    .mean()
    .reindex(graph_beats)
)

missing = centroids.isnull().any(axis=1).sum()
if missing:
    print(f"⚠ Filling {missing} missing centroids with mean")
    centroids = centroids.fillna(centroids.mean())

coords = centroids.values
assert coords.shape == (N, 2), f"Shape mismatch: {coords.shape}"

# =========================
# 4. Haversine distance matrix
# =========================
coords_rad  = np.radians(coords)
dist_matrix = haversine_distances(coords_rad) * 6371  # km

# Log K-th neighbor distance to validate K choice
sorted_dists = np.sort(dist_matrix, axis=1)
print(f"\nDistance to K={K}th neighbor:")
print(f"  Min  : {sorted_dists[:, K].min():.2f} km")
print(f"  Mean : {sorted_dists[:, K].mean():.2f} km")
print(f"  Max  : {sorted_dists[:, K].max():.2f} km")
print(f"  (Chicago beats expected ~1.5 km wide)")

# =========================
# 5. KNN spatial adjacency
#    symmetrize FIRST — normalize AFTER
# =========================
A_sp = np.zeros((N, N), dtype=np.float32)
for i in range(N):
    idx = np.argsort(dist_matrix[i])[1:K + 1]
    A_sp[i, idx] = 1

A_sp = np.maximum(A_sp, A_sp.T)  # symmetrize FIRST

assert np.array_equal(A_sp, A_sp.T), "Not symmetric!"

# Degree AFTER symmetrize
degree       = A_sp.sum(axis=1)
deg_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
D_inv_sqrt   = np.diag(deg_inv_sqrt)
A_sp_norm    = (D_inv_sqrt @ A_sp @ D_inv_sqrt).astype(np.float32)

# =========================
# 6. FC adjacency
# =========================
A_fc      = np.ones((N, N), dtype=np.float32)
np.fill_diagonal(A_fc, 0)
A_fc_norm = (A_fc / A_fc.sum(axis=1, keepdims=True)).astype(np.float32)

# =========================
# 7. Save
# =========================
os.makedirs("data/processed", exist_ok=True)
np.save(OUT_SP, A_sp_norm)
np.save(OUT_FC, A_fc_norm)

# =========================
# 8. Diagnostics
# =========================
neighbors = (A_sp_norm > 0).sum(axis=1)
print(f"\n===== GRAPH STATS =====")
print(f"Shape         : {A_sp_norm.shape}")
print(f"Max value     : {A_sp_norm.max():.4f}")
print(f"Min value(>0) : {A_sp_norm[A_sp_norm > 0].min():.4f}")
print(f"Avg neighbors : {neighbors.mean():.2f}")
print(f"Min neighbors : {neighbors.min()}")
print(f"Max neighbors : {neighbors.max()}")
print(f"Is symmetric  : {np.allclose(A_sp_norm, A_sp_norm.T, atol=1e-6)}")

if (neighbors == 0).any():
    print(f"⚠ {(neighbors == 0).sum()} isolated nodes!")
else:
    print("✅ No isolated nodes")

print(f"\n✅ Saved spatial → {OUT_SP}")
print(f"✅ Saved fc      → {OUT_FC}")