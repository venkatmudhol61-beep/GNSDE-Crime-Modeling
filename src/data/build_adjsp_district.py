import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import os

# =========================
# SETTINGS
# =========================
CRIME_FILE = "data/raw/crimes.csv"
ORDER_FILE = "data/raw/graph_district_order.csv"
OUT_FILE   = "data/processed/chicago_districts_adjacency_spatial.npy"

DATE_COL   = "Date"          # column name in Chicago crimes CSV
DATE_START = "2006-01-01"
DATE_END   = "2024-12-31"

K = 4   # number of nearest neighbors (recommended: 5–10)

# =========================
# 1. Load graph district order
# =========================
graph_districts = (
    pd.read_csv(ORDER_FILE)
    .iloc[:, 0]
    .astype(int)
    .tolist()
)

N = len(graph_districts)
print("Number of districts (graph):", N)   # should be 23

# =========================
# 2. Load raw crime data (WITH coords + dates)
# =========================
df = pd.read_csv(CRIME_FILE)

# --- Date filter: 2006-01 to 2024-12 ---
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
before = len(df)
df = df.dropna(subset=[DATE_COL])
df = df[(df[DATE_COL] >= DATE_START) & (df[DATE_COL] <= DATE_END)]
after = len(df)
print(f"Rows after date filter ({DATE_START} → {DATE_END}): {after:,}  (dropped {before - after:,})")
print(f"Year range in data: {df[DATE_COL].dt.year.min()} – {df[DATE_COL].dt.year.max()}")

# --- Drop rows missing spatial/district info ---
df = df.dropna(subset=["Latitude", "Longitude", "District"])
df["District"] = df["District"].astype(int)

# Filter strictly to graph districts
df = df[df["District"].isin(graph_districts)]

print("Unique districts after filter:", df["District"].nunique())

# =========================
# 3. Compute centroids (STRICT ORDER)
# =========================
centroids = (
    df.groupby("District")[["Latitude", "Longitude"]]
    .mean()
    .reindex(graph_districts)   # enforce order
)

# Warn if any district has no crime data
missing = centroids[centroids.isna().any(axis=1)]
if not missing.empty:
    print(f"⚠ WARNING: {len(missing)} district(s) have no crime data → NaN centroid:")
    print(missing.index.tolist())

coords = centroids.values

if coords.shape[0] != N:
    raise RuntimeError("Centroid mismatch!")

# =========================
# 4. Compute distance matrix
# =========================
coords_rad = np.radians(coords)
dist_matrix = haversine_distances(coords_rad) * 6371  # km

# =========================
# 5. Build KNN adjacency
#    Cap K at N-1 to avoid index errors on small graphs
# =========================
k_actual = min(K, N - 1)
A = np.zeros((N, N), dtype=np.float32)

for i in range(N):
    idx = np.argsort(dist_matrix[i])[1:k_actual + 1]
    A[i, idx] = 1

# Symmetrize
A = np.maximum(A, A.T)

# =========================
# 6. Symmetric normalization (GNN standard)
# =========================
degree = A.sum(axis=1)
degree_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)

D_inv_sqrt = np.diag(degree_inv_sqrt)
A = D_inv_sqrt @ A @ D_inv_sqrt

# =========================
# 7. Save
# =========================
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.save(OUT_FILE, A)

# =========================
# 8. Diagnostics
# =========================
print("\n===== GRAPH STATS =====")
print("Shape:", A.shape)
print("Max value:", A.max())
print("Min value:", A.min())

neighbors = (A > 0).sum(axis=1)
print("Average neighbors:", neighbors.mean())
print("Min neighbors:", neighbors.min())
print("Max neighbors:", neighbors.max())

print("Is symmetric:", np.allclose(A, A.T, atol=1e-6))

# =========================
# 9. Connectivity check
# =========================
isolated = neighbors == 0
if isolated.any():
    isolated_ids = [graph_districts[i] for i in np.where(isolated)[0]]
    print(f"⚠ WARNING: isolated nodes found → districts: {isolated_ids}")
else:
    print("✅ No isolated nodes")

print("\n✅ KNN spatial adjacency saved:", OUT_FILE)