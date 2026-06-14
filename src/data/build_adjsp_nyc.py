import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import os

# ── Settings ──────────────────────────────────────────────────────────
CRIME_FILE = "data/raw/nyc_crimes.csv"
ORDER_FILE = "data/raw/graph_precinct_order.csv"
OUT        = "data/processed/nyc_precincts_adjacency_spatial.npy"
K          = 4

# ── 1. Load ───────────────────────────────────────────────────────────
print("Loading crimes...")
df = pd.read_csv(CRIME_FILE, low_memory=False)
df = df.rename(columns={"ADDR_PCT_CD": "precinct"})
df["precinct"] = pd.to_numeric(df["precinct"], errors="coerce")
df = df.dropna(subset=["precinct", "Latitude", "Longitude"])
df["precinct"] = df["precinct"].astype(int)
df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]
print(f"Records after cleaning : {len(df):,}")

# ── 2. Centroids ──────────────────────────────────────────────────────
centroids = (
    df.groupby("precinct")[["Latitude", "Longitude"]]
    .median()
    .reset_index()
)

# ── 3. Align to model order ───────────────────────────────────────────
order      = pd.read_csv(ORDER_FILE)
region_ids = order.iloc[:, 0].astype(int).tolist()
N          = len(region_ids)
print(f"Precincts in model : {N}")

centroids = centroids.set_index("precinct").reindex(region_ids).reset_index()
missing   = centroids[centroids["Latitude"].isna()]["precinct"].tolist()
if missing:
    print(f"WARNING — missing centroids: {missing}")
    centroids["Latitude"]  = centroids["Latitude"].fillna(centroids["Latitude"].mean())
    centroids["Longitude"] = centroids["Longitude"].fillna(centroids["Longitude"].mean())

coords     = centroids[["Latitude", "Longitude"]].values
coords_rad = np.radians(coords)

# ── 4. KNN via BallTree ───────────────────────────────────────────────
tree = BallTree(coords_rad, metric="haversine")
distances, indices = tree.query(coords_rad, k=K + 1)

print(f"\nDistance to K={K}th neighbor:")
dist_km = distances[:, K] * 6371
print(f"  Min  : {dist_km.min():.2f} km")
print(f"  Mean : {dist_km.mean():.2f} km")
print(f"  Max  : {dist_km.max():.2f} km")

# ── 5. Build binary symmetric adjacency ──────────────────────────────
A = np.zeros((N, N), dtype=np.float32)
for i in range(N):
    for j_pos in range(1, K + 1):
        j = indices[i, j_pos]
        A[i, j] = 1.0
        A[j, i] = 1.0             # symmetrize

np.fill_diagonal(A, 0.0)
assert np.array_equal(A, A.T), "Not symmetric!"

# ── 6. Symmetric normalization D^-0.5 A D^-0.5 ───────────────────────
degree       = A.sum(axis=1)
deg_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
D_inv_sqrt   = np.diag(deg_inv_sqrt)
A_norm       = (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)

# ── 7. Diagnostics ────────────────────────────────────────────────────
neighbors = (A_norm > 0).sum(axis=1)
print(f"\n=== KNN Adjacency (K={K}) ===")
print(f"  Nonzero edges  : {int((A_norm > 0).sum())}")
print(f"  Min neighbors  : {int(neighbors.min())}")
print(f"  Max neighbors  : {int(neighbors.max())}")
print(f"  Mean neighbors : {neighbors.mean():.1f}")
print(f"  Is symmetric   : {np.allclose(A_norm, A_norm.T, atol=1e-6)}")
print(f"  Max value      : {A_norm.max():.4f}")
print(f"  Min value (>0) : {A_norm[A_norm > 0].min():.4f}")

if (neighbors == 0).any():
    print(f"WARNING — {(neighbors == 0).sum()} isolated nodes!")
else:
    print("No isolated nodes")

# ── 8. Save ───────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
np.save(OUT, A_norm)
print(f"\nSaved → {OUT}")