import pandas as pd
import unicodedata
import os

os.makedirs("data/processed", exist_ok=True)

CONFIGS = {
    "chicago_districts": {
        "path":        "data/raw/crimes.csv",
        "date_col":    "Date",
        "region_col":  "District",
        "date_fmt":    "mixed",
        "year_min":    2006,
        "year_max":    2024,
        "century_fix": False,
        "region_min":  None,
        "region_max":  None,
        "graph_order": "data/raw/graph_district_order.csv",
    },
    "chicago_beats": {
        "path":        "data/raw/crimes.csv",
        "date_col":    "Date",
        "region_col":  "Beat",
        "date_fmt":    "mixed",
        "year_min":    2006,
        "year_max":    2024,
        "century_fix": False,
        "region_min":  None,
        "region_max":  None,
        "graph_order": "data/raw/graph_beat_order.csv",
    },
    "nyc_precincts": {
        "path":        "data/raw/nyc_crimes.csv",
        "date_col":    "CMPLNT_FR_DT",
        "region_col":  "ADDR_PCT_CD",
        "date_fmt":    "%m/%d/%Y",
        "year_min":    2006,
        "year_max":    2024,
        "century_fix": False,
        "region_min":  None,
        "region_max":  None,
        "graph_order": "data/raw/graph_precinct_order.csv",
    },
}

def normalize_id(val):
    """Normalize unicode string to consistent form."""
    if isinstance(val, str):
        return unicodedata.normalize("NFC", val.strip())
    return val

def preprocess(dataset_name):
    cfg = CONFIGS[dataset_name]
    print(f"\nProcessing: {dataset_name}")

    df = pd.read_csv(cfg["path"], low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # Parse dates
    df[cfg["date_col"]] = pd.to_datetime(
        df[cfg["date_col"]],
        format=cfg["date_fmt"],
        dayfirst=False,
        errors="coerce"
    )
    df = df.dropna(subset=[cfg["date_col"]])

    # Century fix for NYC
    if cfg["century_fix"]:
        wrong = df[cfg["date_col"]].dt.year < 2000
        print(f"  Century-fixed rows: {wrong.sum():,}")
        df.loc[wrong, cfg["date_col"]] += pd.DateOffset(years=100)

    # Date range filter
    df = df[
        (df[cfg["date_col"]].dt.year >= cfg["year_min"]) &
        (df[cfg["date_col"]].dt.year <= cfg["year_max"])
    ]

    # Clean region
    df = df.dropna(subset=[cfg["region_col"]])

    # Unicode normalization (from Script 3)
    if df[cfg["region_col"]].dtype == object:
        df[cfg["region_col"]] = df[cfg["region_col"]].apply(normalize_id)

    df["region_id"] = pd.to_numeric(df[cfg["region_col"]], errors="coerce")
    df = df.dropna(subset=["region_id"])
    df["region_id"] = df["region_id"].astype(int)

    if cfg["region_min"] is not None:
        df = df[df["region_id"] >= cfg["region_min"]]
    if cfg["region_max"] is not None:
        df = df[df["region_id"] <= cfg["region_max"]]

    # Graph filter
    graph_path = cfg.get("graph_order")
    if graph_path and os.path.exists(graph_path):
        graph_nodes = pd.read_csv(graph_path)
        valid_ids   = set(graph_nodes.iloc[:, 0].astype(int).tolist())
        before      = df["region_id"].nunique()
        df          = df[df["region_id"].isin(valid_ids)]
        after       = df["region_id"].nunique()
        print(f"  Graph filter: {before} → {after} regions")
    else:
        print(f"  Graph filter: skipped (not found: {graph_path})")

    # Monthly time index
    df["month"] = df[cfg["date_col"]].dt.to_period("M").astype(str)
    df = df[
        (df["month"] >= f"{cfg['year_min']}-01") &
        (df["month"] <= f"{cfg['year_max']}-12")
    ]

    # Aggregate
    crime = (
        df.groupby(["region_id", "month"])
          .size()
          .reset_index(name="count")
    )

    # Global max normalization
    crime["C"] = crime["count"] / crime["count"].max()

    crime = crime[["month", "region_id", "C"]]

    out_path = f"data/processed/{dataset_name}_crime_timeseries.csv"
    crime.to_csv(out_path, index=False)

    print(f"  Regions : {crime['region_id'].nunique()}")
    print(f"  Months  : {crime['month'].nunique()}")
    print(f"  Range   : {crime['month'].min()} → {crime['month'].max()}")
    print(f"  C min   : {crime['C'].min():.4f}")
    print(f"  C max   : {crime['C'].max():.4f}")
    print(f"  C mean  : {crime['C'].mean():.4f}")
    print(f"  Rows    : {len(crime):,}")
    print(f"  Saved  → {out_path}")

if __name__ == "__main__":
    for name in CONFIGS:
        preprocess(name)
    print("\nAll 3 datasets processed.")