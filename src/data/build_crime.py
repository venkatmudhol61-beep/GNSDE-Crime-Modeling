import pandas as pd
import unicodedata
import os

os.makedirs("data/processed", exist_ok=True)

TRAIN_RATIO = 0.7   # MUST match TRAIN_RATIO in run_ablations.py /
                     # test_correlated_noise.py / the main training script —
                     # this determines which months' counts are allowed to
                     # contribute to the normalization constant.

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

    # ── Train-only max normalization ────────────────────────────────
    # Fixes leakage: previously used crime["count"].max() over the
    # FULL date range (train+val+test), so the scale of C for every
    # split — including train — depended on values from val/test
    # months. Now the normalization constant is derived strictly from
    # the training-period months, matching the split used downstream.
    months_sorted = sorted(crime["month"].unique())
    n_train       = int(len(months_sorted) * TRAIN_RATIO)
    train_months  = set(months_sorted[:n_train])

    train_max = crime.loc[crime["month"].isin(train_months), "count"].max()
    if pd.isna(train_max) or train_max == 0:
        raise ValueError(
            f"{dataset_name}: train-period max count is 0 or NaN — "
            f"check date filtering / region filtering upstream."
        )

    crime["C"] = crime["count"] / train_max
    # NOTE: val/test months may have C > 1 if their peak count exceeds
    # the train-period max. This is expected and reported, not
    # clipped — clipping would hide distributional shift between
    # splits that a reviewer may want visible.

    crime = crime[["month", "region_id", "C"]]

    out_path = f"data/processed/{dataset_name}_crime_timeseries.csv"
    crime.to_csv(out_path, index=False)

    print(f"  Regions : {crime['region_id'].nunique()}")
    print(f"  Months  : {crime['month'].nunique()}")
    print(f"  Range   : {crime['month'].min()} → {crime['month'].max()}")
    print(f"  Train months     : {len(train_months)} / {len(months_sorted)}")
    print(f"  Train-period max : {train_max}")
    print(f"  C min   : {crime['C'].min():.4f}")
    print(f"  C max   : {crime['C'].max():.4f}")
    print(f"  C mean  : {crime['C'].mean():.4f}")
    over_one = (crime["C"] > 1.0).sum()
    if over_one > 0:
        print(f"  NOTE: {over_one} rows have C > 1.0 "
              f"(val/test counts exceeding train-period max)")
    print(f"  Rows    : {len(crime):,}")
    print(f"  Saved  → {out_path}")

if __name__ == "__main__":
    for name in CONFIGS:
        preprocess(name)
    print("\nAll 3 datasets processed.")