import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

# For beat datasets, geo intersection is applied to ensure
# every node has both crime data and geometry.
GRAPH_ORDER_CONFIGS = {
    "chicago_districts": {
        "processed_path": "data/processed/chicago_districts_crime_timeseries.csv",
        "out_path":       "data/raw/graph_district_order.csv",
        "col_name":       "district",
        "geo_path":       "data/raw/beats.csv",   # ← add this
        "geo_col":        "DISTRICT",             # ← and this
    },
    "chicago_beats": {
        "processed_path": "data/processed/chicago_beats_crime_timeseries.csv",
        "out_path":       "data/raw/graph_beat_order.csv",
        "col_name":       "beat",
        "geo_path":       "data/raw/beats.csv",   # intersection applied
        "geo_col":        "BEAT_NUM",
    },
    "nyc_precincts": {
        "processed_path": "data/processed/nyc_precincts_crime_timeseries.csv",
        "out_path":       "data/raw/graph_precinct_order.csv",
        "col_name":       "precinct",
        "geo_path":       None,  # no intersection needed
        "geo_col":        None,
    },
}

if __name__ == "__main__":
    for name, cfg in GRAPH_ORDER_CONFIGS.items():
        df = pd.read_csv(cfg["processed_path"])
        crime_ids = set(df["region_id"].unique())

        if cfg["geo_path"] is not None:
            geo = pd.read_csv(cfg["geo_path"])
            geo_ids = set(geo[cfg["geo_col"]].astype(int).unique())
            valid_ids = sorted(crime_ids & geo_ids)

            dropped = crime_ids - geo_ids
            if dropped:
                print(f"  [{name}] dropped {len(dropped)} region(s) missing from geo: {sorted(dropped)}")
        else:
            valid_ids = sorted(crime_ids)

        order = pd.DataFrame(valid_ids, columns=[cfg["col_name"]])
        order.to_csv(cfg["out_path"], index=False)
        print(f"{cfg['out_path'].split('/')[-1]} → {len(order)} nodes")

    print("\nAll 3 graph order files created.")