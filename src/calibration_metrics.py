"""
calibration_metrics.py

Usage:
    python src/eval/calibration_metrics.py \
        --dataset chicago_beats --variant latent \
        --pred_csv data/processed/chicago_beats_latent_final_predictions.csv \
        --out data/processed/calibration_summary.csv

Run once per (dataset, variant) pair you've trained; results accumulate
into the same --out CSV (one row per run), which is what you paste into
Table tab:calibration in the manuscript.
run code:
     python src/calibration_metrics.py --dataset chicago_beats --variant latent --pred_csv data/processed/chicago_beats_latent_final_predictions.csv --out data/processed/calibration_summary.csv


"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm


def gaussian_crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS for a Gaussian predictive distribution N(mu, sigma^2)
    evaluated against observed y_true. Standard result (Gneiting & Raftery 2007):

        CRPS(N(mu,sigma^2), y) = sigma * [ z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
        z = (y - mu) / sigma

    Vectorised over arrays. sigma must be > 0 everywhere (clip if needed).
    """
    sigma = np.clip(sigma, 1e-8, None)
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return crps


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Negative log-likelihood under N(mu, sigma^2), per observation."""
    sigma = np.clip(sigma, 1e-8, None)
    return 0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((y_true - mu) / sigma) ** 2


def picp_and_width(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """
    Empirical coverage (PICP) and mean width of a prediction interval
    [lo, hi] against observed y_true. For a 90% PI, PICP should be
    close to 0.90 if the model is well calibrated (over-coverage
    means intervals are too wide / conservative; under-coverage means
    the model is overconfident).
    """
    covered = (y_true >= lo) & (y_true <= hi)
    picp = covered.mean()
    width = (hi - lo).mean()
    return picp, width


def load_true_crime_intensity(dataset: str, processed_dir: Path) -> pd.DataFrame:
    """
    Loads the ground-truth C_i(t) series your build_crime.py already
    produced, so predictions can be joined against the true value on
    (month, region_id). Adjust the filename pattern if yours differs.
    """
    candidate = processed_dir / f"{dataset}_crime_timeseries.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find ground-truth series at {candidate}. "
            f"Point --true_csv at your actual crime_timeseries.csv if the "
            f"naming differs."
        )
    df = pd.read_csv(candidate)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     choices=["chicago_beats", "chicago_districts", "nyc_precincts"])
    ap.add_argument("--variant", required=True,
                     choices=["fc", "spatial", "spatial_attn", "latent"])
    ap.add_argument("--pred_csv", required=True,
                     help="Path to {DATASET}_{VARIANT}_final_predictions.csv")
    ap.add_argument("--true_csv", default=None,
                     help="Path to ground-truth crime_timeseries.csv "
                          "(default: data/processed/{dataset}_crime_timeseries.csv)")
    ap.add_argument("--true_col", default="C_true",
                     help="Column name of the true C_i(t) value in --true_csv "
                          "after merge; adjust if your crime_timeseries.csv "
                          "uses a different column name")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"],
                     help="Which split to evaluate on (default: test, matching "
                          "the rest of the paper's reported metrics)")
    ap.add_argument("--out", default="data/processed/calibration_summary.csv")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)
    required_cols = {"month", "region_id", "C_gnsde", "C_std", "C_p05", "C_p95", "split"}
    missing = required_cols - set(pred.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV is missing expected columns: {missing}. "
            f"Found columns: {list(pred.columns)}"
        )

    if args.split != "all":
        pred = pred[pred["split"] == args.split].copy()
    if len(pred) == 0:
        raise ValueError(f"No rows found for split='{args.split}' in {args.pred_csv}")

    # ---- join ground truth ----
    true_csv = Path(args.true_csv) if args.true_csv else \
        Path("data/processed") / f"{args.dataset}_crime_timeseries.csv"
    true_df = pd.read_csv(true_csv)

    # Try to auto-detect the true-value column if the default isn't present
    true_col = args.true_col
    if true_col not in true_df.columns:
        candidates = [c for c in true_df.columns if c.lower() in
                      ("c_true", "c", "crime_intensity", "c_i")]
        if candidates:
            true_col = candidates[0]
        else:
            raise ValueError(
                f"Could not find true-value column '{args.true_col}' in "
                f"{true_csv}. Available columns: {list(true_df.columns)}. "
                f"Pass --true_col explicitly."
            )

    merged = pred.merge(
        true_df[["month", "region_id", true_col]],
        on=["month", "region_id"], how="inner"
    )
    if len(merged) == 0:
        raise ValueError(
            "Merge produced zero rows — check that 'month' and 'region_id' "
            "columns match in dtype/format between the two CSVs "
            "(e.g. both should be the same date/string format)."
        )
    dropped = len(pred) - len(merged)
    if dropped > 0:
        print(f"[warn] {dropped} prediction rows had no matching ground truth "
              f"and were dropped from evaluation.")

    y = merged[true_col].to_numpy(dtype=float)
    mu = merged["C_gnsde"].to_numpy(dtype=float)
    sigma = merged["C_std"].to_numpy(dtype=float)
    lo = merged["C_p05"].to_numpy(dtype=float)
    hi = merged["C_p95"].to_numpy(dtype=float)

    # ---- point-forecast metrics (sanity check against what you already report) ----
    mae = np.mean(np.abs(y - mu))
    rmse = np.sqrt(np.mean((y - mu) ** 2))
    ss_res = np.sum((y - mu) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    smape = 100.0 * np.mean(2 * np.abs(y - mu) / (np.abs(y) + np.abs(mu) + 1e-12))

    # ---- calibration metrics (the new numbers Reviewer 2 asked for) ----
    crps = gaussian_crps(y, mu, sigma).mean()
    nll = gaussian_nll(y, mu, sigma).mean()
    picp, width = picp_and_width(y, lo, hi)

    row = {
        "dataset": args.dataset,
        "variant": args.variant,
        "split": args.split,
        "n": len(merged),
        "MAE": round(mae, 5),
        "RMSE": round(rmse, 5),
        "R2": round(r2, 5),
        "sMAPE": round(smape, 3),
        "CRPS": round(crps, 5),
        "PICP_90": round(picp * 100, 2),      # as a percentage, target ~90.00
        "mean_interval_width": round(width, 5),
        "NLL": round(nll, 5),
    }

    print("\n=== Calibration summary ===")
    for k, v in row.items():
        print(f"  {k:22s}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if out_path.exists():
        existing = pd.read_csv(out_path)
        # replace any prior row for the same (dataset, variant, split)
        existing = existing[
            ~((existing.dataset == row["dataset"]) &
              (existing.variant == row["variant"]) &
              (existing.split == row["split"]))
        ]
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_csv(out_path, index=False)
    print(f"\nAppended to {out_path}")


if __name__ == "__main__":
    main()