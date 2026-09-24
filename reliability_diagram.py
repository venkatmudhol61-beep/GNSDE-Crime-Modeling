"""
reliability_diagram.py

Produces reliability diagrams (nominal coverage vs. empirical coverage
across a range of confidence levels) -- explicitly requested by
Reviewer 2, point 3, alongside CRPS/PICP/NLL.

A reliability diagram answers a more detailed question than a single
90% PICP number: "at EVERY confidence level (10%, 20%, ..., 90%), does
my predictive interval actually contain the true value that often?"
A perfectly calibrated model traces the diagonal y=x. A model whose
curve falls BELOW the diagonal is overconfident (intervals too narrow);
ABOVE the diagonal means underconfident (intervals too wide/conservative).

Works on the SAME prediction CSVs as calibration_metrics.py and
baseline_probabilistic.py (columns: month, region_id, C_gnsde, C_std,
C_p05, C_p95, split) -- assumes a Gaussian predictive distribution
N(C_gnsde, C_std^2), consistent with how your train_gnsde.py already
constructs C_std (MC variance + sigma_pred^2) and calibrates C_p05/C_p95.

USAGE (single model):
    python src/eval/reliability_diagram.py --pred_csv data/processed/chicago_beats_latent_final_predictions.csv --true_csv data/processed/chicago_beats_crime_timeseries.csv \\
        --label "Latent Enforcement GN-SDE"  --out figures/reliability_latent.png

USAGE (compare several models on one plot -- recommended for the paper):
    python src/eval/reliability_diagram.py  --true_csv data/processed/chicago_beats_crime_timeseries.csv  --compare  "Latent Enforcement GN-SDE=data/processed/chicago_beats_latent_final_predictions.csv" "Persistence+Gaussian=data/processed/chicago_beats_persistgauss_final_predictions.csv" \\
 "GP baseline=data/processed/chicago_beats_gp_final_predictions.csv"  --out figures/reliability_comparison.png

Each --compare entry is "Label=path/to/predictions.csv". Produces one
PNG with all curves overlaid plus the diagonal reference line, and a
companion CSV with the raw nominal/empirical numbers (in case you want
to redraw the figure in a different style for the manuscript, e.g. with
matplotlib directly in a LaTeX-matching font).
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt


NOMINAL_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def load_predictions_with_truth(pred_csv: str, true_csv: str, split: str = "test") -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    required_cols = {"month", "region_id", "C_gnsde", "C_std", "split"}
    missing = required_cols - set(pred.columns)
    if missing:
        raise ValueError(f"{pred_csv} is missing columns: {missing}")

    if split != "all":
        pred = pred[pred["split"] == split].copy()
    if len(pred) == 0:
        raise ValueError(f"No rows for split='{split}' in {pred_csv}")

    true_df = pd.read_csv(true_csv)
    true_col = "C" if "C" in true_df.columns else \
               ("C_true" if "C_true" in true_df.columns else None)
    if true_col is None:
        candidates = [c for c in true_df.columns if c.lower() in
                      ("c", "c_true", "crime_intensity", "c_i")]
        if not candidates:
            raise ValueError(
                f"Could not find true-value column in {true_csv}. "
                f"Available: {list(true_df.columns)}"
            )
        true_col = candidates[0]

    merged = pred.merge(
        true_df[["month", "region_id", true_col]],
        on=["month", "region_id"], how="inner"
    )
    if len(merged) == 0:
        raise ValueError(
            f"Merge produced zero rows between {pred_csv} and {true_csv} -- "
            f"check 'month'/'region_id' dtype and format match."
        )
    merged = merged.rename(columns={true_col: "C_true"})
    return merged


def compute_reliability_curve(merged: pd.DataFrame,
                               nominal_levels=NOMINAL_LEVELS) -> pd.DataFrame:
    """
    For each nominal confidence level p (e.g. 0.90), computes the
    symmetric Gaussian interval [mu - z*sigma, mu + z*sigma] where z is
    the two-sided normal quantile for p, and measures what fraction of
    true values actually fall inside that interval (empirical coverage).
    """
    y = merged["C_true"].to_numpy(dtype=float)
    mu = merged["C_gnsde"].to_numpy(dtype=float)
    sigma = np.clip(merged["C_std"].to_numpy(dtype=float), 1e-8, None)

    rows = []
    for p in nominal_levels:
        z = norm.ppf(0.5 + p / 2.0)  # two-sided z for nominal coverage p
        lo = mu - z * sigma
        hi = mu + z * sigma
        empirical = float(((y >= lo) & (y <= hi)).mean())
        rows.append({"nominal": p, "empirical": empirical})
    return pd.DataFrame(rows)


def plot_reliability(curves: dict, out_png: str, out_csv: str = None):
    """
    curves: dict of {label: reliability_curve_df}, each with columns
    'nominal' and 'empirical' (from compute_reliability_curve).
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            linewidth=1.2, label="Perfect calibration")

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    all_rows = []
    for i, (label, df) in enumerate(curves.items()):
        ax.plot(df["nominal"], df["empirical"],
                marker=markers[i % len(markers)], linewidth=1.6,
                markersize=5, label=label)
        for _, r in df.iterrows():
            all_rows.append({"model": label, "nominal": r["nominal"],
                              "empirical": r["empirical"]})

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Reliability diagram (test split)")
    fig.tight_layout()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"Saved figure -> {out_png}")

    if out_csv:
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"Saved underlying numbers -> {out_csv}")

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true_csv", required=True,
                     help="Path to ground-truth crime_timeseries.csv")
    ap.add_argument("--split", default="test",
                     choices=["test", "val", "train", "all"])

    # Single-model mode
    ap.add_argument("--pred_csv", default=None,
                     help="Single prediction CSV (use with --label)")
    ap.add_argument("--label", default="Model",
                     help="Legend label for single-model mode")

    # Multi-model comparison mode
    ap.add_argument("--compare", nargs="+", default=None,
                     help='One or more "Label=path/to/predictions.csv" entries')

    ap.add_argument("--out", required=True, help="Output PNG path")
    args = ap.parse_args()

    curves = {}

    if args.compare:
        for entry in args.compare:
            if "=" not in entry:
                raise ValueError(
                    f'--compare entries must be "Label=path.csv", got: {entry}'
                )
            label, path = entry.split("=", 1)
            merged = load_predictions_with_truth(path, args.true_csv, args.split)
            curves[label] = compute_reliability_curve(merged)
            print(f"{label}: n={len(merged)}")
    elif args.pred_csv:
        merged = load_predictions_with_truth(args.pred_csv, args.true_csv, args.split)
        curves[args.label] = compute_reliability_curve(merged)
        print(f"{args.label}: n={len(merged)}")
    else:
        raise ValueError("Provide either --pred_csv (single model) or --compare (multiple)")

    out_csv = str(Path(args.out).with_suffix("")) + "_data.csv"
    plot_reliability(curves, args.out, out_csv)

    # Print a quick text summary too
    print("\n=== Reliability summary (nominal -> empirical) ===")
    for label, df in curves.items():
        print(f"\n{label}:")
        for _, r in df.iterrows():
            gap = r["empirical"] - r["nominal"]
            flag = "  (overconfident)" if gap < -0.03 else \
                   ("  (underconfident)" if gap > 0.03 else "")
            print(f"  {r['nominal']*100:5.1f}% -> {r['empirical']*100:5.1f}%{flag}")


if __name__ == "__main__":
    main()