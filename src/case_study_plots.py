"""
case_study_plots.py
Builds interpretability case studies from files already saved by the
GN-SDE training script (spatial_attn m_beta, latent L_latent).
No retraining required -- reads existing CSVs.
Usage:
    python src/case_study_plots.py --dataset chicago_beats
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--outdir", default="data/processed/case_studies")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# ---------- 1. Attention modulator m_beta ----------
mbeta_path = f"data/processed/{args.dataset}_spatial_attn_mbeta.csv"
if os.path.exists(mbeta_path):
    df = pd.read_csv(mbeta_path)
    df["month"] = df["month"].astype(str)

    var_by_region = df.groupby("beat")["m_beta"].std().sort_values()
    low_var_region  = var_by_region.index[0]
    high_var_region = var_by_region.index[-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    for region in [low_var_region, high_var_region]:
        sub = df[df["beat"] == region].sort_values("month")
        ax.plot(sub["month"], sub["m_beta"], marker="o", markersize=2,
                label=f"beat {region}")
    ax.set_xticks(ax.get_xticks()[::max(1, len(ax.get_xticks())//8)])
    ax.set_ylabel("Enforcement sensitivity modulator $m_{\\beta,i}(t)$")
    ax.set_title(f"Attention modulator: lowest- vs highest-variance beat ({args.dataset})")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_fig = f"{args.outdir}/mbeta_case_study_{args.dataset}.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()

    print("=== m_beta case study ===")
    print(f"  Lowest-variance beat : {low_var_region} "
          f"(std={var_by_region.iloc[0]:.4f})")
    print(f"  Highest-variance beat: {high_var_region} "
          f"(std={var_by_region.iloc[-1]:.4f})")
    print(f"  Global m_beta stats: mean={df['m_beta'].mean():.4f} "
          f"std={df['m_beta'].std():.4f}")
    print(f"  Saved figure -> {out_fig}\n")
else:
    print(f"[skip] {mbeta_path} not found -- run spatial_attn training first.")

# ---------- 2. Latent enforcement L_latent ----------
enf_path = f"data/processed/{args.dataset}_latent_final_enforcement.csv"
if os.path.exists(enf_path):
    df = pd.read_csv(enf_path)
    df["month"] = df["month"].astype(str)

    mean_by_region = df.groupby("region_id")["L_latent"].mean().sort_values()
    low_region  = mean_by_region.index[0]
    high_region = mean_by_region.index[-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    for region in [low_region, high_region]:
        sub = df[df["region_id"] == region].sort_values("month")
        ax.plot(sub["month"], sub["L_latent"], marker="o", markersize=2,
                label=f"region {region}")
    ax.set_xticks(ax.get_xticks()[::max(1, len(ax.get_xticks())//8)])
    ax.set_ylabel("Latent enforcement $L_i^{lat}(t)$")
    ax.set_title(f"Latent enforcement: lowest- vs highest-mean region ({args.dataset})")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_fig = f"{args.outdir}/llatent_case_study_{args.dataset}.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()

    print("=== L_latent case study ===")
    print(f"  Lowest-mean region : {low_region} "
          f"(mean={mean_by_region.iloc[0]:.4f})")
    print(f"  Highest-mean region: {high_region} "
          f"(mean={mean_by_region.iloc[-1]:.4f})")
    print(f"  Global L_latent stats: mean={df['L_latent'].mean():.4f} "
          f"std={df['L_latent'].std():.4f}")
    print(f"  Saved figure -> {out_fig}\n")
else:
    print(f"[skip] {enf_path} not found -- run latent training first.")

print("Done.")