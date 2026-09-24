"""
kappa_diagnostics.py

--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from pathlib import Path


class KappaLogger:
    def __init__(self, n_nodes: int, out_csv: str):
        self.n_nodes = n_nodes
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self.records = []   # one row per epoch: mean/std/min kappa_i + trigger flag

    def log_epoch(self, epoch: int, alpha_i: np.ndarray, beta_i: np.ndarray,
                  sigma_i: np.ndarray, L_bar_i: np.ndarray):
        """
        Computes kappa_i for every node at this epoch (manuscript Eq. kappa,
        Definition 4.4) and appends a summary row.
        """
        assert alpha_i.shape == beta_i.shape == sigma_i.shape == L_bar_i.shape == (self.n_nodes,), \
            f"Expected shape ({self.n_nodes},) for all inputs, got " \
            f"alpha={alpha_i.shape}, beta={beta_i.shape}, sigma={sigma_i.shape}, L_bar={L_bar_i.shape}"

        kappa_i = beta_i * L_bar_i - alpha_i - (sigma_i ** 2) / 2.0
        kappa_min = kappa_i.min()
        trigger = bool(kappa_min <= 0)   # matches your existing regularisation condition

        self.records.append({
            "epoch": epoch,
            "kappa_mean": float(kappa_i.mean()),
            "kappa_std": float(kappa_i.std()),
            "kappa_min": float(kappa_min),
            "kappa_max": float(kappa_i.max()),
            "trigger": trigger,
        })
        return kappa_i   # in case you want to feed it straight into your
                          # existing weight-regularisation branch

    def finalize(self):
        df = pd.DataFrame(self.records)
        df.to_csv(self.out_csv, index=False)

        # Convergence-time summary: use the LAST 10% of epochs as a proxy
        # for "at convergence" (post-Phase-2, near the end of training),
        # matching what Table tab:kappa_dist in the manuscript needs.
        n_tail = max(1, int(0.1 * len(df)))
        tail = df.tail(n_tail)

        summary = {
            "mean_kappa_at_convergence": tail["kappa_mean"].mean(),
            "std_kappa_at_convergence": tail["kappa_mean"].std(),   # epoch-to-epoch variation near convergence
            "min_kappa_at_convergence": tail["kappa_min"].min(),
            "trigger_rate_pct_all_training": 100.0 * df["trigger"].mean(),
        }
        print(f"\n=== Kappa stability summary (last {n_tail} epochs) ===")
        for k, v in summary.items():
            print(f"  {k:32s}: {v:.5f}" if isinstance(v, float) else f"  {k:32s}: {v}")
        print(f"\nFull per-epoch log written to {self.out_csv}")
        return summary


def summarize_across_runs(csv_paths_by_model_and_gran: dict, out_csv: str):
    """
    Combines multiple per-run kappa logs (one per variant/granularity,
    produced by KappaLogger.finalize) into the single table your
    manuscript's Table tab:kappa_dist needs.

    csv_paths_by_model_and_gran: dict like
        {
            ("Fully Connected GN-SDE", "Beat-level"): "data/processed/chicago_beats_fc_kappa_log.csv",
            ("Latent Enforcement GN-SDE", "Beat-level"): "data/processed/chicago_beats_latent_kappa_log.csv",
            ...
        }
    """
    rows = []
    for (model_name, gran_name), path in csv_paths_by_model_and_gran.items():
        df = pd.read_csv(path)
        n_tail = max(1, int(0.1 * len(df)))
        tail = df.tail(n_tail)
        rows.append({
            "Model": model_name,
            "Granularity": gran_name,
            "Mean kappa_i": round(tail["kappa_mean"].mean(), 4),
            "Std kappa_i": round(tail["kappa_mean"].std(), 4),
            "Min kappa_i": round(tail["kappa_min"].min(), 4),
            "Trigger rate (%)": round(100.0 * df["trigger"].mean(), 2),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(out_df.to_string(index=False))
    print(f"\nWritten to {out_csv} -- copy these rows into Table tab:kappa_dist")
    return out_df


if __name__ == "__main__":
    print(
        "This module is meant to be imported into train_gnsde.py, not run "
        "standalone -- it needs your actual model's alpha_i/beta_i/sigma_i "
        "tensors, which only exist inside your training script.\n\n"
        "See the integration example at the top of this file (the big "
        "docstring) for exactly what to add to train_gnsde.py:\n\n"
        "    from src.eval.kappa_diagnostics import KappaLogger\n"
        "    kappa_logger = KappaLogger(n_nodes=N, out_csv=...)\n"
        "    # inside your training loop, after each epoch:\n"
        "    kappa_logger.log_epoch(epoch, alpha_i, beta_i, sigma_i, L_bar_i)\n"
        "    # after training finishes:\n"
        "    kappa_logger.finalize()\n"
    )