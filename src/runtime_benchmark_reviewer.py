"""
runtime_benchmark.py

"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def count_params(model) -> int:
    """Total trainable parameter count for a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sync():
    """CUDA sync so timing isn't polluted by async kernel launches."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_one_epoch(step_fn, n_warmup: int = 1, n_measured: int = 5) -> float:
    """
    Times a full training epoch by calling step_fn() repeatedly.
    step_fn should run exactly ONE epoch (one full pass over the
    training data with optimizer steps) and take no arguments --
    wrap your existing epoch loop body in a lambda or small function.

    Returns mean seconds/epoch over n_measured runs (after n_warmup
    untimed warmup runs, to exclude first-call CUDA/cuDNN init cost).
    """
    for _ in range(n_warmup):
        step_fn()
    _sync()

    times = []
    for _ in range(n_measured):
        _sync()
        t0 = time.perf_counter()
        step_fn()
        _sync()
        times.append(time.perf_counter() - t0)

    return float(np.mean(times))


def time_inference(infer_fn, n_warmup: int = 1, n_measured: int = 5) -> float:
    """
    Times one full inference pass (e.g. your N_MC=100 Monte Carlo
    forward-pass procedure over the test set). Same pattern as
    time_one_epoch: infer_fn takes no arguments, wrap your existing
    inference call.
    """
    for _ in range(n_warmup):
        infer_fn()
    _sync()

    times = []
    for _ in range(n_measured):
        _sync()
        t0 = time.perf_counter()
        infer_fn()
        _sync()
        times.append(time.perf_counter() - t0)

    return float(np.mean(times))


def log_runtime_row(dataset: str, variant: str, n_params: int,
                     epoch_time: float, inference_time: float,
                     out_csv: str, device_name: str = None):
    if device_name is None:
        if torch is not None and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        else:
            device_name = "CPU"

    row = {
        "dataset": dataset,
        "variant": variant,
        "device": device_name,
        "n_params": n_params,
        "sec_per_epoch": round(epoch_time, 4),
        "sec_per_inference_pass_NMC100": round(inference_time, 4),
    }
    print("\n=== Runtime summary ===")
    for k, v in row.items():
        print(f"  {k:30s}: {v}")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[
            ~((existing.dataset == row["dataset"]) & (existing.variant == row["variant"]))
        ]
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_csv(out_path, index=False)
    print(f"\nAppended to {out_path}")
    return row


if __name__ == "__main__":
    print(
        "This module is meant to be imported into train_gnsde.py, not run "
        "standalone -- see the docstring at the top of this file for the "
        "integration pattern. It needs your actual model/optimizer/dataloader "
        "objects, which only exist inside your training script."
    )