#!/usr/bin/env python3
"""ETT long-horizon benchmark for base TimesFM 2.5 200M (PyTorch).

Evaluates the pretrained TimesFM 2.5 model onETTh1, ETTh2, ETTm1, ETTm2
using a disjoint-window evaluation strategy (stride = pred_len).

Auto-downloads ETT CSVs from GitHub if not found locally.

Usage:
    python benchmark/run_base_ett.py [--results_dir results/ett]
"""

import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
import tqdm

import timesfm

# Data configs
DATA_DICT = {
    "ettm2": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "datasets/ETT-small/ETTm2.csv",
        "freq": "15min",
    },
    "ettm1": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "datasets/ETT-small/ETTm1.csv",
        "freq": "15min",
    },
    "etth2": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "datasets/ETT-small/ETTh2.csv",
        "freq": "H",
    },
    "etth1": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "datasets/ETT-small/ETTh1.csv",
        "freq": "H",
    },
}

PRED_LENS = [96, 192, 336]
CONTEXT_LEN = 512

ETT_GITHUB_BASE = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
)


# Helpers
def download_ett_if_needed(data_path: str, filename: str):
    """Download ETT CSV from GitHub if it doesn't exist locally."""
    if os.path.exists(data_path):
        return
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    url = ETT_GITHUB_BASE + filename
    print(f"  Downloading {url} → {data_path}")
    urllib.request.urlretrieve(url, data_path)


def _mse(y_pred, y_true):
    return np.square(y_pred - y_true)


def _mae(y_pred, y_true):
    return np.abs(y_pred - y_true)


def _smape(y_pred, y_true, eps=1e-7):
    abs_diff = np.abs(y_pred - y_true)
    abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2
    abs_val = np.where(abs_val > eps, abs_val, 1.0)
    abs_diff = np.where(abs_val > eps, abs_diff, 0.0)
    return abs_diff / abs_val


def create_test_windows(data_df, ts_cols, boundaries, context_len, pred_len, normalize=True):
    """Create disjoint test windows from the ETT data."""
    test_start = boundaries[1]
    test_end = boundaries[2]

    windows_past = []
    windows_actual = []

    pos = test_start
    while pos + pred_len <= test_end:
        ctx_start = max(0, pos - context_len)
        for col in ts_cols:
            series = data_df[col].values.astype(np.float64)

            past = series[ctx_start:pos].copy()
            actual = series[pos:pos + pred_len].copy()

            if normalize:
                mu = np.mean(past)
                std = np.std(past) + 1e-9
                past = (past - mu) / std
                actual = (actual - mu) / std

            windows_past.append(past)
            windows_actual.append(actual)

        pos += pred_len

    return windows_past, windows_actual


# Main eval loop
def run_ett_benchmark(results_dir: str, context_len: int = CONTEXT_LEN):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading TimesFM 2.5 200M (PyTorch)...")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    all_results = {}

    for dataset_name, cfg in DATA_DICT.items():
        data_path = cfg["data_path"]
        filename = os.path.basename(data_path)
        download_ett_if_needed(data_path, filename)

        print(f"\n{'═'*60}")
        print(f"Dataset: {dataset_name} ({data_path})")

        data_df = pd.read_csv(data_path)
        date_col = "date"
        ts_cols = [c for c in data_df.columns if c != date_col]
        boundaries = cfg["boundaries"]

        for pred_len in PRED_LENS:
            print(f"\n  pred_len={pred_len}, context_len={context_len}")

            # Recompile for this horizon
            model.compile(
                timesfm.ForecastConfig(
                    max_context=context_len,
                    max_horizon=pred_len,
                    normalize_inputs=True,
                    per_core_batch_size=32,
                )
            )

            # Create windows
            windows_past, windows_actual = create_test_windows(
                data_df, ts_cols, boundaries, context_len, pred_len,
                normalize=True,
            )
            n_windows = len(windows_past)
            print(f"  Number of windows: {n_windows}")

            if n_windows == 0:
                print("  No windows, skipping.")
                continue

            # Batch forecast
            batch_size = 64
            mse_losses = []
            mae_losses = []
            smape_losses = []
            num_elements = 0
            abs_sum = 0

            t0 = time.time()

            for start_idx in tqdm.tqdm(
                range(0, n_windows, batch_size),
                desc=f"  {dataset_name}_h{pred_len}",
                leave=False,
            ):
                end_idx = min(start_idx + batch_size, n_windows)
                batch_past = windows_past[start_idx:end_idx]
                batch_actual = np.array(windows_actual[start_idx:end_idx])

                point_forecast, _ = model.forecast(
                    horizon=pred_len,
                    inputs=batch_past,
                )
                forecasts = point_forecast[:, :batch_actual.shape[1]]

                mae_losses.append(_mae(forecasts, batch_actual).sum())
                mse_losses.append(_mse(forecasts, batch_actual).sum())
                smape_losses.append(_smape(forecasts, batch_actual).sum())
                num_elements += batch_actual.shape[0] * batch_actual.shape[1]
                abs_sum += np.abs(batch_actual).sum()

            elapsed = time.time() - t0

            mse_val = np.sum(mse_losses) / num_elements
            mae_val = np.sum(mae_losses) / num_elements
            smape_val = np.sum(smape_losses) / num_elements
            wape_val = np.sum(mae_losses) / abs_sum

            result_key = f"{dataset_name}_h{pred_len}"
            result_dict = {
                "mse": float(mse_val),
                "mae": float(mae_val),
                "smape": float(smape_val),
                "wape": float(wape_val),
                "nrmse": float(np.sqrt(mse_val) / (abs_sum / num_elements)),
                "num_windows": n_windows,
                "num_elements": int(num_elements),
                "time": float(elapsed),
                "dataset": dataset_name,
                "pred_len": pred_len,
                "context_len": context_len,
            }
            all_results[result_key] = result_dict

            print(f"    MSE={mse_val:.6f}  MAE={mae_val:.6f}  "
                  f"SMAPE={smape_val:.6f}  WAPE={wape_val:.6f}  "
                  f"Time={elapsed:.2f}s")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Print summary table
    print("\n\n--- ETT Benchmark Summary ---")
    print(f"{'Dataset':<12} {'Horizon':<10} {'MSE':<12} {'MAE':<12} {'SMAPE':<12} {'WAPE':<12} {'Time(s)':<10}")
    print("-" * 80)
    for key, r in sorted(all_results.items()):
        print(f"{r['dataset']:<12} {r['pred_len']:<10} {r['mse']:<12.6f} {r['mae']:<12.6f} {r['smape']:<12.6f} {r['wape']:<12.6f} {r['time']:<10.2f}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETT benchmark for TimesFM 2.5")
    parser.add_argument("--results_dir", default="results/ett", help="Output directory")
    parser.add_argument("--context_len", type=int, default=512, help="Context length")
    args = parser.parse_args()

    run_ett_benchmark(results_dir=args.results_dir, context_len=args.context_len)
