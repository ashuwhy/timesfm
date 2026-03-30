#!/usr/bin/env python3
<<<<<<< HEAD
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ETT long-horizon benchmark for TimesFM 2.5 200M (PyTorch).

Evaluates the pretrained model on ETTh1, ETTh2, ETTm1, ETTm2 datasets
with multiple prediction horizons using disjoint-window evaluation.

Adapted from v1/experiments/long_horizon_benchmarks/run_eval.py for the v2.5 API.

Usage:
    python benchmark/run_base_ett.py [--context_len 512] [--pred_lens 96 192 336]
=======
"""ETT long-horizon benchmark for base TimesFM 2.5 200M (PyTorch).

Evaluates the pretrained TimesFM 2.5 model onETTh1, ETTh2, ETTm1, ETTm2
using a disjoint-window evaluation strategy (stride = pred_len).

Auto-downloads ETT CSVs from GitHub if not found locally.

Usage:
    python benchmark/run_base_ett.py [--results_dir results/ett]
>>>>>>> c04953b (feat: add benchmarking scripts for base Monash and ETT datasets)
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
<<<<<<< HEAD
import timesfm

# ETT dataset configuration
DATA_DICT = {
    "etth1": {
        "boundaries": [8640, 11520, 14400],
        "freq": "H",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    },
    "etth2": {
        "boundaries": [8640, 11520, 14400],
        "freq": "H",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    },
    "ettm1": {
        "boundaries": [34560, 46080, 57600],
        "freq": "15min",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    },
    "ettm2": {
        "boundaries": [34560, 46080, 57600],
        "freq": "15min",
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
    },
}

EPS = 1e-7


def download_ett_csv(dataset_name: str, data_dir: str) -> str:
    """Download ETT CSV from GitHub if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{dataset_name.upper().replace('ETT', 'ETT')}.csv"
    # Map to actual filenames
    name_map = {
        "etth1": "ETTh1.csv",
        "etth2": "ETTh2.csv",
        "ettm1": "ETTm1.csv",
        "ettm2": "ETTm2.csv",
    }
    filename = name_map[dataset_name]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        url = DATA_DICT[dataset_name]["url"]
        print(f"  Downloading {dataset_name} from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  Saved to {filepath}")

    return filepath


def _mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.square(y_pred - y_true)


def _mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.abs(y_pred - y_true)


def _smape(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    abs_diff = np.abs(y_pred - y_true)
    abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2
    abs_val = np.where(abs_val > EPS, abs_val, 1.0)
    abs_diff = np.where(abs_val > EPS, abs_diff, 0.0)
    return abs_diff / abs_val


def evaluate_dataset(
    model: timesfm.TimesFM_2p5_200M_torch,
    dataset_name: str,
    data_dir: str,
    context_len: int,
    pred_len: int,
    normalize: bool = True,
) -> dict:
    """Evaluate model on a single ETT dataset with a given prediction length.

    Follows the standard ETT benchmark protocol:
    - Normalize ALL data using StandardScaler fit on train split
    - Feed normalized context to the model
    - Compute MSE/MAE in normalized space (matching iTransformer, PatchTST, etc.)
    """
    filepath = download_ett_csv(dataset_name, data_dir)
    boundaries = DATA_DICT[dataset_name]["boundaries"]

    data_df = pd.read_csv(filepath)
    ts_cols = [col for col in data_df.columns if col.lower() != "date"]

    # Full data matrix: shape (num_rows, num_cols)
    data_mat = data_df[ts_cols].values.astype(np.float64)

    if normalize:
        # StandardScaler fit on train split (matching v1 data_loader)
        train_mat = data_mat[:boundaries[0], :]
        scaler_mean = np.mean(train_mat, axis=0)
        scaler_std = np.std(train_mat, axis=0) + EPS
        data_mat = (data_mat - scaler_mean) / scaler_std

    test_start = boundaries[1]
    test_end = boundaries[2]

    smape_losses = []
    mse_losses = []
    mae_losses = []
    num_elements = 0
    abs_sum = 0

    # Disjoint-window evaluation: stride = pred_len
    num_windows = (test_end - test_start) // pred_len
    start_time = time.time()

    for window_idx in tqdm.tqdm(range(num_windows), desc=f"  {dataset_name} h={pred_len}"):
        window_start = test_start + window_idx * pred_len
        window_end = window_start + pred_len

        if window_end > test_end:
            break

        # Context: rows before the prediction window
        ctx_start = max(0, window_start - context_len)
        ctx_end = window_start

        # Already-normalized context and actuals
        ctx_values = data_mat[ctx_start:ctx_end, :]   # (context_len, num_cols)
        actual = data_mat[window_start:window_end, :]  # (pred_len, num_cols)

        # Each column is a separate univariate series for v2.5 API
        contexts = [ctx_values[:, i] for i in range(len(ts_cols))]

        # Batch forecast all columns at once
        point_forecast, _ = model.forecast(horizon=pred_len, inputs=contexts)

        # point_forecast shape: (num_cols, max_horizon) -> take first pred_len
        forecast = point_forecast[:, :pred_len].T  # (pred_len, num_cols)

        # Metrics computed in normalized space (standard ETT protocol)
        mae_losses.append(_mae(forecast, actual).sum())
        mse_losses.append(_mse(forecast, actual).sum())
        smape_losses.append(_smape(forecast, actual).sum())
        num_elements += actual.size
        abs_sum += np.abs(actual).sum()

    total_time = time.time() - start_time

    mse_val = float(np.sum(mse_losses) / num_elements)
    result = {
        "dataset": dataset_name,
        "pred_len": pred_len,
        "context_len": context_len,
        "mse": mse_val,
        "mae": float(np.sum(mae_losses) / num_elements),
        "smape": float(np.sum(smape_losses) / num_elements),
        "wape": float(np.sum(mae_losses) / abs_sum),
        "nrmse": float(np.sqrt(mse_val) / (abs_sum / num_elements)),
        "num_elements": int(num_elements),
        "total_time": total_time,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="ETT long-horizon benchmark for TimesFM 2.5"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./datasets/ETT-small",
        help="Directory to store ETT CSV files"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results/ett",
        help="Directory to save results"
    )
    parser.add_argument(
        "--context_len", type=int, default=512,
        help="Context length"
    )
    parser.add_argument(
        "--pred_lens", nargs="+", type=int, default=[96, 192, 336],
        help="Prediction lengths to evaluate"
    )
    parser.add_argument(
        "--datasets", nargs="+", type=str,
        default=["etth1", "etth2", "ettm1", "ettm2"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--normalize", action="store_true", default=True,
        help="Normalize data for evaluation"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    print("Loading TimesFM 2.5 200M model...")
=======

import timesfm

# ═════════════════════════════════════════════════════════════════════════════
# ETT dataset configuration
# ═════════════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═════════════════════════════════════════════════════════════════════════════
def run_ett_benchmark(results_dir: str, context_len: int = CONTEXT_LEN):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading TimesFM 2.5 200M (PyTorch)...")
>>>>>>> c04953b (feat: add benchmarking scripts for base Monash and ETT datasets)

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

<<<<<<< HEAD
    # We'll recompile per pred_len since max_horizon must accommodate it
    max_pred_len = max(args.pred_lens)
    model.compile(
        timesfm.ForecastConfig(
            max_context=args.context_len,
            max_horizon=max_pred_len,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
        )
    )
    print("Model loaded and compiled.")

    all_results = []

    for dataset_name in args.datasets:
        for pred_len in args.pred_lens:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name} | Horizon: {pred_len} | Context: {args.context_len}")
            print(f"{'='*60}")

            result = evaluate_dataset(
                model=model,
                dataset_name=dataset_name,
                data_dir=args.data_dir,
                context_len=args.context_len,
                pred_len=pred_len,
                normalize=args.normalize,
            )

            print(f"  MSE:   {result['mse']:.6f}")
            print(f"  MAE:   {result['mae']:.6f}")
            print(f"  SMAPE: {result['smape']:.6f}")
            print(f"  WAPE:  {result['wape']:.6f}")
            print(f"  Time:  {result['total_time']:.2f}s")

            all_results.append(result)

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} {'Horizon':>8} {'MSE':>10} {'MAE':>10} {'SMAPE':>10} {'WAPE':>10}")
    print("-" * 62)
    for r in all_results:
        print(
            f"{r['dataset']:<12} {r['pred_len']:>8} "
            f"{r['mse']:>10.6f} {r['mae']:>10.6f} "
            f"{r['smape']:>10.6f} {r['wape']:>10.6f}"
        )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
=======
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
    print(f"\n{'═'*80}")
    print("ETT LONG-HORIZON BENCHMARK SUMMARY (TimesFM 2.5 200M)")
    print("═" * 80)
    print(f"{'Dataset':<12} {'Horizon':<10} {'MSE':<12} {'MAE':<12} "
          f"{'SMAPE':<12} {'WAPE':<12} {'Time(s)':<10}")
    print("─" * 80)
    for key, r in sorted(all_results.items()):
        print(f"{r['dataset']:<12} {r['pred_len']:<10} {r['mse']:<12.6f} "
              f"{r['mae']:<12.6f} {r['smape']:<12.6f} {r['wape']:<12.6f} "
              f"{r['time']:<10.2f}")
    print("═" * 80)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETT benchmark for TimesFM 2.5")
    parser.add_argument("--results_dir", default="results/ett", help="Output directory")
    parser.add_argument("--context_len", type=int, default=512, help="Context length")
    args = parser.parse_args()

    run_ett_benchmark(results_dir=args.results_dir, context_len=args.context_len)
>>>>>>> c04953b (feat: add benchmarking scripts for base Monash and ETT datasets)
