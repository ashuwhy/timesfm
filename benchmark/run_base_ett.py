#!/usr/bin/env python3
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

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

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
