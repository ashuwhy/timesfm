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

"""Monash extended benchmark for TimesFM 2.5 200M (PyTorch).

Evaluates the pretrained model on the full Monash dataset suite using gluonts.
Adapted from v1/experiments/extended_benchmarks/run_timesfm.py for the v2.5 API.

Usage:
    python benchmark/run_base_monash.py [--save_dir results/monash] [--max_context 512]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import timesfm

# Add benchmark dir to path so we can import monash_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monash_utils import ExperimentHandler

DATASET_NAMES = [
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    "m3_monthly",
    "m3_other",
    "m3_quarterly",
    "m3_yearly",
    "m4_quarterly",
    "m4_yearly",
    "tourism_monthly",
    "tourism_quarterly",
    "tourism_yearly",
    "nn5_daily_without_missing",
    "m5",
    "nn5_weekly",
    "traffic",
    "weather",
    "australian_electricity_demand",
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "ercot",
    "ett_small_15min",
    "ett_small_1h",
    "exchange_rate",
    "fred_md",
    "hospital",
]

QUANTILES = list(np.arange(1, 10) / 10.0)

# Datasets that benefit from shorter context lengths
CONTEXT_OVERRIDES = {
    "cif_2016": 32,
    "tourism_yearly": 64,
    "covid_deaths": 64,
    "tourism_quarterly": 64,
    "tourism_monthly": 64,
    "m1_monthly": 64,
    "m1_quarterly": 64,
    "m1_yearly": 64,
    "m3_monthly": 64,
    "m3_other": 64,
    "m3_quarterly": 64,
    "m3_yearly": 64,
    "m4_quarterly": 64,
    "m4_yearly": 64,
}

MODEL_NAME = "timesfm"


def forecast_on_df(
    model: timesfm.TimesFM_2p5_200M_torch,
    train_df: pd.DataFrame,
    horizon: int,
    context_len: int,
    model_name: str = "timesfm",
    quantiles: list[float] = QUANTILES,
) -> pd.DataFrame:
    """Forecast using v2.5 API, returning a DataFrame compatible with ExperimentHandler.

    Groups the input DataFrame by unique_id, extracts context, calls model.forecast(),
    and reassembles results into the expected format.
    """
    grouped = train_df.groupby("unique_id", sort=False)

    unique_ids = []
    contexts = []
    last_dates = []
    freqs_per_series = []

    for uid, group in grouped:
        group = group.sort_values("ds")
        values = group["y"].values.astype(np.float64)
        # Take last context_len values
        ctx = values[-context_len:] if len(values) > context_len else values
        contexts.append(ctx)
        unique_ids.append(uid)
        last_dates.append(group["ds"].iloc[-1])
        freqs_per_series.append(
            pd.infer_freq(group["ds"]) if len(group) > 2 else None
        )

    # Run forecast
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon, inputs=contexts
    )
    # point_forecast: (N, H), quantile_forecast: (N, H, Q)

    # Build output DataFrame
    all_rows = []
    for i, uid in enumerate(unique_ids):
        freq = freqs_per_series[i]
        try:
            future_dates = pd.date_range(
                start=last_dates[i], periods=horizon + 1, freq=freq
            )[1:]
        except Exception:
            # Fallback: integer index
            future_dates = list(range(horizon))

        row_data = {
            "unique_id": [uid] * horizon,
            "ds": future_dates,
            model_name: point_forecast[i, :horizon],
        }

        # Add quantile columns
        for qi, q in enumerate(quantiles):
            q_col = f"{model_name}-q-{q}"
            row_data[q_col] = quantile_forecast[i, :horizon, qi]

        all_rows.append(pd.DataFrame(row_data))

    fcsts_df = pd.concat(all_rows, ignore_index=True)
    return fcsts_df


def main():
    parser = argparse.ArgumentParser(
        description="Monash extended benchmark for TimesFM 2.5"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./results/monash",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max_context", type=int, default=512,
        help="Maximum context length"
    )
    parser.add_argument(
        "--max_horizon", type=int, default=128,
        help="Maximum horizon length for model compilation"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Specific datasets to evaluate (default: all)"
    )
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DATASET_NAMES

    # Load model
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
    model.compile(
        timesfm.ForecastConfig(
            max_context=args.max_context,
            max_horizon=args.max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
        )
    )
    print("Model loaded and compiled.")

    results_list = []
    run_id = np.random.randint(100000)

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating {MODEL_NAME} on dataset: {dataset}")
        print(f"{'='*60}")

        try:
            exp = ExperimentHandler(dataset, quantiles=QUANTILES)
        except Exception as e:
            print(f"  SKIP: Failed to load dataset {dataset}: {e}")
            continue

        # Context length override
        context_len = CONTEXT_OVERRIDES.get(dataset, args.max_context)
        horizon = exp.horizon

        # Re-compile if horizon exceeds current max_horizon
        if horizon > args.max_horizon:
            print(f"  Re-compiling model with max_horizon={horizon}...")
            model.compile(
                timesfm.ForecastConfig(
                    max_context=args.max_context,
                    max_horizon=horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=False,
                    force_flip_invariance=True,
                )
            )

        train_df = exp.train_df
        print(f"  Horizon: {horizon}, Context: {context_len}, "
              f"Series count: {train_df['unique_id'].nunique()}")

        init_time = time.time()
        fcsts_df = forecast_on_df(
            model=model,
            train_df=train_df,
            horizon=horizon,
            context_len=context_len,
            model_name=MODEL_NAME,
            quantiles=QUANTILES,
        )
        total_time = time.time() - init_time
        print(f"  Forecast time: {total_time:.2f}s")

        time_df = pd.DataFrame({"time": [total_time], "model": MODEL_NAME})

        try:
            results = exp.evaluate_from_predictions(
                models=[MODEL_NAME],
                fcsts_df=fcsts_df,
                times_df=time_df,
            )
            print(results.to_string(index=False))
            results_list.append(results)
        except Exception as e:
            print(f"  ERROR during evaluation: {e}")
            continue

        # Save incremental results
        if results_list:
            results_full = pd.concat(results_list)
            save_path = os.path.join(args.save_dir, str(run_id))
            os.makedirs(save_path, exist_ok=True)
            results_full.to_csv(f"{save_path}/results.csv", index=False)

    # Final summary
    if results_list:
        results_full = pd.concat(results_list)
        save_path = os.path.join(args.save_dir, str(run_id))
        os.makedirs(save_path, exist_ok=True)
        results_full.to_csv(f"{save_path}/results.csv", index=False)

        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        pivot = results_full.pivot_table(
            index="dataset", columns="metric", values="value", aggfunc="mean"
        )
        print(pivot.to_string())
        print(f"\nResults saved to: {save_path}/results.csv")
    else:
        print("\nNo results were produced.")


if __name__ == "__main__":
    main()
