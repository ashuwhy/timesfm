#!/usr/bin/env python3
"""Monash benchmark for base TimesFM 2.5 200M (PyTorch).

Evaluates the pretrained TimesFM 2.5 model on the extended Monash benchmark
suite (same datasets as v1/experiments/extended_benchmarks) using the v2.5 API.

Usage:
    python benchmark/run_base_monash.py [--save_dir results/monash]
"""

import argparse
import os
import time
from functools import partial
from itertools import repeat
import multiprocessing

import numpy as np
import pandas as pd
import torch

# ── gluonts dataset loading ─────────────────────────────────────────────────
from gluonts.dataset.repository.datasets import (
    dataset_names as gluonts_datasets,
    get_dataset,
)
from gluonts.time_feature.seasonality import get_seasonality

# ── evaluation helpers ───────────────────────────────────────────────────────
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mase, smape

import timesfm

# ═════════════════════════════════════════════════════════════════════════════
# Dataset list — same as v1/experiments/extended_benchmarks/run_timesfm.py
# ═════════════════════════════════════════════════════════════════════════════
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
    "nn5_weekly",
    "traffic",
    "weather",
    "australian_electricity_demand",
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "exchange_rate",
    "fred_md",
    "hospital",
]

# Per-dataset context-length overrides (shorter series benefit from smaller ctx)
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


# ═════════════════════════════════════════════════════════════════════════════
# Helpers: gluonts → pandas conversion
# ═════════════════════════════════════════════════════════════════════════════
def _transform_instance(args):
    ts, last_n = args
    start_period = ts["start"]
    start_ds = start_period.to_timestamp()
    freq = start_period.freq
    target = ts["target"]
    ds = pd.date_range(start=start_ds, freq=freq, periods=len(target))
    if last_n is not None:
        target = target[-last_n:]
        ds = ds[-last_n:]
    return pd.DataFrame({"unique_id": ts["item_id"], "ds": ds, "y": target})


def gluonts_to_df(dataset, last_n=None):
    """Convert a gluonts Dataset to a pandas DataFrame."""
    try:
        with multiprocessing.Pool(min(os.cpu_count() or 1, 4)) as pool:
            results = pool.map(
                _transform_instance, zip(dataset, repeat(last_n))
            )
    except Exception:
        results = [_transform_instance((ts, last_n)) for ts in dataset]
    return pd.concat(results).reset_index(drop=True)


def _maybe_download_m3(dataset_name):
    """Download M3 raw data if needed (gluonts expects it)."""
    from pathlib import Path
    if dataset_name[:2] == "m3":
        m3_file = Path.home() / ".gluonts" / "datasets" / "M3C.xls"
        if not m3_file.exists():
            try:
                from datasetsforecast.m3 import M3
                from datasetsforecast.utils import download_file
                download_file(m3_file.parent, M3.source_url)
            except Exception as e:
                print(f"  Warning: could not download M3 file: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═════════════════════════════════════════════════════════════════════════════
def run_monash_benchmark(save_dir: str, max_context: int = 512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading TimesFM 2.5 200M (PyTorch)...")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=128,
            normalize_inputs=True,
            per_core_batch_size=32,
        )
    )
    print("Model loaded & compiled.\n")

    model_name = "timesfm_2p5"
    results_list = []

    for ds_name in DATASET_NAMES:
        print(f"{'─'*60}")
        print(f"Dataset: {ds_name}")

        # Check availability
        if ds_name not in gluonts_datasets:
            print(f"  Skipping (not in gluonts registry)")
            continue

        _maybe_download_m3(ds_name)

        try:
            gluonts_ds = get_dataset(ds_name)
        except Exception as e:
            print(f"  Skipping (load error: {e})")
            continue

        horizon = gluonts_ds.metadata.prediction_length
        if horizon is None:
            print(f"  Skipping (no prediction_length)")
            continue

        freq = gluonts_ds.metadata.freq
        if freq == "D":
            seasonality = 7
        else:
            seasonality = get_seasonality(freq)

        ctx_len = CONTEXT_OVERRIDES.get(ds_name, max_context)
        print(f"  horizon={horizon}, freq={freq}, seasonality={seasonality}, ctx={ctx_len}")

        # Build train/test DataFrames
        train_df = gluonts_to_df(gluonts_ds.train)
        test_df = gluonts_to_df(gluonts_ds.test, last_n=horizon)
        # Keep only the first backtest window
        test_df = test_df.groupby("unique_id", sort=False).head(horizon)

        # Group by unique_id: build inputs list
        unique_ids = train_df["unique_id"].unique()
        inputs = []
        for uid in unique_ids:
            series = train_df.loc[train_df["unique_id"] == uid, "y"].values
            # Truncate to context length
            if len(series) > ctx_len:
                series = series[-ctx_len:]
            inputs.append(series.astype(np.float64))

        # Forecast
        t0 = time.time()
        point_forecast, quantile_forecast = model.forecast(
            horizon=horizon,
            inputs=inputs,
        )
        elapsed = time.time() - t0
        print(f"  Inference time: {elapsed:.2f}s for {len(inputs)} series")

        # Build forecast DataFrame
        fcst_rows = []
        for i, uid in enumerate(unique_ids):
            uid_test = test_df[test_df["unique_id"] == uid]
            if len(uid_test) == 0:
                continue
            ds_vals = uid_test["ds"].values[:horizon]
            pf = point_forecast[i, :horizon]
            for j in range(min(len(ds_vals), horizon)):
                fcst_rows.append({
                    "unique_id": uid,
                    "ds": ds_vals[j],
                    model_name: pf[j],
                })
        fcsts_df = pd.DataFrame(fcst_rows)

        # Evaluate
        try:
            merged = test_df.merge(fcsts_df, on=["unique_id", "ds"], how="left")
            merged = merged.dropna(subset=[model_name])
            if len(merged) == 0:
                print("  Skipping (no overlapping forecasts)")
                continue

            merged["unique_id"] = merged["unique_id"].astype(str)
            train_df["unique_id"] = train_df["unique_id"].astype(str)

            mase_fn = partial(mase, seasonality=seasonality)
            eval_df = evaluate(
                merged[["unique_id", "ds", "y", model_name]],
                train_df=train_df,
                metrics=[smape, mase_fn, mae],
            )
            eval_df = eval_df.groupby("metric").mean(numeric_only=True).reset_index()
            eval_df = eval_df.melt(
                id_vars="metric", value_name="value", var_name="model"
            )

            time_row = pd.DataFrame({
                "dataset": [ds_name],
                "metric": ["time"],
                "model": [model_name],
                "value": [elapsed],
            })
            eval_df.insert(0, "dataset", ds_name)
            eval_df = pd.concat([eval_df, time_row]).reset_index(drop=True)

            print(eval_df.to_string(index=False))
            results_list.append(eval_df)
        except Exception as e:
            print(f"  Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save combined results
    if results_list:
        all_results = pd.concat(results_list).reset_index(drop=True)
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "results.csv")
        all_results.to_csv(csv_path, index=False)
        print(f"\n{'═'*60}")
        print(f"All results saved to {csv_path}")

        # Print summary table
        summary = all_results[all_results["metric"] != "time"].pivot_table(
            index="dataset", columns="metric", values="value"
        )
        print(f"\n{'═'*60}")
        print("MONASH BENCHMARK SUMMARY (TimesFM 2.5 200M)")
        print("═" * 60)
        print(summary.to_string())
        print("═" * 60)
    else:
        print("No results collected!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monash benchmark for TimesFM 2.5")
    parser.add_argument("--save_dir", default="results/monash", help="Output directory")
    parser.add_argument("--max_context", type=int, default=512, help="Max context length")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    run_monash_benchmark(save_dir=args.save_dir, max_context=args.max_context)
