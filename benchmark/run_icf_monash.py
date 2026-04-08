#!/usr/bin/env python3
"""Monash benchmark for TimesFM-ICF (PyTorch).

This follows the structure of `benchmark/run_base_monash.py`, but uses
`TimesFM_ICF_torch.forecast_icf()` with K in-context examples per target.

Example construction (no leakage):
- For each dataset, we build a pool of fully-observed example segments of length
  T = context_len + horizon from the TRAIN split only.
- For each target series, we sample K segments from the pool to use as in-context
  examples.

Note: Without continued-pretraining on ICF prompts, accuracy may not improve.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from functools import partial

import numpy as np
import pandas as pd
import torch

from gluonts.dataset.repository.datasets import (
  dataset_names as gluonts_datasets,
  get_dataset,
)
from gluonts.time_feature.seasonality import get_seasonality

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mase, smape

import timesfm
from timesfm.timesfm_icf import ICFConfig


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


def gluonts_to_df(dataset, last_n=None):
  rows = []
  for ts in dataset:
    start_period = ts["start"]
    start_ds = start_period.to_timestamp()
    freq = start_period.freq
    target = ts["target"]
    ds = pd.date_range(start=start_ds, freq=freq, periods=len(target))
    if last_n is not None:
      target = target[-last_n:]
      ds = ds[-last_n:]
    rows.append(pd.DataFrame({"unique_id": ts["item_id"], "ds": ds, "y": target}))
  return pd.concat(rows).reset_index(drop=True)


def _make_example_pool(train_series: list[np.ndarray], T: int) -> list[np.ndarray]:
  """Build a pool of fixed-length segments from training series only."""
  pool: list[np.ndarray] = []
  for s in train_series:
    if len(s) < T:
      continue
    # sample a few windows per series to keep pool bounded
    # (for reproducibility, use fixed random sampling downstream).
    step = max(1, (len(s) - T) // 4)
    for start in range(0, len(s) - T + 1, step):
      pool.append(s[start : start + T].astype(np.float64))
  return pool


def run_monash_icf_benchmark(
  save_dir: str,
  max_context: int = 512,
  k_examples: int = 10,
  *,
  only_dataset: str | None = None,
  max_series: int | None = None,
):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Device: {device}")
  print("Loading TimesFM-ICF (from base TimesFM 2.5 checkpoint)...")

  model = timesfm.TimesFM_ICF_torch.from_pretrained_base("google/timesfm-2.5-200m-pytorch")

  # NOTE: We compile per dataset since horizon varies.
  os.makedirs(save_dir, exist_ok=True)

  model_name = "timesfm_icf"
  results_list = []

  rng = random.Random(0)

  for ds_name in DATASET_NAMES:
    if only_dataset is not None and ds_name != only_dataset:
      continue
    print(f"{'─'*60}\nDataset: {ds_name}")

    if ds_name not in gluonts_datasets:
      print("  Skipping (not in gluonts registry)")
      continue

    try:
      gluonts_ds = get_dataset(ds_name)
    except Exception as e:
      print(f"  Skipping (load error: {e})")
      continue

    horizon = gluonts_ds.metadata.prediction_length
    if horizon is None:
      print("  Skipping (no prediction_length)")
      continue

    freq = gluonts_ds.metadata.freq
    seasonality = 7 if freq == "D" else get_seasonality(freq)

    ctx_len = CONTEXT_OVERRIDES.get(ds_name, max_context)
    T = ctx_len + horizon

    print(f"  horizon={horizon}, freq={freq}, seasonality={seasonality}, ctx={ctx_len}, K={k_examples}, T={T}")

    train_df = gluonts_to_df(gluonts_ds.train)
    test_df = gluonts_to_df(gluonts_ds.test, last_n=horizon)
    test_df = test_df.groupby("unique_id", sort=False).head(horizon)

    unique_ids = train_df["unique_id"].unique()
    if max_series is not None:
      unique_ids = unique_ids[:max_series]

    train_series = []
    inputs = []
    for uid in unique_ids:
      series = train_df.loc[train_df["unique_id"] == uid, "y"].values.astype(np.float64)
      train_series.append(series)
      if len(series) > ctx_len:
        series = series[-ctx_len:]
      inputs.append(series)

    pool = _make_example_pool(train_series, T=T)
    if not pool:
      print("  Skipping (no example pool segments of required length)")
      continue

    # Build in-context examples per target series by sampling from pool.
    context_examples = []
    for _ in unique_ids:
      ex = [pool[rng.randrange(len(pool))] for _ in range(k_examples)]
      context_examples.append(ex)

    model.compile(
      timesfm.ForecastConfig(
        max_context=ctx_len,
        max_horizon=horizon,
        normalize_inputs=True,
        per_core_batch_size=32,
      ),
      icf_config=ICFConfig(k_examples=k_examples, example_len=T),
    )

    t0 = time.time()
    point_forecast, _ = model.forecast_icf(horizon=horizon, context_examples=context_examples, target_inputs=inputs)
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.2f}s for {len(inputs)} series")

    fcst_rows = []
    for i, uid in enumerate(unique_ids):
      uid_test = test_df[test_df["unique_id"] == uid]
      if len(uid_test) == 0:
        continue
      ds_vals = uid_test["ds"].values[:horizon]
      pf = point_forecast[i, :horizon]
      for j in range(min(len(ds_vals), horizon)):
        fcst_rows.append({"unique_id": uid, "ds": ds_vals[j], model_name: pf[j]})
    fcsts_df = pd.DataFrame(fcst_rows)

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
      eval_df = eval_df.melt(id_vars="metric", value_name="value", var_name="model")

      time_row = pd.DataFrame({"dataset": [ds_name], "metric": ["time"], "model": [model_name], "value": [elapsed]})
      eval_df.insert(0, "dataset", ds_name)
      eval_df = pd.concat([eval_df, time_row]).reset_index(drop=True)

      print(eval_df.to_string(index=False))
      results_list.append(eval_df)
    except Exception as e:
      print(f"  Evaluation error: {e}")
      import traceback
      traceback.print_exc()
      continue

  if results_list:
    all_results = pd.concat(results_list).reset_index(drop=True)
    out_path = os.path.join(save_dir, "results.csv")
    all_results.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="Monash benchmark for TimesFM-ICF")
  ap.add_argument("--save_dir", default="results/icf/monash")
  ap.add_argument("--max_context", type=int, default=512)
  ap.add_argument("--k", type=int, default=10)
  ap.add_argument("--dataset", default=None, help="Optional: only run one dataset from DATASET_NAMES")
  ap.add_argument("--max_series", type=int, default=None, help="Optional: cap number of series forecasted")
  args = ap.parse_args()

  run_monash_icf_benchmark(
    save_dir=args.save_dir,
    max_context=args.max_context,
    k_examples=args.k,
    only_dataset=args.dataset,
    max_series=args.max_series,
  )
