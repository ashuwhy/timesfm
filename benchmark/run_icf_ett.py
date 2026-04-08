#!/usr/bin/env python3
"""ETT long-horizon benchmark for TimesFM-ICF (PyTorch).

Adapted from `benchmark/run_base_ett.py`. Uses `forecast_icf` with K in-context
examples per target window.

Example construction (no leakage):
- For each test window starting at position `pos`, we create K example segments
  of length T = context_len + pred_len from strictly earlier positions
  (ending before `pos`).
- Each example segment is normalized using the mean/std of its context portion
  (same protocol as the base ETT runner for fairness).

Note: Without continued-pretraining on ICF prompts, accuracy may not improve.
"""

from __future__ import annotations

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
from timesfm.timesfm_icf import ICFConfig


DATA_DICT = {
  "ettm2": {"boundaries": [34560, 46080, 57600], "data_path": "datasets/ETT-small/ETTm2.csv", "freq": "15min"},
  "ettm1": {"boundaries": [34560, 46080, 57600], "data_path": "datasets/ETT-small/ETTm1.csv", "freq": "15min"},
  "etth2": {"boundaries": [8640, 11520, 14400], "data_path": "datasets/ETT-small/ETTh2.csv", "freq": "H"},
  "etth1": {"boundaries": [8640, 11520, 14400], "data_path": "datasets/ETT-small/ETTh1.csv", "freq": "H"},
}

PRED_LENS = [96, 192, 336]
CONTEXT_LEN = 512

ETT_GITHUB_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"


def download_ett_if_needed(data_path: str, filename: str):
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


def _normalize_window(past: np.ndarray, window: np.ndarray) -> np.ndarray:
  mu = float(np.mean(past))
  std = float(np.std(past)) + 1e-9
  return (window - mu) / std


def create_test_windows_with_examples(
  data_df: pd.DataFrame,
  ts_cols: list[str],
  boundaries: list[int],
  context_len: int,
  pred_len: int,
  k_examples: int,
  max_windows: int | None = None,
):
  """Create disjoint test windows + K in-context examples per window."""
  test_start = boundaries[1]
  test_end = boundaries[2]

  windows_past = []
  windows_actual = []
  windows_examples = []  # list per window, list of K example segments

  pos = test_start
  windows_created = 0
  while pos + pred_len <= test_end:
    ctx_start = max(0, pos - context_len)

    for col in ts_cols:
      series = data_df[col].values.astype(np.float64)

      past = series[ctx_start:pos].copy()
      actual = series[pos : pos + pred_len].copy()

      past_n = _normalize_window(past, past)
      actual_n = _normalize_window(past, actual)

      # Build K examples from earlier positions.
      T = context_len + pred_len
      examples = []

      # Candidate end positions must be <= pos (no overlap into target future).
      # We'll take disjoint windows stepping backwards by pred_len.
      cand_end = pos
      for i in range(k_examples):
        ex_end = cand_end - i * pred_len
        ex_start = ex_end - T
        if ex_start < 0:
          # Not enough history; pad with zeros.
          ex_seg = np.zeros(T, dtype=np.float64)
          examples.append(ex_seg)
          continue
        ex_seg_raw = series[ex_start:ex_end].copy()
        ex_past = ex_seg_raw[:context_len]
        ex_seg = _normalize_window(ex_past, ex_seg_raw)
        examples.append(ex_seg)

      windows_past.append(past_n)
      windows_actual.append(actual_n)
      windows_examples.append(examples)

      windows_created += 1
      if max_windows is not None and windows_created >= max_windows:
        return windows_past, windows_actual, windows_examples

    pos += pred_len

  return windows_past, windows_actual, windows_examples


def run_ett_icf_benchmark(
  results_dir: str,
  context_len: int = CONTEXT_LEN,
  k_examples: int = 10,
  *,
  only_dataset: str | None = None,
  only_pred_len: int | None = None,
  max_windows: int | None = None,
):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Device: {device}")
  print("Loading TimesFM-ICF (from base TimesFM 2.5 checkpoint)...")

  model = timesfm.TimesFM_ICF_torch.from_pretrained_base("google/timesfm-2.5-200m-pytorch")

  all_results: dict[str, dict] = {}

  for dataset_name, cfg in DATA_DICT.items():
    if only_dataset is not None and dataset_name != only_dataset:
      continue
    data_path = cfg["data_path"]
    filename = os.path.basename(data_path)
    download_ett_if_needed(data_path, filename)

    print(f"\n{'═'*60}\nDataset: {dataset_name} ({data_path})")

    data_df = pd.read_csv(data_path)
    date_col = "date"
    ts_cols = [c for c in data_df.columns if c != date_col]
    boundaries = cfg["boundaries"]

    for pred_len in PRED_LENS:
      if only_pred_len is not None and pred_len != only_pred_len:
        continue
      T = context_len + pred_len
      print(f"\n  pred_len={pred_len}, context_len={context_len}, K={k_examples}, T={T}")

      model.compile(
        timesfm.ForecastConfig(
          max_context=context_len,
          max_horizon=pred_len,
          normalize_inputs=False,
          per_core_batch_size=32,
        ),
        icf_config=ICFConfig(k_examples=k_examples, example_len=T),
      )

      windows_past, windows_actual, windows_examples = create_test_windows_with_examples(
        data_df,
        ts_cols,
        boundaries,
        context_len,
        pred_len,
        k_examples,
        max_windows=max_windows,
      )
      n_windows = len(windows_past)
      print(f"  Number of windows: {n_windows}")

      if n_windows == 0:
        print("  No windows, skipping.")
        continue

      batch_size = 32
      mse_losses = []
      mae_losses = []
      smape_losses = []
      num_elements = 0
      abs_sum = 0.0

      t0 = time.time()

      for start_idx in tqdm.tqdm(
        range(0, n_windows, batch_size),
        desc=f"  {dataset_name}_h{pred_len}",
        leave=False,
      ):
        end_idx = min(start_idx + batch_size, n_windows)
        batch_past = windows_past[start_idx:end_idx]
        batch_actual = np.array(windows_actual[start_idx:end_idx])
        batch_examples = windows_examples[start_idx:end_idx]

        # forecast_icf expects list[list[np.ndarray]]
        point_forecast, _ = model.forecast_icf(
          horizon=pred_len,
          context_examples=batch_examples,
          target_inputs=batch_past,
        )
        forecasts = point_forecast[:, : batch_actual.shape[1]]

        mae_losses.append(_mae(forecasts, batch_actual).sum())
        mse_losses.append(_mse(forecasts, batch_actual).sum())
        smape_losses.append(_smape(forecasts, batch_actual).sum())
        num_elements += batch_actual.shape[0] * batch_actual.shape[1]
        abs_sum += np.abs(batch_actual).sum()

      elapsed = time.time() - t0

      mse_val = float(np.sum(mse_losses) / num_elements)
      mae_val = float(np.sum(mae_losses) / num_elements)
      smape_val = float(np.sum(smape_losses) / num_elements)
      wape_val = float(np.sum(mae_losses) / abs_sum)

      result_key = f"{dataset_name}_h{pred_len}"
      all_results[result_key] = {
        "mse": mse_val,
        "mae": mae_val,
        "smape": smape_val,
        "wape": wape_val,
        "nrmse": float(np.sqrt(mse_val) / (abs_sum / num_elements)),
        "num_windows": int(n_windows),
        "num_elements": int(num_elements),
        "time": float(elapsed),
        "dataset": dataset_name,
        "pred_len": int(pred_len),
        "context_len": int(context_len),
        "k_examples": int(k_examples),
      }

      print(f"    MSE={mse_val:.6f}  MAE={mae_val:.6f}  SMAPE={smape_val:.6f}  WAPE={wape_val:.6f}  Time={elapsed:.2f}s")

  os.makedirs(results_dir, exist_ok=True)
  json_path = os.path.join(results_dir, "results.json")
  with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
  print(f"\nResults saved to {json_path}")
  return all_results


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="ETT benchmark for TimesFM-ICF")
  ap.add_argument("--results_dir", default="results/icf/ett")
  ap.add_argument("--context_len", type=int, default=512)
  ap.add_argument("--k", type=int, default=10)
  ap.add_argument("--dataset", default=None, help="Optional: only run one dataset (etth1/etth2/ettm1/ettm2)")
  ap.add_argument("--pred_len", type=int, default=None, help="Optional: only run one horizon (e.g. 96)")
  ap.add_argument("--max_windows", type=int, default=None, help="Optional: cap number of (variable,window) samples")
  args = ap.parse_args()

  run_ett_icf_benchmark(
    results_dir=args.results_dir,
    context_len=args.context_len,
    k_examples=args.k,
    only_dataset=args.dataset,
    only_pred_len=args.pred_len,
    max_windows=args.max_windows,
  )
