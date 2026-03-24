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

"""Live crypto benchmark for TimesFM 2.5 200M (PyTorch).

Fetches live BTC-USD (or other crypto) data via yfinance with maximum available
history, runs a rolling-window backtest, and generates a full benchmark report
with publication-quality charts.

Usage:
    # Full benchmark with charts (daily, max history)
    python benchmark/run_crypto_benchmark.py --ticker BTC-USD --interval 1d

    # Hourly granularity
    python benchmark/run_crypto_benchmark.py --ticker BTC-USD --interval 1h

    # Multiple cryptos
    python benchmark/run_crypto_benchmark.py --ticker BTC-USD ETH-USD SOL-USD

    # Forecast only
    python benchmark/run_crypto_benchmark.py --ticker BTC-USD --mode forecast --horizon 30
"""

import argparse
import datetime
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import torch
import tqdm
import yfinance as yf

import timesfm

# ─── Config ───────────────────────────────────────────────────────────────────

EPS = 1e-7

# Max history per interval
YFINANCE_PERIODS = {
    "1h": "730d",   # yfinance max for hourly
    "1d": "max",    # all available daily data
}

# ─── Styling ──────────────────────────────────────────────────────────────────

# Color palette
C_BG       = "#0D1117"
C_CARD     = "#161B22"
C_GRID     = "#21262D"
C_TEXT     = "#C9D1D9"
C_TITLE    = "#F0F6FC"
C_ACCENT   = "#58A6FF"
C_ORANGE   = "#F0883E"
C_GREEN    = "#3FB950"
C_RED      = "#F85149"
C_PURPLE   = "#BC8CFF"
C_CYAN     = "#39D2C0"

def apply_style():
    """Apply dark benchmark theme."""
    plt.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor": C_CARD,
        "axes.edgecolor": C_GRID,
        "axes.labelcolor": C_TEXT,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "text.color": C_TEXT,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.facecolor": C_CARD,
        "legend.edgecolor": C_GRID,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "sans-serif"],
        "figure.dpi": 150,
    })

# ─── Metrics ──────────────────────────────────────────────────────────────────

def _mse(y_pred, y_true):
    return np.square(y_pred - y_true)

def _mae(y_pred, y_true):
    return np.abs(y_pred - y_true)

def _smape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2
    abs_val = np.where(abs_val > EPS, abs_val, 1.0)
    abs_diff = np.where(abs_val > EPS, abs_diff, 0.0)
    return abs_diff / abs_val

# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_crypto_data(ticker: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance with maximum history."""
    period = YFINANCE_PERIODS.get(interval, "max")
    print(f"  Fetching {ticker} ({interval}) for period={period}...")

    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    data = data.reset_index()

    # Normalize datetime column name
    date_col = [c for c in data.columns if c in ("Date", "Datetime", "index")][0]
    data = data.rename(columns={date_col: "datetime"})

    span = data["datetime"].iloc[-1] - data["datetime"].iloc[0]
    print(f"  Fetched {len(data):,} candles spanning {span.days} days")
    print(f"  Range: {data['datetime'].iloc[0]} → {data['datetime'].iloc[-1]}")
    return data

# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest(
    model,
    prices: np.ndarray,
    datetimes: np.ndarray,
    context_len: int,
    pred_len: int,
    test_fraction: float = 0.2,
) -> dict:
    """Rolling-window backtest with per-window tracking for charts."""
    n = len(prices)
    test_size = int(n * test_fraction)
    train_size = n - test_size

    # Train-set StandardScaler (matching ETT benchmark protocol)
    train_data = prices[:train_size]
    scaler_mean = float(np.mean(train_data))
    scaler_std = float(np.std(train_data)) + EPS
    norm_prices = (prices - scaler_mean) / scaler_std

    test_start = train_size
    test_end = n
    num_windows = (test_end - test_start) // pred_len

    # Per-window tracking
    all_forecasts_raw = []    # raw scale forecasts
    all_actuals_raw = []      # raw scale actuals
    all_forecast_times = []   # datetime indices
    window_mses = []
    window_maes = []

    mse_total = 0.0
    mae_total = 0.0
    smape_total = 0.0
    num_elements = 0
    abs_sum = 0.0
    direction_correct = 0
    direction_total = 0

    start_time = time.time()

    for w in tqdm.tqdm(range(num_windows), desc="  Backtest"):
        window_start = test_start + w * pred_len
        window_end = window_start + pred_len
        if window_end > test_end:
            break

        ctx_start = max(0, window_start - context_len)
        ctx = norm_prices[ctx_start:window_start]
        actual_norm = norm_prices[window_start:window_end]

        point_forecast, _ = model.forecast(horizon=pred_len, inputs=[ctx])
        forecast_norm = point_forecast[0, :pred_len]

        # Normalized-space metrics
        w_mse = float(_mse(forecast_norm, actual_norm).mean())
        w_mae = float(_mae(forecast_norm, actual_norm).mean())
        window_mses.append(w_mse)
        window_maes.append(w_mae)

        mse_total += _mse(forecast_norm, actual_norm).sum()
        mae_total += _mae(forecast_norm, actual_norm).sum()
        smape_total += _smape(forecast_norm, actual_norm).sum()
        num_elements += len(actual_norm)
        abs_sum += np.abs(actual_norm).sum()

        # Raw-scale tracking for charts
        raw_actual = prices[window_start:window_end]
        raw_forecast = forecast_norm * scaler_std + scaler_mean
        all_forecasts_raw.append(raw_forecast)
        all_actuals_raw.append(raw_actual)
        all_forecast_times.append(datetimes[window_start:window_end])

        # Directional accuracy
        for i in range(len(raw_actual) - 1):
            actual_dir = raw_actual[i + 1] > raw_actual[i]
            pred_dir = raw_forecast[i + 1] > raw_forecast[i]
            if actual_dir == pred_dir:
                direction_correct += 1
            direction_total += 1

    total_time = time.time() - start_time
    mse_val = float(mse_total / num_elements)

    result = {
        "mse": mse_val,
        "mae": float(mae_total / num_elements),
        "smape": float(smape_total / num_elements),
        "wape": float(mae_total / abs_sum),
        "directional_accuracy": float(direction_correct / max(direction_total, 1)),
        "num_windows": int(num_windows),
        "num_elements": int(num_elements),
        "train_size": int(train_size),
        "test_size": int(test_size),
        "total_candles": int(n),
        "total_time": total_time,
        # For charting
        "_forecasts_raw": all_forecasts_raw,
        "_actuals_raw": all_actuals_raw,
        "_forecast_times": all_forecast_times,
        "_window_mses": window_mses,
        "_window_maes": window_maes,
        "_raw_prices": prices,
        "_raw_datetimes": datetimes,
        "_train_size": train_size,
    }
    return result

# ─── Charts ───────────────────────────────────────────────────────────────────

def create_benchmark_report(
    ticker: str,
    interval: str,
    bt_result: dict,
    fc_result: dict | None,
    results_dir: str,
):
    """Generate a multi-panel benchmark report."""
    apply_style()
    safe_ticker = ticker.replace("-", "_").lower()
    interval_label = "hours" if "h" in interval else "days"
    bt = bt_result

    has_forecast = fc_result is not None
    # Layout: header + price chart + 3 forecast zooms + error bar + scatter + (optional) forecast
    num_rows = 5 if has_forecast else 4

    fig = plt.figure(figsize=(20, 5 * num_rows))
    gs = GridSpec(num_rows, 3, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[0.5] + [1.0] * (num_rows - 1))

    # ── Row 0: Header ─────────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_axis_off()

    ax_h.text(0.5, 0.92,
              f"TimesFM 2.5  |  {ticker}  Benchmark",
              fontsize=24, fontweight="bold", color=C_TITLE,
              ha="center", va="top", transform=ax_h.transAxes)

    meta = (f"Interval: {interval}   "
            f"Data: {bt['total_candles']:,} candles   "
            f"Train: {bt['train_size']:,}   "
            f"Test: {bt['test_size']:,}   "
            f"Prediction window: {len(bt['_actuals_raw'][0])} {interval_label}")
    ax_h.text(0.5, 0.58, meta, fontsize=11, color="#8B949E",
              ha="center", va="center", transform=ax_h.transAxes)

    # Metric badges
    metrics_data = [
        ("MSE",     f"{bt['mse']:.4f}",                          C_ACCENT),
        ("MAE",     f"{bt['mae']:.4f}",                          C_CYAN),
        ("SMAPE",   f"{bt['smape']*100:.1f}%",                   C_ORANGE),
        ("WAPE",    f"{bt['wape']*100:.1f}%",                    C_PURPLE),
        ("Dir Acc", f"{bt['directional_accuracy']*100:.1f}%",
         C_GREEN if bt['directional_accuracy'] > 0.52 else C_RED),
    ]
    for i, (label, val, col) in enumerate(metrics_data):
        x = 0.1 + i * 0.18
        ax_h.text(x, 0.22, label, fontsize=9, color="#8B949E",
                  ha="center", transform=ax_h.transAxes)
        ax_h.text(x, 0.0, val, fontsize=18, fontweight="bold", color=col,
                  ha="center", transform=ax_h.transAxes)

    # ── Row 1: Full Price History (clean, no overlay) ─────────────────────
    ax_price = fig.add_subplot(gs[1, :])

    raw_prices = bt["_raw_prices"]
    raw_dt = bt["_raw_datetimes"]
    train_size = bt["_train_size"]

    # Train region
    ax_price.plot(raw_dt[:train_size], raw_prices[:train_size],
                  color=C_ACCENT, linewidth=0.9, alpha=0.7, label="Train")
    # Test region
    ax_price.plot(raw_dt[train_size:], raw_prices[train_size:],
                  color=C_TEXT, linewidth=1.0, label="Test")
    # Train/test boundary
    ax_price.axvline(x=raw_dt[train_size], color=C_RED, linestyle="--",
                     alpha=0.5, linewidth=1)

    # Shading for test region
    ax_price.axvspan(raw_dt[train_size], raw_dt[-1], alpha=0.06, color=C_ORANGE)

    ax_price.set_title(f"{ticker} — Full Price History", color=C_TITLE, fontsize=16)
    ax_price.set_ylabel("Price (USD)")
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_price.legend(loc="upper left", framealpha=0.8)

    # ── Row 2: Three Zoomed Forecast Windows ──────────────────────────────
    # Pick best, worst, and most recent window
    w_mses = bt["_window_mses"]
    best_idx = int(np.argmin(w_mses))
    worst_idx = int(np.argmax(w_mses))
    recent_idx = len(w_mses) - 1

    showcase = [
        (best_idx,   f"Best Window (#{best_idx})",   C_GREEN),
        (worst_idx,  f"Worst Window (#{worst_idx})",  C_RED),
        (recent_idx, f"Most Recent (#{recent_idx})",  C_ACCENT),
    ]

    for col_idx, (w_idx, title, color) in enumerate(showcase):
        ax = fig.add_subplot(gs[2, col_idx])

        actual = bt["_actuals_raw"][w_idx]
        pred = bt["_forecasts_raw"][w_idx]
        x = np.arange(len(actual))

        ax.plot(x, actual, color=C_TEXT, linewidth=1.8, label="Actual", zorder=3)
        ax.plot(x, pred, color=color, linewidth=1.8, linestyle="--",
                label="Predicted", zorder=3)

        # Error shading
        ax.fill_between(x, actual, pred, alpha=0.12, color=color)

        # MSE badge
        w_mse = w_mses[w_idx]
        ax.text(0.97, 0.95, f"MSE: {w_mse:.4f}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="right", va="top", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C_CARD,
                          edgecolor=color, alpha=0.9))

        ax.set_title(title, color=C_TITLE, fontsize=12)
        ax.set_xlabel(interval_label.capitalize())
        ax.set_ylabel("Price (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(loc="lower left", fontsize=9)

    # ── Row 3: Error Analysis ─────────────────────────────────────────────
    # Left: Per-window MSE bar chart
    ax_bars = fig.add_subplot(gs[3, :2])

    window_indices = np.arange(len(w_mses))
    bars = ax_bars.bar(window_indices, w_mses, width=0.8,
                       edgecolor="none", alpha=0.9)

    max_mse = max(w_mses) if w_mses else 1
    for bar, mse_val in zip(bars, w_mses):
        ratio = mse_val / max_mse
        if ratio > 0.7:
            bar.set_color(C_RED)
        elif ratio > 0.4:
            bar.set_color(C_ORANGE)
        else:
            bar.set_color(C_GREEN)

    avg_mse = float(np.mean(w_mses))
    ax_bars.axhline(y=avg_mse, color=C_ACCENT, linestyle="--", alpha=0.8, linewidth=1.2)
    ax_bars.text(len(w_mses) - 0.5, avg_mse, f" Mean: {avg_mse:.4f}",
                 color=C_ACCENT, fontsize=10, va="bottom", ha="right")

    ax_bars.set_title("Per-Window MSE", color=C_TITLE)
    ax_bars.set_xlabel("Window Index")
    ax_bars.set_ylabel("MSE (normalized)")

    # Right: Predicted vs Actual scatter (ALL windows)
    ax_scatter = fig.add_subplot(gs[3, 2])

    all_pred = np.concatenate(bt["_forecasts_raw"])
    all_actual = np.concatenate(bt["_actuals_raw"])

    ax_scatter.scatter(all_actual, all_pred, s=6, alpha=0.35, color=C_CYAN,
                       edgecolors="none", rasterized=True)

    # Perfect prediction line
    lo = min(all_actual.min(), all_pred.min())
    hi = max(all_actual.max(), all_pred.max())
    ax_scatter.plot([lo, hi], [lo, hi], "--", color=C_GREEN, alpha=0.7,
                    linewidth=1.5, label="y = x")

    corr = float(np.corrcoef(all_actual, all_pred)[0, 1])
    ax_scatter.text(0.05, 0.95, f"r = {corr:.4f}",
                    transform=ax_scatter.transAxes, fontsize=13,
                    fontweight="bold", color=C_ACCENT, va="top")

    ax_scatter.set_title("Predicted vs Actual (all windows)", color=C_TITLE)
    ax_scatter.set_xlabel("Actual (USD)")
    ax_scatter.set_ylabel("Predicted (USD)")
    ax_scatter.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_scatter.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_scatter.legend(loc="lower right", fontsize=9)

    # ── Row 4 (optional): Live Forecast ───────────────────────────────────
    if has_forecast:
        ax_fc = fig.add_subplot(gs[4, :])
        fc = fc_result

        hist_len = min(120, len(raw_prices))
        hist_prices = raw_prices[-hist_len:]

        ax_fc.plot(range(hist_len), hist_prices, color=C_ACCENT,
                   linewidth=1.5, label="Recent Price")

        horizon = fc["horizon"]
        forecast_x = range(hist_len - 1, hist_len + horizon)
        fv = np.concatenate(([raw_prices[-1]], fc["_forecast_values"]))
        ax_fc.plot(forecast_x, fv, color=C_ORANGE, linewidth=2.5,
                   linestyle="--", label=f"Forecast ({horizon} {interval_label})")

        # Confidence band
        if fc.get("_q10") is not None:
            q10_plot = np.concatenate(([raw_prices[-1]], fc["_q10"]))
            q90_plot = np.concatenate(([raw_prices[-1]], fc["_q90"]))
            ax_fc.fill_between(forecast_x, q10_plot, q90_plot,
                               alpha=0.15, color=C_ORANGE, label="80% CI")

        ax_fc.axvline(x=hist_len - 1, color=C_GRID, linestyle=":", alpha=0.7)

        pct = fc["pct_change"]
        arrow = "+" if pct >= 0 else ""
        ac = C_GREEN if pct >= 0 else C_RED
        ax_fc.text(0.98, 0.95, f"{arrow}{pct:.2f}%",
                   transform=ax_fc.transAxes, fontsize=16, fontweight="bold",
                   color=ac, ha="right", va="top",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=C_CARD,
                             edgecolor=ac, alpha=0.9))

        ax_fc.set_title(f"Live Forecast — Next {horizon} {interval_label.capitalize()}",
                        color=C_TITLE, fontsize=16)
        ax_fc.set_xlabel(f"Time ({interval_label})")
        ax_fc.set_ylabel("Price (USD)")
        ax_fc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_fc.legend(loc="upper left")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"{safe_ticker}_{interval}_report.png")
    fig.savefig(report_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"\n  Report saved to: {report_path}")
    return report_path

# ─── Live Forecast ────────────────────────────────────────────────────────────

def live_forecast(
    model,
    prices: np.ndarray,
    context_len: int,
    horizon: int,
) -> dict:
    """Generate a forward-looking forecast."""
    ctx = prices[-context_len:].astype(np.float64)

    print("  Generating forecast...")
    point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=[ctx])
    forecast_values = point_forecast[0, :horizon]

    q10 = q90 = None
    if quantile_forecast is not None and quantile_forecast.shape[-1] >= 9:
        q10 = quantile_forecast[0, :horizon, 0]
        q90 = quantile_forecast[0, :horizon, 8]

    last_price = prices[-1]
    pct_change = (forecast_values[-1] - last_price) / last_price * 100

    print(f"  Last price: ${last_price:,.2f}")
    print(f"  Predicted range: ${forecast_values.min():,.2f} – ${forecast_values.max():,.2f}")
    print(f"  End forecast: ${forecast_values[-1]:,.2f} ({pct_change:+.2f}%)")

    return {
        "last_price": float(last_price),
        "forecast_first": float(forecast_values[0]),
        "forecast_last": float(forecast_values[-1]),
        "forecast_min": float(forecast_values.min()),
        "forecast_max": float(forecast_values.max()),
        "pct_change": float(pct_change),
        "horizon": horizon,
        "_forecast_values": forecast_values,
        "_q10": q10,
        "_q90": q90,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live crypto benchmark for TimesFM 2.5"
    )
    parser.add_argument(
        "--ticker", nargs="+", default=["BTC-USD"],
        help="Crypto ticker(s) (default: BTC-USD)"
    )
    parser.add_argument(
        "--interval", type=str, default="1d",
        choices=["1h", "1d"],
        help="Data interval (default: 1d)"
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["backtest", "forecast", "both"],
        help="Run mode (default: both)"
    )
    parser.add_argument(
        "--context_len", type=int, default=512,
        help="Context length (default: 512)"
    )
    parser.add_argument(
        "--pred_len", type=int, default=30,
        help="Prediction length for backtest windows (default: 30)"
    )
    parser.add_argument(
        "--horizon", type=int, default=30,
        help="Forecast horizon (default: 30)"
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.2,
        help="Fraction of data for test (default: 0.2)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results/crypto",
        help="Results directory"
    )
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading TimesFM 2.5 200M model...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    max_horizon = max(args.pred_len, args.horizon)
    model.compile(
        timesfm.ForecastConfig(
            max_context=args.context_len,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
        )
    )
    print("Model loaded and compiled.\n")

    all_results = []

    for ticker in args.ticker:
        print(f"\n{'='*70}")
        print(f"  {ticker} ({args.interval})")
        print(f"{'='*70}")

        try:
            data = fetch_crypto_data(ticker, args.interval)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        prices = data["Close"].values.astype(np.float64)
        datetimes = data["datetime"].values

        if len(prices) < args.context_len + args.pred_len:
            print(f"  SKIP: Not enough data ({len(prices):,} candles)")
            continue

        ticker_result = {"ticker": ticker, "interval": args.interval}
        bt_result = None
        fc_result = None

        # Backtest
        if args.mode in ("backtest", "both"):
            print(f"\n  ── Backtest (pred_len={args.pred_len}, context={args.context_len}) ──")
            bt_result = backtest(
                model=model,
                prices=prices,
                datetimes=datetimes,
                context_len=args.context_len,
                pred_len=args.pred_len,
                test_fraction=args.test_fraction,
            )
            print(f"  MSE:   {bt_result['mse']:.6f}")
            print(f"  MAE:   {bt_result['mae']:.6f}")
            print(f"  SMAPE: {bt_result['smape']*100:.2f}%")
            print(f"  WAPE:  {bt_result['wape']*100:.2f}%")
            print(f"  Dir Accuracy: {bt_result['directional_accuracy']*100:.1f}%")
            print(f"  Time:  {bt_result['total_time']:.2f}s")

            # Store serializable version
            ticker_result["backtest"] = {
                k: v for k, v in bt_result.items() if not k.startswith("_")
            }

        # Live forecast
        if args.mode in ("forecast", "both"):
            print(f"\n  ── Live Forecast (horizon={args.horizon}) ──")
            fc_result = live_forecast(
                model=model,
                prices=prices,
                context_len=args.context_len,
                horizon=args.horizon,
            )
            ticker_result["forecast"] = {
                k: v for k, v in fc_result.items() if not k.startswith("_")
            }

        # Generate report chart
        if bt_result is not None:
            report_path = create_benchmark_report(
                ticker=ticker,
                interval=args.interval,
                bt_result=bt_result,
                fc_result=fc_result,
                results_dir=args.results_dir,
            )
            ticker_result["report_path"] = report_path

        all_results.append(ticker_result)

    # Save JSON results
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Summary
    if any("backtest" in r for r in all_results):
        print(f"\n{'='*80}")
        print("BACKTEST SUMMARY")
        print(f"{'='*80}")
        print(f"{'Ticker':<12} {'Candles':>10} {'MSE':>10} {'MAE':>10} {'SMAPE':>10} {'WAPE':>10} {'Dir Acc':>10}")
        print("─" * 74)
        for r in all_results:
            if "backtest" in r:
                bt = r["backtest"]
                print(
                    f"{r['ticker']:<12} "
                    f"{bt['total_candles']:>10,} "
                    f"{bt['mse']:>10.6f} {bt['mae']:>10.6f} "
                    f"{bt['smape']*100:>9.2f}% "
                    f"{bt['wape']*100:>9.2f}% "
                    f"{bt['directional_accuracy']*100:>9.1f}%"
                )


if __name__ == "__main__":
    main()
