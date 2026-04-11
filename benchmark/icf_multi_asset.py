#!/usr/bin/env python3
"""
ICF multi-asset / multi-timeframe benchmark.

Tests all 4 conditions (Base noPE, Base RoPE, Trained noPE, Trained RoPE)
across different tickers and intervals to check generalization of the
trained checkpoints beyond their BTC-USD 1d training domain.

Usage:
  python benchmark/icf_multi_asset.py
  python benchmark/icf_multi_asset.py --tickers BTC-USD:1h BTC-USD:1d ETH-USD:1d EUR=X:1d GC=F:1d SPY:1d
"""
from __future__ import annotations

import sys
import json
import math
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import timesfm
from timesfm.timesfm_icf import TimesFM_ICF_torch, ICFConfig

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "BTC-USD:1d",   # training domain
    "BTC-USD:1h",   # same asset, hourly
    "ETH-USD:1d",   # different crypto
    "EUR=X:1d",     # forex
    "GC=F:1d",      # gold futures
    "SPY:1d",       # equities
]

CONTEXT_LEN = 512
PRED_LEN     = 30
K_EXAMPLES   = 5
BATCH_SIZE   = 4

DEFAULT_CKPT_ROPE = str(ROOT / "results" / "icf_trained_rope" / "btc_usd_icf_rope.pt")
DEFAULT_CKPT_NOPE = str(ROOT / "results" / "icf_trained_nope" / "btc_usd_icf_nope.pt")

CONDITIONS = [
    ("ICF Base (noPE)",    "base",    True),
    ("ICF Base (RoPE)",    "base",    False),
    ("ICF Trained (noPE)", "trained", True),
    ("ICF Trained (RoPE)", "trained", False),
]
COLORS = {
    "ICF Base (noPE)":    "#d62728",
    "ICF Base (RoPE)":    "#2ca02c",
    "ICF Trained (noPE)": "#ff7f0e",
    "ICF Trained (RoPE)": "#1f77b4",
}

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333', 'text.color': '#333',
    'grid.color': '#ddd', 'font.size': 10,
})


# ── data ──────────────────────────────────────────────────────────────────────
def fetch_prices(ticker: str, interval: str) -> np.ndarray:
    period = "730d" if interval == "1h" else "5y"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices = df["Close"].dropna().values.astype(np.float64)
    if len(prices) < CONTEXT_LEN * 2:
        raise ValueError(f"{ticker}/{interval}: only {len(prices)} rows, need >{CONTEXT_LEN*2}")
    return prices


def _round_up(n: int, m: int) -> int:
    return math.ceil(n / m) * m

def _log_returns(seg: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.maximum(seg, 1e-8)))

def make_pool(train: np.ndarray, seg_len: int) -> list[np.ndarray]:
    pool = []
    for s in range(0, len(train) - seg_len + 1, PRED_LEN):
        pool.append(train[s : s + seg_len].copy())
    return pool

def pool_signatures(pool: list[np.ndarray], tail: int = 64) -> np.ndarray:
    sigs = []
    for seg in pool:
        lr = _log_returns(seg)
        sigs.append(lr[-tail:] if len(lr) >= tail else np.pad(lr, (tail - len(lr), 0)))
    return np.stack(sigs)

def select_examples(pool, sigs, ctx, k):
    lr   = _log_returns(ctx)
    tail = sigs.shape[1]
    q    = (lr[-tail:] if len(lr) >= tail else np.pad(lr, (tail - len(lr), 0))).astype(np.float64)
    dists = np.linalg.norm(sigs - q[None, :], axis=1)
    return [pool[int(i)] for i in np.argsort(dists.ravel())[:k]]


# ── model ─────────────────────────────────────────────────────────────────────
_model_cache: dict[str, TimesFM_ICF_torch] = {}

def get_model(kind: str, ckpt_rope: str, ckpt_nope: str, use_nope: bool) -> TimesFM_ICF_torch:
    """Cache models to avoid re-downloading base weights for every asset."""
    key = f"{kind}_{'nope' if use_nope else 'rope'}"
    if key not in _model_cache:
        icf = TimesFM_ICF_torch.from_pretrained_base("google/timesfm-2.5-200m-pytorch")
        if kind == "trained":
            path = ckpt_nope if use_nope else ckpt_rope
            device = icf.model.sep_token.device
            ckpt = torch.load(path, map_location=device, weights_only=True)
            icf.model.load_state_dict(ckpt["model"])
            print(f"    Loaded checkpoint: {Path(path).name}")
        icf.model.train(False)
        _model_cache[key] = icf
    return _model_cache[key]


# ── backtest ──────────────────────────────────────────────────────────────────
def run_backtest(prices, pool, sigs, icf, use_nope, label,
                 context_len, max_horizon, example_len, n_train) -> dict:
    fc = timesfm.ForecastConfig(
        max_context=context_len, max_horizon=max_horizon,
        normalize_inputs=True, per_core_batch_size=BATCH_SIZE,
        use_continuous_quantile_head=False, force_flip_invariance=True,
    )
    icf.compile(fc, icf_config=ICFConfig(
        k_examples=K_EXAMPLES, example_len=example_len, use_nope=use_nope,
    ))

    n = len(prices)
    window_specs = [
        (w, n_train + w * PRED_LEN, n_train + w * PRED_LEN + PRED_LEN)
        for w in range((n - n_train) // PRED_LEN)
        if n_train + w * PRED_LEN + PRED_LEN <= n
    ]

    results = []
    for bs in range(0, len(window_specs), BATCH_SIZE):
        batch = window_specs[bs : bs + BATCH_SIZE]
        ctx_list, act_list, ex_batch = [], [], []
        for _, ws, we in batch:
            ctx = prices[max(0, ws - context_len) : ws]
            ctx_list.append(ctx.astype(np.float64))
            act_list.append(prices[ws:we].astype(np.float64))
            ex_batch.append(select_examples(pool, sigs, ctx, K_EXAMPLES))

        point, _ = icf.forecast_icf(
            horizon=max_horizon, context_examples=ex_batch, target_inputs=ctx_list,
        )
        for i, (w, ws, we) in enumerate(batch):
            pred   = point[i, :PRED_LEN]
            actual = act_list[i]
            scale  = float(np.mean(np.abs(actual))) or 1.0
            results.append({
                "window": w,
                "mse":    float(np.mean((pred - actual) ** 2)),
                "mae":    float(np.mean(np.abs(pred - actual))),
                "nmae":   float(np.mean(np.abs(pred - actual))) / scale,  # normalised
            })

    return {
        "label":     label,
        "use_nope":  use_nope,
        "mse":       float(np.mean([r["mse"]  for r in results])),
        "mae":       float(np.mean([r["mae"]  for r in results])),
        "nmae":      float(np.mean([r["nmae"] for r in results])),
        "n_windows": len(results),
    }


# ── per-asset run ─────────────────────────────────────────────────────────────
def run_asset(ticker: str, interval: str, ckpt_rope: str, ckpt_nope: str) -> dict | None:
    tag = f"{ticker}/{interval}"
    print(f"\n{'━'*60}")
    print(f"  {tag}")
    print(f"{'━'*60}")
    try:
        prices = fetch_prices(ticker, interval)
    except Exception as e:
        print(f"  SKIP: {e}")
        return None

    n_train     = int(len(prices) * 0.8)
    context_len = _round_up(CONTEXT_LEN, 32)
    max_horizon = _round_up(PRED_LEN, 128)
    example_len = _round_up(context_len + max_horizon, 32)

    pool = make_pool(prices[:n_train], example_len)
    if len(pool) < K_EXAMPLES:
        print(f"  SKIP: pool too small ({len(pool)} segments)")
        return None
    sigs = pool_signatures(pool)

    print(f"  rows={len(prices)}  train={n_train}  test={len(prices)-n_train}  pool={len(pool)}")

    asset_results = {}
    shared = dict(prices=prices, pool=pool, sigs=sigs,
                  context_len=context_len, max_horizon=max_horizon,
                  example_len=example_len, n_train=n_train)

    for label, kind, use_nope in CONDITIONS:
        icf = get_model(kind, ckpt_rope, ckpt_nope, use_nope)
        r   = run_backtest(icf=icf, use_nope=use_nope, label=label, **shared)
        asset_results[label] = r
        print(f"    {label:<26}  nMAE={r['nmae']:.4f}  MSE={r['mse']:>14,.0f}  n={r['n_windows']}")

    return asset_results


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_summary(all_results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    assets  = list(all_results.keys())
    labels  = [c[0] for c in CONDITIONS]
    n_a     = len(assets)
    n_c     = len(labels)
    x       = np.arange(n_a)
    width   = 0.18

    # nMAE bar chart (normalised — comparable across price scales)
    fig, ax = plt.subplots(figsize=(max(10, n_a * 2.5), 5))
    for ci, label in enumerate(labels):
        vals = [all_results[a][label]["nmae"] for a in assets]
        offset = (ci - (n_c - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=COLORS[label], alpha=0.85, edgecolor='#333', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=20, ha='right')
    ax.set_ylabel('Normalised MAE (MAE / mean|actual|)', fontweight='bold')
    ax.set_title('ICF 4-way Ablation — Multi-Asset nMAE\n(lower = better; normalised so price scales are comparable)',
                 fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='#333', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "multi_asset_nmae.png")
    plt.savefig(p1, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Chart: {p1}")

    # Heatmap: winner per asset
    fig, ax = plt.subplots(figsize=(max(8, n_a * 1.4), 3.5))
    mat = np.array([[all_results[a][l]["nmae"] for l in labels] for a in assets])
    im = ax.imshow(mat.T, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(n_a)); ax.set_xticklabels(assets, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(n_c)); ax.set_yticklabels(labels, fontsize=9)
    for i in range(n_a):
        for j in range(n_c):
            ax.text(i, j, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=8,
                    color='black' if 0.3 < mat[i,j]/mat.max() < 0.8 else 'white')
    plt.colorbar(im, ax=ax, label='nMAE')
    ax.set_title('nMAE Heatmap — green=better, red=worse', fontweight='bold')
    plt.tight_layout()
    p2 = os.path.join(out_dir, "multi_asset_heatmap.png")
    plt.savefig(p2, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Chart: {p2}")

    # Winner table: which condition wins per asset
    print("\n" + "="*72)
    print(f"{'Asset':<18}  {'Winner':<26}  {'nMAE':>8}  {'2nd':<26}  {'nMAE':>8}")
    print("-"*72)
    for asset in assets:
        ranked = sorted(labels, key=lambda l: all_results[asset][l]["nmae"])
        best   = ranked[0]; second = ranked[1]
        print(f"{asset:<18}  {best:<26}  {all_results[asset][best]['nmae']:>8.4f}"
              f"  {second:<26}  {all_results[asset][second]['nmae']:>8.4f}")
    print("="*72)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="List of TICKER:INTERVAL pairs e.g. BTC-USD:1d ETH-USD:1h")
    parser.add_argument("--ckpt_rope", default=DEFAULT_CKPT_ROPE)
    parser.add_argument("--ckpt_nope", default=DEFAULT_CKPT_NOPE)
    parser.add_argument("--results_dir", default="results/multi_asset")
    args = parser.parse_args()

    all_results = {}
    for spec in args.tickers:
        parts    = spec.split(":")
        ticker   = parts[0]
        interval = parts[1] if len(parts) > 1 else "1d"
        tag      = f"{ticker}/{interval}"
        r = run_asset(ticker, interval, args.ckpt_rope, args.ckpt_nope)
        if r is not None:
            all_results[tag] = r

    os.makedirs(args.results_dir, exist_ok=True)
    json_path = os.path.join(args.results_dir, "results.json")
    # strip per-window detail for clean JSON
    clean = {a: {l: {k: v for k, v in cond.items() if k != "windows"}
                 for l, cond in conds.items()}
             for a, conds in all_results.items()}
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n  JSON: {json_path}")

    print("\nGenerating charts...")
    plot_summary(all_results, args.results_dir)


if __name__ == "__main__":
    main()
