#!/usr/bin/env python3
"""
RoPE ablation: compare ICF inference with use_nope=True (broken) vs False (correct).
"""
import sys
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import timesfm
from timesfm.timesfm_icf import TimesFM_ICF_torch, ICFConfig

TICKER = "BTC-USD"
CONTEXT_LEN = 512
PRED_LEN = 30           # actual forecast horizon we evaluate
K_EXAMPLES = 5
BATCH_SIZE = 4

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333',
    'text.color': '#333',
    'grid.color': '#ddd',
    'font.size': 10,
    'figure.dpi': 150,
})


def fetch_data() -> np.ndarray:
    df = yf.download(TICKER, period="5y", progress=False)
    return df["Close"].values.astype(np.float64).ravel()


def _round_up(n: int, multiple: int) -> int:
    return math.ceil(n / multiple) * multiple


def _log_returns(seg: np.ndarray) -> np.ndarray:
    s = np.maximum(seg, 1e-8)
    return np.diff(np.log(s))


def make_pool(train_prices: np.ndarray, seg_len: int) -> list[np.ndarray]:
    pool = []
    for start in range(0, len(train_prices) - seg_len + 1, PRED_LEN):
        pool.append(train_prices[start : start + seg_len].copy())
    return pool


def pool_signatures(pool: list[np.ndarray], tail: int = 64) -> np.ndarray:
    sigs = []
    for seg in pool:
        lr = _log_returns(seg)
        sigs.append(lr[-tail:] if len(lr) >= tail else np.pad(lr, (tail - len(lr), 0)))
    return np.stack(sigs)


def select_examples(pool: list, sigs: np.ndarray, ctx: np.ndarray, k: int) -> list[np.ndarray]:
    lr = _log_returns(ctx)
    tail = sigs.shape[1]
    q = (lr[-tail:] if len(lr) >= tail else np.pad(lr, (tail - len(lr), 0))).astype(np.float64)
    dists = np.linalg.norm(sigs - q[None, :], axis=1)  # shape (P,)
    idx = np.argsort(dists.ravel())[:k]
    return [pool[int(i)] for i in idx]


def run_icf_backtest(prices: np.ndarray, use_nope: bool, label: str) -> dict:
    patch = 32          # TimesFM input patch length
    out_patch = 128     # TimesFM output patch length
    context_len = _round_up(CONTEXT_LEN, patch)
    max_horizon = _round_up(PRED_LEN, out_patch)   # must be multiple of 128
    example_len = _round_up(context_len + max_horizon, patch)

    n = len(prices)
    n_train = int(n * 0.8)
    train_prices = prices[:n_train]
    test_prices  = prices[n_train:]

    pool = make_pool(train_prices, example_len)
    if len(pool) < K_EXAMPLES:
        raise RuntimeError(f"Pool too small ({len(pool)}). Reduce K_EXAMPLES or use more data.")
    sigs = pool_signatures(pool)

    icf = TimesFM_ICF_torch.from_pretrained_base("google/timesfm-2.5-200m-pytorch")
    fc  = timesfm.ForecastConfig(
        max_context=context_len,
        max_horizon=max_horizon,
        normalize_inputs=True,
        per_core_batch_size=BATCH_SIZE,
        use_continuous_quantile_head=False,
        force_flip_invariance=True,
    )
    icf.compile(fc, icf_config=ICFConfig(
        k_examples=K_EXAMPLES,
        example_len=example_len,
        use_nope=use_nope,
    ))

    num_windows = len(test_prices) // PRED_LEN
    window_specs = [
        (w, n_train + w * PRED_LEN, n_train + w * PRED_LEN + PRED_LEN)
        for w in range(num_windows)
        if n_train + w * PRED_LEN + PRED_LEN <= n
    ]

    results = []
    for batch_start in range(0, len(window_specs), BATCH_SIZE):
        batch = window_specs[batch_start : batch_start + BATCH_SIZE]
        ctx_list, act_list, ex_batch = [], [], []
        for _, ws, we in batch:
            ctx = prices[max(0, ws - context_len) : ws]
            ctx_list.append(ctx.astype(np.float64))
            act_list.append(prices[ws:we].astype(np.float64))
            ex_batch.append(select_examples(pool, sigs, ctx, K_EXAMPLES))

        point, _ = icf.forecast_icf(
            horizon=max_horizon,
            context_examples=ex_batch,
            target_inputs=ctx_list,
        )

        for i, (w, ws, we) in enumerate(batch):
            pred   = point[i, :PRED_LEN]
            actual = act_list[i]
            mse = float(np.mean((pred - actual) ** 2))
            mae = float(np.mean(np.abs(pred - actual)))
            results.append({
                "window": w,
                "mse": mse,
                "mae": mae,
                "pred": pred.tolist(),
                "actual": actual.tolist(),
                "context": ctx_list[i][-64:].tolist(),
            })

    agg_mse = float(np.mean([r["mse"] for r in results]))
    agg_mae = float(np.mean([r["mae"] for r in results]))
    print(f"{label:20s}  MSE={agg_mse:>14,.0f}  MAE={agg_mae:>8,.1f}  windows={len(results)}")
    return {
        "label": label,
        "use_nope": use_nope,
        "mse": agg_mse,
        "mae": agg_mae,
        "n_windows": len(results),
        "windows": results,
    }


def plot_comparison(data_nope, data_rope, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels_list = [data_nope["label"], data_rope["label"]]
    mses = [data_nope["mse"], data_rope["mse"]]
    maes = [data_nope["mae"], data_rope["mae"]]
    
    colors = ['#d62728', '#2ca02c']
    x = np.arange(len(labels_list))
    
    bars1 = axes[0].bar(x, mses, color=colors, alpha=0.8, edgecolor='#333', linewidth=1.2)
    axes[0].set_ylabel('Mean Squared Error', fontweight='bold')
    axes[0].set_title('ICF MSE: noPE vs RoPE', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_list, rotation=15, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, mses):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h, f'{val/1e6:.1f}M', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    bars2 = axes[1].bar(x, maes, color=colors, alpha=0.8, edgecolor='#333', linewidth=1.2)
    axes[1].set_ylabel('Mean Absolute Error', fontweight='bold')
    axes[1].set_title('ICF MAE: noPE vs RoPE', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_list, rotation=15, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, maes):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h, f'{val:,.0f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    path1 = os.path.join(out_dir, "rope_ablation_aggregate.png")
    plt.savefig(path1, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"  Chart: {path1}")
    
    nope_windows = [w["mse"] for w in data_nope["windows"]]
    rope_windows = [w["mse"] for w in data_rope["windows"]]
    n = min(len(nope_windows), len(rope_windows))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x_idx = np.arange(n)
    ax.plot(x_idx, nope_windows[:n], marker='o', markersize=4, label='noPE (broken)', 
            color='#d62728', alpha=0.7, linewidth=1.5)
    ax.plot(x_idx, rope_windows[:n], marker='s', markersize=4, label='RoPE (correct)', 
            color='#2ca02c', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Test Window Index', fontweight='bold')
    ax.set_ylabel('MSE per Window', fontweight='bold')
    ax.set_title('Per-Window MSE: ICF with noPE vs RoPE', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='#333')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path2 = os.path.join(out_dir, "rope_ablation_perwindow.png")
    plt.savefig(path2, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"  Chart: {path2}")
    
    mid_idx = len(data_nope["windows"]) // 2
    w_nope = data_nope["windows"][mid_idx]
    w_rope = data_rope["windows"][mid_idx]
    
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    
    ctx = np.array(w_nope["context"])
    pred_nope = np.array(w_nope["pred"])
    pred_rope = np.array(w_rope["pred"])
    actual = np.array(w_nope["actual"])
    
    ctx_x = np.arange(-len(ctx), 0)
    pred_x = np.arange(0, len(actual))
    
    axes2[0].plot(ctx_x, ctx, color='#999', linewidth=1.5, label='Context', alpha=0.7)
    axes2[0].plot(pred_x, actual, color='#333', linewidth=2, label='Actual', marker='o', markersize=3)
    axes2[0].plot(pred_x, pred_nope, color='#d62728', linewidth=2, label='ICF (noPE)', 
             linestyle='--', marker='x', markersize=4)
    axes2[0].axvline(0, color='#666', linestyle=':', linewidth=1)
    axes2[0].fill_between(pred_x, actual.min(), actual.max(), alpha=0.05, color='#ff7f0e')
    axes2[0].set_xlabel('Time Step', fontweight='bold')
    axes2[0].set_ylabel('BTC-USD Close Price', fontweight='bold')
    axes2[0].set_title(f'ICF with noPE (MSE={w_nope["mse"]:,.0f})', fontweight='bold')
    axes2[0].legend(loc='best', frameon=True, edgecolor='#333')
    axes2[0].grid(alpha=0.3)
    
    axes2[1].plot(ctx_x, ctx, color='#999', linewidth=1.5, label='Context', alpha=0.7)
    axes2[1].plot(pred_x, actual, color='#333', linewidth=2, label='Actual', marker='o', markersize=3)
    axes2[1].plot(pred_x, pred_rope, color='#2ca02c', linewidth=2, label='ICF (RoPE)', 
             linestyle='--', marker='s', markersize=4)
    axes2[1].axvline(0, color='#666', linestyle=':', linewidth=1)
    axes2[1].fill_between(pred_x, actual.min(), actual.max(), alpha=0.05, color='#ff7f0e')
    axes2[1].set_xlabel('Time Step', fontweight='bold')
    axes2[1].set_title(f'ICF with RoPE (MSE={w_rope["mse"]:,.0f})', fontweight='bold')
    axes2[1].legend(loc='best', frameon=True, edgecolor='#333')
    axes2[1].grid(alpha=0.3)
    
    plt.tight_layout()
    path3 = os.path.join(out_dir, "rope_ablation_forecast.png")
    plt.savefig(path3, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"  Chart: {path3}")
    
    return [path1, path2, path3]


def main():
    print(f"Fetching {TICKER} data...")
    prices = fetch_data()
    n_train = int(len(prices) * 0.8)
    print(f"  Total: {len(prices)}, Train: {n_train}, Test: {len(prices) - n_train}\n")

    print("Running ICF with use_nope=True  (broken — RoPE disabled)...")
    data_nope = run_icf_backtest(prices, use_nope=True,  label="ICF (noPE)")

    print("\nRunning ICF with use_nope=False (correct — RoPE enabled)...")
    data_rope = run_icf_backtest(prices, use_nope=False, label="ICF (RoPE)")
    
    out_dir = "results/rope_ablation"
    os.makedirs(out_dir, exist_ok=True)
    
    summary = {
        "noPE": {
            "mse": data_nope["mse"],
            "mae": data_nope["mae"],
            "n_windows": data_nope["n_windows"],
        },
        "RoPE": {
            "mse": data_rope["mse"],
            "mae": data_rope["mae"],
            "n_windows": data_rope["n_windows"],
        },
        "improvement": {
            "mse_reduction_pct": (data_nope["mse"] - data_rope["mse"]) / data_nope["mse"] * 100,
            "mae_reduction_pct": (data_nope["mae"] - data_rope["mae"]) / data_nope["mae"] * 100,
        }
    }
    
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {json_path}")
    
    print("\nGenerating comparison charts...")
    chart_paths = plot_comparison(data_nope, data_rope, out_dir)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"noPE (broken):   MSE={summary['noPE']['mse']:>15,.0f}  MAE={summary['noPE']['mae']:>10,.1f}")
    print(f"RoPE (correct):  MSE={summary['RoPE']['mse']:>15,.0f}  MAE={summary['RoPE']['mae']:>10,.1f}")
    print(f"Improvement:     {summary['improvement']['mse_reduction_pct']:>+14.1f}%  {summary['improvement']['mae_reduction_pct']:>+14.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()
