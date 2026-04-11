#!/usr/bin/env python3
"""
4-way RoPE ablation:
  1. ICF Base   noPE  — base pretrained weights, RoPE disabled at inference
  2. ICF Base   RoPE  — base pretrained weights, RoPE enabled  at inference
  3. ICF Trained noPE — fine-tuned with RoPE disabled, inferred without RoPE
  4. ICF Trained RoPE — fine-tuned with RoPE enabled,  inferred with RoPE

Run training first:
  python benchmark/finetune_crypto_icf.py --results_dir results/icf_trained_rope
  python benchmark/finetune_crypto_icf.py --use_nope    --results_dir results/icf_trained_nope

Then run this script:
  python benchmark/rope_ablation_full.py
"""
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
import yfinance as yf
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import timesfm
from timesfm.timesfm_icf import TimesFM_ICF_torch, ICFConfig

# ── constants ────────────────────────────────────────────────────────────────
TICKER       = "BTC-USD"
CONTEXT_LEN  = 512
PRED_LEN     = 30
K_EXAMPLES   = 5
BATCH_SIZE   = 4

DEFAULT_CKPT_ROPE = str(ROOT / "results" / "icf_trained_rope" / "btc_usd_icf_rope.pt")
DEFAULT_CKPT_NOPE = str(ROOT / "results" / "icf_trained_nope" / "btc_usd_icf_nope.pt")

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333',    'text.color': '#333',
    'grid.color': '#ddd',        'font.size': 10, 'figure.dpi': 150,
})

COLORS = {
    "ICF Base (noPE)":    "#d62728",   # red
    "ICF Base (RoPE)":    "#2ca02c",   # green
    "ICF Trained (noPE)": "#ff7f0e",   # orange
    "ICF Trained (RoPE)": "#1f77b4",   # blue
}


# ── data helpers ─────────────────────────────────────────────────────────────
def fetch_data() -> np.ndarray:
    df = yf.download(TICKER, period="5y", progress=False)
    return df["Close"].values.astype(np.float64).ravel()


def _round_up(n: int, multiple: int) -> int:
    return math.ceil(n / multiple) * multiple


def _log_returns(seg: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.maximum(seg, 1e-8)))


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


def select_examples(pool, sigs, ctx, k):
    lr   = _log_returns(ctx)
    tail = sigs.shape[1]
    q    = (lr[-tail:] if len(lr) >= tail else np.pad(lr, (tail - len(lr), 0))).astype(np.float64)
    dists = np.linalg.norm(sigs - q[None, :], axis=1)
    return [pool[int(i)] for i in np.argsort(dists.ravel())[:k]]


# ── model loaders ─────────────────────────────────────────────────────────────
def load_base_icf() -> TimesFM_ICF_torch:
    icf = TimesFM_ICF_torch.from_pretrained_base("google/timesfm-2.5-200m-pytorch")
    icf.model.train(False)
    return icf


def load_trained_icf(checkpoint_path: str) -> TimesFM_ICF_torch:
    icf = load_base_icf()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run finetune_crypto_icf.py first."
        )
    device = icf.model.sep_token.device
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    icf.model.load_state_dict(ckpt["model"])
    icf.model.train(False)
    print(f"  Loaded: {checkpoint_path}")
    return icf


# ── backtest ──────────────────────────────────────────────────────────────────
def run_backtest(
    prices: np.ndarray,
    pool: list,
    sigs: np.ndarray,
    icf: TimesFM_ICF_torch,
    use_nope: bool,
    label: str,
    context_len: int,
    max_horizon: int,
    example_len: int,
    n_train: int,
) -> dict:
    fc = timesfm.ForecastConfig(
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

    n = len(prices)
    test_prices = prices[n_train:]
    num_windows = len(test_prices) // PRED_LEN
    window_specs = [
        (w, n_train + w * PRED_LEN, n_train + w * PRED_LEN + PRED_LEN)
        for w in range(num_windows)
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
            horizon=max_horizon,
            context_examples=ex_batch,
            target_inputs=ctx_list,
        )
        for i, (w, ws, we) in enumerate(batch):
            pred   = point[i, :PRED_LEN]
            actual = act_list[i]
            results.append({
                "window":  w,
                "mse":     float(np.mean((pred - actual) ** 2)),
                "mae":     float(np.mean(np.abs(pred - actual))),
                "pred":    pred.tolist(),
                "actual":  actual.tolist(),
                "context": ctx_list[i][-64:].tolist(),
            })

    agg_mse = float(np.mean([r["mse"] for r in results]))
    agg_mae = float(np.mean([r["mae"] for r in results]))
    print(f"  {label:<26}  MSE={agg_mse:>14,.0f}  MAE={agg_mae:>8,.1f}  n={len(results)}")
    return {"label": label, "use_nope": use_nope, "mse": agg_mse, "mae": agg_mae,
            "n_windows": len(results), "windows": results}


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_all(runs: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Aggregate bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [r["label"] for r in runs]
    mses   = [r["mse"]   for r in runs]
    maes   = [r["mae"]   for r in runs]
    colors = [COLORS[l] for l in labels]
    x = np.arange(len(labels))

    bars1 = axes[0].bar(x, mses, color=colors, alpha=0.85, edgecolor='#333', linewidth=1.1)
    axes[0].set_title('MSE — 4-way RoPE Ablation', fontweight='bold')
    axes[0].set_ylabel('Mean Squared Error', fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, mses):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val/1e6:.1f}M', ha='center', va='bottom', fontsize=8, fontweight='bold')

    bars2 = axes[1].bar(x, maes, color=colors, alpha=0.85, edgecolor='#333', linewidth=1.1)
    axes[1].set_title('MAE — 4-way RoPE Ablation', fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Error', fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    p1 = os.path.join(out_dir, "ablation_aggregate.png")
    plt.savefig(p1, dpi=220, bbox_inches='tight'); plt.close()
    print(f"  Chart: {p1}")

    # 2. Per-window MSE line chart
    n_win = min(r["n_windows"] for r in runs)
    fig, ax = plt.subplots(figsize=(12, 5))
    markers = ['o', 's', '^', 'D']
    for run, mk in zip(runs, markers):
        ax.plot(np.arange(n_win), [w["mse"] for w in run["windows"][:n_win]],
                marker=mk, markersize=4, label=run["label"],
                color=COLORS[run["label"]], alpha=0.75, linewidth=1.5)
    ax.set_xlabel('Test Window Index', fontweight='bold')
    ax.set_ylabel('MSE per Window', fontweight='bold')
    ax.set_title('Per-Window MSE — 4-way RoPE Ablation', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='#333')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "ablation_perwindow.png")
    plt.savefig(p2, dpi=220, bbox_inches='tight'); plt.close()
    print(f"  Chart: {p2}")

    # 3. Best-window forecast panel (window where best trained model beats base the most)
    base_rope = next(r for r in runs if r["label"] == "ICF Base (RoPE)")
    trained_rope = next(r for r in runs if r["label"] == "ICF Trained (RoPE)")
    n_w = min(base_rope["n_windows"], trained_rope["n_windows"])
    deltas = [base_rope["windows"][i]["mse"] - trained_rope["windows"][i]["mse"] for i in range(n_w)]
    plot_idx = int(np.argmax(deltas))

    fig, axes2 = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    ctx_ref = np.array(runs[0]["windows"][plot_idx]["context"])
    actual  = np.array(runs[0]["windows"][plot_idx]["actual"])
    ctx_x   = np.arange(-len(ctx_ref), 0)
    pred_x  = np.arange(len(actual))

    for ax, run in zip(axes2.flat, runs):
        pred = np.array(run["windows"][plot_idx]["pred"])
        c = COLORS[run["label"]]
        ax.plot(ctx_x, ctx_ref, color='#aaa', linewidth=1.4, label='Context', alpha=0.7)
        ax.plot(pred_x, actual, color='#222', linewidth=2, label='Actual', marker='o', markersize=3)
        ax.plot(pred_x, pred, color=c, linewidth=2, linestyle='--', marker='s', markersize=3,
                label=run["label"])
        ax.axvline(0, color='#888', linestyle=':', linewidth=1)
        ax.fill_between(pred_x, actual.min(), actual.max(), alpha=0.04, color='#ff7f0e')
        ax.set_title(f'{run["label"]}  (MSE={run["windows"][plot_idx]["mse"]:,.0f})', fontweight='bold')
        ax.set_xlabel('Time Step'); ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=True, edgecolor='#444')

    axes2[0,0].set_ylabel('BTC-USD Close', fontweight='bold')
    axes2[1,0].set_ylabel('BTC-USD Close', fontweight='bold')
    fig.suptitle(
        f'Window {plot_idx} — largest improvement of Trained RoPE over Base RoPE',
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    p3 = os.path.join(out_dir, "ablation_forecast.png")
    plt.savefig(p3, dpi=220, bbox_inches='tight'); plt.close()
    print(f"  Chart: {p3}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_rope", default=DEFAULT_CKPT_ROPE,
                        help="Checkpoint trained WITH RoPE (btc_usd_icf_rope.pt)")
    parser.add_argument("--ckpt_nope", default=DEFAULT_CKPT_NOPE,
                        help="Checkpoint trained WITHOUT RoPE (btc_usd_icf_nope.pt)")
    parser.add_argument("--results_dir", default="results/rope_ablation_full")
    args = parser.parse_args()

    print(f"Fetching {TICKER} ...")
    prices  = fetch_data()
    n_train = int(len(prices) * 0.8)
    print(f"  Total={len(prices)}  Train={n_train}  Test={len(prices)-n_train}\n")

    patch       = 32
    out_patch   = 128
    context_len = _round_up(CONTEXT_LEN, patch)
    max_horizon = _round_up(PRED_LEN, out_patch)
    example_len = _round_up(context_len + max_horizon, patch)

    train_prices = prices[:n_train]
    pool = make_pool(train_prices, example_len)
    if len(pool) < K_EXAMPLES:
        raise RuntimeError("Pool too small — not enough training history.")
    sigs = pool_signatures(pool)

    shared = dict(
        prices=prices, pool=pool, sigs=sigs,
        context_len=context_len, max_horizon=max_horizon,
        example_len=example_len, n_train=n_train,
    )

    runs = []

    print("── 1/4  ICF Base (noPE) ──────────────────────────────")
    icf = load_base_icf()
    runs.append(run_backtest(icf=icf, use_nope=True,  label="ICF Base (noPE)",    **shared))

    print("── 2/4  ICF Base (RoPE) ──────────────────────────────")
    icf = load_base_icf()
    runs.append(run_backtest(icf=icf, use_nope=False, label="ICF Base (RoPE)",    **shared))

    print("── 3/4  ICF Trained (noPE) ───────────────────────────")
    icf = load_trained_icf(args.ckpt_nope)
    runs.append(run_backtest(icf=icf, use_nope=True,  label="ICF Trained (noPE)", **shared))

    print("── 4/4  ICF Trained (RoPE) ───────────────────────────")
    icf = load_trained_icf(args.ckpt_rope)
    runs.append(run_backtest(icf=icf, use_nope=False, label="ICF Trained (RoPE)", **shared))

    # Save
    os.makedirs(args.results_dir, exist_ok=True)
    summary = {r["label"]: {"mse": r["mse"], "mae": r["mae"], "n_windows": r["n_windows"]}
               for r in runs}
    base_nope_mse = runs[0]["mse"]
    for r in runs:
        summary[r["label"]]["vs_base_nope_mse_pct"] = (
            (base_nope_mse - r["mse"]) / base_nope_mse * 100
        )
    json_path = os.path.join(args.results_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON: {json_path}")

    print("\nGenerating charts...")
    plot_all(runs, args.results_dir)

    print("\n" + "="*72)
    print("4-WAY ROPE ABLATION SUMMARY")
    print("="*72)
    print(f"{'Condition':<26}  {'MSE':>15}  {'MAE':>10}  {'vs Base noPE':>13}")
    print("-"*72)
    for r in runs:
        pct = summary[r["label"]]["vs_base_nope_mse_pct"]
        print(f"{r['label']:<26}  {r['mse']:>15,.0f}  {r['mae']:>10,.1f}  {pct:>+12.1f}%")
    print("="*72)


if __name__ == "__main__":
    main()
