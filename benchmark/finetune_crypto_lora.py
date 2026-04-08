#!/usr/bin/env python3
"""LoRA fine-tuning for TimesFM 2.5 200M on crypto data.

Injects LoRA adapters into the transformer layers, fine-tunes on BTC-USD
(or any crypto), and benchmarks the improvement.

Usage:
    python benchmark/finetune_crypto_lora.py --ticker BTC-USD --epochs 50
    python benchmark/finetune_crypto_lora.py --ticker BTC-USD --lora_rank 16 --lr 5e-5
"""

import argparse
import json
import math
import os
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tqdm
import yfinance as yf

import timesfm
from timesfm.torch.util import revin, update_running_stats

# ─── Colors (matching crypto benchmark) ──────────────────────────────────────
C_BG = "#0D1117"
C_CARD = "#161B22"
C_GRID = "#21262D"
C_TEXT = "#C9D1D9"
C_TITLE = "#F0F6FC"
C_ACCENT = "#58A6FF"
C_ORANGE = "#F0883E"
C_GREEN = "#3FB950"
C_RED = "#F85149"
C_PURPLE = "#BC8CFF"
C_CYAN = "#39D2C0"

EPS = 1e-7


# LoRA Definition

class LoRALinear(nn.Module):
    """LoRA adapter wrapping an existing nn.Linear layer.

    Computes: output = W(x) + B(A(x)) * (alpha/rank)
    Only A and B are trainable. The original W is frozen.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout_rate: float = 0.1):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout_rate)

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # LoRA A: (in, rank), initialized with Kaiming
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA B: (rank, out), initialized to zero so delta starts at 0
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Freeze original weights
        for p in self.base_linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        base_out = self.base_linear(x)
        # LoRA delta with dropout
        lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                target_modules: str = "all", dropout: float = 0.1) -> dict:
    """Inject LoRA adapters into the model's transformer layers.

    Args:
        model: TimesFM_2p5_200M_torch_module instance
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: 'all', 'attention', or 'ffn'

    Returns:
        Dict with injection statistics
    """
    injected = 0
    total_lora_params = 0
    total_base_params = sum(p.numel() for p in model.parameters())

    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Inject into stacked transformer layers
    for layer_idx, layer in enumerate(model.stacked_xf):
        attn = layer.attn

        if target_modules in ("all", "attention"):
            # QKV projection (fused or separate)
            if hasattr(attn, "qkv_proj") and isinstance(attn.qkv_proj, nn.Linear):
                attn.qkv_proj = LoRALinear(attn.qkv_proj, rank, alpha, dropout)
                injected += 1
            else:
                for name in ("query", "key", "value"):
                    if hasattr(attn, name) and isinstance(getattr(attn, name), nn.Linear):
                        setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha, dropout))
                        injected += 1

            # Output projection
            if isinstance(attn.out, nn.Linear):
                attn.out = LoRALinear(attn.out, rank, alpha, dropout)
                injected += 1

        if target_modules in ("all", "ffn"):
            # Feedforward layers
            if isinstance(layer.ff0, nn.Linear):
                layer.ff0 = LoRALinear(layer.ff0, rank, alpha, dropout)
                injected += 1
            if isinstance(layer.ff1, nn.Linear):
                layer.ff1 = LoRALinear(layer.ff1, rank, alpha, dropout)
                injected += 1

    # Count trainable params
    for p in model.parameters():
        if p.requires_grad:
            total_lora_params += p.numel()

    return {
        "injected_layers": injected,
        "lora_params": total_lora_params,
        "base_params": total_base_params,
        "pct_trainable": total_lora_params / total_base_params * 100,
    }


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA adapter weights."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
    torch.save(lora_state, path)
    print(f"  Saved LoRA weights ({len(lora_state)} tensors) to {path}")


def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA adapter weights."""
    lora_state = torch.load(path, weights_only=True)
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in lora_state and b_key in lora_state:
                module.lora_A.data = lora_state[a_key].to(module.lora_A.device)
                module.lora_B.data = lora_state[b_key].to(module.lora_B.device)
                loaded += 1
    print(f"  Loaded {loaded} LoRA adapter pairs from {path}")


# Dataset Pipeline

class CryptoTimeSeriesDataset(Dataset):
    """Sliding-window dataset for time series fine-tuning.

    Each sample is (context, target) pair of raw prices.
    RevIN inside the training loop handles normalization (matching model.decode()).
    Each sample is (context, target) pair of raw prices.
    Target length matches the prediction horizon (pred_len), NOT the model's
    full output head (o_len=128). We only supervise the horizon we care about.
    """

    def __init__(self, prices: np.ndarray, context_len: int, pred_len: int,
                 stride: int = 1):
        self.context_len = context_len
        self.pred_len = pred_len

        # Normalize globally using train-set statistics
        self.mean = np.mean(prices)
        self.std = np.std(prices) + EPS
        self.norm_prices = (prices - self.mean) / self.std

        # Generate window indices
        total_len = context_len + pred_len
        self.indices = list(range(0, len(self.norm_prices) - total_len + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        context = self.norm_prices[start:start + self.context_len].astype(np.float32)
        target = self.norm_prices[start + self.context_len:
                                  start + self.context_len + self.pred_len].astype(np.float32)
        return torch.from_numpy(context), torch.from_numpy(target)


# Training Loop

def _patch_and_revin(context_batch, patch_len, device):
    """Pad context to patch boundary, compute RevIN stats, return normed patches + stats."""
    batch_size = context_batch.shape[0]
    ctx_len = context_batch.shape[1]
    pad_len = patch_len - (ctx_len % patch_len)
    if pad_len < patch_len:
        padding = torch.zeros(batch_size, pad_len, device=device)
        pad_mask = torch.ones(batch_size, pad_len, dtype=torch.bool, device=device)
        inputs = torch.cat([padding, context_batch], dim=1)
        masks = torch.cat([pad_mask,
                           torch.zeros(batch_size, ctx_len, dtype=torch.bool, device=device)],
                          dim=1)
    else:
        inputs = context_batch
        masks = torch.zeros_like(context_batch, dtype=torch.bool)

    patched_inputs = inputs.reshape(batch_size, -1, patch_len)
    patched_masks  = masks.reshape(batch_size, -1, patch_len)

    n   = torch.zeros(batch_size, device=device)
    mu  = torch.zeros(batch_size, device=device)
    sig = torch.zeros(batch_size, device=device)
    patch_mus, patch_sigs = [], []
    for i in range(patched_inputs.shape[1]):
        (n, mu, sig), _ = update_running_stats(n, mu, sig, patched_inputs[:, i], patched_masks[:, i])
        patch_mus.append(mu)
        patch_sigs.append(sig)
    context_mu    = torch.stack(patch_mus, dim=1)
    context_sigma = torch.stack(patch_sigs, dim=1)

        # Normalize with RevIN
        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        # Forward pass
        (_, _, normed_outputs, _), _ = model_module(normed_inputs, patched_masks)

        o_len = model_module.o   # 128
        q_len = model_module.q   # 10
        normed_outputs = normed_outputs.reshape(batch_size, -1, o_len, q_len)

        # Point forecast: last patch, median quantile (index 5)
        pf = normed_outputs[:, -1, :, 5]  # (batch, o_len)

        # De-normalize predictions
        last_mu = context_mu[:, -1:]
        last_sigma = context_sigma[:, -1:]
        pf_denorm = pf * (last_sigma + EPS) + last_mu

        # Truncate to pred_len
        pf_pred = pf_denorm[:, :pred_len]

        # MSE loss
        loss = torch.mean((pf_pred - target_batch) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model_module, dataloader, device, pred_len, patch_len):
    """Validate, returning average MAE in RevIN-normalized space."""
    model_module.eval()
    total_loss = 0.0
    n_batches  = 0

    for context_batch, target_batch in dataloader:
        context_batch = context_batch.to(device)
        target_batch  = target_batch.to(device)
        batch_size    = context_batch.shape[0]

        normed_inputs, patched_masks, context_mu, context_sigma = \
            _patch_and_revin(context_batch, patch_len, device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, enabled=torch.cuda.is_available()):
            (_, _, normed_outputs, _), _ = model_module(normed_inputs, patched_masks)

        o_len = model_module.o
        q_len = model_module.q
        normed_outputs = normed_outputs.reshape(batch_size, -1, o_len, q_len)

        pf = normed_outputs[:, -1, :, 5]
        last_mu = context_mu[:, -1:]
        last_sigma = context_sigma[:, -1:]
        pf_denorm = pf * (last_sigma + EPS) + last_mu
        pf_pred = pf_denorm[:, :pred_len]

        loss = torch.mean((pf_pred - target_batch) ** 2)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# Backtesting and Evaluation

def run_backtest(model_wrapper, prices, context_len, pred_len, test_fraction=0.2):
    """Backtest returning per-window forecasts + aggregate metrics."""
    n = len(prices)
    test_size = int(n * test_fraction)
    train_size = n - test_size

    test_start = train_size
    test_end   = n
    num_windows = (test_end - test_start) // pred_len

    mse_total, mae_total, num_elements = 0.0, 0.0, 0
    windows = []

    for w in range(num_windows):
        window_start = test_start + w * pred_len
        window_end   = window_start + pred_len
        if window_end > test_end:
            break
        ctx_start = max(0, window_start - context_len)
        ctx = norm_prices[ctx_start:window_start]
        actual_norm = norm_prices[window_start:window_end]
        actual_raw = prices[window_start:window_end]

        # Model receives raw prices; normalize_inputs=True handles RevIN internally
        point_forecast, _ = model_wrapper.forecast(horizon=pred_len, inputs=[ctx])
        forecast_norm = point_forecast[0, :pred_len]
        forecast_raw = forecast_norm * scaler_std + scaler_mean

        w_mse = float(np.mean((forecast_norm - actual_norm) ** 2))
        w_mae = float(np.mean(np.abs(forecast_norm - actual_norm)))
        mse_total += np.sum((forecast_norm - actual_norm) ** 2)
        mae_total += np.sum(np.abs(forecast_norm - actual_norm))
        num_elements += len(actual_norm)

        # Directional accuracy on raw prices (sign of price change)
        actual_dir = np.sign(np.diff(actual_raw))
        pred_dir   = np.sign(np.diff(forecast_raw))
        min_len    = min(len(actual_dir), len(pred_dir))
        dir_acc    = float(np.mean(actual_dir[:min_len] == pred_dir[:min_len])) if min_len > 0 else 0.0

        windows.append({
            "idx": w, "start": window_start, "end": window_end,
            "actual": actual_raw, "forecast": forecast_raw,
            "context": prices[ctx_start:window_start],
            "mse": w_mse, "mae": w_mae, "dir_acc": dir_acc,
        })

    agg_mse = float(mse_total / max(num_elements, 1))
    agg_mae = float(mae_total / max(num_elements, 1))
    agg_dir = float(np.mean([w["dir_acc"] for w in windows])) if windows else 0.0

    return {"mse": agg_mse, "mae": agg_mae, "dir_acc": agg_dir,
            "num_windows": len(windows), "windows": windows}


# Charting Utilities

CHART_RC = {
    "figure.facecolor": C_BG, "axes.facecolor": C_CARD,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
    "text.color": C_TEXT, "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.grid": True, "grid.color": C_GRID, "grid.alpha": 0.4,
    "font.family": "sans-serif", "legend.facecolor": C_CARD,
    "legend.edgecolor": C_GRID,
}


def plot_training_curves(train_losses, val_losses, results_dir, ticker):
    """Panel 1: Training + validation loss over epochs."""
    if not train_losses or not val_losses:
        print("  Skipping training curves (no data)")
        return None
    plt.rcParams.update(CHART_RC)
    fig, ax = plt.subplots(figsize=(12, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=C_ACCENT, lw=2.2, label="Train Loss", alpha=0.9)
    ax.plot(epochs, val_losses, color=C_ORANGE, lw=2.2, label="Val Loss", alpha=0.9)
    best_ep = int(np.argmin(val_losses)) + 1
    ax.axvline(best_ep, color=C_GREEN, ls="--", lw=1.2, alpha=0.7, label=f"Best epoch ({best_ep})")
    ax.scatter([best_ep], [val_losses[best_ep-1]], color=C_GREEN, s=80, zorder=5)
    ax.set_title(f"LoRA Training Curves — {ticker}", fontsize=16, fontweight="bold", color=C_TITLE, pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    path = os.path.join(results_dir, f"{ticker.replace('-','_').lower()}_training_curves.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_metrics_comparison(baseline, finetuned, stats, results_dir, ticker):
    """Panel 2: Side-by-side bar chart of baseline vs fine-tuned metrics."""
    plt.rcParams.update(CHART_RC)
    metrics = ["MSE", "MAE", "Dir Accuracy"]
    base_vals = [baseline["mse"], baseline["mae"], baseline.get("dir_acc", 0)]
    ft_vals = [finetuned["mse"], finetuned["mae"], finetuned.get("dir_acc", 0)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Baseline vs LoRA Fine-Tuned — {ticker}",
                 fontsize=17, fontweight="bold", color=C_TITLE, y=1.02)

    colors_base = [C_RED, C_RED, C_RED]
    colors_ft = [C_GREEN, C_GREEN, C_GREEN]

    for i, (ax, name) in enumerate(zip(axes, metrics)):
        bv, fv = base_vals[i], ft_vals[i]
        bars = ax.bar(["Zero-Shot", "LoRA"], [bv, fv], color=[colors_base[i], colors_ft[i]],
                      width=0.5, edgecolor=C_GRID, linewidth=1.2, alpha=0.85)
        # Value labels
        for bar, val in zip(bars, [bv, fv]):
            fmt = f"{val:.4f}" if name != "Dir Accuracy" else f"{val:.1%}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    fmt, ha="center", va="bottom", fontsize=12, fontweight="bold", color=C_TITLE)
        ax.set_title(name, fontsize=14, fontweight="bold", color=C_TITLE, pad=8)
        ax.set_ylim(0, max(bv, fv) * 1.25)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)

        # Improvement badge
        if name != "Dir Accuracy":
            pct = (bv - fv) / bv * 100 if bv != 0 else 0
            badge_color = C_GREEN if pct > 0 else C_RED
            ax.text(0.5, 0.92, f"{'▼' if pct>0 else '▲'} {abs(pct):.1f}%",
                    transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold",
                    color=badge_color, bbox=dict(boxstyle="round,pad=0.3",
                    facecolor=badge_color + "22", edgecolor=badge_color, lw=1))
        else:
            pct = (fv - bv) * 100
            badge_color = C_GREEN if pct > 0 else C_RED
            ax.text(0.5, 0.92, f"{'▲' if pct>0 else '▼'} {abs(pct):.1f}pp",
                    transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold",
                    color=badge_color, bbox=dict(boxstyle="round,pad=0.3",
                    facecolor=badge_color + "22", edgecolor=badge_color, lw=1))

    fig.tight_layout()
    path = os.path.join(results_dir, f"{ticker.replace('-','_').lower()}_metrics_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_forecast_windows(baseline_bt, finetuned_bt, results_dir, ticker, n_windows=4):
    """Panel 3: Side-by-side forecast plots for selected test windows."""
    plt.rcParams.update(CHART_RC)
    b_wins = baseline_bt["windows"]
    f_wins = finetuned_bt["windows"]
    if not b_wins:
        return None

    # Pick windows: best, worst, median, last
    mse_diffs = [f_wins[i]["mse"] - b_wins[i]["mse"] for i in range(len(b_wins))]
    best_idx = int(np.argmin(mse_diffs))
    worst_idx = int(np.argmax(mse_diffs))
    sorted_by_mse = sorted(range(len(b_wins)), key=lambda i: b_wins[i]["mse"])
    median_idx = sorted_by_mse[len(sorted_by_mse)//2]
    last_idx = len(b_wins) - 1
    chosen = []
    for idx in [best_idx, median_idx, worst_idx, last_idx]:
        if idx not in chosen:
            chosen.append(idx)
    chosen = chosen[:n_windows]

    fig, axes = plt.subplots(len(chosen), 1, figsize=(16, 4.5 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]

    labels = {best_idx: "Best Improvement", worst_idx: "Worst Window",
              median_idx: "Median Window", last_idx: "Most Recent"}

    for ax_i, wi in enumerate(chosen):
        ax = axes[ax_i]
        bw, fw = b_wins[wi], f_wins[wi]
        pred_len = len(bw["actual"])
        ctx_show = min(60, len(bw["context"]))
        ctx_prices = bw["context"][-ctx_show:]
        days_ctx = list(range(-ctx_show, 0))
        days_pred = list(range(0, pred_len))

        # Context
        ax.plot(days_ctx, ctx_prices, color=C_TEXT, lw=1.5, alpha=0.5, label="Context")
        # Actual
        ax.plot(days_pred, bw["actual"], color=C_TITLE, lw=2.2, label="Actual", marker="o", ms=3)
        # Baseline forecast
        ax.plot(days_pred, bw["forecast"], color=C_RED, lw=2, ls="--",
                label=f'Zero-Shot (MSE={bw["mse"]:.4f})', alpha=0.85, marker="s", ms=3)
        # Fine-tuned forecast
        ax.plot(days_pred, fw["forecast"], color=C_GREEN, lw=2, ls="-",
                label=f'LoRA (MSE={fw["mse"]:.4f})', alpha=0.85, marker="^", ms=3)

        ax.axvline(0, color=C_ACCENT, ls=":", lw=1, alpha=0.6)
        ax.fill_betweenx(ax.get_ylim(), -ctx_show, 0, alpha=0.03, color=C_ACCENT)

        label = labels.get(wi, f"Window {wi}")
        ax.set_title(f"Window {wi+1}: {label}", fontsize=14, fontweight="bold",
                     color=C_TITLE, pad=8)
        ax.set_xlabel("Days relative to forecast start", fontsize=11)
        ax.set_ylabel("Price (USD)", fontsize=11)
        ax.legend(fontsize=10, loc="upper left", framealpha=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for spine in ax.spines.values():
            spine.set_color(C_GRID)

    fig.suptitle(f"Forecast Comparison — {ticker}", fontsize=18,
                 fontweight="bold", color=C_TITLE, y=1.01)
    fig.tight_layout()
    path = os.path.join(results_dir, f"{ticker.replace('-','_').lower()}_forecast_windows.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_error_analysis(baseline_bt, finetuned_bt, results_dir, ticker):
    """Panel 4: Per-window MSE comparison + error distribution."""
    plt.rcParams.update(CHART_RC)
    b_wins = baseline_bt["windows"]
    f_wins = finetuned_bt["windows"]
    if not b_wins:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left: per-window MSE bars
    ax = axes[0]
    x = np.arange(len(b_wins))
    w = 0.35
    ax.bar(x - w/2, [w_["mse"] for w_ in b_wins], w, color=C_RED, alpha=0.8,
           label="Zero-Shot", edgecolor=C_GRID)
    ax.bar(x + w/2, [w_["mse"] for w_ in f_wins], w, color=C_GREEN, alpha=0.8,
           label="LoRA", edgecolor=C_GRID)
    ax.set_xlabel("Test Window", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Per-Window MSE Comparison", fontsize=14, fontweight="bold",
                 color=C_TITLE, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{i+1}" for i in x], fontsize=9)
    ax.legend(fontsize=11, framealpha=0.8)

    # Right: histogram of errors
    ax2 = axes[1]
    b_errors = np.concatenate([w_["forecast"] - w_["actual"] for w_ in b_wins])
    f_errors = np.concatenate([w_["forecast"] - w_["actual"] for w_ in f_wins])
    bins = np.linspace(min(b_errors.min(), f_errors.min()),
                       max(b_errors.max(), f_errors.max()), 40)
    ax2.hist(b_errors, bins, color=C_RED, alpha=0.55, label="Zero-Shot Errors", edgecolor=C_GRID)
    ax2.hist(f_errors, bins, color=C_GREEN, alpha=0.55, label="LoRA Errors", edgecolor=C_GRID)
    ax2.axvline(0, color=C_TITLE, ls="--", lw=1.2, alpha=0.7)
    ax2.set_xlabel("Forecast Error (USD)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Error Distribution", fontsize=14, fontweight="bold", color=C_TITLE, pad=8)
    ax2.legend(fontsize=11, framealpha=0.8)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(C_GRID)

    fig.suptitle(f"Error Analysis — {ticker}", fontsize=17, fontweight="bold",
                 color=C_TITLE, y=1.02)
    fig.tight_layout()
    path = os.path.join(results_dir, f"{ticker.replace('-','_').lower()}_error_analysis.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


# Main Entrypoint

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune TimesFM 2.5 on crypto")
    parser.add_argument("--ticker", type=str, nargs="+", default=["BTC-USD"])
    parser.add_argument("--interval", type=str, default="1d", choices=["1h", "1d"])
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=30)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="all",
                        choices=["all", "attention", "ffn"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--results_dir", type=str, default="./results/lora")
    parser.add_argument("--load-only", action="store_true",
                        help="Skip training; load saved LoRA weights and run backtest only")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Fetch data ────────────────────────────────────────────────────────
    period = "max" if args.interval == "1d" else "730d"
    
    train_datasets = []
    val_datasets = []
    
    # We will use the first ticker as the primary for backtests
    primary_prices = None
    
    print("\nLoading TimesFM 2.5 200M...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model_module = model.model

    for ticker in args.ticker:
        print(f"\nFetching {ticker} ({args.interval})...")
        data = yf.download(ticker, period=period, interval=args.interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        prices = data["Close"].dropna().values.astype(np.float64)
        print(f"  {len(prices):,} candles fetched")
        
        if primary_prices is None:
            primary_prices = prices

        n = len(prices)
        test_size = int(n * args.test_fraction)
        val_size = int(n * 0.15)  # 15% for validation — need enough windows
        train_size = n - test_size - val_size
        
        # Auto-set stride to pred_len if not specified
        train_stride = args.stride if args.stride > 0 else args.pred_len
        
        # Pure isolated splits: No overlap for validation/test targets into train
        train_prices = prices[:train_size]
        # Prepend just enough context to start forecasting exactly at train_size cutoff
        val_start = max(0, train_size - args.context_len)
        val_prices = prices[val_start : train_size + val_size]

    train_ds = CryptoTimeSeriesDataset(train_prices, args.context_len, args.pred_len, stride=args.stride)
    val_ds = CryptoTimeSeriesDataset(val_prices, args.context_len, args.pred_len, stride=args.pred_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {test_size:,} candles")

    # ── Load model ────────────────────────────────────────────────────────
    print("\nLoading TimesFM 2.5 200M...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model_module = model.model
    # normalize_inputs=False is CRITICAL: model.decode() already does per-patch
    # RevIN internally (update_running_stats + revin). Setting this True would
    # add a SECOND global mean/std normalization layer on top, creating a
    # train/inference mismatch since training only uses per-patch RevIN.
    fc = timesfm.ForecastConfig(
        max_context=args.context_len, max_horizon=args.pred_len,
        normalize_inputs=False, use_continuous_quantile_head=False,
        force_flip_invariance=True,
    )

    # ── Baseline backtest ─────────────────────────────────────────────────
    primary_ticker = args.ticker[0]
    print(f"\n── Baseline Backtest (zero-shot) on {primary_ticker} ──")
    model.compile(fc)
    baseline_bt = run_backtest(model, primary_prices, args.context_len, args.pred_len, args.test_fraction)
    print(f"  MSE: {baseline_bt['mse']:.6f}  MAE: {baseline_bt['mae']:.6f}  "
          f"Dir Acc: {baseline_bt['dir_acc']:.1%}  Windows: {baseline_bt['num_windows']}")

    # ── Inject LoRA ───────────────────────────────────────────────────────
    print(f"\n── Injecting LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}) ──")
    model_module = model_module.to("cpu")
    stats = inject_lora(model_module, rank=args.lora_rank, alpha=args.lora_alpha,
                        target_modules=args.target_modules, dropout=args.lora_dropout)
    model_module = model_module.to(device)
    model_module.device = device  # sync the attribute TimesFM uses to route inputs
    model.model = model_module
    print(f"  Injected: {stats['injected_layers']} layers")
    print(f"  LoRA params: {stats['lora_params']:,} / {stats['base_params']:,} "
          f"({stats['pct_trainable']:.2f}%)")

    # ── Training ──────────────────────────────────────────────────────────
    patch_len = model_module.p
    os.makedirs(args.results_dir, exist_ok=True)
    lora_path = os.path.join(args.results_dir, f"{primary_ticker.replace('-','_').lower()}_lora.pt")

    train_losses, val_losses, best_epoch = [], [], 0
    if args.load_only:
        print(f"\n── Skipping training; loading weights from {lora_path} ──")
        load_lora_weights(model_module, lora_path)
        total_time = 0.0
    else:
        trainable = [p for p in model_module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

        print(f"\n── Training ({args.epochs} epochs, patience={args.patience}) ──")
        best_val = float("inf")
        t_total = time.time()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tl = train_one_epoch(model_module, train_loader, optimizer, device, args.pred_len, patch_len)
            vl = validate(model_module, val_loader, device, args.pred_len, patch_len)
            scheduler.step()
            train_losses.append(tl)
            val_losses.append(vl)
            dt = time.time() - t0

            improved = ""
            if vl < best_val:
                best_val, best_epoch = vl, epoch
                save_lora_weights(model_module, lora_path)
                improved = " ★"

            if epoch % max(1, args.epochs // 20) == 0 or epoch == 1 or improved:
                print(f"  Epoch {epoch:3d}/{args.epochs}  train={tl:.6f}  val={vl:.6f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}  {dt:.1f}s{improved}")

        total_time = time.time() - t_total
        print(f"\n  Best val: {best_val:.6f} @ epoch {best_epoch}  |  Total: {total_time:.0f}s")
        load_lora_weights(model_module, lora_path)

    # ── Fine-tuned backtest ───────────────────────────────────────────────
    print(f"\n── Fine-Tuned Backtest on {primary_ticker} ──")
    model.compile(fc)
    finetuned_bt = run_backtest(model, primary_prices, args.context_len, args.pred_len, args.test_fraction)
    print(f"  MSE: {finetuned_bt['mse']:.6f}  MAE: {finetuned_bt['mae']:.6f}  "
          f"Dir Acc: {finetuned_bt['dir_acc']:.1%}")

    # ── Summary ───────────────────────────────────────────────────────────
    mse_imp = (baseline_bt["mse"] - finetuned_bt["mse"]) / baseline_bt["mse"] * 100
    mae_imp = (baseline_bt["mae"] - finetuned_bt["mae"]) / baseline_bt["mae"] * 100
    dir_imp = (finetuned_bt["dir_acc"] - baseline_bt["dir_acc"]) * 100

    print(f"\n\n--- RESULTS: {primary_ticker} LoRA Fine-Tuning (rank {args.lora_rank}) ---")
    print(f"{'Metric':<16} {'Zero-Shot':>12} {'LoRA':>12} {'Change':>12}")
    print("-" * 56)
    print(f"{'MSE':<16} {baseline_bt['mse']:>12.6f} {finetuned_bt['mse']:>12.6f} {mse_imp:>+11.1f}%")
    print(f"{'MAE':<16} {baseline_bt['mae']:>12.6f} {finetuned_bt['mae']:>12.6f} {mae_imp:>+11.1f}%")
    print(f"{'Dir Accuracy':<16} {baseline_bt['dir_acc']:>11.1%} {finetuned_bt['dir_acc']:>11.1%} {dir_imp:>+10.1f}pp")
    print(f"Trainable: {stats['pct_trainable']:.2f}% ({stats['lora_params']:,} params) | Training Time: {total_time:.0f}s")

    # ── Generate all charts ───────────────────────────────────────────────
    print(f"\n── Generating Charts ──")
    chart_paths = []
    chart_paths.append(plot_training_curves(train_losses, val_losses, args.results_dir, primary_ticker))
    chart_paths.append(plot_metrics_comparison(baseline_bt, finetuned_bt, stats, args.results_dir, primary_ticker))
    chart_paths.append(plot_forecast_windows(baseline_bt, finetuned_bt, args.results_dir, primary_ticker))
    chart_paths.append(plot_error_analysis(baseline_bt, finetuned_bt, args.results_dir, primary_ticker))

    # ── Save JSON results ─────────────────────────────────────────────────
    results = {
        "ticker": primary_ticker, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules, "epochs": args.epochs, "best_epoch": best_epoch,
        "lora_params": stats["lora_params"], "pct_trainable": stats["pct_trainable"],
        "training_time_s": total_time,
        "baseline": {"mse": baseline_bt["mse"], "mae": baseline_bt["mae"], "dir_acc": baseline_bt["dir_acc"]},
        "finetuned": {"mse": finetuned_bt["mse"], "mae": finetuned_bt["mae"], "dir_acc": finetuned_bt["dir_acc"]},
        "mse_improvement_pct": mse_imp, "mae_improvement_pct": mae_imp,
        "dir_acc_improvement_pp": dir_imp,
        "train_losses": train_losses, "val_losses": val_losses,
        "chart_paths": [p for p in chart_paths if p],
    }
    rp = os.path.join(args.results_dir, "finetune_results.json")
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {rp}")
    print(f"  LoRA weights: {lora_path}")
    print(f"  Charts: {args.results_dir}/")
    print(f"\n  Done! ✓")


if __name__ == "__main__":
    main()
