#!/usr/bin/env python3
"""LoRA fine-tuning for TimesFM 2.5 200M on crypto data.

Injects LoRA adapters into the transformer attention/FFN layers, fine-tunes on
BTC-USD (or any set of tickers), and benchmarks the improvement over zero-shot.

Usage:
    python benchmark/finetune_crypto_lora.py
    python benchmark/finetune_crypto_lora.py --ticker BTC-USD ETH-USD --epochs 50
    python benchmark/finetune_crypto_lora.py --lora_rank 8 --lr 3e-5
"""

import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import yfinance as yf

import timesfm
from timesfm.torch.util import revin, update_running_stats

# ── Dark theme colours ────────────────────────────────────────────────────────
C_BG    = "#0D1117"
C_CARD  = "#161B22"
C_GRID  = "#21262D"
C_TEXT  = "#C9D1D9"
C_TITLE = "#F0F6FC"
C_ACCENT = "#58A6FF"
C_ORANGE = "#F0883E"
C_GREEN  = "#3FB950"
C_RED    = "#F85149"

EPS = 1e-7


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """LoRA adapter wrapping an existing nn.Linear.

    output = W(x) + (A @ B)(x) * (alpha / rank)

    A and B are initialized on the same device and dtype as the base weight,
    which prevents silent float32 / bfloat16 precision mismatches.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout_rate)

        in_features  = base_linear.in_features
        out_features = base_linear.out_features
        # Match dtype and device of the base layer to avoid precision cliffs.
        base_dtype   = base_linear.weight.dtype
        base_device  = base_linear.weight.device

        self.lora_A = nn.Parameter(
            torch.empty(in_features, rank, dtype=base_dtype, device=base_device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features, dtype=base_dtype, device=base_device)
        )

        for p in self.base_linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B * self.scaling
        return base_out + lora_out


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: str = "attention",
    dropout: float = 0.05,
) -> dict:
    """Freeze the base model then inject LoRA adapters into stacked_xf layers."""
    injected = 0
    total_base_params = sum(p.numel() for p in model.parameters())

    for p in model.parameters():
        p.requires_grad = False

    for layer in model.stacked_xf:
        attn = layer.attn

        if target_modules in ("all", "attention"):
            if hasattr(attn, "qkv_proj") and isinstance(attn.qkv_proj, nn.Linear):
                attn.qkv_proj = LoRALinear(attn.qkv_proj, rank, alpha, dropout)
                injected += 1
            else:
                for name in ("query", "key", "value"):
                    if hasattr(attn, name) and isinstance(getattr(attn, name), nn.Linear):
                        setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha, dropout))
                        injected += 1
            if isinstance(attn.out, nn.Linear):
                attn.out = LoRALinear(attn.out, rank, alpha, dropout)
                injected += 1

        if target_modules in ("all", "ffn"):
            if isinstance(layer.ff0, nn.Linear):
                layer.ff0 = LoRALinear(layer.ff0, rank, alpha, dropout)
                injected += 1
            if isinstance(layer.ff1, nn.Linear):
                layer.ff1 = LoRALinear(layer.ff1, rank, alpha, dropout)
                injected += 1

    total_lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "injected_layers": injected,
        "lora_params": total_lora_params,
        "base_params": total_base_params,
        "pct_trainable": total_lora_params / total_base_params * 100,
    }


def save_lora_weights(model: nn.Module, path: str):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            state[f"{name}.lora_B"] = module.lora_B.data.cpu()
    torch.save(state, path)
    print(f"  Saved LoRA weights ({len(state)} tensors) -> {path}")


def load_lora_weights(model: nn.Module, path: str):
    state = torch.load(path, weights_only=True)
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key, b_key = f"{name}.lora_A", f"{name}.lora_B"
            if a_key in state and b_key in state:
                module.lora_A.data = state[a_key].to(module.lora_A.device)
                module.lora_B.data = state[b_key].to(module.lora_B.device)
                loaded += 1
    print(f"  Loaded {loaded} LoRA adapter pairs from {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset  — raw prices only, no global normalisation
# ═══════════════════════════════════════════════════════════════════════════════

class CryptoTimeSeriesDataset(Dataset):
    """Sliding-window dataset.

    Returns raw float32 (context, future) pairs.  All normalisation is handled
    inside the training loop via RevIN, exactly as model.decode() does it.
    Pre-normalising here would create a train / inference distribution mismatch.

    target_len should equal model.o (128) so every output neuron is supervised.
    """

    def __init__(
        self,
        prices: np.ndarray,
        context_len: int,
        target_len: int,
        stride: int = 1,
    ):
        self.prices = prices.astype(np.float32)
        self.context_len = context_len
        self.target_len = target_len
        total = context_len + target_len
        self.indices = list(range(0, len(self.prices) - total + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        context = self.prices[s : s + self.context_len]
        future  = self.prices[s + self.context_len : s + self.context_len + self.target_len]
        return torch.from_numpy(context), torch.from_numpy(future)


# ═══════════════════════════════════════════════════════════════════════════════
# Core training utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_and_revin(
    context_batch: torch.Tensor,
    patch_len: int,
    device: torch.device,
):
    """Left-pad to patch boundary, compute per-patch RevIN stats, return
    normalised patches and the running (mu, sigma) tensors.

    Returns:
        normed_inputs  : (batch, n_patches, patch_len)  — normalised, masked=0
        patched_masks  : (batch, n_patches, patch_len)  — True = padded
        context_mu     : (batch, n_patches)
        context_sigma  : (batch, n_patches)
    """
    batch_size = context_batch.shape[0]
    ctx_len    = context_batch.shape[1]
    remainder  = ctx_len % patch_len
    pad_len    = (patch_len - remainder) if remainder != 0 else 0

    if pad_len > 0:
        padding  = torch.zeros(batch_size, pad_len, device=device, dtype=context_batch.dtype)
        pad_mask = torch.ones(batch_size, pad_len, dtype=torch.bool, device=device)
        inputs   = torch.cat([padding, context_batch], dim=1)
        masks    = torch.cat(
            [pad_mask, torch.zeros(batch_size, ctx_len, dtype=torch.bool, device=device)],
            dim=1,
        )
    else:
        inputs = context_batch
        masks  = torch.zeros(batch_size, ctx_len, dtype=torch.bool, device=device)

    patched_inputs = inputs.reshape(batch_size, -1, patch_len)
    patched_masks  = masks.reshape(batch_size, -1, patch_len)

    n   = torch.zeros(batch_size, device=device, dtype=context_batch.dtype)
    mu  = torch.zeros(batch_size, device=device, dtype=context_batch.dtype)
    sig = torch.zeros(batch_size, device=device, dtype=context_batch.dtype)
    patch_mus, patch_sigs = [], []
    for i in range(patched_inputs.shape[1]):
        (n, mu, sig), _ = update_running_stats(
            n, mu, sig, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mus.append(mu)
        patch_sigs.append(sig)

    context_mu    = torch.stack(patch_mus, dim=1)
    context_sigma = torch.stack(patch_sigs, dim=1)

    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, torch.zeros_like(normed_inputs), normed_inputs)
    return normed_inputs, patched_masks, context_mu, context_sigma


def _build_patch_targets(
    context_batch: torch.Tensor,
    future_batch: torch.Tensor,
    patch_len: int,
    o_len: int,
    n_patches: int,
) -> torch.Tensor:
    """Build staggered per-patch targets of length o_len.

    For patch i (0-indexed), the target is the o_len raw values starting right
    after the patch's last time-step, i.e. from position (i+1)*patch_len.

    Returns: (batch, n_patches, o_len)
    """
    stream  = torch.cat([context_batch, future_batch], dim=1)
    targets = []
    for i in range(n_patches):
        start = (i + 1) * patch_len
        end   = start + o_len
        # Clamp to stream length — last patches may have shorter targets.
        chunk = stream[:, start:end]
        if chunk.shape[1] < o_len:
            pad = torch.zeros(
                chunk.shape[0], o_len - chunk.shape[1],
                device=chunk.device, dtype=chunk.dtype,
            )
            chunk = torch.cat([chunk, pad], dim=1)
        targets.append(chunk)
    return torch.stack(targets, dim=1)  # (batch, n_patches, o_len)


def _compute_loss(
    model_module: nn.Module,
    context_batch: torch.Tensor,
    future_batch: torch.Tensor,
    patch_len: int,
    dir_loss_weight: float = 0.0,
    supervise_last_n: int = 0,
    supervise_horizon: int = 0,
) -> torch.Tensor:
    """MAE loss in RevIN-normalised space, optionally focused on last patches.

    Args:
        supervise_last_n: If >0, only supervise the last N non-padded patches
            instead of all patches.  Focuses adaptation on the forecasting
            boundary that the backtest actually evaluates.
        supervise_horizon: If >0, only supervise the first H time-steps of each
            patch's output (e.g. 30 to match pred_len), not the full o_len=128.
        dir_loss_weight: Weight for directional loss component (0=disabled).
    """
    batch_size = context_batch.shape[0]
    o_len  = model_module.o   # 128
    q_len  = model_module.q   # 10
    aridx  = getattr(model_module, "aridx", 5)

    normed_inputs, patched_masks, context_mu, context_sigma = _patch_and_revin(
        context_batch, patch_len, context_batch.device
    )
    n_patches = normed_inputs.shape[1]

    (_, _, normed_outputs, _), _ = model_module(normed_inputs, patched_masks)
    normed_outputs = normed_outputs.reshape(batch_size, n_patches, o_len, q_len)
    pred_norm = normed_outputs[:, :, :, aridx]  # (batch, n_patches, o_len)

    target_raw  = _build_patch_targets(context_batch, future_batch, patch_len, o_len, n_patches)
    mu_exp      = context_mu[:, :, None]
    sigma_exp   = context_sigma[:, :, None]
    target_norm = (target_raw - mu_exp) / (sigma_exp + EPS)

    valid_patch = ~patched_masks.all(dim=-1)             # (batch, n_patches)

    if supervise_last_n > 0:
        n_valid = valid_patch.sum(dim=1, keepdim=True)   # (batch, 1)
        patch_idx = torch.arange(n_patches, device=valid_patch.device).unsqueeze(0)
        valid_patch = valid_patch & (patch_idx >= (n_valid - supervise_last_n))

    h = min(supervise_horizon, o_len) if supervise_horizon > 0 else o_len
    pred_h   = pred_norm[:, :, :h]
    target_h = target_norm[:, :, :h]
    patch_mask = valid_patch[:, :, None].expand(-1, -1, h)

    abs_err  = torch.abs(pred_h - target_h)
    mae_loss = abs_err[patch_mask].mean()

    if dir_loss_weight > 0.0:
        pred_diff   = pred_h[:, :, 1:] - pred_h[:, :, :-1]
        target_diff = target_h[:, :, 1:] - target_h[:, :, :-1]
        dir_mask    = valid_patch[:, :, None].expand(-1, -1, h - 1)
        wrong_dir   = torch.relu(-pred_diff * target_diff)
        dir_loss    = wrong_dir[dir_mask].mean()
        return mae_loss + dir_loss_weight * dir_loss

    return mae_loss


def train_one_epoch(
    model_module: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    patch_len: int,
    grad_accum: int = 1,
    dir_loss_weight: float = 0.0,
    supervise_last_n: int = 0,
    supervise_horizon: int = 0,
) -> float:
    model_module.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, (context_batch, future_batch) in enumerate(dataloader):
        context_batch = context_batch.to(device)
        future_batch  = future_batch.to(device)
        loss = _compute_loss(model_module, context_batch, future_batch, patch_len,
                             dir_loss_weight, supervise_last_n, supervise_horizon)
        (loss / grad_accum).backward()
        total_loss += float(loss.item())
        n_batches  += 1
        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_module.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model_module: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    patch_len: int,
) -> float:
    model_module.eval()
    total_loss, n_batches = 0.0, 0
    for context_batch, future_batch in dataloader:
        context_batch = context_batch.to(device)
        future_batch  = future_batch.to(device)
        loss = _compute_loss(model_module, context_batch, future_batch, patch_len, dir_loss_weight=0.0)
        total_loss += float(loss.item())
        n_batches  += 1
    return total_loss / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest — raw prices in, raw prices out
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    model_wrapper,
    prices: np.ndarray,
    context_len: int,
    pred_len: int,
    test_fraction: float = 0.2,
) -> dict:
    """Rolling-window backtest on raw prices.

    model_wrapper.forecast() is called with raw prices directly.
    ForecastConfig.normalize_inputs=True means the model applies its own
    internal RevIN, matching what was done during training.
    """
    n          = len(prices)
    test_size  = int(n * test_fraction)
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
        ctx    = prices[ctx_start:window_start]
        actual = prices[window_start:window_end]

        point_forecast, _ = model_wrapper.forecast(horizon=pred_len, inputs=[ctx])
        forecast = point_forecast[0, :pred_len]

        w_mse = float(np.mean((forecast - actual) ** 2))
        w_mae = float(np.mean(np.abs(forecast - actual)))
        mse_total    += np.sum((forecast - actual) ** 2)
        mae_total    += np.sum(np.abs(forecast - actual))
        num_elements += len(actual)

        actual_dir = np.sign(np.diff(actual))
        pred_dir   = np.sign(np.diff(forecast))
        min_len    = min(len(actual_dir), len(pred_dir))
        dir_acc    = (
            float(np.mean(actual_dir[:min_len] == pred_dir[:min_len]))
            if min_len > 0 else 0.0
        )

        windows.append({
            "idx":      w,
            "start":    int(window_start),
            "end":      int(window_end),
            "actual":   actual,
            "forecast": forecast,
            "context":  prices[ctx_start:window_start],
            "mse":      w_mse,
            "mae":      w_mae,
            "dir_acc":  dir_acc,
        })

    return {
        "mse":         float(mse_total / max(num_elements, 1)),
        "mae":         float(mae_total / max(num_elements, 1)),
        "dir_acc":     float(np.mean([w["dir_acc"] for w in windows])) if windows else 0.0,
        "num_windows": len(windows),
        "windows":     windows,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════════

CHART_RC = {
    "figure.facecolor": C_BG, "axes.facecolor": C_CARD,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
    "text.color": C_TEXT, "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.grid": True, "grid.color": C_GRID, "grid.alpha": 0.4,
    "font.family": "sans-serif",
    "legend.facecolor": C_CARD, "legend.edgecolor": C_GRID,
}


def plot_training_curves(train_losses, val_losses, results_dir, ticker):
    if not train_losses or not val_losses:
        print("  Skipping training curves (no data)")
        return None
    plt.rcParams.update(CHART_RC)
    fig, ax = plt.subplots(figsize=(12, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=C_ACCENT, lw=2.2, label="Train MAE", alpha=0.9)
    ax.plot(epochs, val_losses,   color=C_ORANGE, lw=2.2, label="Val MAE",   alpha=0.9)
    best_ep = int(np.argmin(val_losses)) + 1
    ax.axvline(best_ep, color=C_GREEN, ls="--", lw=1.2, alpha=0.7, label=f"Best epoch ({best_ep})")
    ax.scatter([best_ep], [val_losses[best_ep - 1]], color=C_GREEN, s=80, zorder=5)
    ax.set_title(
        f"LoRA Training Curves — {ticker}", fontsize=16, fontweight="bold", color=C_TITLE, pad=12
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MAE (RevIN-normalised space)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    path = os.path.join(results_dir, f"{ticker.replace('-', '_').lower()}_training_curves.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_metrics_comparison(baseline, finetuned, stats, results_dir, ticker):
    plt.rcParams.update(CHART_RC)
    metrics   = ["MSE", "MAE", "Dir Accuracy"]
    base_vals = [baseline["mse"], baseline["mae"], baseline.get("dir_acc", 0)]
    ft_vals   = [finetuned["mse"], finetuned["mae"], finetuned.get("dir_acc", 0)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Baseline vs LoRA Fine-Tuned — {ticker}",
        fontsize=17, fontweight="bold", color=C_TITLE, y=1.02,
    )
    for i, (ax, name) in enumerate(zip(axes, metrics)):
        bv, fv = base_vals[i], ft_vals[i]
        bars = ax.bar(
            ["Zero-Shot", "LoRA"], [bv, fv],
            color=[C_RED, C_GREEN], width=0.5, edgecolor=C_GRID, linewidth=1.2, alpha=0.85,
        )
        for bar, val in zip(bars, [bv, fv]):
            fmt = f"{val:.4f}" if name != "Dir Accuracy" else f"{val:.1%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                fmt, ha="center", va="bottom", fontsize=12, fontweight="bold", color=C_TITLE,
            )
        ax.set_title(name, fontsize=14, fontweight="bold", color=C_TITLE, pad=8)
        ax.set_ylim(0, max(bv, fv) * 1.30)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)
        if name != "Dir Accuracy":
            pct = (bv - fv) / bv * 100 if bv != 0 else 0
            col = C_GREEN if pct > 0 else C_RED
            ax.text(
                0.5, 0.92, f"{'▼' if pct > 0 else '▲'} {abs(pct):.1f}%",
                transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col + "22", edgecolor=col, lw=1),
            )
        else:
            pct = (fv - bv) * 100
            col = C_GREEN if pct > 0 else C_RED
            ax.text(
                0.5, 0.92, f"{'▲' if pct > 0 else '▼'} {abs(pct):.1f}pp",
                transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col + "22", edgecolor=col, lw=1),
            )
    fig.tight_layout()
    path = os.path.join(results_dir, f"{ticker.replace('-', '_').lower()}_metrics_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_forecast_windows(baseline_bt, finetuned_bt, results_dir, ticker, n_windows=4):
    """Forecast comparison across representative test windows.

    Selects 4 windows that tell the full story:
      - Best Improvement: where LoRA helped the most vs zero-shot
      - Median Window:    typical/middle-of-the-road performance
      - Worst Window:     where LoRA hurt the most (honesty check)
      - Most Recent:      the latest test window (most relevant)

    Each card shows the trailing context, actual prices, and both
    forecasts so you can visually compare how each tracks the real data.
    """
    plt.rcParams.update(CHART_RC)
    b_wins = baseline_bt["windows"]
    f_wins = finetuned_bt["windows"]
    if not b_wins:
        return None

    n_wins = min(len(b_wins), len(f_wins))
    mse_diffs = [f_wins[i]["mse"] - b_wins[i]["mse"] for i in range(n_wins)]
    best_idx   = int(np.argmin(mse_diffs))
    worst_idx  = int(np.argmax(mse_diffs))
    sorted_idx = sorted(range(n_wins), key=lambda i: b_wins[i]["mse"])
    median_idx = sorted_idx[len(sorted_idx) // 2]
    last_idx   = n_wins - 1
    chosen = []
    for idx in [best_idx, median_idx, worst_idx, last_idx]:
        if idx not in chosen:
            chosen.append(idx)
    chosen = chosen[:n_windows]

    card_meta = {
        best_idx:   ("Best Improvement",
                     "Window where LoRA reduced error the most vs zero-shot"),
        median_idx: ("Median Window",
                     "Typical performance — middle of the pack by MSE"),
        worst_idx:  ("Worst Window",
                     "Window where LoRA performed worst relative to zero-shot"),
        last_idx:   ("Most Recent",
                     "Latest test window — closest to today's market"),
    }

    fig, axes = plt.subplots(len(chosen), 1, figsize=(18, 5.5 * len(chosen)),
                             gridspec_kw={"hspace": 0.45})
    if len(chosen) == 1:
        axes = [axes]

    for ax_i, wi in enumerate(chosen):
        ax = axes[ax_i]
        bw, fw   = b_wins[wi], f_wins[wi]
        pred_len = len(bw["actual"])
        ctx_show = min(70, len(bw["context"]))
        days_ctx  = list(range(-ctx_show, 0))
        days_pred = list(range(0, pred_len))

        ctx_prices = bw["context"][-ctx_show:]
        all_prices = list(ctx_prices) + list(bw["actual"])
        y_min, y_max = min(all_prices), max(all_prices)
        y_pad = (y_max - y_min) * 0.12
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.fill_between(days_ctx, ctx_prices, y_min - y_pad,
                        alpha=0.06, color=C_TEXT)
        ax.plot(days_ctx, ctx_prices,
                color=C_TEXT, lw=1.2, alpha=0.45, label="Historical context")
        ax.plot(days_pred, bw["actual"], color=C_TITLE, lw=2.5,
                label="Actual price", marker="o", ms=3.5, zorder=5)
        ax.plot(days_pred, bw["forecast"], color=C_RED, lw=2, ls="--",
                label="Zero-shot forecast", alpha=0.8, marker="s", ms=3)
        ax.plot(days_pred, fw["forecast"], color=C_GREEN, lw=2.2,
                label="LoRA forecast", alpha=0.9, marker="^", ms=3.5, zorder=4)

        ax.axvspan(-0.5, pred_len - 0.5, alpha=0.04, color=C_ACCENT)
        ax.axvline(0, color=C_ACCENT, ls=":", lw=1.2, alpha=0.5)
        ax.annotate("Forecast starts", xy=(0.3, 0.97), xycoords=("data", "axes fraction"),
                    fontsize=8, color=C_ACCENT, alpha=0.7, ha="left", va="top")

        mse_change = (bw["mse"] - fw["mse"]) / bw["mse"] * 100 if bw["mse"] > 0 else 0
        badge_color = C_GREEN if mse_change > 0 else C_RED
        badge_sign  = "+" if mse_change > 0 else ""
        badge_label = "LoRA better" if mse_change > 0 else "Zero-shot better"

        title_text, subtitle_text = card_meta.get(wi, (f"Window {wi + 1}", ""))
        ax.set_title(
            f"{title_text}  (Window {wi + 1}/{n_wins})",
            fontsize=15, fontweight="bold", color=C_TITLE, pad=14, loc="left",
        )
        ax.text(1.0, 1.06, f"{badge_sign}{mse_change:.1f}% MSE  {badge_label}",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                color=badge_color, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc=C_CARD, ec=badge_color, alpha=0.9))
        if subtitle_text:
            ax.text(0.0, 1.01, subtitle_text, transform=ax.transAxes,
                    fontsize=9, color=C_TEXT, alpha=0.6, ha="left", va="bottom")

        ax.set_xlabel("Days relative to forecast start", fontsize=10, labelpad=6)
        ax.set_ylabel("Price (USD)", fontsize=10, labelpad=6)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85, borderpad=0.8,
                  handlelength=2.5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.tick_params(axis="both", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)

    fig.suptitle(f"Forecast Comparison — {ticker}",
                 fontsize=20, fontweight="bold", color=C_TITLE, y=1.02)
    fig.text(0.5, 1.005,
             f"Rolling {pred_len}-day forecasts on held-out test data  ·  "
             f"{n_wins} total windows  ·  4 shown below",
             ha="center", fontsize=10, color=C_TEXT, alpha=0.5)

    path = os.path.join(results_dir, f"{ticker.replace('-', '_').lower()}_forecast_windows.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C_BG, pad_inches=0.4)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_error_analysis(baseline_bt, finetuned_bt, results_dir, ticker):
    plt.rcParams.update(CHART_RC)
    b_wins = baseline_bt["windows"]
    f_wins = finetuned_bt["windows"]
    if not b_wins:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    x, w = np.arange(len(b_wins)), 0.35
    axes[0].bar(x - w / 2, [ww["mse"] for ww in b_wins], w, color=C_RED,   alpha=0.8, label="Zero-Shot", edgecolor=C_GRID)
    axes[0].bar(x + w / 2, [ww["mse"] for ww in f_wins], w, color=C_GREEN, alpha=0.8, label="LoRA",       edgecolor=C_GRID)
    axes[0].set_xlabel("Test Window", fontsize=12)
    axes[0].set_ylabel("MSE (USD²)", fontsize=12)
    axes[0].set_title("Per-Window MSE", fontsize=14, fontweight="bold", color=C_TITLE, pad=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"W{i+1}" for i in x], fontsize=9)
    axes[0].legend(fontsize=11)

    b_errors = np.concatenate([ww["forecast"] - ww["actual"] for ww in b_wins])
    f_errors = np.concatenate([ww["forecast"] - ww["actual"] for ww in f_wins])
    vmin = min(b_errors.min(), f_errors.min())
    vmax = max(b_errors.max(), f_errors.max())
    bins = np.linspace(vmin, vmax, 40)
    axes[1].hist(b_errors, bins, color=C_RED,   alpha=0.55, label="Zero-Shot Errors", edgecolor=C_GRID)
    axes[1].hist(f_errors, bins, color=C_GREEN, alpha=0.55, label="LoRA Errors",      edgecolor=C_GRID)
    axes[1].axvline(0, color=C_TITLE, ls="--", lw=1.2, alpha=0.7)
    axes[1].set_xlabel("Forecast Error (USD)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Error Distribution", fontsize=14, fontweight="bold", color=C_TITLE, pad=8)
    axes[1].legend(fontsize=11)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(C_GRID)

    fig.suptitle(f"Error Analysis — {ticker}", fontsize=17, fontweight="bold", color=C_TITLE, y=1.02)
    fig.tight_layout()
    path = os.path.join(results_dir, f"{ticker.replace('-', '_').lower()}_error_analysis.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune TimesFM 2.5 on crypto")
    parser.add_argument("--ticker", type=str, nargs="+", default=["BTC-USD"],
                        help="One or more yfinance tickers; first is used for backtest.")
    parser.add_argument("--interval",       type=str,   default="1d",         choices=["1h", "1d"])
    parser.add_argument("--context_len",    type=int,   default=512)
    parser.add_argument("--pred_len",       type=int,   default=30,
                        help="Horizon used for backtest evaluation.")
    parser.add_argument("--lora_rank",      type=int,   default=8)
    parser.add_argument("--lora_alpha",     type=float, default=16.0)
    parser.add_argument("--lora_dropout",   type=float, default=0.05)
    parser.add_argument("--target_modules", type=str,   default="attention",   choices=["all", "attention", "ffn"])
    parser.add_argument("--lr",             type=float, default=3e-5)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--stride",         type=int,   default=16)
    parser.add_argument("--patience",       type=int,   default=10,
                        help="Early-stop patience in epochs.")
    parser.add_argument("--warmup_epochs", type=int,   default=0,
                        help="Number of linear LR warmup epochs.")
    parser.add_argument("--grad_accum",    type=int,   default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--dir_loss_weight", type=float, default=0.0,
                        help="Weight for directional loss component (0=disabled).")
    parser.add_argument("--supervise_last_n", type=int, default=0,
                        help="Only supervise last N patches (0=all).")
    parser.add_argument("--supervise_horizon", type=int, default=0,
                        help="Only supervise first H steps per patch (0=full o_len).")
    parser.add_argument("--test_fraction",  type=float, default=0.2)
    parser.add_argument("--results_dir",    type=str,   default="./results/lora")
    parser.add_argument("--load-only",      action="store_true",
                        help="Skip training; load saved weights and run backtest only.")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load model once ───────────────────────────────────────────────────────
    print("\nLoading TimesFM 2.5 200M...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model_module = model.model
    o_len = model_module.o   # native output patch length = 128

    # ── Fetch data and build datasets ─────────────────────────────────────────
    period = "max" if args.interval == "1d" else "730d"
    train_sets, val_sets = [], []
    primary_prices = None
    primary_ticker = args.ticker[0]

    for ticker in args.ticker:
        print(f"\nFetching {ticker} ({args.interval})...")
        data = yf.download(ticker, period=period, interval=args.interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        prices = data["Close"].dropna().values.astype(np.float64)
        print(f"  {len(prices):,} candles fetched")

        if primary_prices is None:
            primary_prices = prices

        n         = len(prices)
        test_size = int(n * args.test_fraction)
        val_size  = int(n * 0.15)
        train_size = n - test_size - val_size

        train_prices = prices[:train_size]
        # Prepend enough context so the first val window is fully covered.
        val_start  = max(0, train_size - args.context_len)
        val_prices = prices[val_start : train_size + val_size]

        # Dataset target length = o_len (128) so every output head is supervised.
        train_sets.append(
            CryptoTimeSeriesDataset(train_prices, args.context_len, o_len, stride=args.stride)
        )
        val_sets.append(
            CryptoTimeSeriesDataset(val_prices, args.context_len, o_len, stride=o_len)
        )
        print(f"  Train windows: {len(train_sets[-1]):,}  Val windows: {len(val_sets[-1]):,}")

    train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_ds   = val_sets[0]   if len(val_sets)   == 1 else ConcatDataset(val_sets)

    mps_mode  = device.type == "mps"
    dl_kwargs = dict(num_workers=0, pin_memory=False) if mps_mode else dict(num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False, **dl_kwargs)

    # ── ForecastConfig — normalize_inputs=True feeds raw prices to the model's
    #    built-in RevIN path, matching how the training loop normalises.
    fc = timesfm.ForecastConfig(
        max_context=args.context_len,
        max_horizon=args.pred_len,
        normalize_inputs=True,
        use_continuous_quantile_head=False,
        force_flip_invariance=True,
    )

    # ── Baseline backtest (zero-shot) ─────────────────────────────────────────
    print(f"\n── Baseline Backtest (zero-shot) on {primary_ticker} ──")
    model.compile(fc)
    baseline_bt = run_backtest(model, primary_prices, args.context_len, args.pred_len, args.test_fraction)
    print(
        f"  MSE: {baseline_bt['mse']:.6f}  MAE: {baseline_bt['mae']:.6f}  "
        f"Dir Acc: {baseline_bt['dir_acc']:.1%}  Windows: {baseline_bt['num_windows']}"
    )

    # ── Inject LoRA ───────────────────────────────────────────────────────────
    print(
        f"\n── Injecting LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"modules={args.target_modules}, dropout={args.lora_dropout}) ──"
    )
    model_module = model_module.to("cpu")
    stats = inject_lora(
        model_module,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=args.target_modules,
        dropout=args.lora_dropout,
    )
    model_module = model_module.to(device)
    model_module.device = device
    model.model = model_module
    print(
        f"  Injected {stats['injected_layers']} layers | "
        f"LoRA params: {stats['lora_params']:,} / {stats['base_params']:,} "
        f"({stats['pct_trainable']:.3f}%)"
    )

    # ── Training ──────────────────────────────────────────────────────────────
    patch_len = model_module.p  # 32
    os.makedirs(args.results_dir, exist_ok=True)
    lora_path = os.path.join(
        args.results_dir, f"{primary_ticker.replace('-', '_').lower()}_lora.pt"
    )
    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_epoch = 0
    total_time = 0.0

    if args.load_only:
        print(f"\n── Skipping training; loading weights from {lora_path} ──")
        load_lora_weights(model_module, lora_path)
    else:
        trainable = [p for p in model_module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

        if args.warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr * 0.01
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
            )
        best_val   = float("inf")
        no_improve = 0
        t_total    = time.time()

        print(f"\n── Training ({args.epochs} epochs, patience={args.patience}) ──")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tl = train_one_epoch(model_module, train_loader, optimizer, device, patch_len,
                                    grad_accum=args.grad_accum, dir_loss_weight=args.dir_loss_weight,
                                    supervise_last_n=args.supervise_last_n,
                                    supervise_horizon=args.supervise_horizon)
            vl = validate(model_module, val_loader, device, patch_len)
            scheduler.step()
            train_losses.append(tl)
            val_losses.append(vl)

            improved_str = ""
            if vl < best_val:
                best_val, best_epoch = vl, epoch
                no_improve = 0
                save_lora_weights(model_module, lora_path)
                improved_str = " ★"
            else:
                no_improve += 1

            print(
                f"  Epoch {epoch:3d}/{args.epochs}  "
                f"train={tl:.6f}  val={vl:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"{time.time() - t0:.1f}s{improved_str}"
            )

            if no_improve >= args.patience:
                print(f"\n  Early stop after {args.patience} epochs without val improvement.")
                break

        total_time = time.time() - t_total
        print(f"\n  Best val: {best_val:.6f} @ epoch {best_epoch}  |  Total: {total_time:.0f}s")
        load_lora_weights(model_module, lora_path)

    # ── Fine-tuned backtest ───────────────────────────────────────────────────
    print(f"\n── Fine-Tuned Backtest on {primary_ticker} ──")
    model.compile(fc)
    finetuned_bt = run_backtest(model, primary_prices, args.context_len, args.pred_len, args.test_fraction)
    print(
        f"  MSE: {finetuned_bt['mse']:.6f}  MAE: {finetuned_bt['mae']:.6f}  "
        f"Dir Acc: {finetuned_bt['dir_acc']:.1%}"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    mse_imp = (baseline_bt["mse"] - finetuned_bt["mse"]) / baseline_bt["mse"] * 100
    mae_imp = (baseline_bt["mae"] - finetuned_bt["mae"]) / baseline_bt["mae"] * 100
    dir_imp = (finetuned_bt["dir_acc"] - baseline_bt["dir_acc"]) * 100

    print(f"\n{'=' * 64}")
    print(f"  RESULTS — {primary_ticker}  (LoRA rank {args.lora_rank}, {args.target_modules})")
    print(f"{'=' * 64}")
    print(f"  {'Metric':<16} {'Zero-Shot':>12} {'LoRA':>12} {'Change':>12}")
    print(f"  {'-' * 52}")
    print(f"  {'MSE':<16} {baseline_bt['mse']:>12.6f} {finetuned_bt['mse']:>12.6f} {mse_imp:>+11.1f}%")
    print(f"  {'MAE':<16} {baseline_bt['mae']:>12.6f} {finetuned_bt['mae']:>12.6f} {mae_imp:>+11.1f}%")
    print(f"  {'Dir Accuracy':<16} {baseline_bt['dir_acc']:>11.1%} {finetuned_bt['dir_acc']:>11.1%} {dir_imp:>+10.1f}pp")
    print(
        f"  Trainable: {stats['pct_trainable']:.3f}% ({stats['lora_params']:,} params)  |  "
        f"Time: {total_time:.0f}s"
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\n── Generating Charts ──")
    chart_paths = [
        plot_training_curves(train_losses, val_losses, args.results_dir, primary_ticker),
        plot_metrics_comparison(baseline_bt, finetuned_bt, stats, args.results_dir, primary_ticker),
        plot_forecast_windows(baseline_bt, finetuned_bt, args.results_dir, primary_ticker),
        plot_error_analysis(baseline_bt, finetuned_bt, args.results_dir, primary_ticker),
    ]

    # ── Persist results ───────────────────────────────────────────────────────
    results = {
        "ticker":           primary_ticker,
        "tickers_trained":  args.ticker,
        "lora_rank":        args.lora_rank,
        "lora_alpha":       args.lora_alpha,
        "target_modules":   args.target_modules,
        "lr":               args.lr,
        "epochs_configured": args.epochs,
        "epochs_run":       len(train_losses),
        "best_epoch":       best_epoch,
        "patience":         args.patience,
        "lora_params":      stats["lora_params"],
        "pct_trainable":    stats["pct_trainable"],
        "training_time_s":  total_time,
        "baseline":  {"mse": baseline_bt["mse"],  "mae": baseline_bt["mae"],  "dir_acc": baseline_bt["dir_acc"]},
        "finetuned": {"mse": finetuned_bt["mse"], "mae": finetuned_bt["mae"], "dir_acc": finetuned_bt["dir_acc"]},
        "mse_improvement_pct":    mse_imp,
        "mae_improvement_pct":    mae_imp,
        "dir_acc_improvement_pp": dir_imp,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "chart_paths":  [p for p in chart_paths if p],
    }
    rp = os.path.join(args.results_dir, "finetune_results.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results JSON : {rp}")
    print(f"  LoRA weights : {lora_path}")
    print(f"  Charts       : {args.results_dir}/")
    print("\n  Done! ✓")


if __name__ == "__main__":
    main()
