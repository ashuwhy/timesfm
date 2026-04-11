#!/usr/bin/env python3
"""
Generate two presentation-quality slide figures:
  Slide 1 — LoRA: Aggregate Metrics & Key Results
  Slide 2 — ICF:  Aggregate Metrics & Key Results (4-way RoPE + multi-asset)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT      = Path(__file__).parent.parent
RESULTS   = ROOT / "results"
OUT_DIR   = RESULTS / "slides"
OUT_DIR.mkdir(exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
BLUE    = "#1f77b4"
GREEN   = "#2ca02c"
RED     = "#d62728"
ORANGE  = "#ff7f0e"
GREY    = "#7f7f7f"
LIGHT   = "#f5f5f5"

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#ccc', 'grid.color': '#e0e0e0',
    'text.color': '#222', 'font.family': 'sans-serif',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
})

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1  LoRA
# ─────────────────────────────────────────────────────────────────────────────
def slide_lora():
    # Data
    v7     = json.loads((RESULTS / "lora_v7" / "finetune_results.json").read_text())
    final  = json.loads((RESULTS / "lora_final" / "finetune_results.json").read_text())

    baseline_mse = v7["baseline"]["mse"] / 1e6
    lora_mse     = v7["finetuned"]["mse"] / 1e6
    baseline_mae = v7["baseline"]["mae"]
    lora_mae     = v7["finetuned"]["mae"]
    baseline_dir = v7["baseline"]["dir_acc"] * 100
    lora_dir     = v7["finetuned"]["dir_acc"] * 100
    mse_imp      = v7["mse_improvement_pct"]
    mae_imp      = v7["mae_improvement_pct"]

    train_losses = final["train_losses"]
    val_losses   = final["val_losses"]
    best_epoch   = final["best_epoch"]

    fig = plt.figure(figsize=(16, 7), facecolor='white')
    fig.suptitle("LoRA Fine-Tuning — Aggregate Metrics & Key Results  (BTC-USD)",
                 fontsize=16, fontweight='bold', y=0.98, color='#111')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.88, bottom=0.1)

    # ── 1. Training curves ────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[:, 0])
    epochs = range(1, len(train_losses) + 1)
    ax0.plot(epochs, train_losses, color=BLUE,  linewidth=2.2, marker='o', markersize=4, label='Train Loss')
    ax0.plot(epochs, val_losses,   color=ORANGE, linewidth=2.2, marker='s', markersize=4, label='Val Loss')
    ax0.axvline(best_epoch, color=GREEN, linestyle='--', linewidth=1.5, label=f'Best epoch ({best_epoch})')
    ax0.fill_between(epochs, train_losses, val_losses,
                     where=[v < t for t, v in zip(train_losses, val_losses)],
                     alpha=0.08, color=ORANGE, label='Overfit gap')
    ax0.set_xlabel('Epoch'); ax0.set_ylabel('L1 Loss (normalised)')
    ax0.set_title('Training Convergence', fontweight='bold')
    ax0.legend(fontsize=9, frameon=True, edgecolor='#ccc')
    ax0.grid(True, alpha=0.4)

    # ── 2. MSE & MAE grouped bars ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.array([0, 1])
    bars = ax1.bar(x - 0.18, [baseline_mse, baseline_mae / 1000],
                   0.32, label='Baseline', color=GREY,  alpha=0.85, edgecolor='#555')
    bars2 = ax1.bar(x + 0.18, [lora_mse,     lora_mae / 1000],
                    0.32, label='LoRA',     color=BLUE,  alpha=0.85, edgecolor='#555')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['MSE (M)', 'MAE (÷1k)'])
    ax1.set_title('Metric Comparison', fontweight='bold')
    ax1.legend(fontsize=9, frameon=True, edgecolor='#ccc')
    ax1.grid(axis='y', alpha=0.4)
    for bar, val, imp in zip(bars2, [lora_mse, lora_mae / 1000], [mse_imp, mae_imp]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'−{imp:.1f}%', ha='center', va='bottom', fontsize=9,
                 color=GREEN, fontweight='bold')

    # ── 3. Directional accuracy ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    cats = ['Baseline', 'LoRA']
    vals = [baseline_dir, lora_dir]
    colors = [GREY, BLUE]
    b = ax2.bar(cats, vals, color=colors, alpha=0.85, edgecolor='#555', width=0.45)
    ax2.axhline(50, color='#999', linestyle='--', linewidth=1, label='Random (50%)')
    ax2.set_ylim(48, 54)
    ax2.set_ylabel('Directional Accuracy (%)')
    ax2.set_title('Directional Accuracy', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.4)
    for bar, val in zip(b, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ── 4. Key stats callout ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.axis('off')
    stats = [
        ("LoRA Rank",        f"r = {v7['lora_rank']}"),
        ("LoRA Alpha",       f"α = {int(v7['lora_alpha'])}"),
        ("Target Modules",   v7['target_modules'].capitalize()),
        ("Trainable Params", f"{v7['lora_params'] / 1e6:.2f}M"),
        ("% of Total",       f"{v7['pct_trainable']:.2f}%"),
        ("",                 ""),
        ("MSE Improvement",  f"↓ {mse_imp:.1f}%"),
        ("MAE Improvement",  f"↓ {mae_imp:.1f}%"),
        ("Dir. Acc Δ",       f"{v7['dir_acc_improvement_pp']:+.2f}pp"),
    ]
    y0 = 0.95
    ax3.text(0.5, y0, "Key Parameters", transform=ax3.transAxes,
             fontsize=13, fontweight='bold', ha='center', va='top', color='#111')
    for i, (k, v) in enumerate(stats):
        y = y0 - 0.10 - i * 0.092
        if k == "":
            ax3.plot([0.05, 0.95], [y + 0.04, y + 0.04], color='#ddd',
                     linewidth=1, transform=ax3.transAxes)
            continue
        color = GREEN if k in ("MSE Improvement", "MAE Improvement") else \
                RED   if k == "Dir. Acc Δ"                           else '#444'
        ax3.text(0.08, y, k + ":", transform=ax3.transAxes,
                 fontsize=10, color='#555', va='top')
        ax3.text(0.92, y, v, transform=ax3.transAxes,
                 fontsize=11, color=color, fontweight='bold', ha='right', va='top')

    # border around callout
    fancy = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                           boxstyle="round,pad=0.02", linewidth=1.5,
                           edgecolor='#bbb', facecolor='#f9f9f9',
                           transform=ax3.transAxes, zorder=0)
    ax3.add_patch(fancy)

    out = OUT_DIR / "slide_lora.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2  ICF
# ─────────────────────────────────────────────────────────────────────────────
def slide_icf():
    abl  = json.loads((RESULTS / "rope_ablation_full" / "results.json").read_text())
    ma   = json.loads((RESULTS / "multi_asset" / "results.json").read_text())

    # 4-way ablation data
    conds  = ["ICF Base (noPE)", "ICF Base (RoPE)", "ICF Trained (noPE)", "ICF Trained (RoPE)"]
    colors = [RED, GREEN, ORANGE, BLUE]
    mses   = [abl[c]["mse"] / 1e6   for c in conds]
    maes   = [abl[c]["mae"] / 1e3   for c in conds]
    imps   = [abl[c]["vs_base_nope_mse_pct"] for c in conds]
    short  = ["Base\nnoPE", "Base\nRoPE", "Trained\nnoPE", "Trained\nRoPE"]

    # multi-asset nMAE
    assets = list(ma.keys())
    asset_labels = [a.replace("/", "\n") for a in assets]

    fig = plt.figure(figsize=(18, 8), facecolor='white')
    fig.suptitle("ICF Fine-Tuning — Aggregate Metrics & Key Results",
                 fontsize=16, fontweight='bold', y=0.98, color='#111')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                           left=0.06, right=0.97, top=0.88, bottom=0.12)

    # ── 1. 4-way MSE bar ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(4)
    bars = ax0.bar(x, mses, color=colors, alpha=0.85, edgecolor='#555', width=0.6)
    ax0.set_xticks(x); ax0.set_xticklabels(short, fontsize=9)
    ax0.set_ylabel('MSE (millions)')
    ax0.set_title('BTC-USD MSE — 4-way Ablation', fontweight='bold')
    ax0.grid(axis='y', alpha=0.4)
    for bar, val, imp in zip(bars, mses, imps):
        label = f'{val:.0f}M' if imp == 0 else f'{val:.0f}M\n(−{imp:.0f}%)'
        col   = '#222' if imp == 0 else GREEN
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 label, ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')

    # ── 2. 4-way MAE bar ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    bars2 = ax1.bar(x, maes, color=colors, alpha=0.85, edgecolor='#555', width=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(short, fontsize=9)
    ax1.set_ylabel('MAE (thousands USD)')
    ax1.set_title('BTC-USD MAE — 4-way Ablation', fontweight='bold')
    ax1.grid(axis='y', alpha=0.4)
    for bar, val in zip(bars2, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}k', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ── 3. Multi-asset nMAE grouped bar ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[:, 1:])
    n_a   = len(assets)
    n_c   = len(conds)
    w     = 0.18
    xa    = np.arange(n_a)
    for ci, (cond, col) in enumerate(zip(conds, colors)):
        vals = [ma[a][cond]["nmae"] for a in assets]
        offset = (ci - (n_c - 1) / 2) * w
        bars = ax2.bar(xa + offset, vals, w, label=cond.replace("ICF ", ""),
                       color=col, alpha=0.82, edgecolor='#555', linewidth=0.7)

    # mark winner per asset with star
    for ai, asset in enumerate(assets):
        best_cond = min(conds, key=lambda c: ma[asset][c]["nmae"])
        best_val  = ma[asset][best_cond]["nmae"]
        ci        = conds.index(best_cond)
        offset    = (ci - (n_c - 1) / 2) * w
        ax2.text(xa[ai] + offset, best_val + 0.001, '★', ha='center',
                 va='bottom', fontsize=11, color='#FFD700')

    ax2.set_xticks(xa)
    ax2.set_xticklabels(asset_labels, fontsize=10)
    ax2.set_ylabel('Normalised MAE  (MAE / mean|actual|)', fontweight='bold')
    ax2.set_title('Cross-Asset Generalisation — nMAE  (★ = best per asset, lower = better)',
                  fontweight='bold')
    ax2.legend(fontsize=9, frameon=True, edgecolor='#ccc', ncol=2,
               loc='upper right')
    ax2.grid(axis='y', alpha=0.4)

    out = OUT_DIR / "slide_icf.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating slide figures...")
    slide_lora()
    slide_icf()
    print("Done.")
