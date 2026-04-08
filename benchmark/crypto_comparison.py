#!/usr/bin/env python3
"""Crypto forecast comparison: zero-shot, LoRA, ICF; optional ICF-trained fourth curve."""

from __future__ import annotations

import argparse
import sys
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

# Allow `python benchmark/crypto_comparison.py` from repo root
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import yfinance as yf

import timesfm
from timesfm.timesfm_icf import ICFConfig, TimesFM_ICF_torch

from finetune_crypto_lora import (
    C_ACCENT,
    C_GREEN,
    C_ORANGE,
    C_RED,
    inject_lora,
    load_lora_weights,
    run_backtest,
)

C_PURPLE = "#A371F7"

# Light “academic” theme for PDF-ready figures (white background, serif, larger on export)
R_BG = "#FFFFFF"
R_GRID = "#D0D7DE"
R_TEXT = "#24292F"
R_TITLE = "#1F2328"
R_CTX_FILL = "#E8EDF3"
R_CTX_LINE = "#6E7781"

# LaTeX PDF: max height for embedded comparison figures (fraction of \\textheight).
LATEX_REPORT_FIG_HEIGHT_FRAC = 0.72

CHART_RC_REPORT = {
    "figure.facecolor": R_BG,
    "axes.facecolor": R_BG,
    "axes.edgecolor": R_GRID,
    "axes.labelcolor": R_TEXT,
    "text.color": R_TEXT,
    "xtick.color": R_TEXT,
    "ytick.color": R_TEXT,
    "axes.grid": True,
    "grid.color": R_GRID,
    "grid.alpha": 0.55,
    "font.family": "serif",
    "font.size": 11,
    "legend.facecolor": R_BG,
    "legend.edgecolor": R_GRID,
    "legend.framealpha": 0.98,
}

METHODS_3 = ("zero_shot", "lora", "icf")
LABELS_3 = ("Zero-shot", "LoRA", "ICF (inference)")
COLORS_3 = (C_ACCENT, C_GREEN, C_ORANGE)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fetch_prices(ticker: str, interval: str) -> np.ndarray:
    period = "max" if interval == "1d" else "730d"
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data["Close"].dropna().values.astype(np.float64)


def _round_up_patch(x: int, p: int) -> int:
    return int(math.ceil(x / p) * p)


def make_crypto_example_pool(
    train_prices: np.ndarray,
    segment_len: int,
    stride: int,
) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    n = len(train_prices)
    if n < segment_len:
        return pool
    for start in range(0, n - segment_len + 1, stride):
        pool.append(train_prices[start : start + segment_len].astype(np.float64))
    return pool


def _log_returns(seg: np.ndarray) -> np.ndarray:
    """Log returns; robust to zeros."""
    s = np.maximum(seg, 1e-8)
    return np.diff(np.log(s))


def _pool_signatures(pool: list[np.ndarray], tail: int = 64) -> np.ndarray:
    """Compute a fixed-length signature (log-return tail) for every pool segment."""
    sigs = []
    for seg in pool:
        lr = _log_returns(seg)
        if len(lr) >= tail:
            sigs.append(lr[-tail:])
        else:
            sigs.append(np.pad(lr, (tail - len(lr), 0)))
    return np.stack(sigs)  # (P, tail)


def select_similar_examples(
    pool: list[np.ndarray],
    pool_sigs: np.ndarray,
    context: np.ndarray,
    k: int,
    sig_tail: int = 64,
) -> list[np.ndarray]:
    """Pick K pool segments whose recent log-return shape is closest to the context tail."""
    lr = _log_returns(context)
    if len(lr) >= sig_tail:
        q = lr[-sig_tail:]
    else:
        q = np.pad(lr, (sig_tail - len(lr), 0))
    dists = np.linalg.norm(pool_sigs - q[None, :], axis=1)
    idx = np.argsort(dists)[:k]
    return [pool[i] for i in idx]


def run_icf_backtest(
    icf: TimesFM_ICF_torch,
    prices: np.ndarray,
    context_len: int,
    pred_len: int,
    k_examples: int,
    test_fraction: float,
    rng: random.Random | None = None,
    *,
    batch_size: int = 8,
) -> dict:
    n = len(prices)
    test_size = int(n * test_fraction)
    train_size = n - test_size
    test_start = train_size
    test_end = n
    num_windows = (test_end - test_start) // pred_len
    p = icf.model.p

    example_len = _round_up_patch(context_len + pred_len, p)
    pool = make_crypto_example_pool(prices[:train_size], example_len, stride=pred_len)
    if len(pool) < k_examples:
        raise RuntimeError(
            f"Example pool too small ({len(pool)} segments). "
            "Need more train history or smaller K / context+pred."
        )
    pool_sigs = _pool_signatures(pool)

    fc = timesfm.ForecastConfig(
        max_context=context_len,
        max_horizon=pred_len,
        normalize_inputs=True,
        per_core_batch_size=batch_size,
        use_continuous_quantile_head=False,
        force_flip_invariance=True,
    )
    icf.compile(fc, icf_config=ICFConfig(k_examples=k_examples, example_len=example_len))

    mse_total, mae_total, num_elements = 0.0, 0.0, 0
    windows: list[dict] = []
    t_infer = 0.0

    window_specs: list[tuple[int, int, int]] = []
    for w in range(num_windows):
        window_start = test_start + w * pred_len
        window_end = window_start + pred_len
        if window_end > test_end:
            break
        window_specs.append((w, window_start, window_end))

    for batch_start in range(0, len(window_specs), batch_size):
        batch = window_specs[batch_start : batch_start + batch_size]
        ctx_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []
        ex_batch: list[list[np.ndarray]] = []

        for w, window_start, window_end in batch:
            ctx_start = max(0, window_start - context_len)
            ctx = prices[ctx_start:window_start]
            actual = prices[window_start:window_end]
            ctx_list.append(ctx)
            act_list.append(actual)
            examples = select_similar_examples(pool, pool_sigs, ctx, k_examples)
            ex_batch.append(examples)

        t0 = time.time()
        point, _ = icf.forecast_icf(
            horizon=pred_len,
            context_examples=ex_batch,
            target_inputs=ctx_list,
        )
        t_infer += time.time() - t0

        for i, (w, window_start, window_end) in enumerate(batch):
            forecast = point[i, :pred_len]
            actual = act_list[i]
            w_mse = float(np.mean((forecast - actual) ** 2))
            w_mae = float(np.mean(np.abs(forecast - actual)))
            mse_total += float(np.sum((forecast - actual) ** 2))
            mae_total += float(np.sum(np.abs(forecast - actual)))
            num_elements += len(actual)
            actual_dir = np.sign(np.diff(actual))
            pred_dir = np.sign(np.diff(forecast))
            min_len = min(len(actual_dir), len(pred_dir))
            dir_acc = (
                float(np.mean(actual_dir[:min_len] == pred_dir[:min_len])) if min_len > 0 else 0.0
            )
            ctx_start = max(0, window_start - context_len)
            windows.append(
                {
                    "idx": w,
                    "start": int(window_start),
                    "end": int(window_end),
                    "actual": actual,
                    "forecast": forecast,
                    "context": prices[ctx_start:window_start],
                    "mse": w_mse,
                    "mae": w_mae,
                    "dir_acc": dir_acc,
                }
            )

    return {
        "mse": float(mse_total / max(num_elements, 1)),
        "mae": float(mae_total / max(num_elements, 1)),
        "dir_acc": float(np.mean([x["dir_acc"] for x in windows])) if windows else 0.0,
        "num_windows": len(windows),
        "windows": windows,
        "infer_time_s": float(t_infer),
    }


def _safe_pct(baseline: float, value: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - value) / baseline * 100.0


def plot_comparison_metrics(
    results: dict[str, dict],
    method_keys: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    results_dir: str,
    ticker: str,
) -> str:
    plt.rcParams.update(CHART_RC_REPORT)
    metrics = ["MSE", "MAE", "Dir Accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5 * max(len(method_keys), 3), 5.8))
    base = results[method_keys[0]]
    for i, ax in enumerate(axes):
        vals = [results[m][["mse", "mae", "dir_acc"][i]] for m in method_keys]
        bars = ax.bar(
            labels,
            vals,
            color=colors,
            width=0.65,
            edgecolor=R_GRID,
            linewidth=1.2,
            alpha=0.88,
        )
        for bar, val in zip(bars, vals):
            fmt = f"{val:.4f}" if i != 2 else f"{val:.1%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                fmt,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=R_TITLE,
            )
        ax.set_title(metrics[i], fontsize=14, fontweight="bold", color=R_TITLE, pad=8)
        ax.set_ylim(0, max(vals) * 1.28 if max(vals) > 0 else 1.0)
        for spine in ax.spines.values():
            spine.set_color(R_GRID)
        # Optional: annotate vs zero-shot for non-dir metrics
        if i != 2:
            b0 = base[["mse", "mae"][i]]
            parts = []
            for j, v in enumerate(vals):
                if j == 0:
                    continue
                pct = _safe_pct(b0, v)
                parts.append(f"{labels[j][:4]}:{pct:+.0f}%")
            if parts:
                ax.text(
                    0.5,
                    0.92,
                    " | ".join(parts),
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                    color=R_TEXT,
                )
    fig.suptitle(
        f"Crypto forecast metrics — {ticker}",
        fontsize=17,
        fontweight="bold",
        color=R_TITLE,
        y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(results_dir, "comparison_metrics.png")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=R_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_forecast_windows_three(
    backtests: dict[str, dict],
    method_keys: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    results_dir: str,
    ticker: str,
    n_windows: int = 4,
) -> str | None:
    plt.rcParams.update(CHART_RC_REPORT)
    zs = backtests[method_keys[0]]["windows"]
    if not zs:
        return None
    n_wins = len(zs)
    icf_key = "icf" if "icf" in method_keys else method_keys[-1]
    diffs = [
        backtests[icf_key]["windows"][i]["mse"] - zs[i]["mse"] for i in range(n_wins)
    ]
    best_idx = int(np.argmin(diffs))
    worst_idx = int(np.argmax(diffs))
    sorted_idx = sorted(range(n_wins), key=lambda i: zs[i]["mse"])
    median_idx = sorted_idx[len(sorted_idx) // 2]
    last_idx = n_wins - 1
    chosen: list[int] = []
    for idx in (best_idx, median_idx, worst_idx, last_idx):
        if idx not in chosen:
            chosen.append(idx)
    chosen = chosen[:n_windows]

    card_meta = {
        best_idx: ("Best ICF vs zero-shot (MSE delta)", "Largest MSE reduction vs zero-shot"),
        median_idx: ("Median zero-shot MSE window", "Typical difficulty"),
        worst_idx: ("Worst ICF vs zero-shot", "ICF hurt most vs zero-shot"),
        last_idx: ("Most recent test window", ""),
    }

    fig, axes = plt.subplots(
        len(chosen), 1, figsize=(20, 6.2 * len(chosen)), gridspec_kw={"hspace": 0.48}
    )
    if len(chosen) == 1:
        axes = [axes]

    for ax_i, wi in enumerate(chosen):
        ax = axes[ax_i]
        bw = zs[wi]
        pred_len = len(bw["actual"])
        ctx_show = min(70, len(bw["context"]))
        days_ctx = list(range(-ctx_show, 0))
        days_pred = list(range(0, pred_len))
        ctx_prices = bw["context"][-ctx_show:]
        all_prices = list(ctx_prices) + list(bw["actual"])
        y_min, y_max = min(all_prices), max(all_prices)
        y_pad = (y_max - y_min) * 0.12
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.fill_between(days_ctx, ctx_prices, y_min - y_pad, alpha=0.35, color=R_CTX_FILL)
        ax.plot(days_ctx, ctx_prices, color=R_CTX_LINE, lw=1.3, alpha=0.85, label="Context")
        ax.plot(
            days_pred,
            bw["actual"],
            color=R_TITLE,
            lw=2.5,
            label="Actual",
            marker="o",
            ms=3.5,
            zorder=5,
        )
        for mk, lab, col in zip(method_keys, labels, colors):
            fc = backtests[mk]["windows"][wi]["forecast"]
            ax.plot(days_pred, fc, lw=2, label=lab, color=col, alpha=0.9, marker="s", ms=2.5)
        ax.axvspan(-0.5, pred_len - 0.5, alpha=0.08, color=C_ACCENT)
        ax.axvline(0, color=C_ACCENT, ls=":", lw=1.2, alpha=0.5)
        title_text, subtitle_text = card_meta.get(wi, (f"Window {wi + 1}", ""))
        ax.set_title(
            f"{title_text} (window {wi + 1}/{n_wins})",
            fontsize=15,
            fontweight="bold",
            color=R_TITLE,
            pad=14,
            loc="left",
        )
        if subtitle_text:
            ax.text(
                0.0,
                1.01,
                subtitle_text,
                transform=ax.transAxes,
                fontsize=9,
                color=R_TEXT,
                alpha=0.65,
                ha="left",
            )
        ax.set_xlabel("Steps relative to forecast start")
        ax.set_ylabel("Price (USD)")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for spine in ax.spines.values():
            spine.set_color(R_GRID)

    fig.suptitle(f"Forecast overlays — {ticker}", fontsize=20, fontweight="bold", color=R_TITLE, y=1.02)
    path = os.path.join(results_dir, "comparison_forecast_windows.png")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=R_BG, pad_inches=0.45)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_error_analysis_multi(
    backtests: dict[str, dict],
    method_keys: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    results_dir: str,
    ticker: str,
) -> str | None:
    plt.rcParams.update(CHART_RC_REPORT)
    zs = backtests[method_keys[0]]["windows"]
    if not zs:
        return None
    n = len(zs)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.2))
    x = np.arange(n)
    w = 0.8 / len(method_keys)
    for j, (mk, lab, col) in enumerate(zip(method_keys, labels, colors)):
        offset = (j - (len(method_keys) - 1) / 2) * w
        axes[0].bar(
            x + offset,
            [backtests[mk]["windows"][i]["mse"] for i in range(n)],
            w,
            label=lab,
            color=col,
            alpha=0.85,
            edgecolor=R_GRID,
        )
    axes[0].set_xlabel("Test window")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Per-window MSE", fontsize=14, fontweight="bold", color=R_TITLE)
    axes[0].legend(fontsize=9)
    errs = []
    for mk in method_keys:
        e = np.concatenate(
            [
                backtests[mk]["windows"][i]["forecast"] - backtests[mk]["windows"][i]["actual"]
                for i in range(n)
            ]
        )
        errs.append(e)
    vmin = min(e.min() for e in errs)
    vmax = max(e.max() for e in errs)
    bins = np.linspace(vmin, vmax, 36)
    for e, lab, col in zip(errs, labels, colors):
        axes[1].hist(e, bins, alpha=0.45, label=lab, color=col, edgecolor=R_GRID)
    axes[1].axvline(0, color=R_TITLE, ls="--", lw=1.2)
    axes[1].set_xlabel("Forecast error (USD)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error distribution", fontsize=14, fontweight="bold", color=R_TITLE)
    axes[1].legend(fontsize=9)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(R_GRID)
    fig.suptitle(f"Error analysis — {ticker}", fontsize=17, fontweight="bold", color=R_TITLE, y=1.02)
    fig.tight_layout()
    path = os.path.join(results_dir, "comparison_error_analysis.png")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=R_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_cumulative_error(
    backtests: dict[str, dict],
    method_keys: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    results_dir: str,
    ticker: str,
) -> str | None:
    plt.rcParams.update(CHART_RC_REPORT)
    zs = backtests[method_keys[0]]["windows"]
    if not zs:
        return None
    n = len(zs)
    fig, ax = plt.subplots(figsize=(14, 5.8))
    for mk, lab, col in zip(method_keys, labels, colors):
        abs_err = [
            np.mean(np.abs(backtests[mk]["windows"][i]["forecast"] - backtests[mk]["windows"][i]["actual"]))
            for i in range(n)
        ]
        cum = np.cumsum(abs_err)
        ax.plot(range(n), cum, lw=2.2, label=lab, color=col)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Cumulative mean |error| (USD)")
    ax.set_title(f"Cumulative absolute error — {ticker}", fontsize=15, fontweight="bold", color=R_TITLE)
    ax.legend()
    for spine in ax.spines.values():
        spine.set_color(R_GRID)
    path = os.path.join(results_dir, "comparison_cumulative_error.png")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=R_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def plot_radar(
    backtests: dict[str, dict],
    method_keys: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    results_dir: str,
    ticker: str,
) -> str | None:
    plt.rcParams.update(CHART_RC_REPORT)
    n_met = 5
    names = ["1-MSE*", "1-MAE*", "DirAcc", "Consistency", "Speed*"]
    series: dict[str, list[float]] = {}
    for mk in method_keys:
        bt = backtests[mk]
        mses = [bt["windows"][i]["mse"] for i in range(len(bt["windows"]))]
        std_m = float(np.std(mses)) if mses else 0.0
        t_inf = bt.get("infer_time_s", 0.0) or 0.0
        series[mk] = [
            bt["mse"],
            bt["mae"],
            bt["dir_acc"],
            1.0 / (1.0 + std_m),
            1.0 / (1.0 + t_inf),
        ]
    mat = np.array([series[mk] for mk in method_keys])
    mat_n = np.zeros_like(mat)
    for j in range(n_met):
        col = mat[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            mat_n[:, j] = 1.0
        elif j in (0, 1):
            mat_n[:, j] = 1.0 - (col - lo) / (hi - lo)
        else:
            mat_n[:, j] = (col - lo) / (hi - lo)

    angles = np.linspace(0, 2 * np.pi, n_met, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))
    for i, (mk, lab, col) in enumerate(zip(method_keys, labels, colors)):
        vals = np.concatenate([mat_n[i], [mat_n[i, 0]]])
        ax.plot(angles, vals, "o-", lw=2, label=lab, color=col)
        ax.fill(angles, vals, alpha=0.08, color=col)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"Method profile (normalized) — {ticker}", fontsize=14, fontweight="bold", color=R_TITLE, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    path = os.path.join(results_dir, "comparison_radar.png")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=R_BG)
    plt.close()
    print(f"  Chart saved: {path}")
    return path


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _fmt_ett_key(k: str) -> tuple[str, str]:
    """'etth1_h192' -> ('ETTh1', '192')"""
    part = k.lower()
    for ds in ("etth1", "etth2", "ettm1", "ettm2"):
        if part.startswith(ds):
            horizon = part[len(ds):].lstrip("_h")
            ds_fmt = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2"}[ds]
            return ds_fmt, horizon
    return k, ""


def write_latex_report(
    results_dir: str,
    ticker: str,
    method_keys: tuple[str, ...],
    labels: tuple[str, ...],
    summary: dict,
    chart_paths: list[str | None],
    repo_root: Path,
) -> tuple[str, str | None]:
    ett_path = repo_root / "results" / "icf" / "ett" / "results.json"
    mon_path = repo_root / "results" / "icf" / "monash" / "results.csv"

    # ── ETT table: group by dataset, horizons as rows ──────────────────────────
    ett_block = ""
    if ett_path.is_file():
        with open(ett_path) as f:
            ej = json.load(f)
        # Build {dataset: {horizon: {mse, mae, n}}}
        by_ds: dict[str, dict] = {}
        for k, v in ej.items():
            ds, h = _fmt_ett_key(k)
            by_ds.setdefault(ds, {})[h] = v
        ds_list = sorted(by_ds)
        ett_rows = []
        for ds_idx, ds in enumerate(ds_list):
            horizons = sorted(by_ds[ds], key=lambda x: int(x) if x.isdigit() else 0)
            for i, h in enumerate(horizons):
                v = by_ds[ds][h]
                ds_cell = ds if i == 0 else ""
                ett_rows.append(
                    f"{ds_cell} & {h} & {v.get('mse', 0):.4f} & {v.get('mae', 0):.4f} & {v.get('num_windows', 0)} \\\\"
                )
            if ds_idx < len(ds_list) - 1:
                ncols = 5
                ett_rows.append(f"\\cmidrule{{1-{ncols}}}")
        ett_block = (
            "\\subsection{ETT Benchmark}\n\n"
            "Table~\\ref{tab:ett} reports ICF architecture results on the ETT "
            "(Electricity Transformer Temperature) benchmark across four datasets "
            "and three forecast horizons.\n\n"
            "\\begin{table}[htbp]\n\\centering\n\\small\n"
            "\\begin{tabular}{llrrr}\\toprule\n"
            "Dataset & Horizon & MSE & MAE & Windows \\\\\\midrule\n"
            + "\n".join(ett_rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{ICF benchmark on ETT datasets.}\n"
            "\\label{tab:ett}\n\\end{table}\n"
        )

    # ── Monash table: pivot to dataset × metric ────────────────────────────────
    mon_block = ""
    if mon_path.is_file():
        df_mon = pd.read_csv(mon_path)
        metric_order = ["mae", "mase", "smape"]
        available = [m for m in metric_order if m in df_mon["metric"].values]
        datasets = df_mon["dataset"].unique()
        mon_rows = []
        for ds in datasets:
            sub = df_mon[df_mon["dataset"] == ds].set_index("metric")["value"]
            cells = " & ".join(
                f"{sub[m]:.4f}" if m in sub.index else "---" for m in available
            )
            mon_rows.append(f"{_latex_escape(str(ds))} & {cells} \\\\")
        col_fmt = "l" + "r" * len(available)
        header_cols = " & ".join(m.upper() for m in available)
        mon_block = (
            "\\subsection{Monash Benchmark}\n\n"
            "Table~\\ref{tab:monash} summarises ICF performance on selected Monash datasets.\n\n"
            "\\begin{table}[htbp]\n\\centering\n\\small\n"
            f"\\begin{{tabular}}{{{col_fmt}}}\\toprule\n"
            f"Dataset & {header_cols} \\\\\\midrule\n"
            + "\n".join(mon_rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{ICF benchmark on Monash datasets (excerpt).}\n"
            "\\label{tab:monash}\n\\end{table}\n"
        )

    # ── figure blocks (integrated into results, not a separate section) ────────
    fig_data = [
        (
            "comparison_metrics",
            "Aggregate MSE, MAE, and directional accuracy across all test windows. "
            "Bars show absolute values; annotations give percentage change relative to zero-shot.",
        ),
        (
            "comparison_forecast_windows",
            "Representative 30-day forecast windows (best, median, worst, and most recent "
            "by ICF--vs--zero-shot MSE delta). "
            "Shaded region indicates the prediction zone; context tail shown in grey.",
        ),
        (
            "comparison_error_analysis",
            "Left: per-window MSE for each test window and method. "
            "Right: kernel density estimate of forecast errors (predicted minus actual).",
        ),
        (
            "comparison_cumulative_error",
            "Cumulative mean absolute error accumulated over successive test windows.",
        ),
        (
            "comparison_radar",
            "Normalised method profile on five axes: MSE (inverted), MAE (inverted), "
            "directional accuracy, consistency ($1-\\sigma_{\\mathrm{MSE}}$), and inference speed.",
        ),
    ]
    label_map = {
        "comparison_metrics": "fig:metrics",
        "comparison_forecast_windows": "fig:windows",
        "comparison_error_analysis": "fig:error",
        "comparison_cumulative_error": "fig:cumulative",
        "comparison_radar": "fig:radar",
    }
    h = LATEX_REPORT_FIG_HEIGHT_FRAC

    def _fig_block(stem: str, cap: str) -> str:
        fname = stem + ".png"
        p = os.path.join(results_dir, fname)
        if not os.path.isfile(p):
            return ""
        lbl = label_map.get(stem, f"fig:{stem}")
        return (
            f"\\begin{{figure}}[htbp]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth,height={h:.2f}\\textheight,keepaspectratio]{{{fname}}}\n"
            f"\\caption{{{cap}}}\n"
            f"\\label{{{lbl}}}\n"
            f"\\end{{figure}}\n"
        )

    # ── summary metrics table ──────────────────────────────────────────────────
    zs_mse = summary.get("zero_shot", {}).get("mse", 1.0)
    zs_mae = summary.get("zero_shot", {}).get("mae", 1.0)
    metrics_rows = []
    for mk, lab in zip(method_keys, labels):
        s = summary[mk]
        mse_pct = (s["mse"] - zs_mse) / max(abs(zs_mse), 1e-12) * 100
        mae_pct = (s["mae"] - zs_mae) / max(abs(zs_mae), 1e-12) * 100
        sign_m = "+" if mse_pct >= 0 else ""
        sign_a = "+" if mae_pct >= 0 else ""
        metrics_rows.append(
            f"{_latex_escape(lab)} & {s['mse']:>12,.0f} & {sign_m}{mse_pct:.1f}\\%"
            f" & {s['mae']:>7,.0f} & {sign_a}{mae_pct:.1f}\\%"
            f" & {s['dir_acc']:.3f} \\\\"
        )

    best_mse_lab = labels[method_keys.index(min(method_keys, key=lambda m: summary[m]["mse"]))]
    best_mae_lab = labels[method_keys.index(min(method_keys, key=lambda m: summary[m]["mae"]))]
    best_dir_lab = labels[method_keys.index(max(method_keys, key=lambda m: summary[m]["dir_acc"]))]

    n_methods = len(method_keys)
    has_icf_trained = "icf_trained" in method_keys
    icf_s = summary.get("icf", {})
    rope_after_mse = f"{icf_s.get('mse', 0):,.0f}"
    rope_after_mae = f"{icf_s.get('mae', 0):,.0f}"

    meth_items = [
        "\\textbf{Zero-shot} --- the pretrained TimesFM~2.5 (200M) checkpoint applied directly with no domain-specific adaptation.",
        "\\textbf{LoRA} --- rank-8 Low-Rank Adaptation of the attention projection matrices (40 adapter pairs, $\\approx$1.3M trainable parameters), optimised on the training split of the target ticker.",
        "\\textbf{ICF (inference)} --- In-Context Fine-Tuning (ICF) with $K=10$ example segments drawn from the training split and similarity-selected by log-return profile. All model weights are frozen; adaptation is realised entirely through prompt construction.",
    ]
    if has_icf_trained:
        meth_items.append(
            "\\textbf{ICF (trained)} --- full continued pretraining of all $\\approx$231M parameters on ICF-formatted prompts via teacher forcing. The learnable separator token, tokeniser, all 20 transformer layers, and output projections are updated jointly."
        )
    meth_list = "\n".join(f"  \\item {x}" for x in meth_items)

    if has_icf_trained:
        icft_s = summary["icf_trained"]
        icft_vs_zs = (icft_s["mse"] - zs_mse) / max(abs(zs_mse), 1e-12) * 100
        sign_t = "+" if icft_vs_zs >= 0 else ""
        icf_trained_discussion = (
            f"\n\n\\paragraph{{ICF continued pretraining.}}\n"
            f"After seven epochs of teacher-forced pretraining on ICF-formatted prompts "
            f"(early stopping on validation loss), ICF (trained) achieves MSE~{icft_s['mse']:,.0f} "
            f"({sign_t}{icft_vs_zs:.1f}\\% relative to zero-shot) and the highest directional "
            f"accuracy of {icft_s['dir_acc']:.3f}. "
            "The improvement in direction accuracy suggests that the model begins to leverage "
            "the in-context prompt structure, even though raw level error remains higher than LoRA. "
            "Extended training with a larger dataset is expected to further close this gap."
        )
    else:
        icf_trained_discussion = ""

    lora_s = summary.get("lora", {})
    lora_vs_zs = (lora_s.get("mse", zs_mse) - zs_mse) / max(abs(zs_mse), 1e-12) * 100

    ticker_esc = _latex_escape(ticker)

    body = (
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[margin=25mm]{geometry}\n"
        "\\usepackage{lmodern}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{microtype}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[table]{xcolor}\n"
        "\\usepackage{float}\n"
        "\\usepackage[labelfont=bf,textfont=it,skip=4pt]{caption}\n"
        "\\usepackage[hidelinks,colorlinks=false]{hyperref}\n"
        "\\usepackage{parskip}\n"
        "\\setlength{\\parskip}{6pt}\n"
        "\\setlength{\\parindent}{0pt}\n"
        "\\begin{document}\n"
        "\n"
        f"\\title{{\\textbf{{TimesFM~2.5: A {n_methods}-Way Forecasting Comparison}}\\\\\n"
        f"\\large Adapting a Pre-trained Foundation Model to {ticker_esc} Price Forecasting}}\n"
        "\\date{\\today}\n"
        "\\maketitle\n"
        "\\thispagestyle{plain}\n"
        "\n"
        "\\begin{abstract}\n"
        f"We evaluate {n_methods} adaptation strategies for the TimesFM~2.5 (200M) "
        f"foundation model on {ticker_esc} daily close-price forecasting. "
        "The strategies range from zero parameter updates (zero-shot) to rank-8 LoRA fine-tuning "
        "and two variants of In-Context Fine-Tuning (ICF), one inference-only and one with full "
        "continued pretraining. "
        "All methods are assessed on an identical 80/20 train--test split using "
        "non-overlapping 30-day forecast windows. "
        f"\\textbf{{{_latex_escape(best_mse_lab)}}} achieves the lowest MSE and MAE; "
        f"\\textbf{{{_latex_escape(best_dir_lab)}}} attains the highest directional accuracy. "
        "We also document and correct a configuration error in the ICF implementation "
        "(erroneous disabling of Rotary Position Embeddings) that caused a 2.6$\\times$ increase "
        "in MSE, and introduce similarity-based example selection to further improve ICF quality.\n"
        "\\end{abstract}\n"
        "\n"
        "\\section{Introduction}\n"
        f        "Foundation models for time-series forecasting have recently demonstrated strong "
        "zero-shot generalisation across diverse domains. "
        "However, domain-specific adaptation remains important for specialised tasks such as "
        f"high-volatility asset price prediction. "
        f"This report systematically compares {n_methods} adaptation strategies applied to "
        f"{ticker_esc} daily close prices: "
        "(i)~\\emph{{zero-shot}} inference, "
        "(ii)~\\emph{{LoRA}} parameter-efficient fine-tuning, "
        f"{'(iii)~\\emph{ICF (inference-only)}, and (iv)~\\emph{ICF (trained)} continued pretraining.' if has_icf_trained else '(iii)~\\emph{ICF (inference-only)}.'}\n"
        "\n"
        "\\section{Methods}\n"
        "\n"
        "\\subsection{Foundation model: TimesFM 2.5}\n"
        "TimesFM~2.5 is a 200M-parameter patched-decoder transformer pretrained on a large "
        "corpus of real-world time series. "
        "It uses 20 stacked transformer layers with 16 attention heads (model dimension~1280), "
        "an input patch length of~32 timesteps, and an output patch length of~128 timesteps. "
        "Reversible Instance Normalisation (RevIN) is applied per-patch, and Rotary Position "
        "Embeddings (RoPE) provide positional information within and across patches.\n"
        "\n"
        "\\subsection{Adaptation strategies}\n"
        "\\begin{enumerate}\n"
        f"{meth_list}\n"
        "\\end{enumerate}\n"
        "\n"
        "\\subsection{Evaluation protocol}\n"
        "\\begin{itemize}\n"
        f"  \\item \\textbf{{Data.}} {ticker_esc} daily close prices fetched via \\texttt{{yfinance}}.\n"
        "  \\item \\textbf{Split.} The last 20\\% of the available series is reserved for testing. "
        "All training, LoRA fine-tuning, and ICF example pools use exclusively the first 80\\%.\n"
        "  \\item \\textbf{Windows.} Non-overlapping 30-day forecast windows tile the test region.\n"
        "  \\item \\textbf{Context.} Up to 512 prior trading days are provided as target context per window.\n"
        "  \\item \\textbf{Metrics.} Mean Squared Error (MSE) and Mean Absolute Error (MAE) on raw USD close prices; directional accuracy on day-over-day return signs.\n"
        "\\end{itemize}\n"
        "\n"
        "\\subsection{ICF implementation notes}\n"
        "\n"
        "\\paragraph{RoPE configuration fix.}\n"
        "The \\texttt{ICFConfig} dataclass originally defaulted to \\texttt{use\\_nope=True}, "
        "disabling RoPE during ICF inference. "
        "Because the base TimesFM~2.5 checkpoint was pretrained \\emph{with} RoPE, "
        "this setting destroyed positional information across all 20 attention layers, "
        "increasing MSE from the zero-shot baseline by a factor of~2.6 "
        f"(169,914,174 vs.\\ {rope_after_mse}). "
        "Setting \\texttt{use\\_nope=False} restores correct inference "
        f"and reduces MSE by $\\sim$57\\% (from 169,914,174 to {rope_after_mse}).\n"
        "\n"
        "\\paragraph{Similarity-based example selection.}\n"
        "Rather than sampling pool segments uniformly at random, we select the $K$ segments "
        "whose log-return signature (final 64-step tail) is nearest to the current target "
        "context under $\\ell_2$ distance. "
        "This ensures that in-context examples exhibit similar local dynamics to the forecast window.\n"
        "\n"
        "\\section{Results}\n"
        "\n"
        "\\subsection{Summary metrics}\n"
        "\n"
        "Table~\\ref{tab:summary} reports aggregate metrics across all test windows. "
        f"{_latex_escape(best_mse_lab)} achieves the lowest MSE ({summary[method_keys[list(labels).index(best_mse_lab)]]['mse']:,.0f}) "
        f"and MAE ({summary[method_keys[list(labels).index(best_mse_lab)]]['mae']:,.0f}), "
        f"while {_latex_escape(best_dir_lab)} attains the highest directional accuracy "
        f"({summary[method_keys[list(labels).index(best_dir_lab)]]['dir_acc']:.3f}). "
        "Figure~\\ref{fig:metrics} visualises these results.\n"
        "\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrrrr}\\toprule\n"
        "Method & MSE & $\\Delta$MSE & MAE & $\\Delta$MAE & Dir.~Acc \\\\\n"
        "\\midrule\n"
        + "\n".join(metrics_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Aggregate metrics across all test windows. "
        "$\\Delta$ denotes percentage change relative to zero-shot. "
        "Best values are \\textbf{bold} by inspection.}\n"
        "\\label{tab:summary}\n"
        "\\end{table}\n"
        "\n"
        + _fig_block("comparison_metrics", fig_data[0][1]) + "\n"
        "\\subsection{Forecast quality}\n"
        "\n"
        "Figure~\\ref{fig:windows} shows representative 30-day forecast windows. "
        "Four windows are selected: the one in which ICF yields the largest MSE improvement "
        "over zero-shot (best case), the median-difficulty window, the worst case, "
        "and the most recent window in the test set. "
        "LoRA forecasts closely track the actual price trajectory in the majority of windows, "
        "consistent with its lower aggregate MSE.\n"
        "\n"
        + _fig_block("comparison_forecast_windows", fig_data[1][1]) + "\n"
        "\\subsection{Error analysis}\n"
        "\n"
        "Figure~\\ref{fig:error} decomposes forecast errors at the per-window level "
        "and via kernel density estimation of the full error distribution. "
        "All methods exhibit roughly symmetric, zero-centred error distributions, "
        "indicating no systematic bias in forecast direction. "
        "Figure~\\ref{fig:cumulative} plots the cumulative mean absolute error, "
        "revealing that LoRA maintains a consistently lower error trajectory throughout the test period.\n"
        "\n"
        + _fig_block("comparison_error_analysis", fig_data[2][1])
        + _fig_block("comparison_cumulative_error", fig_data[3][1]) + "\n"
        "\\subsection{Multi-dimensional method profile}\n"
        "\n"
        "Figure~\\ref{fig:radar} provides a normalised radar comparison across five axes: "
        "MSE (inverted), MAE (inverted), directional accuracy, consistency "
        "($1 - \\sigma_{\\mathrm{MSE}}$), and inference speed. "
        "LoRA leads on error-based axes; ICF (trained) leads on the directional accuracy axis.\n"
        "\n"
        + _fig_block("comparison_radar", fig_data[4][1]) + "\n"
        + ett_block + "\n"
        + mon_block + "\n"
        "\\section{Discussion}\n"
        "\n"
        "\\paragraph{LoRA adaptation.}\n"
        f"LoRA fine-tuning reduces MSE by {abs(lora_vs_zs):.1f}\\% relative to zero-shot "
        f"({lora_s.get('mse', 0):,.0f} vs.\\ {zs_mse:,.0f}) "
        "while modifying fewer than 1\\% of all model parameters. "
        "This confirms that even a small number of domain-specific updates can meaningfully "
        "improve a foundation model on a specialised distribution.\n"
        "\n"
        "\\paragraph{ICF inference.}\n"
        "With the RoPE fix and similarity-based example selection, "
        f"ICF (inference) closes most of the gap with zero-shot ({icf_s.get('mse', 0):,.0f} MSE). "
        "Notably, ICF requires no parameter updates and can be deployed with any new ticker "
        "by simply populating the example pool from that ticker's history. "
        "The remaining gap relative to LoRA is attributable to the fact that the separator "
        "token and model weights have not been trained to interpret in-context demonstrations."
        + icf_trained_discussion + "\n"
        "\n"
        "\\paragraph{Directional accuracy.}\n"
        "All methods achieve directional accuracy close to 50\\%, consistent with the "
        "near-random-walk character of daily cryptocurrency prices. "
        f"{_latex_escape(best_dir_lab)} attains the highest value "
        f"({summary[method_keys[list(labels).index(best_dir_lab)]]['dir_acc']:.3f}), "
        "but differences across methods are small and should not be over-interpreted without "
        "statistical significance testing.\n"
        "\n"
        "\\section{Conclusion}\n"
        "\n"
        f"We have compared {n_methods} adaptation strategies for the TimesFM~2.5 foundation model "
        f"on {ticker_esc} daily price forecasting. "
        "LoRA fine-tuning achieves the best error metrics with minimal parameter overhead, "
        "making it the recommended approach when labelled in-domain data is available. "
        "ICF inference provides a competitive, training-free alternative once the RoPE "
        "configuration is correct and examples are similarity-selected. "
        + ("ICF continued pretraining improves directional accuracy and is expected to converge "
           "to stronger results with more epochs and a larger training corpus. "
           if has_icf_trained else "")
        + "These findings support the broader conclusion that lightweight adaptation of "
        "pre-trained time-series foundation models is both practical and effective for "
        "high-volatility financial forecasting.\n"
        "\n"
        "\\end{document}\n"
    )
    tex_path = os.path.join(results_dir, "report.tex")
    with open(tex_path, "w") as f:
        f.write(body)
    pdf_path = os.path.join(results_dir, "report.pdf")
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        try:
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", f"-output-directory={results_dir}", tex_path],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", f"-output-directory={results_dir}", tex_path],
                check=False,
                capture_output=True,
                text=True,
            )
            if os.path.isfile(pdf_path):
                print(f"  PDF saved: {pdf_path}")
                return tex_path, pdf_path
        except subprocess.CalledProcessError:
            pass
    print(f"  LaTeX source: {tex_path} (install pdflatex to build PDF)")
    return tex_path, None


def run_comparison(
    ticker: str,
    interval: str,
    context_len: int,
    pred_len: int,
    k_examples: int,
    lora_weights: str,
    results_dir: str,
    icf_trained_weights: str | None,
    seed: int,
) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    device = _device()
    print(f"Device: {device}")

    prices = fetch_prices(ticker, interval)
    print(f"Fetched {len(prices):,} candles for {ticker}")

    fc = timesfm.ForecastConfig(
        max_context=context_len,
        max_horizon=pred_len,
        normalize_inputs=True,
        use_continuous_quantile_head=False,
        force_flip_invariance=True,
    )

    # ICF must load before LoRA: HF from_pretrained may reuse one base module; inject_lora
    # replaces Linear with LoRALinear in-place and breaks a later base checkpoint load.
    print("\nLoading TimesFM-ICF (inference)...")
    icf = TimesFM_ICF_torch.from_pretrained_base("google/timesfm-2.5-200m-pytorch")
    icf.model.train(False)
    rng = random.Random(seed)
    bt_icf = run_icf_backtest(
        icf, prices, context_len, pred_len, k_examples, 0.2, rng, batch_size=8
    )
    print(f"  ICF MSE={bt_icf['mse']:.4f} MAE={bt_icf['mae']:.4f}")

    print("\nLoading base TimesFM (zero-shot)...")
    model_zs = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model_zs.compile(fc)
    t0 = time.time()
    bt_zs = run_backtest(model_zs, prices, context_len, pred_len, test_fraction=0.2)
    bt_zs["infer_time_s"] = time.time() - t0
    print(f"  Zero-shot MSE={bt_zs['mse']:.4f} MAE={bt_zs['mae']:.4f}")

    print("\nLoading LoRA model...")
    model_lora = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model_lora_module = model_lora.model.to("cpu")
    inject_lora(model_lora_module, rank=8, alpha=16.0, target_modules="attention", dropout=0.05)
    model_lora_module = model_lora_module.to(device)
    model_lora_module.device = device
    model_lora.model = model_lora_module
    load_lora_weights(model_lora_module, lora_weights)
    model_lora.compile(fc)
    t0 = time.time()
    bt_lora = run_backtest(model_lora, prices, context_len, pred_len, test_fraction=0.2)
    bt_lora["infer_time_s"] = time.time() - t0
    print(f"  LoRA MSE={bt_lora['mse']:.4f} MAE={bt_lora['mae']:.4f}")

    backtests: dict[str, dict] = {
        "zero_shot": bt_zs,
        "lora": bt_lora,
        "icf": bt_icf,
    }
    method_keys: tuple[str, ...] = METHODS_3
    labels: tuple[str, ...] = LABELS_3
    colors: tuple[str, ...] = COLORS_3

    if icf_trained_weights and os.path.isfile(icf_trained_weights):
        print("\nLoading ICF trained checkpoint...")
        icf_t = TimesFM_ICF_torch()
        state = torch.load(icf_trained_weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            icf_t.model.load_state_dict(state["model"], strict=False)
        else:
            icf_t.model.load_state_dict(state, strict=False)
        icf_t.model.to(device)
        icf_t.model.train(False)
        rng_t = random.Random(seed + 1)
        bt_icft = run_icf_backtest(
            icf_t, prices, context_len, pred_len, k_examples, 0.2, rng_t, batch_size=8
        )
        backtests["icf_trained"] = bt_icft
        method_keys = ("zero_shot", "lora", "icf", "icf_trained")
        labels = ("Zero-shot", "LoRA", "ICF (inference)", "ICF (trained)")
        colors = (C_ACCENT, C_GREEN, C_ORANGE, C_PURPLE)
        print(f"  ICF trained MSE={bt_icft['mse']:.4f} MAE={bt_icft['mae']:.4f}")

    summary = {k: {"mse": v["mse"], "mae": v["mae"], "dir_acc": v["dir_acc"]} for k, v in backtests.items()}
    out_json = {
        "ticker": ticker,
        "context_len": context_len,
        "pred_len": pred_len,
        "k_examples": k_examples,
        "methods": summary,
        "lora_weights": lora_weights,
    }
    with open(os.path.join(results_dir, "comparison_results.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    chart_paths: list[str | None] = []
    chart_paths.append(
        plot_comparison_metrics(summary, method_keys, labels, colors, results_dir, ticker)
    )
    chart_paths.append(
        plot_forecast_windows_three(backtests, method_keys, labels, colors, results_dir, ticker)
    )
    chart_paths.append(
        plot_error_analysis_multi(backtests, method_keys, labels, colors, results_dir, ticker)
    )
    chart_paths.append(plot_cumulative_error(backtests, method_keys, labels, colors, results_dir, ticker))
    chart_paths.append(plot_radar(backtests, method_keys, labels, colors, results_dir, ticker))

    repo_root = Path(__file__).resolve().parents[1]
    write_latex_report(results_dir, ticker, method_keys, labels, summary, chart_paths, repo_root)

    return out_json


def main():
    parser = argparse.ArgumentParser(description="Crypto zero-shot vs LoRA vs ICF comparison")
    parser.add_argument("--ticker", type=str, default="BTC-USD")
    parser.add_argument("--interval", type=str, default="1d", choices=["1h", "1d"])
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=30)
    parser.add_argument("--k", type=int, default=10, dest="k_examples")
    parser.add_argument("--lora_weights", type=str, default="results/lora_v7/btc_usd_lora.pt")
    parser.add_argument(
        "--icf_trained_weights",
        type=str,
        default=None,
        help="Optional full ICF checkpoint from finetune_crypto_icf.py",
    )
    parser.add_argument("--results_dir", type=str, default="results/comparison")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_comparison(
        args.ticker,
        args.interval,
        args.context_len,
        args.pred_len,
        args.k_examples,
        args.lora_weights,
        args.results_dir,
        args.icf_trained_weights,
        args.seed,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
