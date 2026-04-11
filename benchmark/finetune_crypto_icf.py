#!/usr/bin/env python3
"""Full continued pretraining for TimesFM-ICF on crypto (teacher-forcing via decode_icf)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf
from torch.utils.data import DataLoader, Dataset

import timesfm
from timesfm.timesfm_2p5 import timesfm_2p5_base
from timesfm.timesfm_icf import ICFConfig, TimesFM_ICF_torch, _left_pad_to_multiple


def _to_1d_np(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr.astype(np.float64)


def _round_up_patch(x: int, p: int) -> int:
    return int(math.ceil(x / p) * p)


def icf_prepare_batch_tensors(
    icf: TimesFM_ICF_torch,
    horizon: int,
    batch_examples: list[list[np.ndarray]],
    batch_targets: list[np.ndarray],
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    fc = icf.forecast_config
    icfg = icf.icf_config
    assert fc is not None and icfg is not None

    tgt_vals = []
    tgt_masks = []
    for arr in batch_targets:
        v = timesfm_2p5_base.linear_interpolation(timesfm_2p5_base.strip_leading_nans(_to_1d_np(arr)))
        if len(v) >= fc.max_context:
            v = v[-fc.max_context :]
            m = np.zeros_like(v, dtype=bool)
        else:
            m = np.array([True] * (fc.max_context - len(v)) + [False] * len(v))
            v = np.pad(v, (fc.max_context - len(v), 0), constant_values=0.0)
        tgt_vals.append(v)
        tgt_masks.append(m)

    device = icf.model.sep_token.device
    tgt_t = torch.from_numpy(np.stack(tgt_vals)).to(torch.float32).to(device)
    tgt_m = torch.from_numpy(np.stack(tgt_masks)).to(torch.bool).to(device)

    ex_inputs: list[torch.Tensor] = []
    ex_masks: list[torch.Tensor] = []
    for ex_i in range(icfg.k_examples):
        vals = []
        masks = []
        for ex_list in batch_examples:
            ex_arr = ex_list[ex_i] if ex_i < len(ex_list) else np.array([0.0, 0.0, 0.0])
            v = timesfm_2p5_base.linear_interpolation(timesfm_2p5_base.strip_leading_nans(_to_1d_np(ex_arr)))
            if len(v) >= icfg.example_len:
                v = v[-icfg.example_len :]
                m = np.zeros_like(v, dtype=bool)
            else:
                m = np.array([True] * (icfg.example_len - len(v)) + [False] * len(v))
                v = np.pad(v, (icfg.example_len - len(v), 0), constant_values=0.0)
            vals.append(v)
            masks.append(m)
        ex_inputs.append(torch.from_numpy(np.stack(vals)).to(torch.float32).to(device))
        ex_masks.append(torch.from_numpy(np.stack(masks)).to(torch.bool).to(device))

    def _align(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _left_pad_to_multiple(x, m, icf.model.p)

    aligned_ex_inputs, aligned_ex_masks = [], []
    for x, m in zip(ex_inputs, ex_masks):
        x2, m2 = _align(x, m)
        aligned_ex_inputs.append(x2)
        aligned_ex_masks.append(m2)

    tgt_t2, tgt_m2 = _align(tgt_t, tgt_m)
    return aligned_ex_inputs, aligned_ex_masks, tgt_t2, tgt_m2


def icf_decode_batch(
    icf: TimesFM_ICF_torch,
    horizon: int,
    batch_examples: list[list[np.ndarray]],
    batch_targets: list[np.ndarray],
) -> torch.Tensor:
    ex_in, ex_m, tgt_t, tgt_m = icf_prepare_batch_tensors(icf, horizon, batch_examples, batch_targets)
    apply_rope = not icf.icf_config.use_nope
    point_t, _ = icf.model.decode_icf(
        horizon,
        example_inputs=ex_in,
        example_masks=ex_m,
        target_inputs=tgt_t,
        target_masks=tgt_m,
        apply_rope=apply_rope,
    )
    return point_t


class ICFCryptoDataset(Dataset):
    """Sliding windows on ``window_prices``; K example segments from ``pool_prices`` ending before the window (global index)."""

    def __init__(
        self,
        window_prices: np.ndarray,
        context_len: int,
        pred_len: int,
        k_examples: int,
        stride: int,
        seed: int,
        *,
        pool_prices: np.ndarray | None = None,
        window_global_offset: int = 0,
    ):
        self.prices = window_prices.astype(np.float32)
        pool_src = pool_prices if pool_prices is not None else window_prices
        self.pool_prices = pool_src.astype(np.float32)
        self.window_global_offset = int(window_global_offset)
        self.context_len = context_len
        self.pred_len = pred_len
        self.k_examples = k_examples
        self.rng = random.Random(seed)
        self.example_len = _round_up_patch(context_len + pred_len, 32)
        self.pool: list[tuple[int, np.ndarray]] = []
        pn = len(self.pool_prices)
        for start in range(0, pn - self.example_len + 1, pred_len):
            seg = self.pool_prices[start : start + self.example_len].astype(np.float64).copy()
            self.pool.append((start, seg))
        self.indices: list[int] = []
        n = len(self.prices)
        for s in range(0, n - context_len - pred_len + 1, stride):
            g = self.window_global_offset + s
            if not self._enough_examples(g):
                continue
            self.indices.append(s)

    def _enough_examples(self, window_start_global: int) -> bool:
        n_valid = sum(1 for st, seg in self.pool if st + len(seg) <= window_start_global)
        return n_valid >= self.k_examples

    def _sample_examples(self, window_start_global: int) -> list[np.ndarray]:
        valid = [p for p in self.pool if p[0] + len(p[1]) <= window_start_global]
        if len(valid) < self.k_examples:
            out = [v[1].copy() for v in valid]
            while len(out) < self.k_examples:
                out.append(np.zeros(self.example_len, dtype=np.float64))
            return out
        return [self.rng.choice(valid)[1].copy() for _ in range(self.k_examples)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        s = self.indices[idx]
        g = self.window_global_offset + s
        ctx = self.prices[s : s + self.context_len]
        fut = self.prices[s + self.context_len : s + self.context_len + self.pred_len]
        ex = self._sample_examples(g)
        return (
            torch.from_numpy(ctx.copy()),
            torch.from_numpy(fut.copy()),
            ex,
        )


def collate_icf(batch):
    ctx = torch.stack([b[0] for b in batch])
    fut = torch.stack([b[1] for b in batch])
    k = len(batch[0][2])
    ex_lists = []
    for i in range(len(batch)):
        ex_lists.append([batch[i][2][j] for j in range(k)])
    return ctx, fut, ex_lists


def train_one_epoch(
    icf: TimesFM_ICF_torch,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    horizon: int,
    device: torch.device,
    grad_accum: int,
) -> float:
    icf.model.train(True)
    total, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, (ctx_b, fut_b, ex_lists) in enumerate(loader):
        ctx_b = ctx_b.to(device)
        fut_b = fut_b.to(device)
        batch_targets = [ctx_b[i].detach().cpu().numpy() for i in range(ctx_b.shape[0])]
        pred = icf_decode_batch(icf, horizon, ex_lists, batch_targets)
        target = fut_b[:, :horizon].float()
        loss = F.l1_loss(pred, target)
        (loss / grad_accum).backward()
        total += float(loss.item())
        n_batches += 1
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(icf.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total / max(n_batches, 1)


def validate(icf: TimesFM_ICF_torch, loader: DataLoader, horizon: int, device: torch.device) -> float:
    icf.model.train(False)
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for ctx_b, fut_b, ex_lists in loader:
            ctx_b = ctx_b.to(device)
            fut_b = fut_b.to(device)
            batch_targets = [ctx_b[i].detach().cpu().numpy() for i in range(ctx_b.shape[0])]
            pred = icf_decode_batch(icf, horizon, ex_lists, batch_targets)
            target = fut_b[:, :horizon].float()
            loss = F.l1_loss(pred, target)
            total += float(loss.item())
            n_batches += 1
    return total / max(n_batches, 1)


def fetch_prices(ticker: str, interval: str) -> np.ndarray:
    period = "max" if interval == "1d" else "730d"
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data["Close"].dropna().values.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="ICF continued pretraining on crypto")
    parser.add_argument("--ticker", type=str, default="BTC-USD")
    parser.add_argument("--interval", type=str, default="1d", choices=["1h", "1d"])
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=30)
    parser.add_argument("--k", type=int, default=10, dest="k_examples")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--results_dir", type=str, default="results/icf_trained")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_nope", action="store_true", default=False,
                        help="Disable RoPE during training (train in NoPE mode)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    prices = fetch_prices(args.ticker, args.interval)
    n = len(prices)
    test_size = int(n * args.test_fraction)
    val_size = int(n * 0.15)
    train_size = n - test_size - val_size
    train_prices = prices[:train_size]
    val_start = max(0, train_size - args.context_len)
    val_prices = prices[val_start : train_size + val_size]

    print(f"Train {len(train_prices):,}  Val {len(val_prices):,}  Test reserved {test_size:,}")

    icf = TimesFM_ICF_torch.from_pretrained_base("google/timesfm-2.5-200m-pytorch")
    icf.model.to(device)
    icf.model.train(True)

    example_len = _round_up_patch(args.context_len + args.pred_len, icf.model.p)
    max_horizon = _round_up_patch(args.pred_len, icf.model.o)
    fc = timesfm.ForecastConfig(
        max_context=args.context_len,
        max_horizon=max_horizon,
        normalize_inputs=True,
        per_core_batch_size=args.batch_size,
        use_continuous_quantile_head=False,
        force_flip_invariance=True,
    )
    icf.compile(
        fc,
        icf_config=ICFConfig(k_examples=args.k_examples, example_len=example_len, use_nope=args.use_nope),
    )
    mode_tag = "nope" if args.use_nope else "rope"
    print(f"Positional encoding during training: {'NoPE (disabled)' if args.use_nope else 'RoPE (enabled)'}")

    train_ds = ICFCryptoDataset(
        train_prices, args.context_len, args.pred_len, args.k_examples, args.stride, args.seed
    )
    # Val windows on held-out slice; example pool from train-only (no leakage, enough segments before each window).
    val_ds = ICFCryptoDataset(
        val_prices,
        args.context_len,
        args.pred_len,
        args.k_examples,
        args.stride,
        args.seed + 1,
        pool_prices=train_prices,
        window_global_offset=val_start,
    )
    print(f"Train windows: {len(train_ds):,}  Val windows: {len(val_ds):,}")
    if len(train_ds) == 0:
        raise SystemExit("No training windows — increase data or reduce constraints.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_icf,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_icf,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(icf.model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.results_dir, f"{args.ticker.replace('-', '_').lower()}_icf_{mode_tag}.pt"
    )

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(
            icf, train_loader, optimizer, args.pred_len, device, args.grad_accum
        )
        vl = validate(icf, val_loader, args.pred_len, device)
        scheduler.step()
        train_losses.append(tl)
        val_losses.append(vl)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={tl:.6f}  val={vl:.6f}  {time.time()-t0:.1f}s")
        if vl < best_val:
            best_val, best_epoch = vl, epoch
            no_improve = 0
            torch.save({"model": icf.model.state_dict()}, ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    results = {
        "ticker": args.ticker,
        "use_nope": args.use_nope,
        "mode": mode_tag,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint": ckpt_path,
    }
    with open(os.path.join(args.results_dir, f"icf_train_results_{mode_tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
