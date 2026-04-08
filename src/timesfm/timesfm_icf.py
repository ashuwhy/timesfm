"""TimesFM-ICF (In-Context Fine-Tuning) implementation (PyTorch).

Implements the core architectural changes described in
“In-Context Fine-Tuning for Time-Series Foundation Models” (arXiv:2410.24087):
- Learnable separator token inserted after each in-context example
- Cross-example causal attention (achieved by concatenation order)

This repo-specific implementation intentionally KEEPS RoPE (rotary positional
embeddings) and does NOT implement NoPE.

Note: Without continued-pretraining on prompts containing in-context examples,
this model may not improve accuracy over the base checkpoint, but it enables the
architecture and benchmarking pipeline.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import nn

from . import configs
from .timesfm_2p5 import timesfm_2p5_base
from .timesfm_2p5 import timesfm_2p5_torch
from .torch import dense, transformer, util


_TOL = 1e-6


@dataclass(frozen=True)
class ICFConfig:
  """Configuration for ICF prompting at inference time."""

  k_examples: int = 10
  example_len: int = 0  # if 0, examples are padded/truncated per-example


def _to_1d_np(x: Sequence[float] | np.ndarray) -> np.ndarray:
  arr = np.asarray(x)
  if arr.ndim != 1:
    raise ValueError(f"Expected 1D array, got shape {arr.shape}")
  return arr.astype(np.float64)


def _left_pad_to_multiple(x: torch.Tensor, mask: torch.Tensor, multiple: int) -> tuple[torch.Tensor, torch.Tensor]:
  """Left-pad x/mask so length is a multiple of `multiple`.

  The base TimesFM implementation left-pads so patching aligns.
  """
  length = x.shape[-1]
  pad = multiple - (length % multiple)
  if pad < multiple:
    pad_vals = torch.zeros(*x.shape[:-1], pad, dtype=x.dtype, device=x.device)
    pad_mask = torch.ones(*mask.shape[:-1], pad, dtype=torch.bool, device=mask.device)
    x = torch.cat([pad_vals, x], dim=-1)
    mask = torch.cat([pad_mask, mask], dim=-1)
  return x, mask


def _patchify(x: torch.Tensor, mask: torch.Tensor, patch_len: int) -> tuple[torch.Tensor, torch.Tensor]:
  """Reshape (B, T) into (B, N, P). Assumes T % P == 0."""
  b, t = x.shape
  if t % patch_len != 0:
    raise ValueError("Input length must be a multiple of patch_len")
  return x.reshape(b, -1, patch_len), mask.reshape(b, -1, patch_len)


def _compute_revin_running_stats(
  patched_inputs: torch.Tensor,
  patched_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute per-patch running mu/sigma (as in base decode).

  Returns:
    context_mu: (B, N)
    context_sigma: (B, N)
    last_n: (B,)
    last_mu: (B,)
    last_sigma: (B,)
  """
  b, n_patches, _ = patched_inputs.shape
  n = torch.zeros(b, device=patched_inputs.device)
  mu = torch.zeros(b, device=patched_inputs.device)
  sigma = torch.zeros(b, device=patched_inputs.device)
  patch_mu = []
  patch_sigma = []
  for i in range(n_patches):
    (n, mu, sigma), _ = util.update_running_stats(
      n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
    )
    patch_mu.append(mu)
    patch_sigma.append(sigma)
  context_mu = torch.stack(patch_mu, dim=1)
  context_sigma = torch.stack(patch_sigma, dim=1)
  return context_mu, context_sigma, n, mu, sigma


class TimesFM_ICF_2p5_200M_torch_module(nn.Module):
  """TimesFM 2.5 200M with ICF separator token and ICF-aware decoding."""

  config = timesfm_2p5_base.TimesFM_2p5_200M_Definition()

  def __init__(self):
    super().__init__()

    self.p = self.config.input_patch_len  # 32
    self.o = self.config.output_patch_len  # 128
    self.os = self.config.output_quantile_len  # 1024
    self.m = self.o // self.p  # 4
    self.x = self.config.stacked_transformers.num_layers  # 20
    self.h = self.config.stacked_transformers.transformer.num_heads  # 16
    self.md = self.config.stacked_transformers.transformer.model_dims  # 1280
    self.hd = self.md // self.h
    self.q = len(self.config.quantiles) + 1  # 10
    self.aridx = self.config.decode_index  # 5

    # Base layers.
    self.tokenizer = dense.ResidualBlock(self.config.tokenizer)
    self.stacked_xf = nn.ModuleList(
      [transformer.Transformer(self.config.stacked_transformers.transformer) for _ in range(self.x)]
    )
    self.output_projection_point = dense.ResidualBlock(self.config.output_projection_point)
    self.output_projection_quantiles = dense.ResidualBlock(self.config.output_projection_quantiles)

    # Separator token embedding (paper: a common learnable separator token σ).
    self.sep_token = nn.Parameter(torch.zeros(self.md))

  def load_from_base_state_dict(self, base_state: dict[str, torch.Tensor]) -> None:
    """Load weights from base TimesFM module; ignores separator token."""
    # Only load overlapping keys.
    own = self.state_dict()
    filtered = {k: v for k, v in base_state.items() if k in own and own[k].shape == v.shape}
    missing, unexpected = self.load_state_dict(filtered, strict=False)
    if unexpected:
      logging.info("ICF: unexpected keys when loading base: %s", unexpected)
    # Missing is expected for sep_token.

  def _tokenize_patches(
    self,
    patched_inputs: torch.Tensor,
    patched_masks: torch.Tensor,
    *,
    context_mu: torch.Tensor,
    context_sigma: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """RevIN-normalize, then tokenize patches into embeddings.

    Returns:
      embeddings: (B, N, md)
      patch_mask: (B, N) boolean patch mask
    """
    normed = util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed = torch.where(patched_masks, 0.0, normed)
    tokenizer_inputs = torch.cat([normed, patched_masks.to(normed.dtype)], dim=-1)
    embeddings = self.tokenizer(tokenizer_inputs)
    patch_mask = patched_masks[..., -1]
    return embeddings, patch_mask

  def _forward_embeddings(
    self,
    input_embeddings: torch.Tensor,
    patch_mask: torch.Tensor,
    decode_caches: list[util.DecodeCache] | None = None,
  ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], list[util.DecodeCache]]:
    if decode_caches is None:
      decode_caches = [None] * self.x

    output_embeddings = input_embeddings
    new_decode_caches: list[util.DecodeCache] = []
    for i, layer in enumerate(self.stacked_xf):
      output_embeddings, new_cache = layer(output_embeddings, patch_mask, decode_caches[i])
      new_decode_caches.append(new_cache)

    output_ts = self.output_projection_point(output_embeddings)
    output_quantile_spread = self.output_projection_quantiles(output_embeddings)

    return (input_embeddings, output_embeddings, output_ts, output_quantile_spread), new_decode_caches

  def _make_decode_caches(self, batch_size: int, decode_cache_size: int, device: torch.device) -> list[util.DecodeCache]:
    return [
      util.DecodeCache(
        next_index=torch.zeros(batch_size, dtype=torch.int32, device=device),
        num_masked=torch.zeros(batch_size, dtype=torch.int32, device=device),
        key=torch.zeros(batch_size, decode_cache_size, self.h, self.hd, device=device),
        value=torch.zeros(batch_size, decode_cache_size, self.h, self.hd, device=device),
      )
      for _ in range(self.x)
    ]

  def decode_icf(
    self,
    horizon: int,
    *,
    example_inputs: list[torch.Tensor],
    example_masks: list[torch.Tensor],
    target_inputs: torch.Tensor,
    target_masks: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode with in-context examples.

    Args:
      horizon: forecast horizon in timepoints.
      example_inputs: list of (B, T_ex) float tensors (already padded to fixed length).
      example_masks: list of (B, T_ex) bool tensors (True = masked/padded).
      target_inputs: (B, T_ctx) float tensor.
      target_masks: (B, T_ctx) bool tensor.

    Returns:
      point_forecast: (B, horizon)
      quantile_forecast: (B, horizon, q)
    """
    device = target_inputs.device
    batch_size = target_inputs.shape[0]

    # Compute how many decode steps we need beyond the first output patch.
    num_decode_steps = (horizon - 1) // self.o

    # Build prefill embeddings = [ex1 tokens, SEP, ex2 tokens, SEP, ..., target tokens]
    prefill_embeddings: list[torch.Tensor] = []
    prefill_patch_masks: list[torch.Tensor] = []

    # Examples: independent RevIN.
    for ex_inp, ex_m in zip(example_inputs, example_masks):
      patched_inp, patched_m = _patchify(ex_inp, ex_m, self.p)
      ex_mu, ex_sigma, _, _, _ = _compute_revin_running_stats(patched_inp, patched_m)
      ex_emb, ex_patch_mask = self._tokenize_patches(patched_inp, patched_m, context_mu=ex_mu, context_sigma=ex_sigma)
      prefill_embeddings.append(ex_emb)
      prefill_patch_masks.append(ex_patch_mask)

      # Separator token: always unmasked.
      prefill_embeddings.append(self.sep_token.view(1, 1, -1).expand(batch_size, 1, -1))
      prefill_patch_masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device))

    # Target: compute running stats on context; keep last_n/mu/sigma for AR continuation.
    target_patched_inp, target_patched_m = _patchify(target_inputs, target_masks, self.p)
    tgt_mu_seq, tgt_sigma_seq, tgt_last_n, tgt_last_mu, tgt_last_sigma = _compute_revin_running_stats(
      target_patched_inp, target_patched_m
    )
    tgt_emb, tgt_patch_mask = self._tokenize_patches(
      target_patched_inp, target_patched_m, context_mu=tgt_mu_seq, context_sigma=tgt_sigma_seq
    )
    prefill_embeddings.append(tgt_emb)
    prefill_patch_masks.append(tgt_patch_mask)

    prefill_embeddings_t = torch.cat(prefill_embeddings, dim=1)
    prefill_patch_mask_t = torch.cat(prefill_patch_masks, dim=1)

    # Decode cache size: prefill tokens + future AR tokens.
    prefill_tokens = prefill_embeddings_t.shape[1]
    decode_cache_size = prefill_tokens + num_decode_steps * self.m
    decode_caches = self._make_decode_caches(batch_size, decode_cache_size, device)

    # Prefill forward pass (populates caches).
    (_, _, normed_outputs, normed_quantile_spread), decode_caches = self._forward_embeddings(
      prefill_embeddings_t, prefill_patch_mask_t, decode_caches
    )

    # Extract the last token (corresponding to the last target context patch).
    last_normed_out = normed_outputs[:, -1, :]  # (B, md)
    last_normed_qspread = normed_quantile_spread[:, -1, :]  # (B, 10240)

    # Denormalize using *target* last running stats.
    renormed_last = util.revin(last_normed_out, tgt_last_mu, tgt_last_sigma, reverse=True)
    renormed_last = renormed_last.reshape(batch_size, self.o, self.q)

    renormed_last_qspread = util.revin(last_normed_qspread, tgt_last_mu, tgt_last_sigma, reverse=True)
    renormed_last_qspread = renormed_last_qspread.reshape(batch_size, self.os, self.q)

    # Point forecast uses the decode quantile index (median by default).
    pf_first = renormed_last[:, :, self.aridx]  # (B, o)

    # Autoregressive continuation beyond the first output patch.
    ar_outputs: list[torch.Tensor] = []
    last_renormed_output = pf_first

    last_n, last_mu, last_sigma = tgt_last_n, tgt_last_mu, tgt_last_sigma

    for _ in range(num_decode_steps):
      # Feed last predicted output patch as next inputs, split into m patches.
      new_patched_input = last_renormed_output.reshape(batch_size, self.m, self.p)
      new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool, device=device)

      n, mu, sigma = last_n, last_mu, last_sigma
      new_mus, new_sigmas = [], []
      for i in range(self.m):
        (n, mu, sigma), _ = util.update_running_stats(n, mu, sigma, new_patched_input[:, i], new_mask[:, i])
        new_mus.append(mu)
        new_sigmas.append(sigma)
      last_n, last_mu, last_sigma = n, mu, sigma
      new_mu = torch.stack(new_mus, dim=1)
      new_sigma = torch.stack(new_sigmas, dim=1)

      new_normed_input = util.revin(new_patched_input, new_mu, new_sigma, reverse=False)
      tok_inputs = torch.cat([new_normed_input, new_mask.to(new_normed_input.dtype)], dim=-1)
      new_embeddings = self.tokenizer(tok_inputs)
      new_patch_mask = new_mask[..., -1]

      (_, _, new_normed_output, _), decode_caches = self._forward_embeddings(
        new_embeddings, new_patch_mask, decode_caches
      )

      new_renormed_output = util.revin(new_normed_output, new_mu, new_sigma, reverse=True)
      new_renormed_output = new_renormed_output.reshape(batch_size, self.m, self.o, self.q)

      ar_outputs.append(new_renormed_output[:, -1, :, :])
      last_renormed_output = new_renormed_output[:, -1, :, self.aridx]

    # Assemble full forecast (point + quantiles).
    point_patches = [pf_first]
    quantile_patches = [renormed_last]
    if ar_outputs:
      ar_stack = torch.stack(ar_outputs, dim=1)  # (B, steps, o, q)
      # Flatten to time, take median for points.
      ar_time = ar_stack.reshape(batch_size, -1, self.q)
      point_patches.append(ar_time[:, :, self.aridx])
      quantile_patches.append(ar_time.reshape(batch_size, -1, self.q).reshape(batch_size, -1, self.q))

    point_full = torch.cat(point_patches, dim=1)[:, :horizon]

    # Quantiles: first patch has (B, o, q); AR has (B, steps*o, q)
    if ar_outputs:
      quant_first = renormed_last  # (B, o, q)
      quant_ar = ar_stack.reshape(batch_size, -1, self.q)
      quant_full = torch.cat([quant_first, quant_ar], dim=1)[:, :horizon, :]
    else:
      quant_full = renormed_last[:, :horizon, :]

    # Note: renormed_last_qspread is computed for completeness; not currently returned.
    _ = renormed_last_qspread

    return point_full, quant_full


class TimesFM_ICF_torch:
  """User-facing wrapper for TimesFM-ICF using the base TimesFM 2.5 checkpoint."""

  def __init__(self):
    self.model = TimesFM_ICF_2p5_200M_torch_module()
    self.forecast_config: configs.ForecastConfig | None = None
    self.icf_config: ICFConfig | None = None
    self.global_batch_size: int = 0

  @classmethod
  def from_pretrained_base(
    cls,
    model_id: str = "google/timesfm-2.5-200m-pytorch",
  ) -> "TimesFM_ICF_torch":
    base = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    instance = cls()
    instance.model.load_from_base_state_dict(base.model.state_dict())
    # Put on same device as base module.
    instance.model.to(base.model.device)
    instance.model.eval()
    return instance

  def compile(
    self,
    forecast_config: configs.ForecastConfig,
    *,
    icf_config: ICFConfig | None = None,
  ) -> None:
    # Mirror base compile constraints where applicable.
    fc = forecast_config

    if fc.max_context % self.model.p != 0:
      logging.info(
        "ICF compile: max_context must be multiple of patch size %d; rounding up.",
        self.model.p,
      )
      fc = dataclasses.replace(fc, max_context=math.ceil(fc.max_context / self.model.p) * self.model.p)

    if fc.max_horizon % self.model.o != 0:
      logging.info(
        "ICF compile: max_horizon must be multiple of output patch size %d; rounding up.",
        self.model.o,
      )
      fc = dataclasses.replace(fc, max_horizon=math.ceil(fc.max_horizon / self.model.o) * self.model.o)

    if icf_config is None:
      icf_config = ICFConfig()

    if icf_config.k_examples < 0:
      raise ValueError("k_examples must be >= 0")

    if icf_config.k_examples > 0 and icf_config.example_len <= 0:
      raise ValueError("ICF compile requires example_len > 0 when k_examples > 0")

    if icf_config.k_examples == 0:
      # example_len is irrelevant when there are no examples.
      icf_config = dataclasses.replace(icf_config, example_len=0)

    if icf_config.example_len > 0 and icf_config.example_len % self.model.p != 0:
      logging.info(
        "ICF compile: example_len must be multiple of patch size %d; rounding up.",
        self.model.p,
      )
      icf_config = dataclasses.replace(
        icf_config,
        example_len=math.ceil(icf_config.example_len / self.model.p) * self.model.p,
      )

    total_prompt = icf_config.k_examples * icf_config.example_len + fc.max_context + fc.max_horizon
    if total_prompt > self.model.config.context_limit:
      raise ValueError(
        "ICF prompt too long for model context_limit. "
        f"k_examples*example_len + max_context + max_horizon = {total_prompt} > {self.model.config.context_limit}. "
        "Reduce k_examples, example_len, max_context, or max_horizon."
      )

    self.forecast_config = fc
    self.icf_config = icf_config

    # Simple global batch size (single device for now).
    self.global_batch_size = fc.per_core_batch_size

  def forecast_icf(
    self,
    *,
    horizon: int,
    context_examples: list[list[np.ndarray]],
    target_inputs: list[np.ndarray],
  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecast with in-context examples.

    Args:
      horizon: forecast horizon.
      context_examples: list of length B; each entry is a list of example series.
      target_inputs: list of length B; each entry is the target history series.

    Returns:
      point_forecast: (B, horizon)
      quantile_forecast: (B, horizon, q)
    """
    if self.forecast_config is None or self.icf_config is None:
      raise RuntimeError("Model is not compiled. Call compile() first.")

    if len(target_inputs) != len(context_examples):
      raise ValueError("context_examples and target_inputs must have same length")

    fc = self.forecast_config
    icf = self.icf_config

    # Pad batch to global_batch_size.
    num_inputs = len(target_inputs)
    if (w := num_inputs % self.global_batch_size) != 0:
      pad_n = self.global_batch_size - w
      target_inputs = target_inputs + [np.array([0.0, 0.0, 0.0])] * pad_n
      context_examples = context_examples + [[np.array([0.0, 0.0, 0.0]) for _ in range(icf.k_examples)]] * pad_n

    point_outs: list[np.ndarray] = []
    quant_outs: list[np.ndarray] = []

    # Process in fixed-size batches.
    for start in range(0, len(target_inputs), self.global_batch_size):
      end = start + self.global_batch_size

      batch_targets = target_inputs[start:end]
      batch_examples = context_examples[start:end]

      # Prepare target tensors (B, max_context)
      tgt_vals = []
      tgt_masks = []
      for arr in batch_targets:
        v = timesfm_2p5_base.linear_interpolation(timesfm_2p5_base.strip_leading_nans(_to_1d_np(arr)))
        if len(v) >= fc.max_context:
          v = v[-fc.max_context:]
          m = np.zeros_like(v, dtype=bool)
        else:
          m = np.array([True] * (fc.max_context - len(v)) + [False] * len(v))
          v = np.pad(v, (fc.max_context - len(v), 0), constant_values=0.0)
        tgt_vals.append(v)
        tgt_masks.append(m)

      tgt_t = torch.from_numpy(np.stack(tgt_vals)).to(torch.float32).to(self.model.sep_token.device)
      tgt_m = torch.from_numpy(np.stack(tgt_masks)).to(torch.bool).to(self.model.sep_token.device)

      # Prepare examples as fixed-length tensors (B, example_len)
      ex_inputs: list[torch.Tensor] = []
      ex_masks: list[torch.Tensor] = []
      for ex_i in range(icf.k_examples):
        vals = []
        masks = []
        for ex_list in batch_examples:
          ex_arr = ex_list[ex_i] if ex_i < len(ex_list) else np.array([0.0, 0.0, 0.0])
          v = timesfm_2p5_base.linear_interpolation(timesfm_2p5_base.strip_leading_nans(_to_1d_np(ex_arr)))
          if len(v) >= icf.example_len:
            v = v[-icf.example_len:]
            m = np.zeros_like(v, dtype=bool)
          else:
            m = np.array([True] * (icf.example_len - len(v)) + [False] * len(v))
            v = np.pad(v, (icf.example_len - len(v), 0), constant_values=0.0)
          vals.append(v)
          masks.append(m)
        ex_inputs.append(torch.from_numpy(np.stack(vals)).to(torch.float32).to(tgt_t.device))
        ex_masks.append(torch.from_numpy(np.stack(masks)).to(torch.bool).to(tgt_t.device))

      # Ensure patch-aligned lengths (left pad to multiple of p).
      # examples and targets are already fixed lengths, but they might not be multiples of p.
      def _align(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x2, m2 = _left_pad_to_multiple(x, m, self.model.p)
        # If padding happened, length increased; truncate back to a multiple of p at most the configured len.
        # For simplicity, we allow the extra left pad (still <= context_limit check uses timepoints, not tokens).
        return x2, m2

      aligned_ex_inputs, aligned_ex_masks = [], []
      for x, m in zip(ex_inputs, ex_masks):
        x2, m2 = _align(x, m)
        aligned_ex_inputs.append(x2)
        aligned_ex_masks.append(m2)

      tgt_t2, tgt_m2 = _align(tgt_t, tgt_m)

      with torch.no_grad():
        point_t, quant_t = self.model.decode_icf(
          horizon,
          example_inputs=aligned_ex_inputs,
          example_masks=aligned_ex_masks,
          target_inputs=tgt_t2,
          target_masks=tgt_m2,
        )

      point_outs.append(point_t.detach().cpu().numpy())
      quant_outs.append(quant_t.detach().cpu().numpy())

    point = np.concatenate(point_outs, axis=0)[:num_inputs]
    quant = np.concatenate(quant_outs, axis=0)[:num_inputs]
    return point, quant
