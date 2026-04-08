#!/usr/bin/env python3
"""TimesFM-ICF entrypoint.

Runs TimesFM-ICF (separator tokens + cross-example causal prompting) on existing
benchmarks in this repo.

This script keeps RoPE (rotary position embeddings) and does not implement NoPE.

Examples:
  # Monash (extended)
  python timesfm-icf.py monash --save_dir results/icf/monash --k 10

  # ETT
  python timesfm-icf.py ett --results_dir results/icf/ett --k 10

Notes:
- Benchmarks require extra deps (gluonts, utilsforecast, pandas, tqdm).
- Without continued-pretraining, ICF may not improve vs base.
"""

from __future__ import annotations

import argparse

import timesfm


def main() -> None:
  parser = argparse.ArgumentParser(description="TimesFM-ICF benchmark runner")
  sub = parser.add_subparsers(dest="cmd", required=True)

  p_monash = sub.add_parser("monash", help="Run Monash benchmark with ICF")
  p_monash.add_argument("--save_dir", default="results/icf/monash")
  p_monash.add_argument("--max_context", type=int, default=512)
  p_monash.add_argument("--k", type=int, default=10)
  p_monash.add_argument("--dataset", default=None, help="Optional: only run one dataset")
  p_monash.add_argument("--max_series", type=int, default=None, help="Optional: cap number of series")

  p_ett = sub.add_parser("ett", help="Run ETT benchmark with ICF")
  p_ett.add_argument("--results_dir", default="results/icf/ett")
  p_ett.add_argument("--context_len", type=int, default=512)
  p_ett.add_argument("--k", type=int, default=10)
  p_ett.add_argument("--dataset", default=None, help="Optional: only run one dataset (etth1/etth2/ettm1/ettm2)")
  p_ett.add_argument("--pred_len", type=int, default=None, help="Optional: only run one horizon (e.g. 96)")
  p_ett.add_argument("--max_windows", type=int, default=None, help="Optional: cap number of (variable,window) samples")

  args = parser.parse_args()

  if args.cmd == "monash":
    from benchmark.run_icf_monash import run_monash_icf_benchmark

    run_monash_icf_benchmark(
      save_dir=args.save_dir,
      max_context=args.max_context,
      k_examples=args.k,
      only_dataset=args.dataset,
      max_series=args.max_series,
    )
    return

  if args.cmd == "ett":
    from benchmark.run_icf_ett import run_ett_icf_benchmark

    run_ett_icf_benchmark(
      results_dir=args.results_dir,
      context_len=args.context_len,
      k_examples=args.k,
      only_dataset=args.dataset,
      only_pred_len=args.pred_len,
      max_windows=args.max_windows,
    )
    return


if __name__ == "__main__":
  main()
