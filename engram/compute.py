"""Compute estimator for the experiment plan (issue #12).

Reproduces the formulas in research/experiment-plan.md so the assumptions can be changed in one
place and re-run. Replace the defaults with measured values as soon as real runs exist.

    engram-compute --base 8e9 --tokens 1e10 --ttt-overhead 2 --tflops 400 --price 2.5
"""

from __future__ import annotations

import argparse


def frozen_base_training_flops(n_base: float, tokens: float, n_memory: float = 0.0) -> float:
    """Forward plus activation-backward through a frozen base (~4·N·T), plus full training of the
    memory parameters (~6·N_mem·T)."""
    return 4 * n_base * tokens + 6 * n_memory * tokens


def lora_flops(n_base: float, tokens: float) -> float:
    """LoRA training is dominated by forward plus backward through the base (~6·N·T)."""
    return 6 * n_base * tokens


def gpu_hours(flops: float, tflops_sustained: float) -> float:
    return flops / (tflops_sustained * 1e12) / 3600


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", type=float, default=8e9, help="base parameters (default 8e9)")
    ap.add_argument("--memory", type=float, default=0.02, help="memory params as fraction of base")
    ap.add_argument("--tokens", type=float, default=1e10, help="Phase 1 training tokens")
    ap.add_argument("--ttt-overhead", type=float, nargs=2, default=(1.5, 3.0), metavar=("LO", "HI"))
    ap.add_argument("--retry-factor", type=float, default=2.0, help="ablations and failed runs")
    ap.add_argument("--tflops", type=float, default=400.0, help="sustained TFLOP/s per GPU")
    ap.add_argument("--price", type=float, nargs=2, default=(2.0, 3.0), metavar=("LO", "HI"))
    ap.add_argument("--sleep-tokens", type=float, default=2e5, help="rehearsal tokens per sleep")
    args = ap.parse_args(argv)

    flops = frozen_base_training_flops(args.base, args.tokens, args.memory * args.base)
    base_h = gpu_hours(flops, args.tflops)
    lo = base_h * args.ttt_overhead[0] * args.retry_factor
    hi = base_h * args.ttt_overhead[1] * args.retry_factor
    sleep_h = gpu_hours(lora_flops(args.base, args.sleep_tokens), args.tflops)

    print("ASSUMPTIONS — replace with measurements (issue #12)")
    print(f"  base {args.base:.3g} params, memory {args.memory:.1%}, {args.tokens:.3g} tokens")
    print(f"  {args.tflops:.0f} TFLOP/s sustained, ${args.price[0]}–{args.price[1]}/GPU-hour")
    print(f"\nPhase 1 training FLOPs:            {flops:.3g}")
    print(f"Single clean run:                  {base_h:,.0f} GPU-hours")
    print(
        f"With TTT overhead and retries:     {lo:,.0f}–{hi:,.0f} GPU-hours "
        f"(${lo * args.price[0]:,.0f}–${hi * args.price[1]:,.0f})"
    )
    print(
        f"One sleep (LoRA on {args.sleep_tokens:.3g} tokens): {sleep_h * 3600:,.1f} GPU-seconds "
        "(excluding rehearsal-data generation)"
    )


if __name__ == "__main__":
    main()
