#!/usr/bin/env python3
"""
plot_anytime_convergence.py — Anytime convergence: best PSNR @ target bpf vs wall-time.

각 평가 시점에서 "지금까지 본 alloc 중 bpf ≤ B_target 만족 + 최고 PSNR" 을
시간축(또는 #eval 축) 위에 그린다. RDO 와 sweep 두 곡선이 같은 plateau 에
도달하는데, RDO 가 훨씬 일찍 도달함을 보여준다.

Usage
-----
  python plot_anytime_convergence.py \
      --rdo   output_rd/bonsai_rdo_anytime.json \
      --sweep output_rd/bonsai_sweep.json \
      --target_bpf 5.0 \
      --out   output_rd/bonsai_anytime.png
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cumulative_best(eval_log, target_bpf):
    """
    Returns lists (elapsed, best_psnr) where best_psnr at time t =
    max psnr over evals up to t with bpf ≤ target_bpf. None until first valid.
    """
    elapsed, best = [], []
    cur = -math.inf
    for e in eval_log:
        if e["bpf"] <= target_bpf and e["psnr"] > cur:
            cur = e["psnr"]
        elapsed.append(e["elapsed"])
        best.append(cur if cur > -math.inf else np.nan)
    return np.array(elapsed), np.array(best)


def plateau_value(rd_curve, target_bpf):
    """RD curve 의 (bpf ≤ target_bpf) 영역 최고 PSNR — 두 방법 모두에서 ground truth."""
    valid = [p["psnr"] for p in rd_curve if p["bpf"] <= target_bpf]
    return max(valid) if valid else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdo", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--target_bpf", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xaxis", choices=["time", "eval"], default="time",
                    help="x축: wall-time(sec) 또는 #eval")
    ap.add_argument("--logx", action="store_true",
                    help="x축 로그 스케일")
    args = ap.parse_args()

    rdo = json.load(open(args.rdo))
    sw = json.load(open(args.sweep))

    rdo_log = rdo["eval_log"]
    sw_log = sw["eval_log"]

    rdo_t, rdo_best = cumulative_best(rdo_log, args.target_bpf)
    sw_t, sw_best = cumulative_best(sw_log, args.target_bpf)

    rdo_plateau = plateau_value(rdo["rd_curve"], args.target_bpf)
    sw_plateau = plateau_value(sw["rd_curve"], args.target_bpf)
    baseline = rdo.get("baseline", {}).get("psnr") or sw.get("baseline", {}).get("psnr")

    if args.xaxis == "eval":
        rdo_x = np.arange(1, len(rdo_log) + 1)
        sw_x = np.arange(1, len(sw_log) + 1)
        xlabel = "# evaluations"
    else:
        rdo_x = rdo_t
        sw_x = sw_t
        xlabel = "Wall-time (s)"

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(sw_x, sw_best, color="steelblue", linewidth=1.8,
            label=f"Sweep ({len(sw_log)} evals, {sw_t[-1]:.0f}s)", drawstyle="steps-post")
    ax.plot(rdo_x, rdo_best, color="crimson", linewidth=2.2,
            label=f"RDO ({len(rdo_log)} evals, {rdo_t[-1]:.0f}s)", drawstyle="steps-post")

    # 마지막 점 강조
    ax.scatter([rdo_x[-1]], [rdo_best[-1]], color="crimson", s=70, zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.scatter([sw_x[-1]], [sw_best[-1]], color="steelblue", s=70, zorder=5,
               edgecolor="white", linewidth=1.2)

    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.0,
                   label=f"Baseline (NPZ) {baseline:.2f} dB")

    if args.logx:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f"Best PSNR @ bpf ≤ {args.target_bpf:.1f} (dB)", fontsize=12)
    ax.set_title(f"Anytime convergence — {rdo.get('dataset', 'scene')}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    # 속도 차이 어노테이션
    speedup_t = sw_t[-1] / max(rdo_t[-1], 1e-9)
    speedup_n = len(sw_log) / max(len(rdo_log), 1)
    gap_db = abs((rdo_plateau or 0) - (sw_plateau or 0))
    txt = (f"Final PSNR gap: {gap_db:.3f} dB\n"
           f"Speedup (time): {speedup_t:.1f}×\n"
           f"Speedup (#eval): {speedup_n:.1f}×")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print(f"saved → {args.out}")
    print(f"  RDO   : {len(rdo_log)} evals, {rdo_t[-1]:.1f}s, best={rdo_best[-1]:.4f}")
    print(f"  Sweep : {len(sw_log)} evals, {sw_t[-1]:.1f}s, best={sw_best[-1]:.4f}")
    print(f"  PSNR gap: {gap_db:.4f} dB | Speedup: {speedup_t:.1f}× (time) / {speedup_n:.1f}× (#eval)")


if __name__ == "__main__":
    main()
