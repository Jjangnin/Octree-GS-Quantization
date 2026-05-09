#!/usr/bin/env python3
"""
plot_anytime_all_scenes.py — 2×4 subplot figure of anytime convergence
across all scenes.

각 subplot: scene 한 개의 RDO vs sweep 수렴 곡선.
Y축 = best PSNR @ bpf ≤ target_bpf (어 None: scene별 자동).
X축 = wall-time (sec, log scale by default).

Usage
-----
  python plot_anytime_all_scenes.py \
      --indir output_rd \
      --target_bpf 5.2 \
      --out output_rd/all_scenes_anytime.png
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENES_DEFAULT = ["bicycle", "garden", "stump", "room", "counter", "kitchen", "bonsai"]


def cumulative_best(eval_log, target_bpf):
    elapsed, best = [], []
    cur = -math.inf
    for e in eval_log:
        if e["bpf"] <= target_bpf and e["psnr"] > cur:
            cur = e["psnr"]
        elapsed.append(e["elapsed"])
        best.append(cur if cur > -math.inf else np.nan)
    return np.array(elapsed), np.array(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="output_rd")
    ap.add_argument("--scenes", nargs="+", default=SCENES_DEFAULT)
    ap.add_argument("--target_bpf", type=float, default=5.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--linx", action="store_true",
                    help="x축 선형 스케일 (기본: log)")
    args = ap.parse_args()

    indir = Path(args.indir)
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7.5), sharey=False)
    axes = axes.flatten()

    summary = []  # (scene, rdo_t, sweep_t, rdo_best, sweep_best, gap, speedup_t, speedup_n)

    for i, scene in enumerate(args.scenes):
        ax = axes[i]
        rdo_p = indir / f"{scene}_rdo_anytime.json"
        sw_p  = indir / f"{scene}_sweep_anytime.json"
        if not (rdo_p.exists() and sw_p.exists()):
            ax.text(0.5, 0.5, f"{scene}\n(missing)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        rdo = json.load(open(rdo_p))
        sw  = json.load(open(sw_p))
        rdo_t, rdo_best = cumulative_best(rdo["eval_log"], args.target_bpf)
        sw_t,  sw_best  = cumulative_best(sw["eval_log"],  args.target_bpf)

        ax.plot(sw_t, sw_best, color="steelblue", linewidth=1.5,
                label=f"Sweep ({len(sw['eval_log'])} ev)", drawstyle="steps-post")
        ax.plot(rdo_t, rdo_best, color="crimson", linewidth=1.8,
                label=f"RDO ({len(rdo['eval_log'])} ev)", drawstyle="steps-post")
        ax.scatter([rdo_t[-1]], [rdo_best[-1]], color="crimson", s=35,
                   edgecolor="white", linewidth=0.8, zorder=5)
        ax.scatter([sw_t[-1]], [sw_best[-1]], color="steelblue", s=35,
                   edgecolor="white", linewidth=0.8, zorder=5)

        baseline = rdo.get("baseline", {}).get("psnr")
        if baseline is not None:
            ax.axhline(baseline, color="gray", linestyle=":", linewidth=0.8)

        if not args.linx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{scene}", fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=8)

        # 어노테이션
        gap = abs(rdo_best[-1] - sw_best[-1])
        sp_t = sw_t[-1] / max(rdo_t[-1], 1e-9)
        sp_n = len(sw["eval_log"]) / max(len(rdo["eval_log"]), 1)
        txt = (f"gap {gap:.3f} dB\n{sp_t:.0f}× faster\n({sp_n:.0f}× #eval)")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="lightgray", linewidth=0.5))
        ax.legend(loc="lower right", fontsize=8)
        summary.append((scene, rdo_t[-1], sw_t[-1], rdo_best[-1],
                        sw_best[-1], gap, sp_t, sp_n))

    # 8th cell: summary table
    if len(args.scenes) < rows * cols:
        ax = axes[rows * cols - 1]
        ax.axis("off")
        if summary:
            avg_gap = np.mean([s[5] for s in summary])
            avg_spt = np.mean([s[6] for s in summary])
            avg_spn = np.mean([s[7] for s in summary])
            txt = "Summary\n" + "─" * 22 + "\n"
            for s in summary:
                txt += f"{s[0]:<8} gap={s[5]:5.3f}  {s[6]:4.0f}×\n"
            txt += "─" * 22 + "\n"
            txt += f"{'avg':<8} gap={avg_gap:5.3f}  {avg_spt:4.0f}×"
            ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender",
                              alpha=0.6, edgecolor="gray"))

    # 공통 라벨
    fig.supxlabel("Wall-time (s)", fontsize=12)
    fig.supylabel(f"Best PSNR @ bpf ≤ {args.target_bpf:.1f} (dB)", fontsize=12)
    fig.suptitle(
        f"Anytime convergence: RDO vs brute-force sweep "
        f"(target bpf ≤ {args.target_bpf:.1f})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved → {args.out}")
    print(f"\nSummary @ target_bpf ≤ {args.target_bpf:.1f}:")
    print(f"  {'scene':<10}{'RDO_t(s)':>10}{'Sw_t(s)':>10}{'RDO_psnr':>11}{'Sw_psnr':>11}{'gap(dB)':>10}{'speed(t)':>10}")
    for s in summary:
        print(f"  {s[0]:<10}{s[1]:>10.1f}{s[2]:>10.1f}{s[3]:>11.4f}{s[4]:>11.4f}{s[5]:>10.4f}{s[6]:>9.1f}×")


if __name__ == "__main__":
    main()
