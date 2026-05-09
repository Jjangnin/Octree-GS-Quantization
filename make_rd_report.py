#!/usr/bin/env python3
"""
make_rd_report.py — RD curve PNG + 마크다운 리포트 생성

rd_sweep.json (sweep 결과) 또는 compression_results.json (단일 결과) 모두 지원.

Usage:
  python make_rd_report.py --input <model_path>/rd_sweep.json \
                            --output <model_path>
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────
# Pareto frontier
# ────────────────────────────────────────────────────────────────

def pareto_frontier(points):
    """bpf 오름차순 정렬 기준, PSNR이 단조 증가하는 점만 남긴다."""
    sorted_pts = sorted(points, key=lambda r: r["bpf"])
    pareto, best = [], -float("inf")
    for r in sorted_pts:
        if r["psnr"] >= best:
            best = r["psnr"]
            pareto.append(r)
    return pareto


# ────────────────────────────────────────────────────────────────
# Sweep 결과 처리 (rd_sweep.json 포맷)
# ────────────────────────────────────────────────────────────────

def handle_sweep(data, output_dir):
    baseline = data["baseline"]
    rd_curve = data["rd_curve"]
    model_path = data.get("model_path", "")
    iteration = data.get("iteration", "?")
    name = os.path.basename(model_path.rstrip("/"))

    pareto = pareto_frontier(rd_curve)

    # ── RD curve PNG ────────────────────────────────────────────
    xs = [r["bpf"] for r in rd_curve]
    ys = [r["psnr"] for r in rd_curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.35, s=18, color="steelblue", label="All allocations")
    ax.plot([r["bpf"] for r in pareto], [r["psnr"] for r in pareto],
            "r-o", linewidth=2, markersize=5, label="Pareto frontier", zorder=5)
    ax.axhline(baseline["psnr"], linestyle="--", color="gray", linewidth=1.2,
               label=f"Baseline {baseline['psnr']:.2f} dB")

    ax.set_xlabel("Avg bits per feature element (bpf)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title(f"Rate-Distortion — {name} (iter {iteration})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    rd_path = os.path.join(output_dir, "rd_curve.png")
    plt.savefig(rd_path, dpi=150)
    plt.close()
    print(f"RD curve 저장 → {rd_path}")

    # ── 마크다운 리포트 ──────────────────────────────────────────
    md_lines = [
        f"# {name} — RD Sweep Report (iter {iteration})\n",
        f"- Baseline PSNR: **{baseline['psnr']:.4f} dB** | SSIM: {baseline['ssim']:.6f}",
        f"- 탐색 포인트: {len(rd_curve)}개 | Pareto: {len(pareto)}개\n",
        "## Pareto Frontier",
        "| bpf | PSNR (dB) | SSIM | Size (MB) | PSNR drop | Allocation |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for r in pareto:
        alloc_str = ", ".join(f"L{k}={v}b" for k, v in r["allocation"].items())
        md_lines.append(
            f"| {r['bpf']:.2f} | {r['psnr']:.4f} | {r['ssim']:.6f} "
            f"| {r['size_mb']:.2f} | {r['psnr_drop']:+.4f} | {alloc_str} |"
        )

    md_path = os.path.join(output_dir, "rd_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"MD 리포트 저장 → {md_path}")

    # 터미널 요약
    best = min(pareto, key=lambda r: r["psnr_drop"])
    print(f"\n=== 요약 ===")
    print(f"  Baseline : PSNR {baseline['psnr']:.4f} dB")
    print(f"  Best(PSNR drop 최소): {best['psnr']:.4f} dB "
          f"| drop {best['psnr_drop']:+.4f} dB "
          f"| bpf {best['bpf']:.2f} "
          f"| alloc {best['allocation']}")


# ────────────────────────────────────────────────────────────────
# 단일 실험 결과 처리 (compression_results.json 포맷, 하위 호환)
# ────────────────────────────────────────────────────────────────

def handle_single(data, output_dir):
    baseline = data["baseline"]
    exp_a = data["experiment_A"]
    model_path = data.get("model_path", "")
    iteration = data.get("iteration", "?")
    name = os.path.basename(model_path.rstrip("/"))

    # PNG
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([baseline["size_mb"]], [baseline["psnr"]],
               marker="o", s=80, color="steelblue", label="Baseline (float32)", zorder=5)
    ax.scatter([exp_a["size_mb"]], [exp_a["psnr"]],
               marker="^", s=80, color="tomato", label="A codec (LOD-quant)", zorder=5)
    ax.annotate("", xy=(exp_a["size_mb"], exp_a["psnr"]),
                xytext=(baseline["size_mb"], baseline["psnr"]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    for x, y, label in [
        (baseline["size_mb"], baseline["psnr"],
         f"Baseline\n{baseline['psnr']:.2f} dB\n{baseline['size_mb']:.1f} MB"),
        (exp_a["size_mb"], exp_a["psnr"],
         f"A codec\n{exp_a['psnr']:.2f} dB\n{exp_a['size_mb']:.1f} MB"),
    ]:
        ax.annotate(label, xy=(x, y), xytext=(10, 6), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Size (MB)", fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.set_title(f"Rate-Distortion — {name}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    rd_path = os.path.join(output_dir, "rd_curve.png")
    plt.savefig(rd_path, dpi=150)
    plt.close()
    print(f"RD curve 저장 → {rd_path}")

    # MD
    md_lines = [
        f"# {name} — Compression Report (iter {iteration})\n",
        "| Model | PSNR (dB) | SSIM | Size (MB) | Ratio |",
        "|---|---:|---:|---:|---:|",
        "| Baseline (float32) | {:.4f} | {:.6f} | {:.2f} | 1.00x |".format(
            baseline["psnr"], baseline.get("ssim") or 0.0, baseline["size_mb"]),
        "| A codec (LOD-adaptive) | {:.4f} | {:.6f} | {:.2f} | {:.2f}x |".format(
            exp_a["psnr"], exp_a["ssim"], exp_a["size_mb"], exp_a["ratio"]),
        "",
        f"- PSNR drop : {exp_a['psnr_drop']:+.4f} dB",
        f"- Size reduction : {100*(1 - exp_a['size_mb']/baseline['size_mb']):.1f}%",
    ]

    md_path = os.path.join(output_dir, "compression_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"MD 저장 → {md_path}")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="rd_sweep.json 또는 compression_results.json 경로")
    parser.add_argument("--output", "-o", required=True, help="결과 저장 폴더")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    os.makedirs(args.output, exist_ok=True)

    if "rd_curve" in data:
        handle_sweep(data, args.output)
    elif "experiment_A" in data:
        handle_single(data, args.output)
    else:
        raise ValueError("지원하지 않는 JSON 포맷: 'rd_curve' 또는 'experiment_A' 키가 필요합니다.")


if __name__ == "__main__":
    main()
