#!/usr/bin/env python3
"""
compare_fixed_bpf.py
고정 bpf에서 여러 데이터셋의 baseline vs compressed PSNR을 비교한다.
sweep JSON에서 target_bpf에 가장 가까운 Pareto 점을 찾아 표로 출력한다.

Usage:
  python compare_fixed_bpf.py \
      --jsons output_rd/bits3468/garden_30k_compress_result.json \
              output_rd/bits3468/bonsai_30k_compress_result.json \
              output_rd/bits3468/bartender_30k_compress_result.json \
      --target_bpf 5.0 \
      --output output_rd/compare_bpf5.png
"""

import json, argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pareto_frontier(rd_results):
    sorted_pts = sorted(rd_results, key=lambda r: r["bpf"])
    frontier, best = [], -float("inf")
    for r in sorted_pts:
        if r["psnr"] > best:
            best = r["psnr"]
            frontier.append(r)
    return frontier


def find_nearest_bpf(rd_results, target_bpf, use_pareto=True):
    """target_bpf에 가장 가까운 점 반환 (Pareto frontier 위에서)"""
    pts = pareto_frontier(rd_results) if use_pareto else rd_results
    return min(pts, key=lambda r: abs(r["bpf"] - target_bpf))


def make_comparison_table(rows, target_bpf, out_path):
    """
    rows: list of dict with keys:
        dataset, baseline_psnr, compressed_psnr, psnr_drop,
        actual_bpf, allocation, ply_mb, npz_mb
    """
    col_labels = ["Dataset", "Baseline PSNR", "Compressed PSNR",
                  "PSNR drop", "actual bpf", "PLY (MB)", "NPZ (MB)", "압축률"]

    cell_text = []
    for r in rows:
        ratio = r["ply_mb"] / r["npz_mb"] if r["npz_mb"] > 0 else 0
        cell_text.append([
            r["dataset"],
            f"{r['baseline_psnr']:.4f} dB",
            f"{r['compressed_psnr']:.4f} dB",
            f"{r['psnr_drop']:+.4f} dB",
            f"{r['actual_bpf']:.2f}",
            f"{r['ply_mb']:.1f}",
            f"{r['npz_mb']:.1f}",
            f"{ratio:.2f}×",
        ])

    fig, ax = plt.subplots(figsize=(13, len(rows) * 0.7 + 1.2))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # 헤더 스타일
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # 행 배경 + PSNR drop 색상
    for i, r in enumerate(rows, start=1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
        drop = r["psnr_drop"]
        tbl[i, 3].set_text_props(
            color="green" if drop <= 0.1 else ("orange" if drop <= 0.3 else "red"),
            fontweight="bold"
        )

    ax.set_title(f"Fixed bpf ≈ {target_bpf} 비교 (Pareto frontier 기준)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"비교 테이블 저장 → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", nargs="+", required=True,
                        help="compress_result.json 파일들")
    parser.add_argument("--target_bpf", type=float, default=5.0,
                        help="비교 기준 bpf (default: 5.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="저장할 PNG 경로")
    args = parser.parse_args()

    target_bpf = args.target_bpf
    rows = []

    print(f"\n=== bpf ≈ {target_bpf} 고정 비교 ===\n")
    print(f"{'Dataset':<20} {'Baseline':>12} {'Compressed':>12} {'Drop':>10} {'bpf':>6} {'PLY':>8} {'NPZ':>8} {'ratio':>7}")
    print("-" * 90)

    for json_path in args.jsons:
        with open(json_path) as f:
            data = json.load(f)

        rd_results = data["rd_curve"]

        # baseline_psnr alias: rd_sweep.py 는 baseline.psnr (중첩) 형식
        if "baseline_psnr" in data:
            baseline_psnr = data["baseline_psnr"]
        elif "baseline" in data and isinstance(data["baseline"], dict):
            baseline_psnr = data["baseline"]["psnr"]
        else:
            raise KeyError(f"{json_path}: baseline_psnr 키를 찾을 수 없음")

        ply_mb = data.get("ply_mb", 0.0)

        pt = find_nearest_bpf(rd_results, target_bpf)

        # dataset 이름 fallback: model_path 마지막이 generic이면 부모까지
        if "dataset" in data:
            dataset = data["dataset"]
        else:
            mp_parts = data.get("model_path", json_path).rstrip("/").split("/")
            if len(mp_parts) >= 2 and mp_parts[-1].lower() in ("baseline", "default", "exp"):
                dataset = f"{mp_parts[-2]}/{mp_parts[-1]}"
            else:
                dataset = mp_parts[-1]

        # npz_mb 추정: ply_mb * (pt["bpf"] / 32) — 실제 값은 JSON에 없으므로 근사
        # 단, bpf 기반 feature만 고려하므로 실제와 다를 수 있음
        npz_mb = data.get("npz_mb", 0.0)  # 최적 allocation의 npz_mb
        # target_bpf 점의 npz_mb는 size_mb 키가 있으면 사용
        pt_npz_mb = pt.get("size_mb", npz_mb)

        row = {
            "dataset":         dataset,
            "baseline_psnr":   baseline_psnr,
            "compressed_psnr": pt["psnr"],
            "psnr_drop":       pt["psnr_drop"],
            "actual_bpf":      pt["bpf"],
            "allocation":      pt["allocation"],
            "ply_mb":          ply_mb,
            "npz_mb":          pt_npz_mb,
        }
        rows.append(row)

        ratio = ply_mb / pt_npz_mb if pt_npz_mb > 0 else 0
        print(f"{dataset:<20} {baseline_psnr:>12.4f} {pt['psnr']:>12.4f} "
              f"{pt['psnr_drop']:>+10.4f} {pt['bpf']:>6.2f} "
              f"{ply_mb:>8.1f} {pt_npz_mb:>8.1f} {ratio:>7.2f}×")
        print(f"  allocation: {pt['allocation']}")

    print()

    out_path = args.output or f"output_rd/compare_bpf{target_bpf}.png"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    make_comparison_table(rows, target_bpf, out_path)


if __name__ == "__main__":
    main()
