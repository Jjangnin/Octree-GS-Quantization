#!/usr/bin/env python3
"""
find_common_bpf.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
여러 데이터셋의 rd_sweep / compress_optimal JSON을 읽어서
'모든 데이터셋에서 PSNR drop ≤ threshold 를 동시에 만족하는
 최소 공통 BPF' 를 찾는다.

알고리즘:
  1. 각 데이터셋 Pareto frontier 구성 (bpf 오름차순, PSNR 단조 증가)
  2. 각 데이터셋에서 psnr_drop ≤ threshold 를 처음 만족하는 최소 bpf 계산
  3. 공통 BPF = max(per-dataset 최소 bpf)  — 이 값 이상이면 모두 만족
  4. 각 데이터셋에서 공통 BPF 에 가장 가까운 Pareto 점 보고

Usage:
  # 방법 1: compress_optimal.py 가 저장한 JSON 사용
  python find_common_bpf.py \\
      --jsons output_rd/garden_compress_result.json \\
              output_rd/bonsai_compress_result.json \\
              output_rd/bartender_compress_result.json

  # 방법 2: rd_sweep.py 가 저장한 JSON 사용 (같은 형식)
  python find_common_bpf.py \\
      --jsons output_rd/garden_rd_sweep.json \\
              output_rd/bonsai_rd_sweep.json \\
              output_rd/bartender_rd_sweep.json \\
      --threshold 0.3

  # 방법 3: 공통 BPF를 직접 지정해 각 데이터셋 결과만 확인
  python find_common_bpf.py --jsons ... --force_bpf 5.5

  옵션:
    --threshold   PSNR drop 허용 한계 (dB, default: 0.3)
    --force_bpf   이 BPF를 강제로 공통 BPF 로 사용 (탐색 생략)
    --output_dir  PNG 저장 디렉터리 (default: output_rd/common_bpf)
    --no_plot     플롯 생략
"""

import json
import argparse
import os
import sys
import numpy as np


# ────────────────────────────────────────────────────────────────
# Pareto frontier
# ────────────────────────────────────────────────────────────────

def pareto_frontier(rd_results: list) -> list:
    """bpf 오름차순으로 PSNR 단조 증가하는 Pareto 점 추출"""
    sorted_pts = sorted(rd_results, key=lambda r: r["bpf"])
    frontier, best = [], -float("inf")
    for r in sorted_pts:
        if r["psnr"] > best:
            best = r["psnr"]
            frontier.append(r)
    return frontier


def best_psnr_drop_at_bpf(frontier: list, target_bpf: float) -> float:
    """
    frontier 위에서 bpf ≤ target_bpf 인 점 중 최고 PSNR 의 drop 반환.
    해당 점이 없으면 frontier 전체에서 최소 drop 반환 (bpf 초과).
    """
    candidates = [r for r in frontier if r["bpf"] <= target_bpf + 1e-6]
    if candidates:
        return min(r["psnr_drop"] for r in candidates)
    # target_bpf 미달 → frontier 의 첫 점(최소 bpf) 의 drop 반환
    return frontier[0]["psnr_drop"]


def min_bpf_for_threshold(frontier: list, threshold: float) -> float | None:
    """
    psnr_drop ≤ threshold 를 처음 만족하는 최소 bpf.
    Pareto 위에서 bpf 오름차순으로 탐색.
    threshold 를 만족하는 점이 없으면 None 반환.
    """
    for r in frontier:
        if r["psnr_drop"] <= threshold:
            return r["bpf"]
    return None


def nearest_pareto_point(frontier: list, target_bpf: float) -> dict:
    """target_bpf 에 가장 가까운 Pareto 점"""
    return min(frontier, key=lambda r: abs(r["bpf"] - target_bpf))


# ────────────────────────────────────────────────────────────────
# JSON 로딩 — rd_sweep.py / compress_optimal.py 둘 다 지원
# ────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    # rd_sweep.py: baseline = {"psnr": ..., "ssim": ...}
    # compress_optimal.py: baseline_psnr = float
    if "baseline" in data and "psnr" in data["baseline"]:
        data["baseline_psnr"] = data["baseline"]["psnr"]
    elif "baseline_psnr" not in data:
        raise KeyError(f"{path}: baseline_psnr 키를 찾을 수 없음")

    if "dataset" not in data:
        # model_path 에서 dataset 이름 추출
        data["dataset"] = os.path.basename(
            data.get("model_path", path).rstrip("/")
        )

    return data


# ────────────────────────────────────────────────────────────────
# 플롯
# ────────────────────────────────────────────────────────────────

def plot_rd_curves(datasets_info: list, common_bpf: float, threshold: float, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음 — 플롯 스킵")
        return

    colors = ["steelblue", "darkorange", "forestgreen",
              "orchid", "saddlebrown", "teal"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, info in enumerate(datasets_info):
        c = colors[i % len(colors)]
        bpfs  = [r["bpf"]  for r in info["rd"]]
        psnrs = [r["psnr"] for r in info["rd"]]
        ax.scatter(bpfs, psnrs, s=12, alpha=0.25, color=c)

        fp = info["frontier"]
        ax.plot([r["bpf"] for r in fp], [r["psnr"] for r in fp],
                "-o", color=c, linewidth=1.8, markersize=4,
                label=f"{info['name']} (baseline {info['baseline_psnr']:.2f} dB)")

        # 공통 BPF 에서의 점 표시
        pt = info["common_pt"]
        ax.scatter([pt["bpf"]], [pt["psnr"]], s=140, color=c,
                   marker="*", zorder=6)
        ax.annotate(
            f"  {pt['psnr']:.2f}↓{pt['psnr_drop']:+.3f}",
            (pt["bpf"], pt["psnr"]), fontsize=8, color=c
        )

    ax.axvline(common_bpf, linestyle="--", color="red", linewidth=1.5,
               label=f"Common BPF = {common_bpf:.3f}\n(max drop ≤ {threshold} dB)")

    ax.set_xlabel("Avg bits per feature element (bpf)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title(f"Common BPF search — PSNR drop ≤ {threshold} dB 공통 최소 BPF", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"RD curve 플롯 저장 → {out_path}")


def plot_summary_table(rows: list, common_bpf: float, threshold: float, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    col_labels = ["Dataset", "Baseline PSNR", "PSNR @ common BPF",
                  "PSNR drop", "actual bpf", "Allocation", "Satisfy?"]
    cell_text = []
    for r in rows:
        satisfy = "✓" if r["psnr_drop"] <= threshold else "✗"
        cell_text.append([
            r["dataset"],
            f"{r['baseline_psnr']:.4f} dB",
            f"{r['compressed_psnr']:.4f} dB",
            f"{r['psnr_drop']:+.4f} dB",
            f"{r['actual_bpf']:.3f}",
            str(r["allocation"]),
            satisfy,
        ])

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(14, n_rows * 0.75 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, r in enumerate(rows, start=1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
        drop = r["psnr_drop"]
        color = "green" if drop <= threshold * 0.5 else \
                ("orange" if drop <= threshold else "red")
        tbl[i, 3].set_text_props(color=color, fontweight="bold")
        tbl[i, 6].set_text_props(
            color="green" if r["psnr_drop"] <= threshold else "red",
            fontweight="bold"
        )

    ax.set_title(
        f"Common BPF = {common_bpf:.3f}  (threshold: drop ≤ {threshold} dB)",
        fontsize=12, fontweight="bold", pad=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"요약 테이블 저장 → {out_path}")


# ────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="모든 데이터셋에서 PSNR drop ≤ threshold를 만족하는 최소 공통 BPF 탐색"
    )
    parser.add_argument("--jsons", nargs="+", required=True,
                        help="rd_sweep.json 또는 compress_result.json 파일들")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="PSNR drop 허용 한계 dB (default: 0.3)")
    parser.add_argument("--force_bpf", type=float, default=None,
                        help="이 BPF를 강제로 공통 BPF로 사용 (탐색 생략)")
    parser.add_argument("--output_dir", type=str, default="output_rd/common_bpf",
                        help="PNG 저장 디렉터리")
    parser.add_argument("--no_plot", action="store_true",
                        help="플롯 생략")
    args = parser.parse_args()

    threshold = args.threshold

    # ── JSON 로드 ────────────────────────────────────────────────
    all_data = []
    for path in args.jsons:
        if not os.path.exists(path):
            print(f"[ERROR] 파일 없음: {path}", file=sys.stderr)
            sys.exit(1)
        d = load_json(path)
        frontier = pareto_frontier(d["rd_curve"])
        all_data.append({
            "name":          d["dataset"],
            "baseline_psnr": d["baseline_psnr"],
            "ply_mb":        d.get("ply_mb", 0.0),
            "rd":            d["rd_curve"],
            "frontier":      frontier,
        })

    # ── 각 데이터셋의 최소 bpf 계산 ────────────────────────────
    per_dataset_min_bpf = []
    print(f"\n=== PSNR drop ≤ {threshold} dB 를 만족하는 최소 BPF (데이터셋별) ===\n")
    for info in all_data:
        mb = min_bpf_for_threshold(info["frontier"], threshold)
        info["min_bpf_needed"] = mb
        if mb is None:
            print(f"  {info['name']:<30} → threshold 만족 불가 "
                  f"(최소 drop = {min(r['psnr_drop'] for r in info['frontier']):.4f} dB)")
        else:
            per_dataset_min_bpf.append(mb)
            print(f"  {info['name']:<30} → 최소 BPF = {mb:.4f}  "
                  f"(baseline {info['baseline_psnr']:.4f} dB)")

    if not per_dataset_min_bpf:
        print("\n[ERROR] 어떤 데이터셋도 threshold를 만족하는 BPF를 찾지 못함.")
        sys.exit(1)

    unsatisfied = [d for d in all_data if d["min_bpf_needed"] is None]
    if unsatisfied:
        print(f"\n[WARNING] {[d['name'] for d in unsatisfied]} 데이터셋은 어떤 BPF에서도 "
              f"drop ≤ {threshold} dB를 만족하지 못합니다.")

    # ── 공통 BPF 결정 ───────────────────────────────────────────
    if args.force_bpf is not None:
        common_bpf = args.force_bpf
        print(f"\n공통 BPF 강제 지정: {common_bpf:.4f}")
    else:
        common_bpf = max(per_dataset_min_bpf)
        print(f"\n★ 공통 최소 BPF = max({[f'{v:.4f}' for v in per_dataset_min_bpf]})"
              f" = {common_bpf:.4f}")

    # ── 공통 BPF에서 각 데이터셋의 결과 확인 ───────────────────
    rows = []
    datasets_info_plot = []
    all_satisfy = True

    print(f"\n=== 공통 BPF = {common_bpf:.4f} 에서 각 데이터셋 결과 ===\n")
    print(f"{'Dataset':<28} {'Baseline':>12} {'Compressed':>12} "
          f"{'Drop':>9} {'actual bpf':>11} {'PLY MB':>8} {'NPZ MB':>8} {'ratio':>7} {'OK?':>5}")
    print("─" * 110)

    for info in all_data:
        pt = nearest_pareto_point(info["frontier"], common_bpf)
        drop_ok = pt["psnr_drop"] <= threshold
        if not drop_ok:
            all_satisfy = False

        ply_mb = info["ply_mb"]
        # rd_sweep.py가 기록한 size_mb가 있으면 그 값을 우선 사용한다.
        # 없을 때만 legacy bpf 비율 추정으로 fallback한다.
        npz_mb = pt.get("size_mb")
        if npz_mb is None:
            npz_mb = ply_mb * (pt["bpf"] / 32.0) if ply_mb > 0 else 0.0
        ratio  = ply_mb / npz_mb if npz_mb > 0 else 0.0

        symbol = "✓" if drop_ok else "✗"
        print(f"  {info['name']:<26} {info['baseline_psnr']:>12.4f} "
              f"{pt['psnr']:>12.4f} {pt['psnr_drop']:>+9.4f} "
              f"{pt['bpf']:>11.4f} {ply_mb:>8.2f} {npz_mb:>8.2f} {ratio:>7.2f}×  {symbol}")
        print(f"    allocation = {pt['allocation']}")

        rows.append({
            "dataset":          info["name"],
            "baseline_psnr":    info["baseline_psnr"],
            "compressed_psnr":  pt["psnr"],
            "psnr_drop":        pt["psnr_drop"],
            "actual_bpf":       pt["bpf"],
            "allocation":       pt["allocation"],
            "ply_mb":           round(ply_mb, 3),
            "npz_mb_est":       round(npz_mb, 3),
            "ratio_est":        round(ratio, 3),
        })
        info["common_pt"] = pt
        datasets_info_plot.append(info)

    print()
    if all_satisfy:
        print(f"[OK] 모든 데이터셋이 BPF = {common_bpf:.4f} 에서 drop ≤ {threshold} dB 만족 ✓")
    else:
        failed = [r["dataset"] for r in rows if r["psnr_drop"] > threshold]
        print(f"[WARN] 다음 데이터셋이 threshold를 초과함: {failed}")

    # ── 플롯 ────────────────────────────────────────────────────
    if not args.no_plot:
        os.makedirs(args.output_dir, exist_ok=True)

        rd_png = os.path.join(args.output_dir, "common_bpf_rd_curves.png")
        plot_rd_curves(datasets_info_plot, common_bpf, threshold, rd_png)

        tbl_png = os.path.join(args.output_dir, "common_bpf_table.png")
        plot_summary_table(rows, common_bpf, threshold, tbl_png)

    # ── JSON 저장 ────────────────────────────────────────────────
    result_json = {
        "threshold_db":   threshold,
        "common_bpf":     common_bpf,
        "all_satisfy":    all_satisfy,
        "per_dataset":    rows,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    json_out = os.path.join(args.output_dir, "common_bpf_result.json")
    with open(json_out, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"\n결과 JSON 저장 → {json_out}")


if __name__ == "__main__":
    main()
