#!/usr/bin/env python3
"""
resave_optimal.py
RD sweep / compress result JSON에서 원하는 operating point의
bit allocation을 찾아 모델을 재저장한다.

Usage:
  python resave_optimal.py \
      --sweep  output_rd/garden30k_rd_sweep.json \
      --model  outputA/garden_30k \
      --source /data3/isjang/Octree-GS/data/mipnerf360/garden \
      [--max_drop 0.1]   # 허용 PSNR drop (dB), 기본 0.05
      [--force_bpf 6.0]  # 이 BPF에 가장 가까운 Pareto 점 사용
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

from scene import Scene
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser


def pareto_frontier(rd_results):
    """bpf 오름차순 정렬 후 PSNR 단조증가 점만 추출"""
    sorted_pts = sorted(rd_results, key=lambda r: r["bpf"])
    frontier = []
    best_psnr = -float("inf")
    for r in sorted_pts:
        if r["psnr"] > best_psnr:
            best_psnr = r["psnr"]
            frontier.append(r)
    return frontier


def nearest_pareto_point(frontier, target_bpf):
    """target_bpf 에 가장 가까운 Pareto 점"""
    return min(frontier, key=lambda r: (abs(r["bpf"] - target_bpf), r["bpf"]))


def load_result_json(path):
    with open(path) as f:
        data = json.load(f)

    if "baseline" in data and "psnr" in data["baseline"]:
        data["baseline_psnr"] = data["baseline"]["psnr"]
    elif "baseline_psnr" not in data:
        raise KeyError(f"{path}: baseline_psnr 키를 찾을 수 없음")
    return data


def output_stem(path):
    name = os.path.basename(path)
    for suffix in ("_rd_sweep.json", "_compress_result.json", ".json"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def plot_rd_with_optimal(sweep, optimal, out_path):
    rd_results = sweep["rd_curve"]
    baseline_psnr = sweep["baseline_psnr"]

    bpfs  = [r["bpf"]  for r in rd_results]
    psnrs = [r["psnr"] for r in rd_results]

    frontier = pareto_frontier(rd_results)
    f_bpf  = [r["bpf"]  for r in frontier]
    f_psnr = [r["psnr"] for r in frontier]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bpfs, psnrs, s=20, alpha=0.5, color="steelblue", label="Sweep points")
    ax.plot(f_bpf, f_psnr, color="tomato", linewidth=1.5, marker="o",
            markersize=4, label="Pareto frontier")
    ax.axhline(baseline_psnr, color="gray", linestyle="--", linewidth=1.2,
               label=f"Baseline PSNR ({baseline_psnr:.2f} dB)")

    # 최적 점 강조
    ax.scatter([optimal["bpf"]], [optimal["psnr"]], s=150, color="gold",
               edgecolors="black", linewidths=1, zorder=5,
               label=f"Optimal ({optimal['bpf']:.2f} bpf, drop {optimal['psnr_drop']:+.3f} dB)")

    ax.set_xlabel("bpf (bits per feature element)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"RD Curve — {os.path.basename(out_path).replace('_optimal_rd.png', '')}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"RD curve 저장 → {out_path}")


def make_result_table(dataset_name, baseline_psnr, opt_psnr, psnr_drop,
                      ply_mb, npz_mb, alloc, bpf, out_path):
    """압축 결과 요약 테이블을 이미지로 저장"""
    ratio      = ply_mb / npz_mb if npz_mb > 0 else float("inf")
    reduction  = (1 - npz_mb / ply_mb) * 100 if ply_mb > 0 else 0.0

    alloc_str = ", ".join(f"LOD{k}:{v}b" for k, v in sorted(alloc.items()))

    rows = [
        ("Dataset",               dataset_name),
        ("Baseline PSNR (float32)", f"{baseline_psnr:.4f} dB"),
        ("Compressed PSNR",       f"{opt_psnr:.4f} dB"),
        ("PSNR drop",             f"{psnr_drop:+.4f} dB"),
        ("bpf",                   f"{bpf:.2f} bits/element"),
        ("Bit allocation",        alloc_str),
        ("Original size (PLY)",   f"{ply_mb:.2f} MB"),
        ("Compressed size (NPZ)", f"{npz_mb:.2f} MB"),
        ("Compression ratio",     f"{ratio:.2f}×"),
        ("Size reduction",        f"{reduction:.1f}%"),
    ]

    fig, ax = plt.subplots(figsize=(7, len(rows) * 0.55 + 0.8))
    ax.axis("off")

    col_labels = ["Metric", "Value"]
    cell_text  = [[k, v] for k, v in rows]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width([0, 1])

    # 헤더 색
    for j in range(2):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # 짝수 행 배경
    for i in range(1, len(rows) + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(2):
            tbl[i, j].set_facecolor(color)

    # PSNR drop 행 강조 (4번째 row → index 4)
    drop_row = 4
    tbl[drop_row, 1].set_text_props(
        color="green" if psnr_drop >= -0.1 else "red", fontweight="bold"
    )
    # 압축률 행 강조 (index 9)
    tbl[9, 1].set_text_props(color="#1a5276", fontweight="bold")

    ax.set_title(f"Compression Result — {dataset_name}",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"결과 테이블 저장 → {out_path}")


def find_optimal(rd_results, max_drop):
    """Pareto frontier에서 허용 drop 이내의 최소 bpf allocation을 반환"""
    sorted_pts = sorted(rd_results, key=lambda r: r["bpf"])

    # drop <= max_drop 중 최소 bpf
    candidates = [r for r in sorted_pts if r["psnr_drop"] <= max_drop]
    if candidates:
        return candidates[0]
    # 없으면 drop이 가장 작은 것
    return min(rd_results, key=lambda r: r["psnr_drop"])


def choose_operating_point(rd_results, max_drop, force_bpf=None):
    frontier = pareto_frontier(rd_results)
    if force_bpf is None:
        return find_optimal(frontier, max_drop)

    valid = [r for r in frontier if r["psnr_drop"] <= max_drop]
    candidates = valid if valid else frontier
    return nearest_pareto_point(candidates, force_bpf)


def main():
    parser = ArgumentParser()
    model_params  = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sweep",     required=True, help="rd_sweep.json 경로")
    parser.add_argument("--max_drop",  default=0.05, type=float,
                        help="허용 PSNR drop (dB). 기본 0.05")
    parser.add_argument("--force_bpf", default=None, type=float,
                        help="이 BPF에 가장 가까운 Pareto 점을 사용")
    parser.add_argument("--quiet",     action="store_true")
    args = get_combined_args(parser)

    # sweep JSON 읽기
    sweep_path = getattr(args, "sweep", None)
    max_drop   = getattr(args, "max_drop", 0.05)
    force_bpf  = getattr(args, "force_bpf", None)
    sweep = load_result_json(sweep_path)

    optimal = choose_operating_point(sweep["rd_curve"], max_drop, force_bpf=force_bpf)
    alloc   = {int(k): int(v) for k, v in optimal["allocation"].items()}

    if force_bpf is None:
        print(f"\n=== 최적 allocation (max_drop={max_drop} dB) ===")
    else:
        print(f"\n=== allocation @ target bpf={force_bpf:.4f} (max_drop={max_drop} dB) ===")
    print(f"  bpf        : {optimal['bpf']:.2f}")
    print(f"  PSNR drop  : {optimal['psnr_drop']:+.4f} dB")
    print(f"  allocation : {alloc}")

    # 모델 로드
    safe_state(getattr(args, "quiet", False))
    dataset  = model_params.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork,
            dataset.use_feat_bank, dataset.appearance_dim,
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
            dataset.add_level, dataset.visible_threshold, dataset.dist2level,
            dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=args.iteration,
                      shuffle=False, resolution_scales=dataset.resolution_scales)

    # Scene이 NPZ를 로드해 _anchor_feat_quantized=True가 되므로
    # PLY(float32)를 다시 읽어 float32 feature를 복원한다
    iteration = scene.loaded_iter
    out_dir   = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
    ply_load_path = os.path.join(out_dir, "point_cloud.ply")
    if os.path.exists(ply_load_path):
        print(f"float32 PLY 재로드 → {ply_load_path}")
        gaussians.load_ply_sparse_gaussian(ply_load_path)
    else:
        raise FileNotFoundError(f"PLY not found: {ply_load_path}")

    npz_path  = os.path.join(out_dir, "point_cloud_quantized.npz")

    print(f"\n재저장 중 → {npz_path}")
    gaussians.save_gaussian(npz_path, lod_bits_dict=alloc)
    print(f"완료! allocation={alloc}")

    # 파일 크기 계산
    ply_path = os.path.join(out_dir, "point_cloud.ply")
    ply_mb   = os.path.getsize(ply_path) / 1e6 if os.path.exists(ply_path) else 0.0
    npz_mb   = os.path.getsize(npz_path) / 1e6 if os.path.exists(npz_path) else 0.0

    dataset_name = os.path.basename(dataset.model_path.rstrip("/"))
    sweep_dir    = os.path.dirname(os.path.abspath(sweep_path))
    stem         = output_stem(sweep_path)

    # RD curve 플롯
    if force_bpf is None:
        plot_name = f"{stem}_optimal_rd.png"
    else:
        plot_name = f"{stem}_bpf_{optimal['bpf']:.4f}_rd.png"
    plot_path = os.path.join(sweep_dir, plot_name)
    plot_rd_with_optimal(sweep, optimal, plot_path)

    # 압축 결과 테이블 이미지
    if force_bpf is None:
        table_name = f"{stem}_result_table.png"
    else:
        table_name = f"{stem}_bpf_{optimal['bpf']:.4f}_result_table.png"
    table_path = os.path.join(sweep_dir, table_name)
    make_result_table(
        dataset_name   = dataset_name,
        baseline_psnr  = sweep["baseline_psnr"],
        opt_psnr       = optimal["psnr"],
        psnr_drop      = optimal["psnr_drop"],
        ply_mb         = ply_mb,
        npz_mb         = npz_mb,
        alloc          = alloc,
        bpf            = optimal["bpf"],
        out_path       = table_path,
    )

    # 터미널 요약 출력
    ratio     = ply_mb / npz_mb if npz_mb > 0 else float("inf")
    reduction = (1 - npz_mb / ply_mb) * 100 if ply_mb > 0 else 0.0
    print(f"\n=== 압축 결과 요약 ===")
    print(f"  원본 (PLY)     : {ply_mb:.2f} MB")
    print(f"  압축 (NPZ)     : {npz_mb:.2f} MB")
    print(f"  압축률         : {ratio:.2f}× ({reduction:.1f}% 감소)")
    print(f"  PSNR           : {optimal['psnr']:.4f} dB  (drop {optimal['psnr_drop']:+.4f} dB)")

    # 요약 JSON
    summary = {
        "allocation"          : alloc,
        "bpf"                 : optimal["bpf"],
        "psnr_drop"           : optimal["psnr_drop"],
        "psnr"                : optimal["psnr"],
        "sweep_baseline_psnr" : sweep["baseline_psnr"],
        "target_bpf"          : force_bpf,
        "max_drop"            : max_drop,
        "ply_mb"              : round(ply_mb, 4),
        "npz_mb"              : round(npz_mb, 4),
        "compression_ratio"   : round(ratio, 4),
        "size_reduction_pct"  : round(reduction, 2),
    }
    summary_path = os.path.join(dataset.model_path, "optimal_alloc.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"요약 저장 → {summary_path}")


if __name__ == "__main__":
    main()
