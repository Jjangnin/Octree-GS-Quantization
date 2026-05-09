#!/usr/bin/env python3
"""
compress_optimal.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOD별 bit 조합을 전부 탐색(sweep)해서 최적 allocation을 찾고,
그 allocation으로 column-wise 양자화 NPZ를 저장한다.
RD curve + 압축 결과 테이블 이미지도 함께 저장한다.

Usage:
  python compress_optimal.py \
      -m outputA/garden_30k \
      -s /data3/isjang/Octree-GS/data/mipnerf360/garden \
      [--allowed_bits 4 6 8] \
      [--max_drop 0.05] \
      [--output_dir output_rd]
"""

import os, json, math
import numpy as np
import torch
import torch.nn as nn
from itertools import product
from tqdm import tqdm
from argparse import ArgumentParser

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scene import Scene
from gaussian_renderer import render, prefilter_voxel, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn
from arguments import ModelParams, PipelineParams, get_combined_args
from rdo import run_rdo


# ────────────────────────────────────────────────────────────────
# Quantization helpers (sweep 평가용)
# ────────────────────────────────────────────────────────────────

def quantize_np(feat: np.ndarray, bits: int):
    levels = (1 << bits) - 1
    vmin = feat.min(axis=0, keepdims=True)
    vmax = feat.max(axis=0, keepdims=True)
    scale = (vmax - vmin) / levels
    scale = np.where(scale == 0.0, 1.0, scale)
    q = np.clip(np.round((feat - vmin) / scale), 0, levels).astype(np.uint8)
    return q, scale.squeeze(0), vmin.squeeze(0)


def dequantize_np(q, scale, zero):
    return q.astype(np.float32) * scale + zero


def bpf_for_allocation(levels: torch.Tensor, feat_dim: int, allocation: dict) -> float:
    N = levels.shape[0]
    total_bits = sum(
        int((levels == lod).sum()) * feat_dim * allocation.get(lod, 8)
        for lod in range(int(levels.max()) + 1)
    )
    return total_bits / (N * feat_dim) if N > 0 else 0.0


def apply_allocation(gaussians, orig_feats: torch.Tensor, allocation: dict):
    """allocation으로 양자화→역양자화 후 gaussians에 임시 적용"""
    levels = gaussians._level.squeeze(1)
    result = torch.zeros_like(orig_feats)
    for lod in range(int(levels.max()) + 1):
        mask = levels == lod
        if not mask.any():
            continue
        bits = allocation.get(lod, 8)
        feat_np = orig_feats[mask].cpu().numpy()
        q, scale, zero = quantize_np(feat_np, bits)
        result[mask] = torch.tensor(dequantize_np(q, scale, zero),
                                    dtype=torch.float32, device="cuda")
    gaussians._anchor_feat_quantized = False
    gaussians._anchor_feat = nn.Parameter(result, requires_grad=False)


# ────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(gaussians, cameras, pipeline, background, desc="eval"):
    psnr_vals, ssim_vals = [], []
    for view in tqdm(cameras, desc=desc, leave=False):
        gaussians.set_anchor_mask(view.camera_center, 1_000_000, view.resolution_scale)
        visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        pkg = render(view, gaussians, pipeline, background, visible_mask=visible_mask)
        rendered = torch.clamp(pkg["render"], 0.0, 1.0)
        gt = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)[:3]
        psnr_vals.append(psnr_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).mean().item())
        ssim_vals.append(ssim_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).item())
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


# ────────────────────────────────────────────────────────────────
# Sweep
# ────────────────────────────────────────────────────────────────

def run_sweep(gaussians, cameras, pipeline, background, allowed_bits):
    levels = gaussians._level.squeeze(1)
    max_lod = int(levels.max().item())
    feat_dim = gaussians._anchor_feat.shape[1]
    N = gaussians._anchor.shape[0]

    print("\nLOD 분포:")
    for lod in range(max_lod + 1):
        n = int((levels == lod).sum())
        print(f"  LOD {lod}: {n:,} anchors ({n/N*100:.1f}%)")

    # float32 feature 추출 (NPZ에서 역양자화)
    all_mask = torch.ones(N, dtype=torch.bool, device="cuda")
    orig_feats = gaussians.dequantize_visible_anchor_feat(all_mask).detach()

    # baseline 평가 (현재 NPZ 상태)
    print("\nBaseline 평가...")
    baseline_psnr, baseline_ssim = evaluate(gaussians, cameras, pipeline,
                                            background, desc="Baseline")
    print(f"  PSNR: {baseline_psnr:.4f} dB | SSIM: {baseline_ssim:.4f}")

    all_combos = list(product(allowed_bits, repeat=max_lod + 1))
    print(f"\n{len(all_combos)}가지 조합 탐색 중 (bits={allowed_bits}, LOD 0~{max_lod})...\n")

    results = []
    for combo in tqdm(all_combos, desc="Sweep"):
        allocation = {lod: b for lod, b in enumerate(combo)}
        apply_allocation(gaussians, orig_feats, allocation)
        psnr_val, ssim_val = evaluate(gaussians, cameras, pipeline, background, desc="")

        # 원래 quantized 모드 복원
        gaussians._anchor_feat_quantized = True
        gaussians._anchor_feat = nn.Parameter(
            torch.empty((0, feat_dim), dtype=torch.float32, device="cuda"),
            requires_grad=False
        )

        bpf = bpf_for_allocation(levels, feat_dim, allocation)
        results.append({
            "allocation": {str(k): v for k, v in allocation.items()},
            "bpf": round(bpf, 4),
            "psnr": round(psnr_val, 4),
            "ssim": round(ssim_val, 6),
            "psnr_drop": round(baseline_psnr - psnr_val, 4),
        })

    results.sort(key=lambda x: x["bpf"])
    return baseline_psnr, baseline_ssim, orig_feats, results


# ────────────────────────────────────────────────────────────────
# Find optimal
# ────────────────────────────────────────────────────────────────

def find_optimal(rd_results, max_drop, max_bpf=None, target_bpf=None):
    """
    Pick an allocation from rd_results under optional constraints.

    target_bpf != None  (method B — "nearest to target bpf"):
      - 후보 = rd_results (max_bpf 있으면 bpf ≤ max_bpf 로 먼저 필터)
      - drop ≤ max_drop 만족하는 후보 중 |bpf - target_bpf| 최소 (tie: drop 최소)
      - 만족 없으면 전체 후보 중 |bpf - target_bpf| 최소 (tie: drop 최소)

    target_bpf == None  (legacy: smallest bpf under drop-cap):
      - bpf ≤ max_bpf (지정시) 내에서 drop ≤ max_drop 중 bpf 최소
      - 만족 없으면 bpf ≤ max_bpf 중 drop 최소
      - 그것도 없으면 전체에서 drop 최소
    """
    pts = rd_results
    if max_bpf is not None:
        pts = [r for r in rd_results if r["bpf"] <= max_bpf]
    if not pts:
        pts = rd_results

    if target_bpf is not None:
        under_drop = [r for r in pts if r["psnr_drop"] <= max_drop]
        pool = under_drop if under_drop else pts
        return min(pool, key=lambda r: (abs(r["bpf"] - target_bpf), r["psnr_drop"]))

    under_drop = [r for r in pts if r["psnr_drop"] <= max_drop]
    if under_drop:
        return min(under_drop, key=lambda r: (r["bpf"], r["psnr_drop"]))
    return min(pts, key=lambda r: (r["psnr_drop"], r["bpf"]))


# ────────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────────

def pareto_frontier(rd_results):
    sorted_pts = sorted(rd_results, key=lambda r: r["bpf"])
    frontier, best = [], -float("inf")
    for r in sorted_pts:
        if r["psnr"] > best:
            best = r["psnr"]
            frontier.append(r)
    return frontier


def plot_rd_curve(rd_results, baseline_psnr, optimal, out_path, dataset_name):
    bpfs  = [r["bpf"]  for r in rd_results]
    psnrs = [r["psnr"] for r in rd_results]
    frontier = pareto_frontier(rd_results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(bpfs, psnrs, s=18, alpha=0.4, color="steelblue", label="All allocations")
    ax.plot([r["bpf"] for r in frontier], [r["psnr"] for r in frontier],
            "r-o", linewidth=2, markersize=5, label="Pareto frontier", zorder=5)
    ax.axhline(baseline_psnr, linestyle="--", color="gray", linewidth=1.2,
               label=f"Baseline {baseline_psnr:.2f} dB")
    ax.scatter([optimal["bpf"]], [optimal["psnr"]], s=180, color="gold",
               edgecolors="black", linewidths=1.2, zorder=10,
               label=f"Optimal ({optimal['bpf']:.2f} bpf, drop {optimal['psnr_drop']:+.3f} dB)")

    ax.set_xlabel("Avg bits per feature element (bpf)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title(f"Rate-Distortion — {dataset_name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"RD curve 저장 → {out_path}")


def make_result_table(dataset_name, baseline_psnr, opt_psnr, psnr_drop,
                      ply_mb, npz_mb, alloc, bpf, out_path):
    ratio     = ply_mb / npz_mb if npz_mb > 0 else float("inf")
    reduction = (1 - npz_mb / ply_mb) * 100 if ply_mb > 0 else 0.0
    alloc_str = ", ".join(f"LOD{k}:{v}b" for k, v in sorted(alloc.items()))

    rows = [
        ("Dataset",                dataset_name),
        ("Baseline PSNR (원본 NPZ)", f"{baseline_psnr:.4f} dB"),
        ("Compressed PSNR",        f"{opt_psnr:.4f} dB"),
        ("PSNR drop",              f"{psnr_drop:+.4f} dB"),
        ("bpf",                    f"{bpf:.2f} bits/element"),
        ("Bit allocation",         alloc_str),
        ("Original size (PLY)",    f"{ply_mb:.2f} MB"),
        ("Compressed size (NPZ)",  f"{npz_mb:.2f} MB"),
        ("Compression ratio",      f"{ratio:.2f}×"),
        ("Size reduction",         f"{reduction:.1f}%"),
    ]

    fig, ax = plt.subplots(figsize=(7, len(rows) * 0.55 + 0.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Metric", "Value"],
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width([0, 1])

    for j in range(2):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(2):
            tbl[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")

    # PSNR drop 강조 (row index 4 = 4번째 데이터행)
    tbl[4, 1].set_text_props(
        color="green" if psnr_drop >= -0.1 else "red", fontweight="bold"
    )
    # 압축률 강조
    tbl[9, 1].set_text_props(color="#1a5276", fontweight="bold")

    ax.set_title(f"Compression Result — {dataset_name}",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"결과 테이블 저장 → {out_path}")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    model_params   = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration",    default=-1,     type=int)
    parser.add_argument("--allowed_bits", nargs="+",      type=int, default=[4, 6, 8])
    parser.add_argument("--max_drop",     default=0.3,    type=float,
                        help="허용 PSNR drop (dB). 기본 0.3")
    parser.add_argument("--max_bpf",      default=None,   type=float,
                        help="허용 bpf 상한 (bits/feature). 지정하면 bpf ≤ max_bpf 중에서 선택.")
    parser.add_argument("--target_bpf",   default=None,   type=float,
                        help="목표 bpf (method B). 지정 시 |bpf - target_bpf| 최소인 allocation 선택.")
    parser.add_argument("--output_dir",   default="output_rd", type=str,
                        help="결과 저장 디렉토리")
    parser.add_argument("--method",       default="rdo",  choices=["rdo", "sweep"],
                        help="bit allocation 탐색 방법. rdo=Lagrangian RDO (기본, 빠름), "
                             "sweep=brute-force allowed_bits^num_LOD (legacy)")
    parser.add_argument("--num_lambdas",  default=400,    type=int,
                        help="RDO λ sweep grid 크기 (기본 400, 렌더링과 무관)")
    parser.add_argument("--quiet",        action="store_true")
    args = get_combined_args(parser)

    safe_state(getattr(args, "quiet", False))
    dataset      = model_params.extract(args)
    pipeline     = pipeline_params.extract(args)
    allowed_bits = sorted(set(getattr(args, "allowed_bits", [4, 6, 8])))
    max_drop     = getattr(args, "max_drop", 0.05)
    max_bpf      = getattr(args, "max_bpf", None)
    target_bpf   = getattr(args, "target_bpf", None)
    output_dir   = getattr(args, "output_dir", "output_rd")
    method       = getattr(args, "method", "rdo")
    num_lambdas  = getattr(args, "num_lambdas", 400)
    os.makedirs(output_dir, exist_ok=True)

    # dataset_name: 우선 source_path의 basename(=scene name)을 사용. model_path의 timestamp가 아니라.
    dataset_name = os.path.basename(dataset.source_path.rstrip("/"))

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
        gaussians.eval()

        cameras = scene.getTestCameras()
        if len(cameras) == 0:
            cameras = scene.getTrainCameras()
            print("Test cameras 없음 → train cameras 사용")

        bg_color = [1., 1., 1.] if dataset.white_background else [0., 0., 0.]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print(f"\n모델     : {dataset.model_path}")
        print(f"Iteration: {scene.loaded_iter}")
        print(f"Cameras  : {len(cameras)}")
        print(f"Bits 후보: {allowed_bits}")
        print(f"Max drop : {max_drop} dB")
        print(f"Max bpf  : {max_bpf if max_bpf is not None else 'unbounded'}")
        print(f"Target bpf: {target_bpf if target_bpf is not None else 'n/a (using bpf-minimization)'}")
        print(f"Method   : {method}")

        # ── Step 1: bit allocation 탐색 ───────────────────────────
        if method == "rdo":
            (baseline_psnr, baseline_ssim,
             orig_feats, rd_results, _cost_table) = run_rdo(
                gaussians, cameras, pipeline, background, allowed_bits,
                include_size_mb=False,
                num_lambdas=num_lambdas,
            )
        else:
            baseline_psnr, baseline_ssim, orig_feats, rd_results = run_sweep(
                gaussians, cameras, pipeline, background, allowed_bits,
            )

    # ── Step 2: 최적 allocation 선택 ──────────────────────────────
    optimal = find_optimal(rd_results, max_drop, max_bpf=max_bpf, target_bpf=target_bpf)
    alloc   = {int(k): int(v) for k, v in optimal["allocation"].items()}

    print(f"\n=== 최적 allocation (max_drop={max_drop} dB, max_bpf={max_bpf}, target_bpf={target_bpf}) ===")
    print(f"  bpf       : {optimal['bpf']:.2f}")
    print(f"  PSNR drop : {optimal['psnr_drop']:+.4f} dB")
    print(f"  allocation: {alloc}")
    if max_bpf is not None and optimal["bpf"] > max_bpf:
        print(f"  WARNING  : selected bpf {optimal['bpf']:.2f} > max_bpf {max_bpf:.2f}")
    if optimal["psnr_drop"] > max_drop:
        print(f"  WARNING  : selected drop {optimal['psnr_drop']:+.4f} dB exceeds max_drop {max_drop:.4f} dB")
    if target_bpf is not None:
        print(f"  |bpf - target| = {abs(optimal['bpf'] - target_bpf):.4f}")

    # ── Step 3: 최적 allocation으로 NPZ 저장 ─────────────────────
    iteration = scene.loaded_iter
    out_dir   = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
    npz_path  = os.path.join(out_dir, "point_cloud_quantized.npz")

    # orig_feats(float32)를 gaussians에 적용한 상태로
    # geometry + feature를 함께 저장하는 geoQ 포맷으로 저장
    gaussians._anchor_feat_quantized = False
    gaussians._anchor_feat = nn.Parameter(orig_feats, requires_grad=False)

    print(f"\n최적 allocation으로 NPZ 저장 중 → {npz_path}")
    gaussians.save_gaussian(npz_path, lod_bits_dict=alloc)
    print("NPZ 저장 완료!")

    # ── Step 4: 파일 크기 ─────────────────────────────────────────
    ply_path = os.path.join(out_dir, "point_cloud.ply")
    ply_mb   = os.path.getsize(ply_path) / 1e6 if os.path.exists(ply_path) else 0.0
    npz_mb   = os.path.getsize(npz_path) / 1e6 if os.path.exists(npz_path) else 0.0
    ratio    = ply_mb / npz_mb if npz_mb > 0 else float("inf")
    reduction = (1 - npz_mb / ply_mb) * 100 if ply_mb > 0 else 0.0

    # ── Step 5: RD curve 이미지 ───────────────────────────────────
    rd_png = os.path.join(output_dir, f"{dataset_name}_rd_curve.png")
    plot_rd_curve(rd_results, baseline_psnr, optimal, rd_png, dataset_name)

    # ── Step 6: 결과 테이블 이미지 ───────────────────────────────
    table_png = os.path.join(output_dir, f"{dataset_name}_result_table.png")
    make_result_table(
        dataset_name  = dataset_name,
        baseline_psnr = baseline_psnr,
        opt_psnr      = optimal["psnr"],
        psnr_drop     = optimal["psnr_drop"],
        ply_mb        = ply_mb,
        npz_mb        = npz_mb,
        alloc         = alloc,
        bpf           = optimal["bpf"],
        out_path      = table_png,
    )

    # ── Step 7: JSON 저장 ─────────────────────────────────────────
    json_path = os.path.join(output_dir, f"{dataset_name}_compress_result.json")
    with open(json_path, "w") as f:
        json.dump({
            "dataset":          dataset_name,
            "method":           method,
            "allowed_bits":     allowed_bits,
            "max_drop":         max_drop,
            "max_bpf":          max_bpf,
            "target_bpf":       target_bpf,
            "baseline_psnr":    round(baseline_psnr, 4),
            "opt_psnr":         optimal["psnr"],
            "psnr_drop":        optimal["psnr_drop"],
            "allocation":       alloc,
            "bpf":              optimal["bpf"],
            "ply_mb":           round(ply_mb, 4),
            "npz_mb":           round(npz_mb, 4),
            "compression_ratio": round(ratio, 4),
            "size_reduction_pct": round(reduction, 2),
            "rd_curve":         rd_results,
        }, f, indent=2)

    # ── 터미널 최종 요약 ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Dataset          : {dataset_name}")
    print(f"  Method           : {method}")
    print(f"  Anchors          : {gaussians._anchor.shape[0]:,}")
    print(f"  Baseline PSNR    : {baseline_psnr:.4f} dB")
    print(f"  Compressed PSNR  : {optimal['psnr']:.4f} dB")
    print(f"  PSNR drop        : {optimal['psnr_drop']:+.4f} dB")
    print(f"  Bit allocation   : {alloc}")
    print(f"  bpf              : {optimal['bpf']:.2f}")
    print(f"  원본 (PLY)       : {ply_mb:.2f} MB")
    print(f"  압축 (NPZ)       : {npz_mb:.2f} MB")
    print(f"  압축률           : {ratio:.2f}× ({reduction:.1f}% 감소)")
    print("=" * 55)


if __name__ == "__main__":
    main()
