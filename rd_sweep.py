#!/usr/bin/env python3
"""
rd_sweep.py — LOD bit allocation sweep for RD curve analysis

LOD별 bit 조합을 전부 탐색해서 (bitrate, PSNR) 쌍을 모은 뒤
make_rd_report.py 가 읽을 수 있는 JSON 으로 저장한다.

Usage:
  python rd_sweep.py -m <model_path> -s <source_path> \
      [--allowed_bits 4 6 8] [--output path/to/rd_sweep.json] [--plot]
"""

import os
import json
import math
import time
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

from scene import Scene
from gaussian_renderer import render, prefilter_voxel, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn
from arguments import ModelParams, PipelineParams, get_combined_args
from rdo import run_rdo


# ────────────────────────────────────────────────────────────────
# Quantization helpers
# ────────────────────────────────────────────────────────────────

def quantize_np(feat: np.ndarray, bits: int):
    levels = (1 << bits) - 1
    vmin = feat.min(axis=0, keepdims=True)
    vmax = feat.max(axis=0, keepdims=True)
    scale = (vmax - vmin) / levels
    scale = np.where(scale == 0.0, 1.0, scale)
    q = np.clip(np.round((feat - vmin) / scale), 0, levels).astype(np.uint8)
    return q, scale.squeeze(0), vmin.squeeze(0)


def dequantize_np(q: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale + zero


# ────────────────────────────────────────────────────────────────
# Bitrate / size helpers
# ────────────────────────────────────────────────────────────────

def bpf_for_allocation(levels: torch.Tensor, feat_dim: int, allocation: dict) -> float:
    """평균 bits per feature element = 총 feature bits / (N * D)"""
    N = levels.shape[0]
    total_bits = sum(
        int((levels == lod).sum()) * feat_dim * allocation.get(lod, 8)
        for lod in range(int(levels.max()) + 1)
    )
    return total_bits / (N * feat_dim) if N > 0 else 0.0


def feat_bytes_for_allocation(levels: torch.Tensor, feat_dim: int, allocation: dict) -> int:
    """압축된 feature codec bytes.

    q payload뿐 아니라 LOD별 index / scale / zero / metadata까지 포함한다.
    """
    total = 0
    for lod in range(int(levels.max()) + 1):
        n = int((levels == lod).sum())
        if n == 0:
            continue
        bits = allocation.get(lod, 8)
        if bits == 4:
            total += n * math.ceil(feat_dim / 2)
        else:
            total += n * math.ceil(feat_dim * bits / 8)
        total += n * np.dtype(np.int32).itemsize  # indices_lod_* (int32)
        total += feat_dim * 4 * 2  # scale + zero (per-dim float32)
        total += np.dtype(np.int32).itemsize * 2  # bits_lod_* + packed_lod_*
    return total


def non_feat_model_bytes(model_path: str, iteration: int) -> int:
    """MLP 파일 + feature codec을 제외한 NPZ payload 바이트.

    현재 checkpoint가 old-format(offset float32)인지 geoQ-format(q_offset)인지와
    무관하게 고정 크기 부분을 실제 파일 기준으로 계산한다.

    NPZ가 PLY-fallback 강제를 위해 `.bak_*` 로 옮겨져 있으면 그 백업본을 사용한다
    (없으면 MLP 바이트만 반환 — 누락된 비-feature payload 만큼 size_mb가 작게 나옴).
    """
    import glob as _glob
    ckpt_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    mlp_bytes = 0
    for fname in ["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt", "embedding_appearance.pt"]:
        p = os.path.join(ckpt_dir, fname)
        if os.path.exists(p):
            mlp_bytes += os.path.getsize(p)

    npz_path = os.path.join(ckpt_dir, "point_cloud_quantized.npz")
    if not os.path.exists(npz_path):
        # PLY-fallback 강제로 옮겨진 백업본에서 비-feature payload 읽기
        baks = sorted(_glob.glob(npz_path + ".bak_*"), reverse=True)
        if not baks:
            return mlp_bytes
        npz_path = baks[0]

    payload = np.load(npz_path, allow_pickle=False)
    feature_prefixes = (
        "indices_lod_",
        "scale_lod_",
        "zero_lod_",
        "bits_lod_",
        "q_lod_",
        "packed_lod_",
    )
    npz_bytes = sum(
        payload[key].nbytes
        for key in payload.files
        if not key.startswith(feature_prefixes)
    )
    return mlp_bytes + npz_bytes


# ────────────────────────────────────────────────────────────────
# Apply / restore allocation
# ────────────────────────────────────────────────────────────────

def apply_allocation(gaussians, orig_feats: torch.Tensor, allocation: dict):
    """allocation에 따라 양자화 → 역양자화해서 g._anchor_feat에 적용"""
    levels = gaussians._level.squeeze(1)
    feat_dim = orig_feats.shape[1]
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


def restore_quantized_mode(gaussians, feat_dim: int):
    """NPZ 양자화 모드로 복원 (codec dict는 메모리에 그대로 유지)"""
    gaussians._anchor_feat_quantized = True
    gaussians._anchor_feat = torch.empty((0, feat_dim), dtype=torch.float32, device="cuda")


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

def sweep_rd(gaussians, cameras, pipeline, background, allowed_bits, model_path, iteration, eval_log=None):
    levels = gaussians._level.squeeze(1)
    max_lod = int(levels.max().item())
    feat_dim = gaussians._anchor_feat.shape[1]
    N = gaussians._anchor.shape[0]

    print("\nLOD 분포:")
    for lod in range(max_lod + 1):
        n = int((levels == lod).sum())
        print(f"  LOD {lod}: {n:,} anchors ({n/N*100:.1f}%)")
    print(f"  feat_dim={feat_dim}, max_lod={max_lod}")

    # NPZ에서 float32 feature 복원 (sweep의 입력 기준)
    all_mask = torch.ones(N, dtype=torch.bool, device="cuda")
    orig_feats = gaussians.dequantize_visible_anchor_feat(all_mask).detach()

    # baseline: 원본 NPZ codec 그대로 평가
    print("\nBaseline 평가 (원본 NPZ codec)...")
    baseline_psnr, baseline_ssim = evaluate(gaussians, cameras, pipeline, background, desc="Baseline")
    print(f"  PSNR: {baseline_psnr:.4f} dB | SSIM: {baseline_ssim:.4f}")

    fixed_bytes = non_feat_model_bytes(model_path, iteration)

    all_combos = list(product(allowed_bits, repeat=max_lod + 1))
    print(f"\n{len(all_combos)}가지 조합 탐색 중 (bits={allowed_bits}, LOD 0~{max_lod})...\n")

    t0 = time.perf_counter()
    results = []
    for combo in tqdm(all_combos, desc="Sweep"):
        allocation = {lod: b for lod, b in enumerate(combo)}

        apply_allocation(gaussians, orig_feats, allocation)
        psnr_val, ssim_val = evaluate(gaussians, cameras, pipeline, background, desc="")
        restore_quantized_mode(gaussians, feat_dim)

        bpf = bpf_for_allocation(levels, feat_dim, allocation)
        feat_bytes = feat_bytes_for_allocation(levels, feat_dim, allocation)
        total_mb = (feat_bytes + fixed_bytes) / 1e6

        if eval_log is not None:
            eval_log.append({
                "elapsed": time.perf_counter() - t0,
                "allocation": {str(k): int(v) for k, v in allocation.items()},
                "bpf": bpf,
                "psnr": psnr_val,
                "phase": "sweep",
            })

        results.append({
            "allocation": {str(k): v for k, v in allocation.items()},
            "bpf": round(bpf, 4),
            "psnr": round(psnr_val, 4),
            "ssim": round(ssim_val, 6),
            "psnr_drop": round(baseline_psnr - psnr_val, 4),
            "size_mb": round(total_mb, 3),
        })

    results.sort(key=lambda x: x["bpf"])
    return baseline_psnr, baseline_ssim, results


# ────────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────────

def plot_rd_curve(results, baseline_psnr, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음, 플롯 스킵")
        return

    xs = [r["bpf"] for r in results]
    ys = [r["psnr"] for r in results]

    # Pareto frontier
    pareto, best = [], -float("inf")
    for r in results:  # bpf 오름차순 정렬됨
        if r["psnr"] >= best:
            best = r["psnr"]
            pareto.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.35, s=18, color="steelblue", label="All allocations")
    ax.plot([r["bpf"] for r in pareto], [r["psnr"] for r in pareto],
            "r-o", linewidth=2, markersize=5, label="Pareto frontier", zorder=5)
    ax.axhline(baseline_psnr, linestyle="--", color="gray", linewidth=1.2,
               label=f"Baseline (NPZ) {baseline_psnr:.2f} dB")

    ax.set_xlabel("Avg bits per feature element (bpf)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion — LOD bit allocation sweep", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"RD curve 저장 → {output_path}")


# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="LOD bit allocation sweep for RD curve")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--allowed_bits", nargs="+", type=int, default=[4, 6, 8],
                        help="탐색할 bit 후보 (default: 4 6 8)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="저장할 JSON 경로 (default: <model_path>/rd_sweep.json)")
    parser.add_argument("--plot", action="store_true",
                        help="RD curve PNG 바로 저장")
    parser.add_argument("--method", default="rdo", choices=["rdo", "sweep"],
                        help="bit allocation 탐색 방법. rdo=Lagrangian RDO (기본, 빠름), "
                             "sweep=brute-force allowed_bits^num_LOD (legacy)")
    parser.add_argument("--num_lambdas", default=400, type=int,
                        help="RDO λ sweep grid 크기 (기본 400, 렌더링과 무관)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(getattr(args, "quiet", False))
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    allowed_bits = sorted(set(getattr(args, "allowed_bits", [4, 6, 8])))
    method = getattr(args, "method", "rdo")
    num_lambdas = getattr(args, "num_lambdas", 400)

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

        if dataset.white_background:
            bg = [1.0, 1.0, 1.0]
        elif dataset.random_background:
            bg = [np.random.random()] * 3
        else:
            bg = [0.0, 0.0, 0.0]
        background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        print(f"모델     : {dataset.model_path}")
        print(f"Iteration: {scene.loaded_iter}")
        print(f"Cameras  : {len(cameras)}")
        print(f"Bits 후보: {allowed_bits}")
        print(f"Method   : {method}")

        eval_log = []
        if method == "rdo":
            fixed_bytes = non_feat_model_bytes(dataset.model_path, scene.loaded_iter)
            (baseline_psnr, baseline_ssim,
             _orig_feats, rd_results, _cost_table) = run_rdo(
                gaussians, cameras, pipeline, background, allowed_bits,
                fixed_bytes=fixed_bytes,
                include_size_mb=True,
                num_lambdas=num_lambdas,
                eval_log=eval_log,
            )
        else:
            baseline_psnr, baseline_ssim, rd_results = sweep_rd(
                gaussians, cameras, pipeline, background,
                allowed_bits, dataset.model_path, scene.loaded_iter,
                eval_log=eval_log,
            )

    output_path = getattr(args, "output", None) or os.path.join(dataset.model_path, "rd_sweep.json")

    # dataset name: model_path 형태별 처리
    #   .../scene/exp/<timestamp>  →  scene/exp
    #   .../scene/exp              →  scene/exp
    #   .../scene                  →  scene
    import re as _re
    _TS = _re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$")
    _GENERIC = ("baseline", "default", "exp")
    mp_parts = dataset.model_path.rstrip("/").split("/")
    if (len(mp_parts) >= 3
            and _TS.match(mp_parts[-1])
            and mp_parts[-2].lower() in _GENERIC):
        dataset_name = f"{mp_parts[-3]}/{mp_parts[-2]}"
    elif len(mp_parts) >= 2 and mp_parts[-1].lower() in _GENERIC:
        dataset_name = f"{mp_parts[-2]}/{mp_parts[-1]}"
    else:
        dataset_name = mp_parts[-1]

    # PLY 크기: compress 비교용 ratio 계산을 위해 JSON에 포함
    ply_path_for_size = os.path.join(
        dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "point_cloud.ply",
    )
    ply_mb = round(os.path.getsize(ply_path_for_size) / 1e6, 4) if os.path.exists(ply_path_for_size) else 0.0

    payload = {
        "model_path": dataset.model_path,
        "dataset": dataset_name,
        "iteration": scene.loaded_iter,
        "allowed_bits": allowed_bits,
        "method": method,
        "ply_mb": ply_mb,
        "baseline": {
            "psnr": round(baseline_psnr, 4),
            "ssim": round(baseline_ssim, 6),
        },
        "rd_curve": rd_results,
        "eval_log": eval_log,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n결과 저장 → {output_path} ({len(rd_results)} 포인트)")

    if getattr(args, "plot", False):
        plot_path = output_path.replace(".json", ".png")
        plot_rd_curve(rd_results, baseline_psnr, plot_path)

    # ── 최적 allocation 요약 출력 ────────────────────────────────
    N = gaussians._anchor.shape[0]
    sorted_pts = sorted(rd_results, key=lambda r: r["bpf"])

    # baseline과 동일하거나 더 좋은 PSNR을 내는 최소 bpf 점
    optimal = next((r for r in sorted_pts if r["psnr_drop"] <= 0.0), None)
    # 없으면 PSNR drop이 가장 작은 점
    if optimal is None:
        optimal = min(rd_results, key=lambda r: r["psnr_drop"])

    # baseline bpf: NPZ에서 직접 읽기
    ckpt_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}")
    npz_path = os.path.join(ckpt_dir, "point_cloud_quantized.npz")
    baseline_bpf = None
    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        bits_keys = sorted([k for k in npz.keys() if k.startswith("bits_lod_")])
        idx_keys  = sorted([k for k in npz.keys() if k.startswith("indices_lod_")])
        if bits_keys and idx_keys:
            feat_dim = int(npz["feat_dim"][0]) if "feat_dim" in npz else 32
            total_bits = sum(int(npz[b][0]) * len(npz[i]) * feat_dim
                             for b, i in zip(bits_keys, idx_keys))
            baseline_bpf = total_bits / (N * feat_dim)

    print("\n" + "=" * 55)
    print(f"  Dataset  : {os.path.basename(dataset.model_path.rstrip('/'))}")
    print(f"  Anchors  : {N:,}")
    print(f"  Baseline PSNR : {baseline_psnr:.4f} dB")
    if baseline_bpf:
        print(f"  Baseline bpf  : {baseline_bpf:.2f}")
    print(f"  최적 allocation (baseline PSNR 달성 최소 bpf):")
    print(f"    bpf   = {optimal['bpf']:.2f}")
    print(f"    PSNR  = {optimal['psnr']:.4f} dB  (drop {optimal['psnr_drop']:+.4f})")
    print(f"    alloc = {optimal['allocation']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
