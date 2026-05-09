#!/usr/bin/env python3
"""
rdo.py — Lagrangian Rate-Distortion Optimization for LOD bit allocation.

기존 brute-force sweep (allowed_bits^num_LOD 조합 평가) 을
per-LOD Lagrangian RDO 로 대체한다.

핵심 알고리즘
─────────────
1. (Phase 1) marginal cost table 측정:
   - baseline = 모든 LOD가 max(allowed_bits)
   - 각 (LOD l, bit b) 쌍에 대해 LOD l 만 b로 바꾼 alloc 평가
   - D_l(b) = MSE = 10^(-PSNR/10),  R_l(b) = N_l · feat_dim · b
   - 총 K·L 번 평가 (K = |allowed_bits|, L = num_LOD)

2. (Phase 2) λ sweep 으로 Lagrangian-optimal alloc 들을 모두 수집:
   - 각 LOD에서 b*(l, λ) = argmin_b [D_l(b) + λ·R_l(b)]
   - cost table 에서 유도한 |ΔD/ΔR| slope 분포로 λ 범위 자동 설정
   - 별도 렌더링 없이 cost table 만으로 결정

3. (Phase 3) 각 unique alloc 을 실제로 평가해 RD 점 (bpf, PSNR) 산출
   - separability 가정이 깨질 때를 위한 검증
   - 보통 |unique_allocs| ≪ K^L

복잡도: brute-force K^L 대신 K·L + |unique_allocs|.
예: K=3 (4,6,8 bit), L=5 → 243 → 15 + 보통 5~15 ≈ 30 회 (약 8× 단축).
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from gaussian_renderer import render, prefilter_voxel
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn


# ────────────────────────────────────────────────────────────────
# Quantization helpers (sweep 평가용 — 실제 NPZ 저장은 GaussianModel.save_gaussian)
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


# ────────────────────────────────────────────────────────────────
# Bitrate / size helpers
# ────────────────────────────────────────────────────────────────

def bpf_for_allocation(levels: torch.Tensor, feat_dim: int, allocation: dict) -> float:
    """평균 bits per feature element = 총 feature bits / (N · D)."""
    N = levels.shape[0]
    if N == 0:
        return 0.0
    max_lod = int(levels.max())
    total_bits = sum(
        int((levels == lod).sum()) * feat_dim * allocation.get(lod, 8)
        for lod in range(max_lod + 1)
    )
    return total_bits / (N * feat_dim)


def feat_bytes_for_allocation(levels: torch.Tensor, feat_dim: int, allocation: dict) -> int:
    """압축된 feature codec bytes (q payload + LOD 별 metadata)."""
    total = 0
    if levels.numel() == 0:
        return total
    max_lod = int(levels.max())
    for lod in range(max_lod + 1):
        n = int((levels == lod).sum())
        if n == 0:
            continue
        bits = allocation.get(lod, 8)
        if bits == 4:
            total += n * math.ceil(feat_dim / 2)
        else:
            total += n * math.ceil(feat_dim * bits / 8)
        total += n * np.dtype(np.int32).itemsize  # indices_lod_*
        total += feat_dim * 4 * 2                 # scale + zero per-dim float32
        total += np.dtype(np.int32).itemsize * 2  # bits_lod_* + packed_lod_*
    return total


# ────────────────────────────────────────────────────────────────
# Apply / restore allocation
# ────────────────────────────────────────────────────────────────

def apply_allocation(gaussians, orig_feats: torch.Tensor, allocation: dict):
    """allocation에 따라 양자화 → 역양자화해서 g._anchor_feat에 적용."""
    levels = gaussians._level.squeeze(1)
    result = torch.zeros_like(orig_feats)

    if levels.numel() > 0:
        max_lod = int(levels.max())
        for lod in range(max_lod + 1):
            mask = levels == lod
            if not mask.any():
                continue
            bits = allocation.get(lod, 8)
            feat_np = orig_feats[mask].cpu().numpy()
            q, scale, zero = quantize_np(feat_np, bits)
            result[mask] = torch.tensor(
                dequantize_np(q, scale, zero),
                dtype=torch.float32, device="cuda",
            )

    gaussians._anchor_feat_quantized = False
    gaussians._anchor_feat = nn.Parameter(result, requires_grad=False)


def restore_quantized_mode(gaussians, feat_dim: int):
    """NPZ 양자화 모드로 복원 (codec dict는 메모리에 그대로 유지)."""
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
# Phase 1 — marginal cost table
# ────────────────────────────────────────────────────────────────

def measure_marginal_cost_table(
    gaussians, cameras, pipeline, background,
    allowed_bits, orig_feats, levels, feat_dim,
    eval_log=None, t0=None,
):
    """각 (LOD, bit) 쌍의 marginal D, R 측정.

    baseline alloc = {모든 LOD: max(allowed_bits)} 에서
    한 LOD 만 b 로 바꾼 alloc 을 평가한다 (separability 가정).

    Returns
    -------
    cost_table : dict
        cost_table[lod][bits] = {"psnr", "ssim", "D_mse", "R_bits", "n_anchors"}
    base_psnr, base_ssim : float
        모든 LOD가 max_bit 일 때의 측정값.
    """
    max_lod = int(levels.max().item()) if levels.numel() > 0 else -1
    max_bit = max(allowed_bits)
    n_per_lod = {l: int((levels == l).sum()) for l in range(max_lod + 1)}

    # baseline (all max_bit): cost_table 의 (l, max_bit) 항목으로 재사용
    baseline_alloc = {l: max_bit for l in range(max_lod + 1)}
    apply_allocation(gaussians, orig_feats, baseline_alloc)
    base_psnr, base_ssim = evaluate(
        gaussians, cameras, pipeline, background,
        desc=f"RDO baseline (all={max_bit}b)",
    )
    base_mse = 10.0 ** (-base_psnr / 10.0)
    if eval_log is not None:
        eval_log.append({
            "elapsed": time.perf_counter() - t0,
            "allocation": {str(k): int(v) for k, v in baseline_alloc.items()},
            "bpf": bpf_for_allocation(levels, feat_dim, baseline_alloc),
            "psnr": base_psnr,
            "phase": "phase1",
        })

    cost_table = {l: {} for l in range(max_lod + 1)}
    for l in range(max_lod + 1):
        cost_table[l][max_bit] = {
            "psnr": base_psnr,
            "ssim": base_ssim,
            "D_mse": base_mse,
            "R_bits": n_per_lod[l] * feat_dim * max_bit,
            "n_anchors": n_per_lod[l],
        }

    other_bits = [b for b in allowed_bits if b != max_bit]
    total_runs = (max_lod + 1) * len(other_bits)
    print(f"\n[Phase 1] marginal cost table 측정 ({total_runs} runs)...")

    pbar = tqdm(total=total_runs, desc="Marginal cost")
    for l in range(max_lod + 1):
        if n_per_lod[l] == 0:
            for b in other_bits:
                cost_table[l][b] = {
                    "psnr": base_psnr, "ssim": base_ssim, "D_mse": base_mse,
                    "R_bits": 0, "n_anchors": 0,
                }
                pbar.update(1)
            continue
        for b in other_bits:
            alloc = dict(baseline_alloc)
            alloc[l] = b
            apply_allocation(gaussians, orig_feats, alloc)
            psnr_val, ssim_val = evaluate(
                gaussians, cameras, pipeline, background, desc=f"L{l}={b}b",
            )
            if eval_log is not None:
                eval_log.append({
                    "elapsed": time.perf_counter() - t0,
                    "allocation": {str(k): int(v) for k, v in alloc.items()},
                    "bpf": bpf_for_allocation(levels, feat_dim, alloc),
                    "psnr": psnr_val,
                    "phase": "phase1",
                })
            cost_table[l][b] = {
                "psnr": psnr_val,
                "ssim": ssim_val,
                "D_mse": 10.0 ** (-psnr_val / 10.0),
                "R_bits": n_per_lod[l] * feat_dim * b,
                "n_anchors": n_per_lod[l],
            }
            pbar.update(1)
    pbar.close()

    restore_quantized_mode(gaussians, feat_dim)
    return cost_table, base_psnr, base_ssim


# ────────────────────────────────────────────────────────────────
# Phase 2 — λ sweep
# ────────────────────────────────────────────────────────────────

def lambda_to_allocation(cost_table: dict, lambda_val: float, allowed_bits) -> dict:
    """주어진 λ에서 per-LOD argmin → allocation."""
    alloc = {}
    for lod, bits_dict in cost_table.items():
        best_b = min(
            allowed_bits,
            key=lambda b: bits_dict[b]["D_mse"] + lambda_val * bits_dict[b]["R_bits"],
        )
        alloc[lod] = best_b
    return alloc


def enumerate_pareto_allocations(cost_table, allowed_bits, num_lambdas=400):
    """λ ∈ [λ_min, λ_max] 를 sweep해서 Lagrangian-optimal alloc 들을 모두 수집.

    λ 범위는 cost table 의 인접 bit 간 |ΔD/ΔR| 분포에서 자동 결정한다.
    """
    sorted_bits = sorted(allowed_bits)
    slopes = []
    for bits_dict in cost_table.values():
        for b1, b2 in zip(sorted_bits[:-1], sorted_bits[1:]):
            d1, r1 = bits_dict[b1]["D_mse"], bits_dict[b1]["R_bits"]
            d2, r2 = bits_dict[b2]["D_mse"], bits_dict[b2]["R_bits"]
            if r2 > r1 and d1 > d2:
                slopes.append((d1 - d2) / (r2 - r1))

    if slopes:
        lam_min = max(min(slopes) * 1e-3, 1e-30)
        lam_max = max(slopes) * 1e3
    else:
        lam_min, lam_max = 1e-15, 1e0

    lambda_grid = np.geomspace(lam_min, lam_max, num_lambdas)
    # corner: λ=0 (max bit), λ=∞ (min bit)
    grid = np.concatenate([[0.0], lambda_grid, [np.inf]])

    seen = set()
    allocs = []
    for lam in grid:
        if lam == np.inf:
            alloc = {l: min(allowed_bits) for l in cost_table}
        else:
            alloc = lambda_to_allocation(cost_table, float(lam), allowed_bits)
        key = tuple(sorted(alloc.items()))
        if key not in seen:
            seen.add(key)
            allocs.append(alloc)
    return allocs


# ────────────────────────────────────────────────────────────────
# Phase 3 — evaluate selected allocations
# ────────────────────────────────────────────────────────────────

def evaluate_allocations(
    gaussians, cameras, pipeline, background,
    allocations, orig_feats, levels, feat_dim,
    baseline_psnr, fixed_bytes=0, include_size_mb=True,
    eval_log=None, t0=None,
):
    results = []
    for alloc in tqdm(allocations, desc="[Phase 3] Pareto eval"):
        apply_allocation(gaussians, orig_feats, alloc)
        psnr_val, ssim_val = evaluate(
            gaussians, cameras, pipeline, background, desc="",
        )
        restore_quantized_mode(gaussians, feat_dim)

        bpf = bpf_for_allocation(levels, feat_dim, alloc)
        if eval_log is not None:
            eval_log.append({
                "elapsed": time.perf_counter() - t0,
                "allocation": {str(k): int(v) for k, v in alloc.items()},
                "bpf": bpf,
                "psnr": psnr_val,
                "phase": "phase3",
            })
        item = {
            "allocation": {str(k): int(v) for k, v in alloc.items()},
            "bpf": round(bpf, 4),
            "psnr": round(psnr_val, 4),
            "ssim": round(ssim_val, 6),
            "psnr_drop": round(baseline_psnr - psnr_val, 4),
        }
        if include_size_mb:
            feat_bytes = feat_bytes_for_allocation(levels, feat_dim, alloc)
            item["size_mb"] = round((feat_bytes + fixed_bytes) / 1e6, 3)
        results.append(item)

    results.sort(key=lambda x: x["bpf"])
    return results


# ────────────────────────────────────────────────────────────────
# Top-level
# ────────────────────────────────────────────────────────────────

def run_rdo(
    gaussians, cameras, pipeline, background,
    allowed_bits, fixed_bytes=0, include_size_mb=True,
    num_lambdas=400, eval_log=None,
):
    """Lagrangian RDO entry point.

    sweep_rd / run_sweep 와 같은 자리에서 호출되도록 동일한 시그니처/반환을 따른다.

    Returns
    -------
    baseline_psnr, baseline_ssim : float
        원본 NPZ codec 그대로 평가한 값.
    orig_feats : torch.Tensor
        NPZ에서 복원한 float32 anchor feature (caller가 final NPZ 저장 시 재사용).
    rd_results : list[dict]
        sweep 코드와 동일 포맷의 RD 점 리스트.
    cost_table : dict
        Phase 1 marginal cost table (분석/디버깅용).
    """
    levels = gaussians._level.squeeze(1)
    max_lod = int(levels.max().item()) if levels.numel() > 0 else -1
    feat_dim = int(gaussians.feat_dim)
    N = gaussians._anchor.shape[0]

    print("\nLOD 분포:")
    for lod in range(max_lod + 1):
        n = int((levels == lod).sum())
        print(f"  LOD {lod}: {n:,} anchors ({n/N*100:.1f}%)")
    print(f"  feat_dim={feat_dim}, max_lod={max_lod}, allowed_bits={allowed_bits}")

    # NPZ → float32 feature
    all_mask = torch.ones(N, dtype=torch.bool, device="cuda")
    orig_feats = gaussians.dequantize_visible_anchor_feat(all_mask).detach()

    t0 = time.perf_counter()

    # Baseline = 원본 NPZ codec (sweep 과 동일 의미)
    print("\nBaseline 평가 (원본 NPZ codec)...")
    baseline_psnr, baseline_ssim = evaluate(
        gaussians, cameras, pipeline, background, desc="Baseline (NPZ codec)",
    )
    print(f"  PSNR: {baseline_psnr:.4f} dB | SSIM: {baseline_ssim:.4f}")

    # Phase 1: marginal cost table
    cost_table, _allmax_psnr, _allmax_ssim = measure_marginal_cost_table(
        gaussians, cameras, pipeline, background,
        allowed_bits, orig_feats, levels, feat_dim,
        eval_log=eval_log, t0=t0,
    )

    # Phase 2: λ sweep → unique alloc
    pareto_allocs = enumerate_pareto_allocations(
        cost_table, allowed_bits, num_lambdas=num_lambdas,
    )
    print(f"\n[Phase 2] λ sweep → {len(pareto_allocs)} 개 Lagrangian-optimal allocation\n")

    # Phase 3: 실제 평가
    rd_results = evaluate_allocations(
        gaussians, cameras, pipeline, background,
        pareto_allocs, orig_feats, levels, feat_dim,
        baseline_psnr=baseline_psnr,
        fixed_bytes=fixed_bytes,
        include_size_mb=include_size_mb,
        eval_log=eval_log, t0=t0,
    )

    return baseline_psnr, baseline_ssim, orig_feats, rd_results, cost_table
