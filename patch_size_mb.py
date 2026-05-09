#!/usr/bin/env python3
"""
patch_size_mb.py — 기존 RDO/sweep JSON 의 rd_curve[*].size_mb 를 재계산.

문제 배경:
  PLY fallback 강제를 위해 NPZ 를 .bak_* 으로 mv 한 상태에서 RDO 를 돌리면
  rd_sweep.py 의 non_feat_model_bytes() 가 NPZ 를 못 찾아 MLP 바이트만 반환.
  → size_mb 에 NPZ 의 비-feature payload (anchor, q_offset, q_opacity, q_scaling,
     q_rotation 등 약 5~15 MB) 가 누락 → ratio 가 부풀려져 표시됨.

이 스크립트:
  1) JSON 의 model_path/iteration 으로 ckpt 디렉토리 찾기
  2) NPZ 또는 NPZ.bak_* 에서 levels/feat_dim/비-feature payload 합산
  3) 각 rd_curve 점의 allocation 으로 feat_bytes 재계산
  4) size_mb = (feat_bytes + non_feat_bytes) / 1e6 갱신

Usage:
  python patch_size_mb.py outputs/rdo_test/clean/*.json
"""
import glob
import json
import math
import os
import sys
import numpy as np


def find_npz(ckpt_dir: str):
    p = os.path.join(ckpt_dir, "point_cloud_quantized.npz")
    if os.path.exists(p):
        return p
    baks = sorted(glob.glob(p + ".bak_*"), reverse=True)
    return baks[0] if baks else None


def feat_bytes_for_allocation(levels: np.ndarray, feat_dim: int, allocation: dict) -> int:
    total = 0
    if levels.size == 0:
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
        total += feat_dim * 4 * 2                  # scale + zero (per-dim float32)
        total += np.dtype(np.int32).itemsize * 2  # bits + packed flag
    return total


def non_feat_model_bytes(model_path: str, iteration: int):
    """MLP + NPZ 의 비-feature payload 바이트. NPZ.bak_* 도 인식."""
    ckpt_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    mlp_bytes = 0
    for fname in ["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt", "embedding_appearance.pt"]:
        p = os.path.join(ckpt_dir, fname)
        if os.path.exists(p):
            mlp_bytes += os.path.getsize(p)

    npz_path = find_npz(ckpt_dir)
    if not npz_path:
        return mlp_bytes, None, None

    payload = np.load(npz_path, allow_pickle=False)
    feature_prefixes = (
        "indices_lod_", "scale_lod_", "zero_lod_",
        "bits_lod_", "q_lod_", "packed_lod_",
    )
    npz_bytes = sum(
        payload[k].nbytes
        for k in payload.files
        if not k.startswith(feature_prefixes)
    )
    levels = payload["level"].astype(np.int32).reshape(-1)
    feat_dim = int(payload["feat_dim"][0]) if "feat_dim" in payload.files else 32
    return mlp_bytes + npz_bytes, levels, feat_dim


def patch(json_path: str) -> None:
    d = json.load(open(json_path))
    if "model_path" not in d or "iteration" not in d or "rd_curve" not in d:
        print(f"  [skip] {json_path}: 필수 키 부재")
        return

    fixed_bytes, levels, feat_dim = non_feat_model_bytes(d["model_path"], d["iteration"])
    if levels is None:
        print(f"  [skip] {json_path}: NPZ/NPZ.bak 둘 다 없음 — 재계산 불가")
        return

    n_changed = 0
    for pt in d["rd_curve"]:
        alloc = {int(k): int(v) for k, v in pt["allocation"].items()}
        feat = feat_bytes_for_allocation(levels, feat_dim, alloc)
        new_size_mb = round((feat + fixed_bytes) / 1e6, 3)
        if abs(new_size_mb - pt.get("size_mb", -1.0)) > 1e-3:
            pt["size_mb"] = new_size_mb
            n_changed += 1

    json.dump(d, open(json_path, "w"), indent=2)
    print(f"  [ok] {os.path.basename(json_path)}: "
          f"{n_changed}/{len(d['rd_curve'])} 점 갱신, "
          f"non_feat_bytes={fixed_bytes/1e6:.2f} MB")


def main():
    if len(sys.argv) < 2:
        print("Usage: patch_size_mb.py <json> [more.json ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        patch(path)


if __name__ == "__main__":
    main()
