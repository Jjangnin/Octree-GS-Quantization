#!/usr/bin/env python3
"""
save_no_deflate.py — deflate ablation용 스크립트.

iteration_{src_iter}/ 의 weight를 그대로 로드해서
iteration_{dst_iter}/ 에 deflate=False로 재저장한다.

NPZ 외 PLY와 MLP(.pt)는 src_iter에서 dst_iter로 그대로 복사.
"""
import os
import shutil
import subprocess
import numpy as np

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
if len(result) > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

import torch
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args


def main():
    parser = ArgumentParser(description="Re-save NPZ with deflate=False (ablation)")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--src_iter", type=int, default=35000)
    parser.add_argument("--dst_iter", type=int, default=35001)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(getattr(args, "quiet", False))
    dataset = mp.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork,
            dataset.use_feat_bank, dataset.appearance_dim,
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
            dataset.add_level, dataset.visible_threshold, dataset.dist2level,
            dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=args.src_iter,
                      shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()

    src_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.src_iter}")
    dst_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.dst_iter}")
    os.makedirs(dst_dir, exist_ok=True)

    # 로드한 양자화 상태 그대로의 LOD bits
    lod_bits = {int(lod): int(b) for lod, b in gaussians._anchor_feat_bits_by_lod.items()}
    print(f"src_iter={args.src_iter}, dst_iter={args.dst_iter}")
    print(f"lod_bits: {lod_bits}")

    # NPZ를 deflate=False로 재저장
    dst_npz = os.path.join(dst_dir, "point_cloud_quantized.npz")
    print(f"saving NPZ (deflate=False) → {dst_npz}")
    gaussians.save_gaussian(dst_npz, lod_bits_dict=lod_bits, deflate=False)

    # PLY와 MLP(.pt) 복사
    for fname in ["point_cloud.ply"]:
        src_f = os.path.join(src_dir, fname)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(dst_dir, fname))
            print(f"copied {fname}")
    for fname in os.listdir(src_dir):
        if fname.endswith(".pt"):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
            print(f"copied {fname}")

    src_size = os.path.getsize(os.path.join(src_dir, "point_cloud_quantized.npz"))
    dst_size = os.path.getsize(dst_npz)
    print(f"\nNPZ 사이즈 비교:")
    print(f"  src (deflate=ON ) : {src_size/1e6:8.3f} MB")
    print(f"  dst (deflate=OFF): {dst_size/1e6:8.3f} MB")
    print(f"  deflate 절감     : {(dst_size-src_size)/1e6:+8.3f} MB ({(1-src_size/dst_size)*100:+.1f}%)")


if __name__ == "__main__":
    main()
