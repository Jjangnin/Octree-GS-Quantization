#!/bin/bash
# Print PSNR/SSIM/LPIPS for all 7 scenes at 40k+TF32 OFF.
cd /data2/MatrixCity/Octree-GS_QAT-Deflate_40k

PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python

$PY - <<'PY'
import json, os, glob

scenes = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
print(f"{'scene':<10} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}   model_path")
print("-" * 80)

psnr_sum = ssim_sum = lpips_sum = 0.0
n = 0
for s in scenes:
    base = f"outputs/mipnerf360/{s}/baseline_40k_tf32off"
    if not os.path.isdir(base):
        print(f"{s:<10} {'-':>7} {'-':>7} {'-':>7}   (no run yet)")
        continue
    runs = sorted(glob.glob(f"{base}/*"))
    if not runs:
        print(f"{s:<10} {'-':>7} {'-':>7} {'-':>7}   (no run yet)")
        continue
    latest = runs[-1]
    rj = os.path.join(latest, "results.json")
    if not os.path.isfile(rj):
        print(f"{s:<10} {'-':>7} {'-':>7} {'-':>7}   (training in progress: {os.path.basename(latest)})")
        continue
    d = json.load(open(rj))
    key = "ours_40000" if "ours_40000" in d else next(iter(d))
    psnr  = d[key]["PSNR"]
    ssim  = d[key]["SSIM"]
    lpips = d[key]["LPIPS"]
    psnr_sum += psnr; ssim_sum += ssim; lpips_sum += lpips; n += 1
    print(f"{s:<10} {psnr:>7.2f} {ssim:>7.4f} {lpips:>7.4f}   {os.path.basename(latest)}")

if n:
    print("-" * 80)
    print(f"{'avg':<10} {psnr_sum/n:>7.2f} {ssim_sum/n:>7.4f} {lpips_sum/n:>7.4f}   ({n} scenes)")
PY
