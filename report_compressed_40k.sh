#!/bin/bash
# Aggregate report: 40k baseline + 45k QAT-compressed metrics + sizes.
cd /data2/MatrixCity/Octree-GS_QAT-Deflate_40k
PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python

$PY - <<'PY'
import json, os, glob

scenes = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]

def latest(scene):
    p = f"outputs/mipnerf360/{scene}/baseline_40k_tf32off"
    if not os.path.isdir(p): return None
    runs = sorted(glob.glob(f"{p}/*"))
    return runs[-1] if runs else None

def fsize(p):
    return os.path.getsize(p)/1e6 if os.path.isfile(p) else None

print(f"{'scene':<10} | {'40k(base)':>23} | {'45k(QAT+Defl)':>23} | {'PLY':>7} {'NPZ':>7} {'ratio':>6}")
print(f"{'':<10} | {'PSNR  SSIM   LPIPS':>23} | {'PSNR  SSIM   LPIPS':>23} | {'(MB)':>7} {'(MB)':>7} {'(x)':>6}")
print("-"*100)

agg40 = [0,0,0,0]; agg45 = [0,0,0,0]; ply_t=0; npz_t=0
for s in scenes:
    mp = latest(s)
    if not mp:
        print(f"{s:<10} | {'(no run)':>23} |"); continue
    rj = os.path.join(mp, "results.json")
    if not os.path.isfile(rj):
        print(f"{s:<10} | {'(no results.json)':>23} |"); continue
    d = json.load(open(rj))
    r40 = d.get("ours_40000")
    r45 = d.get("ours_45000")
    s40 = f"{r40['PSNR']:>6.2f} {r40['SSIM']:.4f} {r40['LPIPS']:.4f}" if r40 else f"{'-':>6} {'-':>6} {'-':>6}"
    s45 = f"{r45['PSNR']:>6.2f} {r45['SSIM']:.4f} {r45['LPIPS']:.4f}" if r45 else f"{'-':>6} {'-':>6} {'-':>6}"
    ply = fsize(os.path.join(mp, "point_cloud/iteration_40000/point_cloud.ply"))
    npz45 = fsize(os.path.join(mp, "point_cloud/iteration_45000/point_cloud_quantized.npz"))
    npz40 = fsize(os.path.join(mp, "point_cloud/iteration_40000/point_cloud_quantized.npz"))
    npz = npz45 if npz45 else npz40
    ratio = (ply/npz) if (ply and npz) else None
    ply_s = f"{ply:>6.1f}" if ply else f"{'-':>6}"
    npz_s = f"{npz:>6.1f}" if npz else f"{'-':>6}"
    rat_s = f"{ratio:>5.1f}x" if ratio else f"{'-':>5}"
    print(f"{s:<10} | {s40:>23} | {s45:>23} | {ply_s:>7} {npz_s:>7} {rat_s:>6}")
    if r40:
        agg40[0]+=r40['PSNR']; agg40[1]+=r40['SSIM']; agg40[2]+=r40['LPIPS']; agg40[3]+=1
    if r45:
        agg45[0]+=r45['PSNR']; agg45[1]+=r45['SSIM']; agg45[2]+=r45['LPIPS']; agg45[3]+=1
    if ply: ply_t += ply
    if npz: npz_t += npz

print("-"*100)
if agg40[3]:
    n=agg40[3]; print(f"{'avg(40k)':<10} | {agg40[0]/n:>6.2f} {agg40[1]/n:.4f} {agg40[2]/n:.4f} |")
if agg45[3]:
    n=agg45[3]; print(f"{'avg(45k)':<10} |                         | {agg45[0]/n:>6.2f} {agg45[1]/n:.4f} {agg45[2]/n:.4f} | {ply_t:>6.1f} {npz_t:>6.1f} {(ply_t/npz_t if npz_t else 0):>5.1f}x")
PY

echo ""
echo "[RDO summary from output_rd_40k/]"
for s in bicycle bonsai counter garden kitchen room stump; do
  f="output_rd_40k/${s}_compress_result.json"
  [ -f "$f" ] && $PY -c "
import json; d=json.load(open('$f'))
print(f\"  $s: bpf={d['bpf']:.2f} ply={d['ply_mb']:.1f} npz={d['npz_mb']:.2f} reduction={d['size_reduction_pct']:.1f}% rdo_psnr={d['opt_psnr']:.2f} drop={d['psnr_drop']:.2f}\")
"
done
