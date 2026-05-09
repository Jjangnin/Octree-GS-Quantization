#!/bin/bash
# Full compression pipeline on top of 40k baselines:
#   RDO @ iter 40000 (target_bpf=5) → QAT to iter 45000 → render → metrics
# Outputs:
#   - <model>/point_cloud/iteration_40000/point_cloud_quantized.npz  (RDO-allocated, backed up to .original)
#   - <model>/point_cloud/iteration_45000/                           (QAT fine-tuned, same allocation)
#   - <model>/test/ours_45000/                                       (rendered images)
#   - <model>/results.json                                           (PSNR/SSIM/LPIPS at ours_45000)
#   - output_rd_40k/<scene>_compress_result.json                     (baseline vs RDO PSNR + sizes)
set -e
cd /data2/MatrixCity/Octree-GS_QAT-Deflate_40k
PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python
mkdir -p logs output_rd_40k

PRE_ITER=40000
QAT_ITERS=5000
POST_ITER=$((PRE_ITER + QAT_ITERS))   # 45000
TARGET_BPF=5

echo "═══ NPZ prep: restore .original if present, else back up ═══"
find outputs/mipnerf360 -path "*/baseline_40k_tf32off/*/point_cloud/iteration_${PRE_ITER}/point_cloud_quantized.npz" -not -name '*.original' | while read npz; do
  orig="${npz}.original"
  if [ -f "$orig" ]; then cp "$orig" "$npz"; echo "  restored: $npz"
  else cp "$npz" "$orig"; echo "  backup:   $orig"; fi
done
echo ""

for scene in bicycle bonsai counter garden kitchen room stump; do
  ts=$(date +"%Y-%m-%d_%H:%M:%S")
  log="logs/comp_${scene}_${ts}.log"
  model_path=$(ls -td outputs/mipnerf360/${scene}/baseline_40k_tf32off/*/ 2>/dev/null | head -1)
  if [ -z "$model_path" ]; then
    echo "✗ $(date +%T) ${scene}: no model_path, skipping"; continue
  fi
  model_path="${model_path%/}"
  echo "▶ $(date +%T) ${scene} → ${model_path}"

  # 1) RDO @ iter 40000 (target_bpf=5)
  echo "  · RDO    $(date +%T)"
  $PY compress_optimal.py \
    -m "${model_path}" \
    -s data/mipnerf360/${scene} \
    --iteration ${PRE_ITER} \
    --allowed_bits 2 3 4 5 6 7 8 \
    --target_bpf ${TARGET_BPF} \
    --max_drop 999 \
    --output_dir output_rd_40k \
    > "${log}" 2>&1

  # 2) QAT 40000 → 45000
  echo "  · QAT    $(date +%T)"
  $PY train_qat.py \
    -m "${model_path}" \
    -s data/mipnerf360/${scene} \
    --data_device cpu \
    --pretrained_iteration ${PRE_ITER} \
    --qat_iterations ${QAT_ITERS} \
    --lr_scale 0.1 \
    >> "${log}" 2>&1

  # 3) render @ iter 45000
  echo "  · Render $(date +%T)"
  $PY render.py -m "${model_path}" --iteration ${POST_ITER} >> "${log}" 2>&1

  # 4) metrics → results.json[ours_45000]
  echo "  · Metric $(date +%T)"
  $PY metrics.py -m "${model_path}" >> "${log}" 2>&1

  echo "✓ $(date +%T) ${scene} done"
done
echo "════ COMPRESS+QAT+METRIC ALL DONE $(date '+%F %T') ════"
