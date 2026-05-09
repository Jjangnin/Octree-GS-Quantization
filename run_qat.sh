#!/bin/bash
cd /data2/MatrixCity/Octree-GS_QAT-Deflate
PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python
mkdir -p logs

# Pre-condition: each scene has iteration_30000/{point_cloud.ply, point_cloud_quantized.npz}
#   where the NPZ holds the RDO-allocated bits (from compress_optimal.py).
# Output: iteration_35000/ (= 30000 + 5000 qat steps), with the SAME LOD bits.
for scene in bicycle garden stump room counter kitchen bonsai; do
  time=$(date +"%Y-%m-%d_%H:%M:%S")
  log="logs/qat_${scene}_${time}.log"
  model_path=$(ls -td outputs/mipnerf360/${scene}/baseline/*/ 2>/dev/null | head -1)
  if [ -z "$model_path" ]; then
    echo "✗ $(date +%H:%M:%S) ${scene}: no model_path found, skipping"
    continue
  fi
  model_path="${model_path%/}"
  echo "▶ $(date +%H:%M:%S) ${scene} → ${model_path}  log=${log}"
  $PY train_qat.py \
    -m "${model_path}" \
    -s data/mipnerf360/${scene} \
    --data_device cpu \
    --pretrained_iteration 30000 \
    --qat_iterations 5000 \
    --lr_scale 0.1 \
    > "${log}" 2>&1
  echo "✓ $(date +%H:%M:%S) ${scene} done"
done
