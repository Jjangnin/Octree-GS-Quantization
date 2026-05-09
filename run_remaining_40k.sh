#!/bin/bash
# Train remaining 6 scenes at 40k iter (TF32 OFF patched in train.py).
# bonsai already done at outputs/mipnerf360/bonsai/baseline_40k_tf32off/.
# Output dir: outputs/mipnerf360/<scene>/baseline_40k_tf32off/<timestamp>
cd /data2/MatrixCity/Octree-GS_QAT-Deflate_40k
PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python
mkdir -p logs

for scene in bicycle garden stump room counter kitchen; do
  time=$(date +"%Y-%m-%d_%H:%M:%S")
  port=$((10000 + RANDOM % 20000))
  log="logs/${scene}_40k_${time}.log"
  model_path="outputs/mipnerf360/${scene}/baseline_40k_tf32off/${time}"
  echo "▶ $(date +%H:%M:%S) ${scene} → ${model_path}"
  $PY train.py --eval \
    -s data/mipnerf360/${scene} \
    -r -1 --gpu -1 --fork 2 --ratio 1 \
    --data_device cpu \
    --iterations 40000 --port ${port} \
    -m ${model_path} \
    --appearance_dim 0 \
    --visible_threshold -1 --base_layer 10 \
    --dist2level round --update_ratio 0.2 \
    --progressive --init_level -1 \
    --dist_ratio 0.999 --levels -1 \
    --extra_ratio 0.25 --extra_up 0.01 \
    > "${log}" 2>&1
  echo "✓ $(date +%H:%M:%S) ${scene} done"
done
echo "════ ALL DONE $(date '+%F %T') ════"
