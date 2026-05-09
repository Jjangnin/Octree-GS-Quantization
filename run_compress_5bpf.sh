#!/bin/bash
cd /data2/MatrixCity/Octree-GS_QAT-Deflate
PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python
mkdir -p logs output_rd

# 1) NPZ 정합성 보장 — .original이 있으면 거기로부터 복구, 없으면 백업
#    (idempotent: 매 실행 시 default-8bit baseline에서 시작 보장)
echo "═══ Preparing NPZs (restore from .original if exists, else backup) ═══"
find outputs/mipnerf360 -name 'point_cloud_quantized.npz' -not -name '*.original' | while read npz; do
  orig="${npz}.original"
  if [ -f "$orig" ]; then
    cp "$orig" "$npz"
    echo "  restored: $npz"
  else
    cp "$npz" "$orig"
    echo "  backup:   $orig"
  fi
done
echo ""

# 2) 각 scene 순차 RDO 압축 (target_bpf=5, max_drop 비활성)
for scene in bicycle garden stump room counter kitchen bonsai; do
  time=$(date +"%Y-%m-%d_%H:%M:%S")
  log="logs/compress_${scene}_${time}.log"
  model_path=$(ls -td outputs/mipnerf360/${scene}/baseline/*/ 2>/dev/null | head -1)
  if [ -z "$model_path" ]; then
    echo "✗ $(date +%H:%M:%S) ${scene}: no model_path found, skipping"
    continue
  fi
  model_path="${model_path%/}"
  echo "▶ $(date +%H:%M:%S) ${scene} → ${model_path}  log=${log}"
  $PY compress_optimal.py \
    -m "${model_path}" \
    -s data/mipnerf360/${scene} \
    --iteration 30000 \
    --allowed_bits 2 3 4 5 6 7 8 \
    --target_bpf 5 \
    --max_drop 999 \
    --output_dir output_rd \
    > "${log}" 2>&1
  echo "✓ $(date +%H:%M:%S) ${scene} done"
done
