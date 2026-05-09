#!/bin/bash
# Deflate ablation: 같은 weight를 deflate=False로 iteration_35001에 재저장 → render → metrics
cd /data2/MatrixCity/Octree-GS_QAT-Deflate
PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python
mkdir -p logs

SRC_ITER=35000
DST_ITER=35001

for SCENE in bicycle garden stump room counter kitchen bonsai; do
  T=$(date +"%H:%M:%S")
  LOG="logs/ablation_${SCENE}.log"
  MP=$(ls -td outputs/mipnerf360/${SCENE}/baseline/*/ 2>/dev/null | head -1)
  if [ -z "$MP" ]; then
    echo "✗ $T ${SCENE}: no model_path, skipping"
    continue
  fi
  MP="${MP%/}"
  echo "▶ $T ${SCENE} → ${MP}"

  # 1. iteration_35001/ 에 deflate=False NPZ 생성
  $PY save_no_deflate.py -m "${MP}" -s data/mipnerf360/${SCENE} \
    --src_iter ${SRC_ITER} --dst_iter ${DST_ITER} > "${LOG}" 2>&1

  # 2. iteration_35001 렌더링
  $PY render.py -m "${MP}" --iteration ${DST_ITER} >> "${LOG}" 2>&1

  # 3. metrics.py — results.json에 ours_35000 + ours_35001 둘 다 갱신됨
  $PY metrics.py -m "${MP}" >> "${LOG}" 2>&1

  T=$(date +"%H:%M:%S")
  echo "✓ $T ${SCENE} done"
done

echo "[$(date +%T)] === ABLATION DONE ==="
