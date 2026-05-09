#!/bin/bash
# train_remaining.sh
# ─────────────────────────────────────────────────────────────────
# mipnerf360 의 학습되지 않은 4개 씬 (bicycle, kitchen, room, stump)
# 을 30k iterations 로 학습한다. RTX 5090 두 장에 페어로 분산.
#
# Group 1: bicycle (GPU 0) + room    (GPU 1) — 동시
# Group 2: stump   (GPU 0) + kitchen (GPU 1) — 동시
#
# 출력: outputs/mipnerf360/<scene>/baseline/<timestamp>/
#       (timestamp 가 들어가서 덮어쓰기 절대 없음)
# 로그: logs/<scene>_<timestamp>.log
#
# Usage:
#   bash train_remaining.sh
#   tail -f logs/bicycle_*.log    # 다른 터미널에서 진행 모니터링
# ─────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

# cu128 환경의 python 을 PATH 에 추가 → train.py 가 그 python 으로 실행됨
export PATH=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin:$PATH

PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python

EXP_NAME="baseline"
ITERATIONS=30000          # 기존 bonsai/counter/garden 과 동일
RATIO=1
RESOLUTION=-1
APPEARANCE_DIM=0
FORK=2
BASE_LAYER=10
VISIBLE_THRESHOLD=-1
DIST2LEVEL="round"
UPDATE_RATIO=0.2
DIST_RATIO=0.999
LEVELS=-1
INIT_LEVEL=-1
EXTRA_RATIO=0.25
EXTRA_UP=0.01

mkdir -p logs

train_one() {
    local scene=$1
    local gpu=$2
    local time=$(date "+%Y-%m-%d_%H:%M:%S")
    local model_path="outputs/mipnerf360/${scene}/${EXP_NAME}/${time}"
    local port=$((10000 + RANDOM % 20000))
    local log="logs/${scene}_$(date +%Y%m%d_%H%M%S).log"

    echo "▶ [$(date +%H:%M:%S)] $scene  GPU $gpu  →  $model_path"
    echo "    log: $log"
    $PY train.py --eval \
        -s data/mipnerf360/${scene} \
        -r ${RESOLUTION} \
        --gpu ${gpu} \
        --fork ${FORK} --ratio ${RATIO} \
        --iterations ${ITERATIONS} \
        --port ${port} \
        -m ${model_path} \
        --appearance_dim ${APPEARANCE_DIM} \
        --visible_threshold ${VISIBLE_THRESHOLD} \
        --base_layer ${BASE_LAYER} \
        --dist2level ${DIST2LEVEL} \
        --update_ratio ${UPDATE_RATIO} \
        --progressive \
        --init_level ${INIT_LEVEL} \
        --dist_ratio ${DIST_RATIO} \
        --levels ${LEVELS} \
        --extra_ratio ${EXTRA_RATIO} --extra_up ${EXTRA_UP} \
        > "$log" 2>&1
    echo "✓ [$(date +%H:%M:%S)] $scene 완료"
}

T_START=$(date +%s)

# 환경 변수 GPU_PLAN=parallel 이면 GPU 0+1 동시, 그 외엔 GPU 0 만 직렬.
# 기본값 serial — GPU 1 이 다른 사람 거일 수 있음.
GPU_PLAN=${GPU_PLAN:-serial}

if [ "$GPU_PLAN" = "parallel" ]; then
    echo "═══ 병렬 (GPU 0 + 1) 모드 ═══"
    # Group 1: bicycle (GPU 0) + room (GPU 1)
    train_one bicycle 0 &
    sleep 30
    train_one room 1 &
    wait
    # Group 2: stump (GPU 0) + kitchen (GPU 1)
    train_one stump 0 &
    sleep 30
    train_one kitchen 1 &
    wait
else
    echo "═══ 직렬 (GPU 0 만) 모드 ═══"
    train_one bicycle 0
    train_one room 0
    train_one stump 0
    train_one kitchen 0
fi

T_END=$(date +%s)
echo "전체 학습 시간: $((T_END - T_START)) 초 ($((($T_END - $T_START) / 60)) 분)"

echo ""
echo "학습된 모델 디렉토리:"
ls -d outputs/mipnerf360/*/baseline/*/

echo ""
echo "다음 단계: NPZ 백업 (RDO 실험 누적 위험 방지)"
echo "  find outputs/mipnerf360-name 'point_cloud_quantized.npz' \\"
echo "       -exec sh -c 'cp \"\$1\" \"\${1}.original\"' _ {} \\;"
