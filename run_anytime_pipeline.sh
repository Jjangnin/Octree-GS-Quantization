#!/bin/bash
# Per-scene anytime convergence data generation:
#   1) restore .original NPZ (8-bit) at iteration_30000
#   2) run instrumented RDO   → output_rd/<scene>_rdo_anytime.json
#   3) run instrumented Sweep → output_rd/<scene>_sweep_anytime.json
#
# Scenes ordered by sweep cost (fast first).
# Run with: nohup ./run_anytime_pipeline.sh > logs/anytime_pipeline.log 2>&1 &

set -e
cd /data2/MatrixCity/Octree-GS_QAT-Deflate
PY=/data3/isjang/.micromamba/envs/octree-gs-cu128/bin/python
mkdir -p logs output_rd

# bonsai already done; remaining 6 scenes ordered fast→slow
SCENES=(counter kitchen garden bicycle stump room)

for scene in "${SCENES[@]}"; do
  echo "════════════════════════════════════════════════════════════════"
  echo "▶ $(date '+%F %T')  scene=${scene}"
  echo "════════════════════════════════════════════════════════════════"

  model_path=$(ls -td outputs/mipnerf360/${scene}/baseline/*/ 2>/dev/null | head -1 | sed 's:/$::')
  if [ -z "$model_path" ]; then
    echo "  ✗ no model_path, skip"; continue
  fi
  ckpt="${model_path}/point_cloud/iteration_30000"
  npz="${ckpt}/point_cloud_quantized.npz"
  orig="${ckpt}/point_cloud_quantized.npz.original"

  # 1) restore .original (8-bit)
  if [ -f "$orig" ]; then
    cp "$orig" "$npz"
    echo "  ✓ restored 8-bit NPZ"
  else
    echo "  ✗ no .original, skip"; continue
  fi

  # 2) RDO
  rdo_out="output_rd/${scene}_rdo_anytime.json"
  rdo_log="logs/${scene}_rdo_anytime.log"
  if [ -f "$rdo_out" ]; then
    echo "  · RDO already exists ($rdo_out), skip"
  else
    echo "  · RDO start  $(date '+%T')"
    $PY rd_sweep.py -m "${model_path}" -s data/mipnerf360/${scene} \
      --iteration 30000 --allowed_bits 2 3 4 5 6 7 8 \
      --method rdo --output "$rdo_out" > "$rdo_log" 2>&1
    echo "  ✓ RDO done   $(date '+%T')"
  fi

  # 3) Sweep
  sw_out="output_rd/${scene}_sweep_anytime.json"
  sw_log="logs/${scene}_sweep_anytime.log"
  if [ -f "$sw_out" ]; then
    echo "  · Sweep already exists ($sw_out), skip"
  else
    echo "  · Sweep start $(date '+%T')"
    $PY rd_sweep.py -m "${model_path}" -s data/mipnerf360/${scene} \
      --iteration 30000 --allowed_bits 2 3 4 5 6 7 8 \
      --method sweep --output "$sw_out" > "$sw_log" 2>&1
    echo "  ✓ Sweep done  $(date '+%T')"
  fi
done

echo ""
echo "════ ALL DONE  $(date '+%F %T') ════"
