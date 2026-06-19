# Octree-GS: QAT + Deflate Compression (40k Baseline)

Compression pipeline on top of Octree-GS: **Lagrangian RDO → QAT fine-tuning → Deflate (NPZ)**.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate octree-gs
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Key dependencies: Python 3.7.13, PyTorch 1.12.1, CUDA 11.6, pytorch-scatter.

> **Note on the convenience scripts.** `run_compress_qat_40k.sh`, `report_40k.sh`, and
> `report_compressed_40k.sh` contain a hardcoded working directory (`cd /data2/MatrixCity/...`)
> and a hardcoded interpreter path (`PY=/data3/isjang/.../bin/python`). Before running them on a
> fresh clone, edit those two lines to match your environment, or just use the equivalent
> `python ...` commands shown in each step below.

---

## Dataset

MipNeRF-360 (7 scenes): `bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, `stump`

```
data/mipnerf360/
├── bicycle/
├── bonsai/
├── counter/
├── garden/
├── kitchen/
├── room/
└── stump/
```

---

## Usage

### 1. Baseline Training (40k iterations)

```bash
bash train_mipnerf360.sh   # launches all scenes in parallel via train.sh
```

`train_mipnerf360.sh` dispatches each scene through `train.sh` (the shared argument wrapper),
which finally calls `train.py`. To train a single scene manually:

```bash
python train.py --eval \
  -s data/mipnerf360/<scene> \
  -r -1 --gpu -1 --fork 2 --ratio 1 \
  --data_device cpu --iterations 40000 \
  -m outputs/mipnerf360/<scene>/baseline_40k_tf32off/<timestamp> \
  --appearance_dim 0 --visible_threshold -1 --base_layer 10 \
  --dist2level round --update_ratio 0.2 --progressive \
  --init_level -1 --dist_ratio 0.999 --levels -1 \
  --extra_ratio 0.25 --extra_up 0.01
```

### 2. Compression Pipeline (RDO → QAT → Render → Metrics)

Run all 7 scenes end-to-end:

```bash
bash run_compress_qat_40k.sh
```

This executes the following steps per scene:

**Step 1 — RDO: find optimal per-LOD bit allocation**

```bash
python compress_optimal.py \
  -m <model_path> -s data/mipnerf360/<scene> \
  --iteration 40000 \
  --allowed_bits 2 3 4 5 6 7 8 \
  --target_bpf 5 --max_drop 999 \
  --output_dir output_rd_40k
```

**Step 2 — QAT fine-tuning (40k → 45k)**

```bash
python train_qat.py \
  -m <model_path> -s data/mipnerf360/<scene> \
  --data_device cpu \
  --pretrained_iteration 40000 \
  --qat_iterations 5000 --lr_scale 0.1
```

Geometry parameters are frozen; `anchor_feat` is fake-quantized per-LOD with STE.

**Step 3 — Render & Metrics**

```bash
python render.py -m <model_path> --iteration 45000
python metrics.py -m <model_path>
```

Results are written to `<model_path>/results.json` under key `ours_45000`.

### 3. Report

```bash
bash report_40k.sh             # baseline metrics at iter 40000
bash report_compressed_40k.sh  # baseline vs QAT+Deflate + compression stats
```

---

## Output Structure

```
outputs/mipnerf360/<scene>/baseline_40k_tf32off/<timestamp>/
├── point_cloud/
│   ├── iteration_40000/
│   │   ├── point_cloud.ply              # float32 original
│   │   └── point_cloud_quantized.npz   # RDO bit allocation result
│   └── iteration_45000/
│       └── point_cloud.ply              # QAT fine-tuned
├── test/
│   ├── ours_40000/renders/
│   └── ours_45000/renders/
└── results.json

output_rd_40k/
├── <scene>_compress_result.json         # compression stats (bpf, size, PSNR)
└── <scene>_rd_curve.png
```

---

## Results

Settings: target bpf = 5, allowed bits = {2,3,4,5,6,7,8}, QAT 5000 steps (lr_scale = 0.1)

> PSNR values below are after RDO quantization (before QAT fine-tuning).

| Scene | Baseline PSNR | RDO PSNR | Drop | BPF | PLY (MB) | NPZ (MB) | Ratio |
|---|---|---|---|---|---|---|---|
| bicycle | 24.91 | 24.76 | 0.16 | 5.13 | 230.3 | 59.1 | 3.90× |
| bonsai | 31.62 | 31.25 | 0.37 | 5.11 | 55.2 | 13.9 | 3.98× |
| counter | 29.46 | 29.03 | 0.43 | 5.00 | 65.8 | 16.9 | 3.90× |
| garden | 27.46 | 27.09 | 0.38 | 5.00 | 206.7 | 54.2 | 3.81× |
| kitchen | 31.09 | 30.55 | 0.54 | 5.00 | 55.5 | 14.3 | 3.89× |

---

## Key Scripts

| File | Description |
|---|---|
| `train_mipnerf360.sh` | Launch baseline training for all 7 scenes |
| `train.sh` | Argument wrapper around `train.py` (used by `train_mipnerf360.sh`) |
| `run_compress_qat_40k.sh` | Full RDO → QAT → render → metrics pipeline |
| `report_40k.sh` / `report_compressed_40k.sh` | Aggregate metric tables |
| `train.py` | Octree-GS baseline training |
| `train_qat.py` | QAT fine-tuning |
| `compress_optimal.py` | RDO / sweep-based optimal bit allocation |
| `rdo.py` | Lagrangian RDO implementation |
| `rd_sweep.py` | Brute-force RD curve generation |
| `render.py` | Test-view rendering |
| `metrics.py` | PSNR / SSIM / LPIPS evaluation |
| `make_rd_report.py` | RD curve and table visualization |
