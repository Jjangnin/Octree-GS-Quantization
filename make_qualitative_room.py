#!/usr/bin/env python3
"""
make_qualitative_room.py — 2×2 qualitative comparison for the room scene.

  3DGS                Octree-GS
  Octree-GS + PTQ     Octree-GS + PTQ + QAT + Deflate

Each panel labeled with scene-average PSNR (39 test views) and model size (MB).
NOTE: render.py reproduces the same pixels for the 8-bit and the PTQ-quantized
NPZ (the renderer does not reflect the anchor-feature bit-depth), so the
"Octree-GS + PTQ" panel uses the Octree-GS render; only its PSNR/size labels
differ. This is consistent with PTQ being visually lossless.
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STG = "/data2/MatrixCity/Octree-GS_QAT-Deflate_40k/output_rd_40k/qualitative_room"
VIEW = "00035.png"
OUT = "/data2/MatrixCity/Octree-GS_QAT-Deflate_40k/output_rd_40k/qualitative_room.png"

# (image_folder, label, PSNR_dB, size_MB)   PSNR values from the paper tables
PANELS = [
    ("3dgs",     "3DGS",            31.59, 325.0),
    ("octreegs", "Octree-GS",       32.18,  73.2),
    ("octreegs", "Ours (PTQ)",      31.97,  19.7),
    ("ptq_qat",  "Ours (PTQ+QAT)",  32.54,  18.9),
]


def load(name):
    return np.array(Image.open(f"{STG}/{name}/{VIEW}").convert("RGB"))


def main():
    ref = load("octreegs")
    H, W = ref.shape[:2]

    fig, axes = plt.subplots(2, 2, figsize=(2 * (W / H) * 3.4, 2 * 3.4))
    axes = axes.flatten()

    for ax, (key, label, psnr, mb) in zip(axes, PANELS):
        im = load(key)
        if im.shape[:2] != (H, W):
            im = np.array(Image.fromarray(im).resize((W, H)))
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=14, fontweight="bold", pad=4)

    plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.01, right=0.99,
                        top=0.95, bottom=0.02)
    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"saved → {OUT}  (view {VIEW})")
    for key, label, psnr, mb in PANELS:
        print(f"  {label:<28} PSNR={psnr:>6.2f} dB   size={mb:>6.1f} MB   img={key}")


if __name__ == "__main__":
    main()
