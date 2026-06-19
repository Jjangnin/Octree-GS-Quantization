"""
Octree LOD point cloud visualization from Octree-GS PLY checkpoints.
Usage:
    python visualize_lod.py [ply_path] [out_dir]

Defaults to bonsai scene if no args given.
"""
import sys, os, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── helpers ────────────────────────────────────────────────────────────────────

def read_ply(path):
    """Read binary PLY, return (n×k float32 array, list of prop names)."""
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        props = [l.split()[-1] for l in header_lines if l.startswith('property')]
        n_pts = int([l for l in header_lines if l.startswith('element vertex')][0].split()[-1])
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n_pts, len(props))
    return data, props


LOD_COLORS = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d']   # 0=fine … 3=coarse
LOD_LABELS = [f'LOD {i}' for i in range(4)]

FONT_TITLE  = 20
FONT_LABEL  = 16
FONT_TICK   = 14
FONT_LEGEND = 14


def scatter2d(ax, x, y, levels, alpha=0.4, s=0.8):
    for lv in sorted(np.unique(levels)):
        mask = levels == lv
        c = LOD_COLORS[int(lv) % len(LOD_COLORS)]
        ax.scatter(x[mask], y[mask], c=c, s=s, alpha=alpha, rasterized=True, label=LOD_LABELS[int(lv)])


# ── main ───────────────────────────────────────────────────────────────────────

def visualize(ply_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    data, props = read_ply(ply_path)
    x, y, z  = data[:,0], data[:,1], data[:,2]
    levels    = data[:,3].astype(int)
    n_lods    = len(np.unique(levels))
    # path: …/outputs/<dataset>/<scene>/<run>/<timestamp>/point_cloud/<iter>/point_cloud.ply
    parts = os.path.normpath(ply_path).split(os.sep)
    try:
        pc_idx = parts.index('point_cloud')
        scene  = parts[pc_idx - 3]        # 3 levels above point_cloud/ is <scene>
    except ValueError:
        scene  = parts[-5]                 # fallback

    print(f"Scene : {scene}")
    print(f"Points: {len(x):,}")
    for lv in sorted(np.unique(levels)):
        print(f"  LOD {lv}: {(levels==lv).sum():,} pts")

    legend_patches = [mpatches.Patch(color=LOD_COLORS[lv], label=f'LOD {lv}: {(levels==lv).sum():,} pts')
                      for lv in sorted(np.unique(levels))]

    # ── 1. Top-down (XY) ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    scatter2d(ax, x, y, levels)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Octree LOD – top-down (XY)  |  {scene}')
    ax.legend(handles=legend_patches, markerscale=8, loc='upper right')
    out = os.path.join(out_dir, f'{scene}_lod_topdown.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ── 2. Front (XZ) ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    scatter2d(ax, x, z, levels)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(f'Octree LOD – front (XZ)  |  {scene}')
    ax.legend(handles=legend_patches, markerscale=8, loc='upper right')
    out = os.path.join(out_dir, f'{scene}_lod_front.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ── 3. Side (YZ) ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    scatter2d(ax, y, z, levels)
    ax.set_aspect('equal')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title(f'Octree LOD – side (YZ)  |  {scene}')
    ax.legend(handles=legend_patches, markerscale=8, loc='upper right')
    out = os.path.join(out_dir, f'{scene}_lod_side.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ── 4. 3-panel overview ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (xa, ya, xl, yl) in zip(axes, [(x,y,'X','Y'),(x,z,'X','Z'),(y,z,'Y','Z')]):
        scatter2d(ax, xa, ya, levels, alpha=0.3, s=0.5)
        ax.set_aspect('equal')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    axes[1].set_title(f'Octree LOD – {scene}  ({len(x):,} anchors, LOD 0–{n_lods-1})')
    axes[2].legend(handles=legend_patches, markerscale=8, loc='upper right')
    out = os.path.join(out_dir, f'{scene}_lod_overview.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ── 5. Per-LOD subplots ───────────────────────────────────────────────────
    unique_lvs = sorted(np.unique(levels))
    ncols = len(unique_lvs)
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 7))
    if ncols == 1:
        axes = [axes]
    for ax, lv in zip(axes, unique_lvs):
        mask = levels == lv
        ax.scatter(x[mask], y[mask], c=LOD_COLORS[lv % len(LOD_COLORS)],
                   s=0.8, alpha=0.5, rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'LOD {lv}  ({mask.sum():,} pts)')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
    fig.suptitle(f'Per-level top-down  |  {scene}', fontsize=14)
    out = os.path.join(out_dir, f'{scene}_lod_per_level.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == '__main__':
    default_ply = (
        '/data2/MatrixCity/Octree-GS_QAT-Deflate_40k/outputs/mipnerf360/'
        'bonsai/baseline_40k_tf32off/2026-05-09_11:35:32/'
        'point_cloud/iteration_40000/point_cloud.ply'
    )
    ply_path = sys.argv[1] if len(sys.argv) > 1 else default_ply
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else '/data2/MatrixCity/lod_vis'
    visualize(ply_path, out_dir)
