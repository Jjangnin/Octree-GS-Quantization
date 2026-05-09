#!/usr/bin/env python3
"""
train_qat.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Post-training Quantization-Aware fine-tuning on top of a pretrained
Octree-GS checkpoint.

Phase 1: fake-quantize anchor_feat per-LOD (bits from RDO-allocated NPZ),
         fine-tune anchor_feat + MLPs (+ appearance embedding) with STE.
         Other geo attributes (anchor, offset, scaling, opacity, rotation)
         are frozen.

Output goes to <model_path>/point_cloud/iteration_{pretrained+qat_iterations}/
so render.py can pick it up naturally via searchForMaxIteration.

Usage:
  python train_qat.py \
      -m outputs/mipnerf360/garden/baseline/<ts> \
      -s /path/to/mipnerf360/garden \
      --pretrained_iteration 40000 \
      [--qat_iterations 5000] \
      [--lr_scale 0.1]
"""

import os
import sys
import subprocess
import numpy as np

# auto-pick least-loaded GPU (consistent with train.py / compress_optimal.py)
_cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
_result = subprocess.run(_cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
if len(_result) > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in _result[:-1]]))

import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser
import logging

from utils.loss_utils import l1_loss, ssim
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from gaussian_renderer import prefilter_voxel, render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def get_logger(path):
    os.makedirs(path, exist_ok=True)
    logger = logging.getLogger("train_qat")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(path, "train_qat.log"))
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def load_bits_per_lod_from_npz(npz_path):
    """Extract {lod: bits} from the RDO-allocated NPZ (compress_optimal.py output)."""
    data = np.load(npz_path, allow_pickle=False)
    bits = {}
    for k in data.files:
        if k.startswith("bits_lod_"):
            lod = int(k.split("_")[-1])
            bits[lod] = int(data[k][0])
    return bits


def make_qat_optimizer(gaussians, opt, lr_scale):
    """Build Adam covering only anchor_feat + MLPs (+embedding / +feat_bank).
    All other Gaussian attrs must already have requires_grad=False."""
    groups = [
        {'params': [gaussians._anchor_feat],                        'lr': opt.feature_lr         * lr_scale, 'name': 'anchor_feat'},
        {'params': list(gaussians.mlp_opacity.parameters()),        'lr': opt.mlp_opacity_lr_init * lr_scale, 'name': 'mlp_opacity'},
        {'params': list(gaussians.mlp_cov.parameters()),            'lr': opt.mlp_cov_lr_init     * lr_scale, 'name': 'mlp_cov'},
        {'params': list(gaussians.mlp_color.parameters()),          'lr': opt.mlp_color_lr_init   * lr_scale, 'name': 'mlp_color'},
    ]
    if gaussians.appearance_dim > 0:
        groups.append({'params': list(gaussians.embedding_appearance.parameters()),
                       'lr': opt.appearance_lr_init * lr_scale, 'name': 'embedding_appearance'})
    if gaussians.use_feat_bank:
        groups.append({'params': list(gaussians.mlp_feature_bank.parameters()),
                       'lr': opt.mlp_featurebank_lr_init * lr_scale, 'name': 'mlp_featurebank'})
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def make_qat_schedulers(opt, lr_scale, qat_iters, has_appearance, use_feat_bank):
    """Per-group exponential decay: init = <opt>*lr_scale, final follows the
    corresponding opt.*_lr_final field (or init*0.1 for feature_lr which has no final)."""
    def mk(init, final):
        return get_expon_lr_func(
            lr_init=init * lr_scale,
            lr_final=final * lr_scale,
            lr_delay_mult=0.01,
            max_steps=qat_iters,
        )
    scheds = {
        'anchor_feat': mk(opt.feature_lr,        opt.feature_lr * 0.1),
        'mlp_opacity': mk(opt.mlp_opacity_lr_init, opt.mlp_opacity_lr_final),
        'mlp_cov':     mk(opt.mlp_cov_lr_init,     opt.mlp_cov_lr_final),
        'mlp_color':   mk(opt.mlp_color_lr_init,   opt.mlp_color_lr_final),
    }
    if has_appearance:
        scheds['embedding_appearance'] = mk(opt.appearance_lr_init, opt.appearance_lr_final)
    if use_feat_bank:
        scheds['mlp_featurebank'] = mk(opt.mlp_featurebank_lr_init, opt.mlp_featurebank_lr_final)
    return scheds


def update_qat_lr(optimizer, schedulers, iteration):
    for pg in optimizer.param_groups:
        name = pg.get('name')
        if name in schedulers:
            pg['lr'] = schedulers[name](iteration)


@torch.no_grad()
def eval_set(gaussians, cameras, pipe, bg_color, desc="eval"):
    gaussians.eval()
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    psnrs, ssims = [], []
    for cam in tqdm(cameras, desc=desc, leave=False):
        gaussians.set_anchor_mask(cam.camera_center, 1_000_000, cam.resolution_scale)
        vis_mask = prefilter_voxel(cam, gaussians, pipe, bg)
        img = torch.clamp(render(cam, gaussians, pipe, bg, visible_mask=vis_mask)['render'], 0.0, 1.0)
        gt = torch.clamp(cam.original_image.to('cuda'), 0.0, 1.0)[:3]
        psnrs.append(psnr(img.unsqueeze(0), gt.unsqueeze(0)).mean().item())
        ssims.append(ssim(img.unsqueeze(0), gt.unsqueeze(0)).item())
    gaussians.train()
    return float(np.mean(psnrs)) if psnrs else 0.0, float(np.mean(ssims)) if ssims else 0.0


# ────────────────────────────────────────────────────────────────
# Main QAT loop
# ────────────────────────────────────────────────────────────────

def run_qat(dataset, opt, pipe, args, logger):
    pretrained_iter = args.pretrained_iteration
    qat_iters       = args.qat_iterations
    save_iter       = pretrained_iter + qat_iters
    lr_scale        = args.lr_scale

    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.add_opacity_dist, dataset.add_cov_dist,
        dataset.add_color_dist, dataset.add_level, dataset.visible_threshold,
        dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
    )
    # Scene.__init__ with load_iteration=N triggers load_gaussian(NPZ) + load_mlp_checkpoints.
    # After this, _anchor_feat is an empty quantized placeholder; we reload it from PLY below.
    scene = Scene(dataset, gaussians, load_iteration=pretrained_iter,
                  shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)

    iter_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{pretrained_iter}")
    ply_path = os.path.join(iter_dir, "point_cloud.ply")
    npz_path = os.path.join(iter_dir, "point_cloud_quantized.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Expected RDO-allocated NPZ at {npz_path}. "
            "Run compress_optimal.py first to produce per-LOD bit allocation."
        )
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing {ply_path}. QAT needs float weights from PLY.")

    # 1) extract per-LOD bits from the NPZ that compress_optimal.py saved
    bits_per_lod = load_bits_per_lod_from_npz(npz_path)
    if not bits_per_lod:
        raise RuntimeError(f"No bits_lod_* keys found in {npz_path}")
    logger.info(f"bits_per_lod (from NPZ): {bits_per_lod}")

    # 2) force float anchor_feat (+ offset/anchor/etc) via PLY reload.
    #    load_ply_sparse_gaussian resets _anchor_feat_quantized and _geo_quantized to False.
    logger.info(f"Reloading float weights from {ply_path}")
    gaussians.load_ply_sparse_gaussian(ply_path)

    # 3) freeze everything except anchor_feat in Phase 1
    gaussians._anchor.requires_grad_(False)
    gaussians._offset.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    # _opacity, _rotation already have requires_grad=False from load_ply_sparse_gaussian
    logger.info("Frozen params: anchor, offset, scaling, opacity, rotation")

    # 4) enable QAT fake quantization on anchor_feat
    gaussians.enable_qat(bits_per_lod, offset_bits=8)
    logger.info(f"QAT enabled: qat_bits_per_lod={gaussians.qat_bits_per_lod}")

    # 5) kill progressive + densification
    gaussians.progressive = False
    gaussians.coarse_intervals = []

    # 6) optimizer + schedulers (scoped to anchor_feat + MLPs (+ embedding / +feat_bank))
    qat_optimizer = make_qat_optimizer(gaussians, opt, lr_scale)
    schedulers    = make_qat_schedulers(opt, lr_scale, qat_iters,
                                        gaussians.appearance_dim > 0, gaussians.use_feat_bank)

    # baseline eval (fake-quantized forward = what QAT effectively starts from)
    test_cams = scene.getTestCameras()
    bg_color = [1., 1., 1.] if dataset.white_background else [0., 0., 0.]
    if test_cams:
        p0, s0 = eval_set(gaussians, test_cams, pipe, bg_color, desc="Pre-QAT test")
        logger.info(f"[Pre-QAT]  test PSNR: {p0:.4f} dB | SSIM: {s0:.4f}")

    # 7) training loop
    gaussians.train()
    viewpoint_stack = None
    ema = 0.0
    pbar = tqdm(range(1, qat_iters + 1), desc="QAT")
    for it in pbar:
        update_qat_lr(qat_optimizer, schedulers, it)

        if dataset.random_background:
            bg = [np.random.random(), np.random.random(), np.random.random()]
        elif dataset.white_background:
            bg = [1.0, 1.0, 1.0]
        else:
            bg = [0.0, 0.0, 0.0]
        background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        gaussians.set_anchor_mask(cam.camera_center, it, cam.resolution_scale)
        vis_mask = prefilter_voxel(cam, gaussians, pipe, background)
        pkg = render(cam, gaussians, pipe, background, visible_mask=vis_mask, retain_grad=False)

        image   = pkg["render"]
        scaling = pkg["scaling"]
        gt      = cam.original_image.cuda()

        Ll1   = l1_loss(image, gt)
        Lssim = 1.0 - ssim(image, gt)
        if scaling.shape[0] > 0:
            Lscale = scaling.prod(dim=1).mean()
        else:
            Lscale = torch.tensor(0.0, device="cuda")
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim + 0.01 * Lscale

        loss.backward()
        qat_optimizer.step()
        qat_optimizer.zero_grad(set_to_none=True)

        ema = 0.4 * loss.item() + 0.6 * ema
        if it % 10 == 0:
            pbar.set_postfix({"loss": f"{ema:.5f}"})
    pbar.close()

    # 8) save to iteration_{pretrained+qat}/ — NPZ keeps the same RDO per-LOD bits
    logger.info(f"Saving QAT output at iteration {save_iter} (pretrained {pretrained_iter} + qat {qat_iters})")
    scene.save(save_iter, lod_bits_dict=gaussians.qat_bits_per_lod)

    # 9) post-QAT test eval for sanity
    if test_cams:
        p1, s1 = eval_set(gaussians, test_cams, pipe, bg_color, desc="Post-QAT test")
        logger.info(f"[Post-QAT] test PSNR: {p1:.4f} dB | SSIM: {s1:.4f}")


# ────────────────────────────────────────────────────────────────
# Entry
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(description="QAT fine-tuning on top of a pretrained Octree-GS + compress_optimal NPZ")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--pretrained_iteration", type=int, required=True,
                        help="iteration directory to load pretrained weights from (e.g. 40000)")
    parser.add_argument("--qat_iterations", type=int, default=5000,
                        help="QAT fine-tuning steps (default 5000)")
    parser.add_argument("--lr_scale", type=float, default=0.1,
                        help="multiplier applied to all base LRs during QAT (default 0.1)")
    parser.add_argument("--gpu", type=str, default='-1')
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    logger = get_logger(args.model_path)
    logger.info(f"QAT args: {args}")

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(False)

    dataset = lp.extract(args)
    opt     = op.extract(args)
    pipe    = pp.extract(args)

    run_qat(dataset, opt, pipe, args, logger)
    logger.info("QAT complete.")
