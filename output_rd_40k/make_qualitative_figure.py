"""
GT 기준으로 각 방법별 로컬 품질 차이를 계산해,
PTQ+QAT가 가장 눈에 띄게 좋은 이미지 + 패치를 찾아
논문용 figure를 생성한다.

필터 조건:
  - GT 패치 평균 밝기 > MIN_BRIGHTNESS (어두운 영역 제외)
  - GT 패치 표준편차 > MIN_STD (단조로운 영역 제외)
  - ptq_qat SSIM > QAT_MIN_SSIM (방법 자체가 망가진 경우 제외)
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

BASE    = "/data2/MatrixCity/Octree-GS_QAT-Deflate_40k/output_rd_40k/qualitative_room"
GT_DIR  = os.path.join(BASE, "gt")
OUT_PATH = os.path.join(BASE, "..", "qualitative_room_annotated.png")

METHODS = {
    "3DGS":           "3dgs",
    "Octree-GS":      "octreegs",
    "Ours (PTQ)":     "ptq",
    "Ours (PTQ+QAT)": "ptq_qat",
}

QAT_KEY    = "Ours (PTQ+QAT)"
OTHER_KEYS = [k for k in METHODS if k != QAT_KEY]

PATCH_SIZE     = 220    # 패치 크기
PATCH_STRIDE   = 40     # 탐색 간격
MIN_BRIGHTNESS = 0.18   # GT 패치 최소 평균 밝기 (0~1)
MIN_STD        = 0.06   # GT 패치 최소 표준편차 (텍스처 요구)
QAT_MIN_SSIM   = 0.60   # PTQ+QAT 최소 SSIM (완전 망가진 패치 제외)

OVERVIEW_SCALE = 0.40   # 상단 overview 이미지 축소 비율
ZOOM_SIZE      = 320    # crop 인셋 크기 (px)
BOX_COLOR      = (220, 30, 30)
BOX_WIDTH      = 5
GAP            = 8
LABEL_H        = 52
TOP_N          = 5      # 상위 N개 후보 저장

TARGET_SIZE = None  # (W, H), GT 기준


def load_gray(path):
    img = Image.open(path).convert("L")
    if TARGET_SIZE and img.size != TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    if TARGET_SIZE and img.size != TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return np.array(img)


def try_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def scan_image(img_idx_str):
    """
    한 이미지에 대해 유효 패치 중 best score를 반환.
    score = ssim(qat, gt) - mean(ssim(others, gt))
           * [qat > ALL others 조건 보너스]
    """
    fname = img_idx_str + ".png"
    gt_g  = load_gray(os.path.join(GT_DIR, fname))
    H, W  = gt_g.shape
    ps    = PATCH_SIZE
    st    = PATCH_STRIDE

    method_g = {lbl: load_gray(os.path.join(BASE, METHODS[lbl], fname))
                for lbl in METHODS}

    candidates = []
    for y in range(0, H - ps, st):
        for x in range(0, W - ps, st):
            gt_patch = gt_g[y:y+ps, x:x+ps]

            # ── 필터: 밝기 & 분산 ──
            if gt_patch.mean() < MIN_BRIGHTNESS:
                continue
            if gt_patch.std() < MIN_STD:
                continue

            # ── 각 방법 SSIM ──
            scores = {}
            for lbl in METHODS:
                p = method_g[lbl][y:y+ps, x:x+ps]
                scores[lbl] = ssim(p, gt_patch, data_range=1.0)

            qat_s = scores[QAT_KEY]
            if qat_s < QAT_MIN_SSIM:
                continue

            others_s = [scores[k] for k in OTHER_KEYS]
            mean_others = np.mean(others_s)

            # qat가 모든 others보다 좋을 때 보너스
            all_better = all(qat_s > s for s in others_s)
            bonus = 0.01 if all_better else 0.0

            score = (qat_s - mean_others) + bonus
            candidates.append((score, y, x, scores))

    if not candidates:
        return None, -np.inf

    candidates.sort(key=lambda c: -c[0])
    best = candidates[0]
    return (best[1], best[2]), best[0]


def find_best(top_n=TOP_N):
    fnames = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".png"))
    print(f"이미지 {len(fnames)}장 탐색 중 (patch={PATCH_SIZE}, stride={PATCH_STRIDE})...")

    ranking = []
    for fname in fnames:
        idx = fname.replace(".png", "")
        yx, score = scan_image(idx)
        if yx is None:
            print(f"  {fname}: 유효 패치 없음 (필터 탈락)")
            continue
        print(f"  {fname}: score={score:.4f}  patch=({yx[0]},{yx[1]})")
        ranking.append((score, idx, yx))

    ranking.sort(key=lambda r: -r[0])
    print(f"\n상위 {top_n}개:")
    for rank, (sc, idx, yx) in enumerate(ranking[:top_n]):
        print(f"  [{rank+1}] {idx}.png  score={sc:.4f}  patch=({yx[0]},{yx[1]})")

    return ranking


def draw_box(draw, ox, oy, px, py, ps, width, color):
    box = [ox + px, oy + py, ox + px + ps, oy + py + ps]
    for bw in range(width):
        draw.rectangle([box[0]-bw, box[1]-bw, box[2]+bw, box[3]+bw], outline=color)


def compute_psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")


def make_figure(img_idx, patch_yx, out_path=None):
    if out_path is None:
        out_path = OUT_PATH

    fname = img_idx + ".png"
    py, px = patch_yx
    ps     = PATCH_SIZE

    imgs = {lbl: load_rgb(os.path.join(BASE, METHODS[lbl], fname)) for lbl in METHODS}
    imgs["GT"] = load_rgb(os.path.join(GT_DIR, fname))

    # 각 방법의 전체 이미지 PSNR (GT 대비)
    psnr_vals = {lbl: compute_psnr(imgs[lbl], imgs["GT"]) for lbl in METHODS}

    H, W, _ = imgs["GT"].shape
    font_lg  = try_font(32)
    font_sm  = try_font(28)

    ov_w = int(W * OVERVIEW_SCALE)
    ov_h = int(H * OVERVIEW_SCALE)
    box_px = int(px * OVERVIEW_SCALE)
    box_py = int(py * OVERVIEW_SCALE)
    box_ps = int(ps * OVERVIEW_SCALE)

    top_labels = list(METHODS.keys())
    top_total_w = len(top_labels) * ov_w + (len(top_labels) - 1) * GAP
    top_total_h = ov_h + LABEL_H

    top_img = Image.new("RGB", (top_total_w, top_total_h), (255, 255, 255))
    draw_top = ImageDraw.Draw(top_img)

    best_psnr = max(psnr_vals.values())
    font_psnr_top = try_font(28)
    for i, lbl in enumerate(top_labels):
        ox = i * (ov_w + GAP)
        thumb = Image.fromarray(imgs[lbl]).resize((ov_w, ov_h), Image.LANCZOS)
        top_img.paste(thumb, (ox, LABEL_H))
        draw_box(draw_top, ox, LABEL_H, box_px, box_py, box_ps,
                 max(2, BOX_WIDTH - 1), BOX_COLOR)
        # 레이블 + PSNR 한 줄로
        pv = psnr_vals[lbl]
        is_best = abs(pv - best_psnr) < 0.01
        label_text = f"{lbl}   {pv:.2f} dB"
        color = (200, 120, 0) if is_best else (30, 30, 30)
        tw = draw_top.textlength(label_text, font=font_psnr_top)
        draw_top.text((ox + ov_w//2 - tw//2, 10), label_text,
                      fill=color, font=font_psnr_top)

    # ── 하단 crop row: GT | 3DGS | Octree | PTQ | PTQ+QAT ──
    zoom_labels = ["GT"] + top_labels
    bot_total_w = len(zoom_labels) * ZOOM_SIZE + (len(zoom_labels) - 1) * GAP
    bot_total_h = ZOOM_SIZE + LABEL_H

    bot_img = Image.new("RGB", (bot_total_w, bot_total_h), (255, 255, 255))
    draw_bot = ImageDraw.Draw(bot_img)

    font_psnr = try_font(24)

    for i, lbl in enumerate(zoom_labels):
        ox = i * (ZOOM_SIZE + GAP)
        src = imgs[lbl]
        crop = Image.fromarray(src[py:py+ps, px:px+ps]).resize(
            (ZOOM_SIZE, ZOOM_SIZE), Image.LANCZOS)

        if lbl != "GT":
            frame = Image.new("RGB", (ZOOM_SIZE, ZOOM_SIZE), BOX_COLOR)
            inner = ZOOM_SIZE - BOX_WIDTH * 2
            frame.paste(crop.resize((inner, inner), Image.LANCZOS), (BOX_WIDTH, BOX_WIDTH))
            crop = frame

        bot_img.paste(crop, (ox, LABEL_H))

        # 레이블
        color = (100, 100, 100) if lbl == "GT" else (30, 30, 30)
        tw = draw_bot.textlength(lbl, font=font_sm)
        draw_bot.text((ox + ZOOM_SIZE//2 - tw//2, 8), lbl, fill=color, font=font_sm)


    # ── 합치기 ──
    sep    = 12
    fin_w  = max(top_total_w, bot_total_w)
    fin_h  = top_total_h + sep + bot_total_h
    final  = Image.new("RGB", (fin_w, fin_h), (255, 255, 255))
    final.paste(top_img, ((fin_w - top_total_w) // 2, 0))
    final.paste(bot_img, ((fin_w - bot_total_w) // 2, top_total_h + sep))

    final.save(out_path)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    sample_gt  = Image.open(os.path.join(GT_DIR, "00000.png"))
    TARGET_SIZE = sample_gt.size
    print(f"기준 해상도: {TARGET_SIZE}")

    ranking = find_best()

    if not ranking:
        print("유효한 후보가 없습니다.")
        exit(1)

    # 1위로 메인 figure 생성
    best_score, best_idx, best_yx = ranking[0]
    print(f"\n최종 선택: {best_idx}.png  score={best_score:.4f}  patch={best_yx}")
    make_figure(best_idx, best_yx)

    # 상위 5개 미리보기 저장 (확인용)
    preview_dir = os.path.join(BASE, "..", "qualitative_candidates")
    os.makedirs(preview_dir, exist_ok=True)
    for rank, (sc, idx, yx) in enumerate(ranking[:TOP_N]):
        p = os.path.join(preview_dir, f"rank{rank+1}_{idx}_score{sc:.4f}.png")
        make_figure(idx, yx, out_path=p)
    print(f"상위 {TOP_N}개 후보 저장: {preview_dir}/")
