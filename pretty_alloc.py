#!/usr/bin/env python3
"""
pretty_alloc.py — 여러 데이터셋의 JSON을 받아 target_bpf 근처의 LOD별 bit allocation을 표 형태로 출력.

Usage:
  python pretty_alloc.py outputs/rdo_test/clean/*.json --target_bpf 5.0
"""
import argparse
import json
import os
import sys


def pareto(rd):
    pts = sorted(rd, key=lambda x: x["bpf"])
    out, best = [], -1e9
    for p in pts:
        if p["psnr"] > best:
            best = p["psnr"]
            out.append(p)
    return out


def nearest(frontier, target_bpf):
    return min(frontier, key=lambda p: abs(p["bpf"] - target_bpf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--target_bpf", type=float, default=5.0)
    args = ap.parse_args()

    rows = []
    max_lod_seen = -1
    for path in args.jsons:
        if not os.path.exists(path):
            print(f"[skip] {path} 없음", file=sys.stderr)
            continue
        d = json.load(open(path))
        # dataset name fallback
        name = d.get("dataset")
        if not name:
            mp_parts = d.get("model_path", path).rstrip("/").split("/")
            if len(mp_parts) >= 2 and mp_parts[-1].lower() in ("baseline", "default", "exp"):
                name = f"{mp_parts[-2]}/{mp_parts[-1]}"
            else:
                name = mp_parts[-1]

        baseline_psnr = d.get("baseline_psnr") or d["baseline"]["psnr"]
        ply_mb = d.get("ply_mb", 0.0)

        pt = nearest(pareto(d["rd_curve"]), args.target_bpf)
        alloc = {int(k): int(v) for k, v in pt["allocation"].items()}
        max_lod_seen = max(max_lod_seen, max(alloc) if alloc else -1)

        rows.append({
            "name": name,
            "baseline": baseline_psnr,
            "psnr": pt["psnr"],
            "drop": pt["psnr_drop"],
            "bpf": pt["bpf"],
            "size_mb": pt.get("size_mb", 0.0),
            "ply_mb": ply_mb,
            "alloc": alloc,
        })

    if not rows:
        print("결과 없음", file=sys.stderr)
        sys.exit(1)

    L = max_lod_seen + 1
    print(f"\n=== Target bpf ≈ {args.target_bpf:.2f} | LOD별 bit allocation ===\n")
    name_w = max(len(r["name"]) for r in rows)
    name_w = max(name_w, 12)

    # Header
    lod_hdr = "  ".join(f"L{l}b" for l in range(L))
    header = (f"  {'Dataset':<{name_w}}  {'base':>7}  {'comp':>7}  {'drop':>7}  "
              f"{'bpf':>5}  {'NPZ MB':>7}  {'PLY MB':>7}  {'ratio':>6}  {lod_hdr}")
    print(header)
    print("─" * len(header))
    for r in rows:
        ratio = r["ply_mb"] / r["size_mb"] if r["size_mb"] > 0 else 0.0
        bits_str = "  ".join(
            f"{r['alloc'].get(l, '-'):>3}" for l in range(L)
        )
        print(
            f"  {r['name']:<{name_w}}  "
            f"{r['baseline']:>7.4f}  {r['psnr']:>7.4f}  {r['drop']:>+7.4f}  "
            f"{r['bpf']:>5.2f}  {r['size_mb']:>7.2f}  {r['ply_mb']:>7.2f}  "
            f"{ratio:>5.2f}×  {bits_str}"
        )

    # LOD별 bit를 데이터셋 간 평균/최빈도로 요약
    print("\n=== LOD별 bit 분포 (데이터셋 간) ===")
    for l in range(L):
        bits = [r["alloc"].get(l) for r in rows if l in r["alloc"]]
        if not bits:
            continue
        avg = sum(bits) / len(bits)
        modes = max(set(bits), key=bits.count)
        print(f"  LOD {l}: avg={avg:.2f} bits, mode={modes} bits, values={bits}")


if __name__ == "__main__":
    main()
