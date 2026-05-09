#!/usr/bin/env python3
"""
RDO vs brute-force sweep 결과 빠른 비교.

Usage:
  python compare_rdo_vs_sweep.py outputs/rdo_test/bonsai/rdo.json \
                                  outputs/rdo_test/bonsai/sweep.json
"""
import json
import sys


def pareto(rd):
    pts = sorted(rd, key=lambda x: x["bpf"])
    out, best = [], -1e9
    for p in pts:
        if p["psnr"] > best:
            best = p["psnr"]
            out.append(p)
    return out


def main():
    if len(sys.argv) != 3:
        print("Usage: compare_rdo_vs_sweep.py <rdo.json> <sweep.json>")
        sys.exit(1)
    rdo_path, sweep_path = sys.argv[1], sys.argv[2]
    r = json.load(open(rdo_path))
    s = json.load(open(sweep_path))

    rdo_pf = pareto(r["rd_curve"])
    sw_pf = pareto(s["rd_curve"])

    print(f"{'method':<8}{'#points':>10}{'#pareto':>10}")
    print(f"{'rdo':<8}{len(r['rd_curve']):>10}{len(rdo_pf):>10}")
    print(f"{'sweep':<8}{len(s['rd_curve']):>10}{len(sw_pf):>10}")
    print()
    print(f"baseline psnr  rdo: {r['baseline']['psnr']:.4f}  sweep: {s['baseline']['psnr']:.4f}")

    print("\nRDO Pareto:")
    for p in rdo_pf:
        print(f"  bpf={p['bpf']:.3f}  psnr={p['psnr']:.4f}  drop={p['psnr_drop']:+.4f}  alloc={p['allocation']}")

    print("\nSweep Pareto:")
    for p in sw_pf:
        print(f"  bpf={p['bpf']:.3f}  psnr={p['psnr']:.4f}  drop={p['psnr_drop']:+.4f}  alloc={p['allocation']}")

    # 같은 bpf 영역에서 PSNR 차이 측정 (RDO Pareto 점마다 sweep Pareto 위에서 가장 가까운 bpf 점과 비교)
    print("\n=== gap 분석 (같은 alloc 또는 bpf-인접) ===")
    sw_by_alloc = {tuple(sorted(p["allocation"].items())): p for p in s["rd_curve"]}
    max_gap = 0.0
    for p in rdo_pf:
        key = tuple(sorted(p["allocation"].items()))
        sw_match = sw_by_alloc.get(key)
        if sw_match is None:
            print(f"  bpf={p['bpf']:.3f} alloc={p['allocation']} → sweep에 같은 alloc 없음")
            continue
        gap = sw_match["psnr"] - p["psnr"]
        max_gap = max(max_gap, abs(gap))
        marker = "  " if abs(gap) < 0.01 else "* "
        print(f"{marker}bpf={p['bpf']:.3f}  rdo psnr={p['psnr']:.4f}  sweep psnr={sw_match['psnr']:.4f}  Δ={gap:+.4f}")
    print(f"\n최대 PSNR gap: {max_gap:.4f} dB")
    print("기준: ≤0.05 dB → RDO 정확,  >0.10 dB → separability 가정 약함")


if __name__ == "__main__":
    main()
