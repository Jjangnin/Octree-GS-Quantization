#!/usr/bin/env python3
"""
patch_jsons.py — rd_sweep.py 가 만든 기존 JSON에 빠진/잘못된 키(ply_mb, dataset)를 보강.

dataset 키가 timestamp 형식(YYYY-MM-DD_HH:MM:SS) 으로 들어가 있으면 재계산한다.

Usage:
  python patch_jsons.py outputs/rdo_test/multi/*.json
  python patch_jsons.py path/to/rd.json [more.json ...]
"""
import json
import os
import re
import sys


# timestamp 형식: 2026-04-23_14:10:30
TS_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$")
GENERIC_LEAF = ("baseline", "default", "exp")


def derive_dataset_name(model_path: str) -> str:
    """model_path 에서 사람이 알아볼 수 있는 dataset 이름 추출.

    형식 1: .../<scene>/<exp>/<timestamp>   → "<scene>/<exp>"   (예: bicycle/baseline)
    형식 2: .../<scene>/<exp>               → "<scene>/<exp>"   (예: bonsai/baseline)
    형식 3: .../<scene>                     → "<scene>"
    """
    parts = model_path.rstrip("/").split("/")
    if (len(parts) >= 3
            and TS_PAT.match(parts[-1])
            and parts[-2].lower() in GENERIC_LEAF):
        return f"{parts[-3]}/{parts[-2]}"
    if len(parts) >= 2 and parts[-1].lower() in GENERIC_LEAF:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def patch(path: str) -> None:
    d = json.load(open(path))
    changed = False

    if "model_path" not in d:
        print(f"  [skip] {path}: model_path 키 없음 — 건드릴 수 없음")
        return

    mp = d["model_path"].rstrip("/")

    # dataset 키가 없거나, leaf 가 timestamp 형식이면 재계산
    cur_ds = d.get("dataset")
    needs_redo = (cur_ds is None) or TS_PAT.match(str(cur_ds).split("/")[-1])
    if needs_redo:
        d["dataset"] = derive_dataset_name(mp)
        changed = True

    if "ply_mb" not in d:
        ply = os.path.join(mp, "point_cloud", f"iteration_{d['iteration']}", "point_cloud.ply")
        d["ply_mb"] = round(os.path.getsize(ply) / 1e6, 4) if os.path.exists(ply) else 0.0
        changed = True

    if changed:
        json.dump(d, open(path, "w"), indent=2)
        print(f"  [ok] {path}: dataset={d['dataset']}, ply_mb={d['ply_mb']:.2f}")
    else:
        print(f"  [skip] {path}: 이미 보강됨")


def main():
    if len(sys.argv) < 2:
        print("Usage: patch_jsons.py <json> [more.json ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        patch(path)


if __name__ == "__main__":
    main()
