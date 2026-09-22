"""이미지 리사이즈 캐시 생성 — CPU 학습 속도 확보용.

1024x1024 PNG 디코딩이 CPU 학습의 병목이다(이미지당 ~40ms).
해상도별 uint8 배열을 .npy 로 미리 만들어 memmap 으로 읽는다. 무손실.

산출물: data/cache/img_{res}.npy, data/cache/keys_{res}.json

  img_{res}.npy  shape (N, res, res, 3) uint8, RGB
  keys_{res}.json  행 순서와 1:1 대응하는 image key 목록

메모리: 640px 기준 374 x 640 x 640 x 3 = 459 MB
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg, write_json  # noqa: E402

RESOLUTIONS = [640, 384, 256]


def main() -> int:
    cfg = load_cfg()
    src_root = Path(cfg["source_root"])
    cache_dir = Path(cfg["data_root"]) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(cfg["data_root"]) / "labels_index.csv", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["key"])

    for res in RESOLUTIONS:
        npy = cache_dir / f"img_{res}.npy"
        keys_path = cache_dir / f"keys_{res}.json"
        if npy.exists() and keys_path.exists():
            print(f"  {res}px: 이미 존재 — 건너뜀 ({npy.stat().st_size / 1e6:.0f} MB)")
            continue

        arr = np.lib.format.open_memmap(
            npy, mode="w+", dtype=np.uint8, shape=(len(rows), res, res, 3)
        )
        for i, r in enumerate(rows):
            p = src_root / r["src_image"]
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)   # BGR
            if img is None:
                raise IOError(f"읽기 실패: {p}")
            # 원본이 정사각(1024x1024)이므로 letterbox 불필요, 종횡비 보존됨
            assert img.shape[0] == img.shape[1] == cfg["image_size"], f"{p}: {img.shape}"
            small = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
            arr[i] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            if (i + 1) % 100 == 0:
                print(f"    {res}px {i+1}/{len(rows)}")
        arr.flush()
        del arr
        write_json(keys_path, [r["key"] for r in rows])
        print(f"  {res}px: {npy} ({npy.stat().st_size / 1e6:.0f} MB)")

    print(f"[CACHE] 완료 — {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
