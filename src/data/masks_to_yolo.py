"""SOP-DATA-02 — 마스크 → YOLO bbox 라벨 변환 (SOP-MVTEC-CABLE-001 §4.2)

산출물:
  data/yolo/multi/{images,labels}/    8클래스 검출용
  data/yolo/binary/{images,labels}/   OK/NG 단일클래스 검출용
  data/labels_index.csv
  reports/label_check/*.png           육안 검증용 오버레이 (게이트 항목)
  reports/label_conversion_cable.md

원본 이미지는 하드링크로 연결한다(용량 중복 없음, 원본 불변).
"""
from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import env_stamp, image_key, load_cfg, load_classes, set_seed  # noqa: E402

N_CHECK_DEFECT = 20   # 게이트: 육안 검증 20장
N_CHECK_GOOD = 4


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)           # NTFS 하드링크 (동일 볼륨)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def components_to_boxes(mask: np.ndarray, min_area_px: int, connectivity: int):
    """이진 마스크 → [(x0,y0,x1,y1,area), ...], 제거된 요소 수"""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    boxes, removed = [], 0
    for i in range(1, n):  # 0 = 배경
        x, y, w, h, area = stats[i]
        if area < min_area_px:
            removed += 1
            continue
        boxes.append((int(x), int(y), int(x + w - 1), int(y + h - 1), int(area)))
    boxes.sort(key=lambda b: -b[4])
    return boxes, removed


def to_yolo(box, W: int, H: int):
    x0, y0, x1, y1, _ = box
    cx = (x0 + x1 + 1) / 2 / W
    cy = (y0 + y1 + 1) / 2 / H
    w = (x1 - x0 + 1) / W
    h = (y1 - y0 + 1) / H
    clip = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return clip(cx), clip(cy), clip(w), clip(h)


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    set_seed(cfg["seed"])

    root = Path(cfg["source_root"])
    yolo_root = Path(cfg["yolo_root"])
    reports = Path(cfg["reports_dir"])
    check_dir = reports / "label_check"
    check_dir.mkdir(parents=True, exist_ok=True)

    W = H = cfg["image_size"]
    min_area_px = int(cfg["min_component_area_ratio"] * W * H)
    thr = cfg["mask_binarize_threshold"]
    conn = cfg["connectivity"]

    name_to_id = {v: k for k, v in classes["multi"].items()}

    index_rows = []
    per_class = defaultdict(lambda: {"imgs": 0, "boxes": 0, "removed": 0})
    total_removed = 0
    defect_keys, good_keys = [], []

    for split in ("train", "test"):
        for d in sorted((root / split).iterdir()):
            if not d.is_dir():
                continue
            cls = d.name
            is_defect = cls != "good"
            for img_path in sorted(d.glob("*.png")):
                key = image_key(split, cls, img_path.stem)

                for variant in ("multi", "binary"):
                    link_or_copy(img_path, yolo_root / variant / "images" / f"{key}.png")

                boxes, removed = [], 0
                if is_defect:
                    mp = root / "ground_truth" / cls / f"{img_path.stem}_mask.png"
                    if not mp.exists():
                        raise FileNotFoundError(f"마스크 없음: {mp} (§4.1 검증 선행 필요)")
                    m = np.array(Image.open(mp).convert("L"))
                    mb = (m > thr).astype(np.uint8)
                    boxes, removed = components_to_boxes(mb, min_area_px, conn)
                    if not boxes:
                        raise ValueError(
                            f"{key}: 유효 연결요소 0개 (min_area_px={min_area_px}). "
                            f"임계값 재검토 필요"
                        )
                    total_removed += removed
                    defect_keys.append((key, cls, img_path, mp, boxes))
                else:
                    good_keys.append((key, img_path))

                cid = name_to_id[cls] if is_defect else None
                multi_lines, binary_lines = [], []
                for b in boxes:
                    cx, cy, bw, bh = to_yolo(b, W, H)
                    multi_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    binary_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                # good → 0바이트 빈 라벨 (배경 샘플, §4.2-5)
                for variant, lines in (("multi", multi_lines), ("binary", binary_lines)):
                    lp = yolo_root / variant / "labels" / f"{key}.txt"
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    lp.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

                area_ratio = sum(b[4] for b in boxes) / (W * H) if boxes else 0.0
                index_rows.append({
                    "key": key, "orig_split": split, "class": cls,
                    "stem": img_path.stem,
                    "image_label": 1 if is_defect else 0,
                    "class_id": cid if cid is not None else "",
                    "n_boxes": len(boxes),
                    "n_removed_components": removed,
                    "defect_area_ratio": f"{area_ratio:.6f}",
                    "src_image": str(img_path.relative_to(root)),
                })
                per_class[cls]["imgs"] += 1
                per_class[cls]["boxes"] += len(boxes)
                per_class[cls]["removed"] += removed

    # --- labels_index.csv ----------------------------------------------------
    idx_path = Path(cfg["data_root"]) / "labels_index.csv"
    with open(idx_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)

    # --- 육안 검증 오버레이 (게이트) -----------------------------------------
    for f in check_dir.glob("*.png"):
        f.unlink()
    rng = random.Random(cfg["seed"])
    # 클래스 고르게: 8클래스에서 순회하며 표본 추출
    by_cls = defaultdict(list)
    for item in defect_keys:
        by_cls[item[1]].append(item)
    picked = []
    cls_order = sorted(by_cls)
    i = 0
    while len(picked) < N_CHECK_DEFECT:
        c = cls_order[i % len(cls_order)]
        pool = [x for x in by_cls[c] if x not in picked]
        if pool:
            picked.append(rng.choice(pool))
        i += 1
        if i > 500:
            break

    for key, cls, img_path, mp, boxes in picked:
        img = cv2.imread(str(img_path))
        m = np.array(Image.open(mp).convert("L"))
        overlay = img.copy()
        overlay[m > thr] = (0, 0, 255)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        lines = (yolo_root / "multi" / "labels" / f"{key}.txt").read_text(encoding="utf-8").split("\n")
        for ln in [l for l in lines if l.strip()]:
            cid, cx, cy, bw, bh = ln.split()
            cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
            x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
            x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 4)
            cv2.putText(img, classes["multi"][int(cid)], (x0, max(y0 - 12, 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.putText(img, f"{key}  boxes={len(boxes)}", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        cv2.imwrite(str(check_dir / f"{key}.png"), img)

    for key, img_path in rng.sample(good_keys, min(N_CHECK_GOOD, len(good_keys))):
        img = cv2.imread(str(img_path))
        cv2.putText(img, f"{key}  GOOD (empty label)", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        cv2.imwrite(str(check_dir / f"{key}.png"), img)

    # --- 리포트 --------------------------------------------------------------
    env = env_stamp()
    n_def = sum(1 for r in index_rows if r["image_label"] == 1)
    n_good = len(index_rows) - n_def
    n_boxes = sum(r["n_boxes"] for r in index_rows)
    multi_cnt = sum(1 for _ in (yolo_root / "multi" / "labels").glob("*.txt"))

    md = [
        "# 라벨 변환 리포트 — MVTec AD `cable`",
        "",
        "- SOP: SOP-MVTEC-CABLE-001 §4.2 (SOP-DATA-02)",
        f"- 실행일: {date.today().isoformat()}",
        f"- git: `{env['git_rev']}`",
        f"- 시드: {cfg['seed']}",
        "",
        "## 파라미터",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 마스크 이진화 임계 | {thr} |",
        f"| 연결성 | {conn}-연결 |",
        f"| 최소 연결요소 면적 | {cfg['min_component_area_ratio']*100:.3f}% = {min_area_px} px |",
        f"| 좌표 포맷 | `<class_id> <cx/W> <cy/H> <w/W> <h/H>` (0~1 정규화) |",
        "",
        "## 결과 요약",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 전체 이미지 | {len(index_rows)} |",
        f"| 불량 이미지 | {n_def} |",
        f"| 정상 이미지 (빈 라벨) | {n_good} |",
        f"| 생성 bbox | {n_boxes} |",
        f"| 노이즈 제거 연결요소 | {total_removed} |",
        f"| 라벨 파일 수 (multi / binary) | {multi_cnt} / {multi_cnt} |",
        "",
        "## 클래스별",
        "",
        "| 클래스 | class_id | 이미지 | bbox | 이미지당 bbox | 제거된 요소 |",
        "|---|---|---|---|---|---|",
    ]
    for cls in sorted(per_class):
        s = per_class[cls]
        cid = name_to_id.get(cls, "-")
        avg = s["boxes"] / s["imgs"] if s["imgs"] else 0
        md.append(f"| {cls} | {cid} | {s['imgs']} | {s['boxes']} | {avg:.2f} | {s['removed']} |")

    md += [
        "",
        "## 육안 검증 (게이트 항목)",
        "",
        f"- `reports/label_check/` 에 불량 {len(picked)}장 + 정상 {min(N_CHECK_GOOD, len(good_keys))}장 오버레이 저장",
        "- 빨강 반투명 = ground truth 마스크, 초록 사각형 = 생성된 bbox",
        "- **확인 항목**: bbox가 마스크를 정확히 감싸는가 / 좌표축이 뒤바뀌지 않았는가 / 다중 결함이 개별 bbox로 분리되었는가",
        "",
        "## 비고",
        "",
        "- 원본 이미지는 하드링크로 연결됨 (용량 중복 없음, 원본 디렉토리 불변)",
        "- 정상 이미지는 0바이트 라벨 파일을 생성해 배경 샘플로 학습에 포함시킨다",
        f"- 상세: `data/labels_index.csv`",
    ]
    (reports / "label_conversion_cable.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[SOP-DATA-02] 이미지 {len(index_rows)} (불량 {n_def} / 정상 {n_good}), bbox {n_boxes}, "
          f"노이즈 제거 {total_removed}")
    print(f"  -> {yolo_root}/multi, {yolo_root}/binary")
    print(f"  -> {idx_path}")
    print(f"  -> {check_dir} ({len(list(check_dir.glob('*.png')))}장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
