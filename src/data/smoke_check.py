"""P1 게이트 보조 — 라벨 유효성 + Ultralytics 데이터셋 로딩 스모크 테스트.

SOP §4.2 검증 항목(좌표 범위, 클래스 ID, 빈 라벨) + 실제 학습 파이프라인이
분할 파일을 읽을 수 있는지 확인한다. 학습은 하지 않는다.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg, load_classes  # noqa: E402


def check_labels(cfg, classes) -> list[str]:
    problems = []
    for variant in ("multi", "binary"):
        valid_ids = set(classes["multi" if variant == "multi" else "binary"])
        lbl_dir = Path(cfg["yolo_root"]) / variant / "labels"
        img_dir = Path(cfg["yolo_root"]) / variant / "images"
        n_empty = n_box = 0
        for lp in sorted(lbl_dir.glob("*.txt")):
            if not (img_dir / f"{lp.stem}.png").exists():
                problems.append(f"{variant}: 이미지 없는 라벨 {lp.name}")
            txt = lp.read_text(encoding="utf-8").strip()
            if not txt:
                n_empty += 1
                continue
            for i, ln in enumerate(txt.split("\n"), 1):
                parts = ln.split()
                if len(parts) != 5:
                    problems.append(f"{variant}/{lp.name}:{i} 필드 {len(parts)}개 (5개여야 함)")
                    continue
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                if cid not in valid_ids:
                    problems.append(f"{variant}/{lp.name}:{i} 미정의 class_id {cid}")
                for nm, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                    if not 0.0 <= v <= 1.0:
                        problems.append(f"{variant}/{lp.name}:{i} {nm}={v} 범위 이탈")
                if w <= 0 or h <= 0:
                    problems.append(f"{variant}/{lp.name}:{i} 퇴화 bbox w={w} h={h}")
                if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                    problems.append(f"{variant}/{lp.name}:{i} bbox 이미지 경계 이탈")
                n_box += 1
        print(f"  {variant}: 라벨 {len(list(lbl_dir.glob('*.txt')))} (빈 라벨 {n_empty}), bbox {n_box}")
    return problems


def check_ultralytics(cfg) -> list[str]:
    from ultralytics.data.utils import check_det_dataset
    problems = []
    for variant in ("multi", "binary"):
        y = Path(cfg["splits_dir"]) / "yolo" / variant / "dataset_fold0.yaml"
        try:
            d = check_det_dataset(str(y))
            ntr = len(Path(d["train"]).read_text(encoding="utf-8").strip().split("\n"))
            nva = len(Path(d["val"]).read_text(encoding="utf-8").strip().split("\n"))
            print(f"  {variant}/fold0: names={len(d['names'])}클래스, train={ntr}, val={nva}")
        except Exception as e:
            problems.append(f"{variant}: Ultralytics 데이터셋 로딩 실패 — {e}")
    return problems


def check_splits(cfg) -> list[str]:
    import json
    problems = []
    sd = Path(cfg["splits_dir"])
    hold = set(json.loads((sd / "holdout.json").read_text(encoding="utf-8"))["keys"])
    all_cv = None
    for k in range(cfg["n_folds"]):
        f = json.loads((sd / f"fold{k}.json").read_text(encoding="utf-8"))
        tr, va = set(f["train"]), set(f["val"])
        if tr & va:
            problems.append(f"fold{k}: train∩val {len(tr & va)}건")
        if (tr | va) & hold:
            problems.append(f"fold{k}: 홀드아웃 누수 {len((tr | va) & hold)}건")
        if all_cv is None:
            all_cv = tr | va
        elif all_cv != (tr | va):
            problems.append(f"fold{k}: CV 풀이 fold0과 불일치")
    # 전체 커버리지
    import csv
    with open(Path(cfg["data_root"]) / "labels_index.csv", encoding="utf-8") as fh:
        keys = {r["key"] for r in csv.DictReader(fh)}
    missing = keys - (all_cv | hold)
    if missing:
        problems.append(f"어느 분할에도 속하지 않는 이미지 {len(missing)}건: {sorted(missing)[:5]}")
    print(f"  분할 커버리지: CV {len(all_cv)} + 홀드아웃 {len(hold)} = {len(all_cv | hold)} / 전체 {len(keys)}")
    return problems


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    problems = []
    print("[1/3] 라벨 유효성")
    problems += check_labels(cfg, classes)
    print("[2/3] 분할 무결성")
    problems += check_splits(cfg)
    print("[3/3] Ultralytics 데이터셋 로딩")
    problems += check_ultralytics(cfg)

    print()
    if problems:
        print(f"FAIL — 문제 {len(problems)}건")
        for p in problems[:30]:
            print("  !", p)
        return 1
    print("PASS — 문제 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
