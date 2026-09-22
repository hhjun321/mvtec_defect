"""SOP-DATA-01 — 데이터 무결성 검증 (SOP-MVTEC-CABLE-001 §4.1)

산출물: reports/data_integrity_cable.csv, reports/data_integrity_cable.md
합격 기준: 손상 0건, 규격 불일치 0건, 결손 0건, 인덱스 정합 OK
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import env_stamp, load_cfg, load_classes  # noqa: E402

# MVTec AD `cable` 공식 수량 (SOP §3.1)
OFFICIAL = {
    "train/good": 224,
    "test/good": 58,
    "test/bent_wire": 13,
    "test/cable_swap": 12,
    "test/combined": 11,
    "test/cut_inner_insulation": 14,
    "test/cut_outer_insulation": 10,
    "test/missing_cable": 12,
    "test/missing_wire": 10,
    "test/poke_insulation": 10,
    "ground_truth": 92,
}


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    root = Path(cfg["source_root"])
    size = cfg["image_size"]
    reports = Path(cfg["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    rows = []
    problems = []

    # --- 1. 이미지 파일 검사 -------------------------------------------------
    for split in ("train", "test"):
        for d in sorted((root / split).iterdir()):
            if not d.is_dir():
                continue
            paths = sorted(d.glob("*.png"))
            for p in paths:
                rec = {
                    "kind": "image", "split": split, "class": d.name,
                    "stem": p.stem, "path": str(p.relative_to(root)),
                }
                try:
                    im = Image.open(p)
                    im.load()
                    rec["size"] = f"{im.size[0]}x{im.size[1]}"
                    rec["mode"] = im.mode
                    rec["ok"] = True
                    if im.size != (size, size):
                        rec["ok"] = False
                        problems.append(f"규격 불일치 {rec['path']}: {im.size}")
                    if im.mode != "RGB":
                        rec["ok"] = False
                        problems.append(f"채널 불일치 {rec['path']}: mode={im.mode}")
                except Exception as e:  # 손상 파일
                    rec.update(size="", mode="", ok=False)
                    problems.append(f"손상 {rec['path']}: {e}")
                rows.append(rec)

    # --- 2. 마스크 파일 검사 -------------------------------------------------
    for d in sorted((root / "ground_truth").iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")):
            rec = {
                "kind": "mask", "split": "ground_truth", "class": d.name,
                "stem": p.stem, "path": str(p.relative_to(root)),
            }
            try:
                im = Image.open(p)
                a = np.array(im)
                rec["size"] = f"{im.size[0]}x{im.size[1]}"
                rec["mode"] = im.mode
                rec["ok"] = True
                if im.size != (size, size):
                    rec["ok"] = False
                    problems.append(f"마스크 규격 불일치 {rec['path']}: {im.size}")
                if im.mode != "L":
                    rec["ok"] = False
                    problems.append(f"마스크 채널 불일치 {rec['path']}: mode={im.mode}")
                extra = set(np.unique(a)) - {0, 255}
                rec["nonbinary_values"] = len(extra)
                if extra:
                    problems.append(
                        f"비이진 마스크 {rec['path']}: 중간값 {len(extra)}종 "
                        f"→ 임계 {cfg['mask_binarize_threshold']}로 이진화 필요"
                    )
            except Exception as e:
                rec.update(size="", mode="", ok=False, nonbinary_values=-1)
                problems.append(f"마스크 손상 {rec['path']}: {e}")
            rows.append(rec)

    # --- 3. 수량 대조 --------------------------------------------------------
    counts = defaultdict(int)
    for r in rows:
        if r["kind"] == "image":
            counts[f"{r['split']}/{r['class']}"] += 1
    counts["ground_truth"] = sum(1 for r in rows if r["kind"] == "mask")

    count_table = []
    for key, expected in OFFICIAL.items():
        actual = counts.get(key, 0)
        match = actual == expected
        if not match:
            problems.append(f"수량 불일치 {key}: 보유 {actual} / 공식 {expected}")
        count_table.append((key, actual, expected, "일치" if match else "불일치"))

    # --- 4. 인덱스 정합 ------------------------------------------------------
    align_table = []
    align_ok = True
    for d in sorted((root / "test").iterdir()):
        if not d.is_dir():
            continue
        cls = d.name
        timgs = {p.stem for p in d.glob("*.png")}
        gt_dir = root / "ground_truth" / cls
        gmasks = {p.stem.replace("_mask", "") for p in gt_dir.glob("*.png")} if gt_dir.is_dir() else set()
        if cls == "good":
            if gmasks:
                align_ok = False
                problems.append(f"test/good 에 마스크 존재: {sorted(gmasks)[:5]}")
            align_table.append((cls, len(timgs), len(gmasks), "-", "-", "정상(마스크 없음)"))
            continue
        no_mask = sorted(timgs - gmasks)
        orphan = sorted(gmasks - timgs)
        if no_mask or orphan:
            align_ok = False
            if no_mask:
                problems.append(f"{cls}: 마스크 없는 이미지 {no_mask}")
            if orphan:
                problems.append(f"{cls}: 대응 이미지 없는 마스크 {orphan}")
        align_table.append(
            (cls, len(timgs), len(gmasks),
             ",".join(no_mask) or "-", ",".join(orphan) or "-",
             "OK" if not (no_mask or orphan) else "FAIL")
        )

    # --- 5. 산출물 -----------------------------------------------------------
    csv_path = reports / "data_integrity_cable.csv"
    fields = ["kind", "split", "class", "stem", "path", "size", "mode", "ok", "nonbinary_values"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_img = sum(1 for r in rows if r["kind"] == "image")
    n_mask = sum(1 for r in rows if r["kind"] == "mask")
    sizes = sorted({r["size"] for r in rows if r.get("size")})
    modes_img = sorted({r["mode"] for r in rows if r["kind"] == "image" and r.get("mode")})
    modes_mask = sorted({r["mode"] for r in rows if r["kind"] == "mask" and r.get("mode")})
    n_nonbin = sum(1 for r in rows if r.get("nonbinary_values", 0) > 0)
    passed = not problems

    env = env_stamp()
    md = [
        "# 데이터 무결성 검증 리포트 — MVTec AD `cable`",
        "",
        "- SOP: SOP-MVTEC-CABLE-001 §4.1 (SOP-DATA-01)",
        f"- 실행일: {date.today().isoformat()}",
        f"- 대상: `{root}`",
        f"- git: `{env['git_rev']}`",
        f"- python: `{env['python'].splitlines()[0]}`",
        "",
        f"## 판정: **{'PASS' if passed else 'FAIL'}**",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 이미지 파일 | {n_img} |",
        f"| 마스크 파일 | {n_mask} |",
        f"| 해상도 (전체) | {', '.join(sizes)} |",
        f"| 이미지 mode | {', '.join(modes_img)} |",
        f"| 마스크 mode | {', '.join(modes_mask)} |",
        f"| 비이진 마스크 | {n_nonbin} |",
        f"| 인덱스 정합 | {'OK' if align_ok else 'FAIL'} |",
        f"| 문제 건수 | {len(problems)} |",
        "",
        "## 수량 대조",
        "",
        "| 경로 | 보유 | 공식 | 상태 |",
        "|---|---|---|---|",
    ]
    md += [f"| {k} | {a} | {e} | {s} |" for k, a, e, s in count_table]
    md += [
        "",
        "## 인덱스 정합 (test 이미지 ↔ ground_truth 마스크)",
        "",
        "| 클래스 | test | mask | 마스크 없음 | 고아 마스크 | 상태 |",
        "|---|---|---|---|---|---|",
    ]
    md += [f"| {c} | {t} | {g} | {nm} | {om} | {st} |" for c, t, g, nm, om, st in align_table]

    md += ["", "## 문제 목록", ""]
    md += ["문제 없음."] if passed else [f"- {p}" for p in problems]
    md += [
        "",
        "## 비고",
        "",
        f"- 노이즈 제거 임계: 연결요소 면적 < {cfg['min_component_area_ratio']*100:.3f}% "
        f"(≈ {int(cfg['min_component_area_ratio'] * cfg['image_size']**2)} px) — §4.2에서 적용",
        f"- 구조적 결함 클래스(§3.2 가설): {', '.join(classes['structural_defects'])}",
        f"- 해상도 민감 감시 클래스: {', '.join(classes['small_defect_watch'])}",
        f"- 상세 파일별 결과: `{csv_path.name}`",
    ]

    md_path = reports / "data_integrity_cable.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[SOP-DATA-01] {'PASS' if passed else 'FAIL'} — 이미지 {n_img}, 마스크 {n_mask}, 문제 {len(problems)}")
    for p in problems[:20]:
        print("  !", p)
    print(f"  -> {md_path}")
    print(f"  -> {csv_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
