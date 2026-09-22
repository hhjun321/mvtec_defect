"""연구후보 문서의 전제 검증 — `cable` 데이터셋의 환경 변수 실측.

`docs/연구후보/`의 두 제안 문서는 다음을 해결 대상으로 삼는다:
  (1) 미세 회전/이동 오차  (2) 빛 굴절·반사·명암 불균일  (3) 색바램

MVTec AD `cable`에서 이 변동이 실제로 얼마나 존재하는지 측정한다.
정상 이미지(train/good + test/good)로 한정한다 — 결함 변동과 섞지 않기 위해.

측정 방법
  기하: 외피 원판을 Otsu + 최대 연결요소 + 최소외접원으로 검출해 중심/반지름을 잡고,
        녹색 가닥 무게중심의 방위각으로 회전을 잰다.
        (위상상관은 배경 텍스처에 물려 실패하므로 쓰지 않는다)
  조명: LAB L 채널의 이미지 간 편차 + 포화 화소 비율
  색상: 색도(chromaticity) r,g,b 의 이미지 간 편차

산출물: reports/dataset_variation_cable.md
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg  # noqa: E402

RES = 384
GREEN_HSV = ((30, 40, 40), (85, 255, 255))   # 녹색 가닥 — 배경/외피와 충분히 분리됨


def disk_and_rotation(im_rgb: np.ndarray):
    """(cx, cy, r, 녹색 가닥 방위각deg) 또는 None"""
    bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    (ux, uy), ur = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(GREEN_HSV[0]), np.array(GREEN_HSV[1]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cn, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ang = np.nan
    if cn:
        M = cv2.moments(max(cn, key=cv2.contourArea))
        if M["m00"] > 0:
            ang = np.degrees(np.arctan2(M["m01"] / M["m00"] - uy, M["m10"] / M["m00"] - ux))
    return ux, uy, ur, ang


def main() -> int:
    cfg = load_cfg()
    cache = Path(cfg["data_root"]) / "cache"
    keys = json.loads((cache / f"keys_{RES}.json").read_text(encoding="utf-8"))
    arr = np.load(cache / f"img_{RES}.npy", mmap_mode="r")
    with open(Path(cfg["data_root"]) / "labels_index.csv", encoding="utf-8") as f:
        meta = {r["key"]: r for r in csv.DictReader(f)}

    good_idx = [i for i, k in enumerate(keys) if meta[k]["image_label"] == "0"]
    def_idx = [i for i, k in enumerate(keys) if meta[k]["image_label"] == "1"]
    imgs = np.stack([np.asarray(arr[i]) for i in good_idx])
    scale = cfg["image_size"] / RES

    # ---------- (1) 기하 ----------
    det = [disk_and_rotation(im) for im in imgs]
    det = [d for d in det if d is not None]
    cx = np.array([d[0] for d in det]); cy = np.array([d[1] for d in det])
    rad = np.array([d[2] for d in det]); ang = np.array([d[3] for d in det])
    ang = ang[~np.isnan(ang)]
    dev = np.hypot(cx - cx.mean(), cy - cy.mean())
    z = np.exp(1j * np.radians(ang))
    R = abs(z.mean())
    circ_std = float(np.degrees(np.sqrt(-2 * np.log(R)))) if R > 0 else float("nan")
    ang_c = np.degrees(np.angle(z / z.mean()))   # 평균 대비 편차

    # ---------- (2) 조명 ----------
    labs = np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2LAB) for im in imgs])
    L = labs[..., 0].astype(np.float32) * (100.0 / 255.0)
    L_mean = L.reshape(len(L), -1).mean(1)
    L_within = L.reshape(len(L), -1).std(1)
    sat = (imgs.max(axis=3) >= 250).reshape(len(imgs), -1).mean(1) * 100

    # ---------- (3) 색상 ----------
    f32 = imgs.astype(np.float32) + 1e-6
    chrom = f32 / f32.sum(axis=3, keepdims=True)
    chrom_mean = chrom.reshape(len(chrom), -1, 3).mean(1)
    chrom_std = chrom_mean.std(0)

    # ---------- (4) 정상 변동 vs 결함 신호 ----------
    mean_img = imgs.astype(np.float32).mean(0)
    px_std = float(imgs.astype(np.float32).std(0).mean())
    d_imgs = np.stack([np.asarray(arr[i]) for i in def_idx]).astype(np.float32)
    good_dev = np.abs(imgs.astype(np.float32) - mean_img).mean(axis=(1, 2, 3))
    def_dev = np.abs(d_imgs - mean_img).mean(axis=(1, 2, 3))

    out_lines = []
    A = out_lines.append
    A("# `cable` 데이터셋 환경 변수 실측")
    A("")
    A("- 목적: `docs/연구후보/`의 두 제안 문서가 해결 대상으로 삼은 환경 변수"
      "(미세 정렬 오차 / 조명·반사 / 색바램)가 이 데이터셋에 실제로 존재하는지 확인")
    A(f"- 생성일: {date.today().isoformat()}")
    A(f"- 대상: 정상 {len(good_idx)}장 (train/good + test/good) / 비교군 불량 {len(def_idx)}장")
    A(f"- 해상도: {RES}px 캐시 (원본 {cfg['image_size']}px 환산치 병기)")
    A("")
    A("> 기하 측정에 위상상관(phase correlation)을 먼저 써봤으나 배경 화강암 텍스처에 물려 "
      "실패했다(이동량 중앙값 270px 등 비현실적 값). 외피 원판 검출 방식으로 교체했다.")
    A("")
    A("## 1. 기하 변동 — 이동 · 크기 · 회전")
    A("")
    A(f"외피 원판 검출 성공 {len(det)}/{len(good_idx)}장. 평균 반지름 "
      f"{rad.mean():.1f}px ({rad.mean()*scale:.0f}px @1024).")
    A("")
    A("| 항목 | 값 (384px) | 1024px 환산 | 원판 반지름 대비 |")
    A("|---|---|---|---|")
    A(f"| 중심 이탈 중앙값 | {np.median(dev):.2f} px | {np.median(dev)*scale:.1f} px | "
      f"{np.median(dev)/rad.mean()*100:.2f} % |")
    A(f"| 중심 이탈 p95 | {np.percentile(dev,95):.2f} px | {np.percentile(dev,95)*scale:.1f} px | "
      f"{np.percentile(dev,95)/rad.mean()*100:.2f} % |")
    A(f"| 중심 이탈 최대 | {dev.max():.2f} px | {dev.max()*scale:.1f} px | "
      f"{dev.max()/rad.mean()*100:.2f} % |")
    A(f"| 반지름 표준편차 | {rad.std():.2f} px | {rad.std()*scale:.1f} px | "
      f"변동계수 {rad.std()/rad.mean()*100:.2f} % |")
    A("")
    A("**회전** — 녹색 가닥 무게중심의 방위각(원판 중심 기준)")
    A("")
    A("| 항목 | 값 |")
    A("|---|---|")
    A(f"| 평균 방위각 | {np.degrees(np.angle(z.mean())):.2f}° (상단 고정) |")
    A(f"| **원형 표준편차** | **{circ_std:.2f}°** |")
    A(f"| 평균 대비 편차 범위 | {ang_c.min():.1f}° ~ {ang_c.max():.1f}° |")
    A(f"| 편차 p95 (절대값) | {np.percentile(np.abs(ang_c),95):.1f}° |")
    A("")
    A("## 2. 조명 / 명암 변동 (LAB L, 0~100)")
    A("")
    A("| 항목 | 값 |")
    A("|---|---|")
    A(f"| 이미지별 평균 L — 전체 평균 | {L_mean.mean():.2f} |")
    A(f"| 이미지별 평균 L — **이미지 간 표준편차** | **{L_mean.std():.2f}** (상대 {L_mean.std()/L_mean.mean()*100:.1f} %) |")
    A(f"| 이미지별 평균 L — 최소 / 최대 | {L_mean.min():.2f} / {L_mean.max():.2f} |")
    A(f"| 이미지 내 L 표준편차 (평균) | {L_within.mean():.2f} |")
    A(f"| 포화 화소(≥250) 비율 평균 / 최대 | {sat.mean():.2f} % / {sat.max():.2f} % |")
    A("")
    A("## 3. 색상 변동 (색도 = 채널/합)")
    A("")
    A("| 채널 | 평균 색도 | **이미지 간 표준편차** | 상대 변동 |")
    A("|---|---|---|---|")
    for i, ch in enumerate("rgb"):
        A(f"| {ch} | {chrom_mean[:,i].mean():.4f} | **{chrom_std[i]:.5f}** | "
          f"{chrom_std[i]/chrom_mean[:,i].mean()*100:.2f} % |")
    A("")
    A("## 4. 정상 간 변동 vs 결함 신호")
    A("")
    A("정상 평균 영상 대비 화소 절대 편차(0~255).")
    A("")
    A("| 집단 | 평균 | 표준편차 | 최소 | 최대 |")
    A("|---|---|---|---|---|")
    A(f"| 정상 {len(good_idx)} | {good_dev.mean():.2f} | {good_dev.std():.2f} | "
      f"{good_dev.min():.2f} | {good_dev.max():.2f} |")
    A(f"| 불량 {len(def_idx)} | {def_dev.mean():.2f} | {def_dev.std():.2f} | "
      f"{def_dev.min():.2f} | {def_dev.max():.2f} |")
    A("")
    A(f"정상 화소별 표준편차(전 채널 평균) **{px_std:.2f}/255**. "
      "정상과 불량의 전역 편차 분포가 거의 겹친다 — 단순 화소 차분으로는 판별 불가이며, "
      "결함 신호가 국소적이라는 뜻이다.")
    A("")
    A("## 5. 판정")
    A("")
    A("| 문서가 상정한 환경 변수 | 실측 | 판정 |")
    A("|---|---|---|")
    A(f"| 미세 위치 오차 | 중심 이탈 중앙 {np.median(dev)*scale:.0f}px / 반지름의 "
      f"{np.median(dev)/rad.mean()*100:.1f} %, p95 {np.percentile(dev,95)/rad.mean()*100:.1f} % | 소폭 존재 |")
    A(f"| 미세 크기 오차 | 반지름 변동계수 {rad.std()/rad.mean()*100:.2f} % | 소폭 존재 |")
    A(f"| 미세 회전 오차 | 원형 표준편차 **{circ_std:.1f}°**, 편차 범위 "
      f"{ang_c.min():.0f}° ~ {ang_c.max():.0f}° | **존재 — 셋 중 가장 큼** |")
    A(f"| 빛 굴절 / 조명 불균일 | 이미지 간 평균 L 표준편차 {L_mean.std():.2f}/100 "
      f"(상대 {L_mean.std()/L_mean.mean()*100:.1f} %) | 거의 없음 |")
    A(f"| 색바램 | 색도 이미지 간 표준편차 {chrom_std.max():.5f} "
      f"(상대 {(chrom_std/chrom_mean.mean(0)).max()*100:.2f} %) | **사실상 없음** |")
    A("")
    A("포화 화소가 평균 6% 수준으로 존재하나 이는 절단면 구리 소선의 금속 반사로, "
      "**촬영 환경 변동이 아니라 피사체 고유 속성**이다. 전 이미지에 일관되게 나타난다.")
    A("")

    out = Path(cfg["reports_dir"]) / "dataset_variation_cable.md"
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("\n".join(out_lines[7:]))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
