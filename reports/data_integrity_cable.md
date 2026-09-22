# 데이터 무결성 검증 리포트 — MVTec AD `cable`

- SOP: SOP-MVTEC-CABLE-001 §4.1 (SOP-DATA-01)
- 실행일: 2026-09-21
- 대상: `D:\project\mvtec_defect\mvtec\cable`
- git: `HEAD-dirty`
- python: `3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)]`

## 판정: **PASS**

| 항목 | 값 |
|---|---|
| 이미지 파일 | 374 |
| 마스크 파일 | 92 |
| 해상도 (전체) | 1024x1024 |
| 이미지 mode | RGB |
| 마스크 mode | L |
| 비이진 마스크 | 0 |
| 인덱스 정합 | OK |
| 문제 건수 | 0 |

## 수량 대조

| 경로 | 보유 | 공식 | 상태 |
|---|---|---|---|
| train/good | 224 | 224 | 일치 |
| test/good | 58 | 58 | 일치 |
| test/bent_wire | 13 | 13 | 일치 |
| test/cable_swap | 12 | 12 | 일치 |
| test/combined | 11 | 11 | 일치 |
| test/cut_inner_insulation | 14 | 14 | 일치 |
| test/cut_outer_insulation | 10 | 10 | 일치 |
| test/missing_cable | 12 | 12 | 일치 |
| test/missing_wire | 10 | 10 | 일치 |
| test/poke_insulation | 10 | 10 | 일치 |
| ground_truth | 92 | 92 | 일치 |

## 인덱스 정합 (test 이미지 ↔ ground_truth 마스크)

| 클래스 | test | mask | 마스크 없음 | 고아 마스크 | 상태 |
|---|---|---|---|---|---|
| bent_wire | 13 | 13 | - | - | OK |
| cable_swap | 12 | 12 | - | - | OK |
| combined | 11 | 11 | - | - | OK |
| cut_inner_insulation | 14 | 14 | - | - | OK |
| cut_outer_insulation | 10 | 10 | - | - | OK |
| good | 58 | 0 | - | - | 정상(마스크 없음) |
| missing_cable | 12 | 12 | - | - | OK |
| missing_wire | 10 | 10 | - | - | OK |
| poke_insulation | 10 | 10 | - | - | OK |

## 문제 목록

문제 없음.

## 비고

- 노이즈 제거 임계: 연결요소 면적 < 0.020% (≈ 209 px) — §4.2에서 적용
- 구조적 결함 클래스(§3.2 가설): cable_swap, missing_cable, missing_wire
- 해상도 민감 감시 클래스: poke_insulation, cut_outer_insulation, missing_wire
- 상세 파일별 결과: `data_integrity_cable.csv`
