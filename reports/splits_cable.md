# 데이터 분할 리포트 — MVTec AD `cable`

- SOP: SOP-MVTEC-CABLE-001 §4.3 (SOP-DATA-03)
- 실행일: 2026-09-21
- git: `HEAD-dirty`
- 시드: 42 (분할은 JSON으로 고정 저장, 실행 시마다 재추첨하지 않음)

## 분할 A — 비지도 이상탐지용 (원본 유지, 변경 금지)

| 항목 | 값 |
|---|---|
| train (정상만) | 224 |
| test | 150 (불량 92 / 정상 58) |
| sha1 | `a749d16b84f3` |

## 홀드아웃 (분할 B, 최종 판정 전용)

| 항목 | 값 |
|---|---|
| 불량 | 16 (클래스당 2) |
| 정상 | 49 |
| 합계 | 65 |
| sha1 | `2e3f627995a5` |

> **홀드아웃은 §1.3 DoD 최종 판정에만 1회 사용한다.** 모델 선택, 하이퍼파라미터 튜닝, 임계값 τ 결정(§7.2)에 사용하면 그 결과는 무효다.

## 5-fold 교차검증 (분할 B)

| fold | train 불량 | train 정상 | val 불량 | val 정상 | sha1 |
|---|---|---|---|---|---|
| 0 | 60 | 186 | 16 | 47 | `e49ccc6119d2` |
| 1 | 61 | 186 | 15 | 47 | `b1e3cb1d0b67` |
| 2 | 61 | 186 | 15 | 47 | `402214d6780d` |
| 3 | 61 | 187 | 15 | 46 | `c54e2e4cf088` |
| 4 | 61 | 187 | 15 | 46 | `0a2aa51b57b0` |

### fold별 val 불량 클래스 분포

| fold | bent_wire | cable_swap | combined | cut_inner_insulation | cut_outer_insulation | missing_cable | missing_wire | poke_insulation |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 2 | 1 | 3 | 1 | 2 | 2 | 2 |
| 1 | 2 | 2 | 2 | 3 | 1 | 2 | 2 | 1 |
| 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 |
| 4 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 |

## 누수 방지 검증

- 모든 fold에서 train ∩ val = ∅ (assert 통과)
- 모든 fold에서 (train ∪ val) ∩ holdout = ∅ (assert 통과)
- 분할 단위 = 이미지 파일. 파생 크롭·증강본은 원본과 같은 분할에 속한다.

## 생성 파일

```
data/splits/split_A.json          분할 A (비지도)
data/splits/holdout.json          홀드아웃
data/splits/fold{0..4}.json       5-fold
data/splits/yolo/multi/           8클래스 검출용 Ultralytics 설정
data/splits/yolo/binary/          OK/NG 단일클래스 검출용 Ultralytics 설정
  fold{k}_train.txt / fold{k}_val.txt / holdout.txt
  splitA_train.txt / splitA_test.txt
  dataset_fold{k}.yaml            → yolo train data=... 에 지정
```

## 통계적 한계 (해석 시 필수 고려)

- fold당 val 불량이 약 15장이므로, recall 1건 차이가 약 6.6%p로 나타난다.
- 홀드아웃 불량 16장 기준 recall의 95% 신뢰구간은 ±20%p 수준이다. 홀드아웃 단독 수치로 모델 우열을 주장하지 않는다.
- 보고는 5-fold 평균 ± 표준편차로 한다(§7.4).
