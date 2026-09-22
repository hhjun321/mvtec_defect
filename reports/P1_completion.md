# P1 완료 리포트 — 환경 구축 · 라벨 변환 · 분할 확정

- SOP: SOP-MVTEC-CABLE-001 §2.2, §4.1~4.4, §12 P1
- 완료일: 2026-09-21
- 게이트: **육안 검증 20장 통과** → PASS
- 전체 재실행: `powershell -ExecutionPolicy Bypass -File src\data\run_p1.ps1`

---

## 1. 환경 구축

| 항목 | 값 |
|---|---|
| venv | `D:\project\mvtec_defect\.venv` (Python 3.12.6) |
| torch | 2.14.0+cpu / torchvision 0.29.0+cpu |
| ultralytics | 8.4.157 |
| onnx / onnxruntime | 1.23.0 / 1.30.0 |
| GPU | 미검출 (`torch.cuda.is_available() = False`) |
| 잠금파일 | `requirements.lock.txt` (55개) |

SOP는 Python 3.11을 권장했으나 미설치여서 3.12.6을 채택했다. 3.13(시스템 기본)은 휠 지원이 늦어 배제했다.

> **P2 블로커**: GPU 없음. §2.2 대안 (a)/(b)/(c) 중 선택 필요.

---

## 2. SOP-DATA-01 — 무결성 검증 (`src/data/verify_dataset.py`)

**판정: PASS**

| 항목 | 값 |
|---|---|
| 이미지 | 374 (train/good 224 + test 150) |
| 마스크 | 92 |
| 해상도 | 1024x1024 (전체 동일) |
| 이미지 mode | RGB |
| 마스크 mode | L, 값 {0,255} — 비이진 0건 |
| 수량 대조 | 공식 MVTec AD와 전 항목 일치 |
| 인덱스 정합 | OK (마스크 없는 이미지 0, 고아 마스크 0) |
| 문제 건수 | **0** |

산출물: `reports/data_integrity_cable.md`, `reports/data_integrity_cable.csv`

---

## 3. SOP-DATA-02 — 라벨 변환 (`src/data/masks_to_yolo.py`)

파라미터: 이진화 임계 127, 8-연결, 최소 연결요소 면적 0.020% = 209 px

| 클래스 | class_id | 이미지 | bbox | 이미지당 bbox | 노이즈 제거 |
|---|---|---|---|---|---|
| bent_wire | 0 | 13 | 18 | 1.38 | 0 |
| cable_swap | 1 | 12 | 12 | 1.00 | 0 |
| combined | 2 | 11 | 41 | 3.73 | 0 |
| cut_inner_insulation | 3 | 14 | 20 | 1.43 | 0 |
| cut_outer_insulation | 4 | 10 | 17 | 1.70 | 0 |
| missing_cable | 5 | 12 | 12 | 1.00 | 0 |
| missing_wire | 6 | 10 | 12 | 1.20 | 0 |
| poke_insulation | 7 | 10 | 19 | 1.90 | 0 |
| good | — | 282 | 0 | 0.00 | 0 |
| **합계** | | **374** | **151** | | **0** |

**생성물**
```
data/classes.yaml                  클래스 ID 매핑 (고정, 변경 금지)
data/yolo/multi/{images,labels}/   8클래스 검출용
data/yolo/binary/{images,labels}/  OK/NG 단일클래스 검출용
data/labels_index.csv              이미지별 라벨 메타
```
원본 이미지는 **하드링크**로 연결했다(용량 중복 없음, `mvtec/cable/` 불변).
정상 282장은 0바이트 라벨 파일을 생성해 배경 샘플로 학습에 포함된다.

**노이즈 제거 0건** — 최소 면적 임계 209 px가 실제 결함을 하나도 깎지 않았다. 최소 결함(`poke_insulation` 0.14% ≈ 1,470 px)이 임계보다 7배 크다. 임계를 올릴 필요도, 내릴 필요도 없다.

### 3.1 육안 검증 (게이트 항목) — PASS

`reports/label_check/` 에 24장(불량 20 + 정상 4) 오버레이 저장. 빨강 = GT 마스크, 초록 = 생성 bbox.
8개 클래스 전부 표본 포함. 확인 결과:

- bbox가 마스크를 정확히 외접 — 좌표축 뒤바뀜 없음
- 다중 결함이 개별 bbox로 분리됨 (`combined`, `bent_wire`, `poke_insulation`, `cut_outer_insulation`)
- 정상 이미지는 빈 라벨, bbox 없음

**관찰 — 이후 단계에 영향**

1. `cable_swap`의 마스크는 **케이블 단면 전체**를 덮는다(국소 결함 아님). 이미지당 bbox 정확히 1개. 색이 뒤바뀐 케이블을 통째로 지정한 것으로, §3.2의 "구조적 결함" 가설과 일치한다. 패치 단위 이상탐지가 불리할 후보.
2. `bent_wire`와 `cut_outer_insulation`의 마스크는 **가늘고 휜 곡선 형태**다. 이 경우 bbox가 실제 결함 면적보다 훨씬 넓은 영역을 덮는다(bbox 내 결함 픽셀 비율이 낮음). → **mAP@IoU 기반 지표가 검출 품질을 과대평가할 수 있다.** §7.1의 픽셀 레벨 지표(PRO)를 반드시 병행하고, Track A 단독 mAP로 결론내지 않는다.
3. `combined`는 이미지당 평균 3.73개로 최다. 한 이미지에 이종 결함이 섞이므로 다중 클래스 학습 시 라벨이 모두 `combined`(id 2)로 부여된다. 이는 MVTec 원본 분류 체계를 따른 것이나, **의미상 `combined`는 결함 유형이 아니라 "복수 결함이 있음"이라는 메타 라벨**이다. 8클래스 검출의 클래스 혼동 원인이 될 수 있어, binary 변형(OK/NG)을 병행 준비해 두었다.

---

## 4. SOP-DATA-03 — 분할 확정 (`src/data/make_splits.py`)

### 분할 A — 비지도 이상탐지용 (원본 유지, 변경 금지)

| 항목 | 값 |
|---|---|
| train (정상만) | 224 |
| test | 150 (불량 92 / 정상 58) |
| sha1 | 리포트 참조 |

문헌값과 직접 비교 가능한 유일한 설정.

### 홀드아웃 — 최종 DoD 판정 전용

| 항목 | 값 |
|---|---|
| 불량 | 16 (클래스당 2) |
| 정상 | 49 (전체 불량:정상 비율 유지) |
| 합계 | 65 |

**모델 선택·튜닝·임계값 τ 결정에 사용 금지. §1.3 최종 판정에만 1회 사용.**

### 5-fold 교차검증 (분할 B)

| fold | train 불량 | train 정상 | val 불량 | val 정상 |
|---|---|---|---|---|
| 0 | 60 | 186 | 16 | 47 |
| 1 | 61 | 186 | 15 | 47 |
| 2 | 61 | 186 | 15 | 47 |
| 3 | 61 | 187 | 15 | 46 |
| 4 | 61 | 187 | 15 | 46 |

클래스 층화 적용 — fold별 val 불량 클래스 분포는 클래스당 1~3장으로 고르다(`reports/splits_cable.md`).

**생성물**
```
data/splits/split_A.json / holdout.json / fold{0..4}.json
data/splits/yolo/{multi,binary}/
    fold{k}_train.txt, fold{k}_val.txt, holdout.txt
    splitA_train.txt, splitA_test.txt
    dataset_fold{k}.yaml     → yolo train data=... 에 그대로 지정
```

---

## 5. 스모크 체크 (`src/data/smoke_check.py`) — PASS

| 검사 | 결과 |
|---|---|
| 라벨 필드 수·클래스 ID·좌표 0~1 범위·퇴화 bbox·경계 이탈 | 문제 0건 |
| 라벨 374 / 빈 라벨 282 / bbox 151 (multi, binary 동일) | 일치 |
| fold별 train ∩ val | ∅ |
| (train ∪ val) ∩ holdout | ∅ |
| 분할 커버리지 | CV 309 + 홀드아웃 65 = 374 / 전체 374 |
| Ultralytics `check_det_dataset` 로딩 | multi 8클래스 / binary 1클래스, train 246 · val 63 |

---

## 6. P1 게이트 판정

| 게이트 | 기준 | 결과 |
|---|---|---|
| P1 | 육안 검증 20장 통과 | **PASS** |

---

## 7. P2 착수 전 남은 항목

1. **GPU 확보 여부 결정** (§2.2) — 유일한 실질 블로커
2. §2.1 미확정 4항목 — 타깃 하드웨어 / 택트타임 / 미검:과검 비용비 / 판정 출력 형태
   - 1~3번은 §7.2 임계값과 §9 벤치마크에 필요하므로 P4~P5 전까지 확정하면 된다
   - 4번(출력 형태)은 §6 본선 트랙 선정 입력값이므로 **P2 착수 시점에 필요**하다. 다만 multi/binary 라벨을 모두 생성해 두었으므로 어느 쪽이든 즉시 학습 가능하다

## 8. 재현성 메모

- 전역 시드 42 (`configs/paths.yaml`), 분할은 JSON으로 고정 저장 — 실행 시마다 재추첨하지 않는다
- 분할 식별 해시는 train/val 경계를 구분한다(합집합만 해시하면 모든 fold가 동일 해시가 되는 문제를 수정함)
- `data/classes.yaml`의 클래스 ID는 **고정**이다. 변경 시 전체 라벨 재생성 + 실험 전면 재실행
