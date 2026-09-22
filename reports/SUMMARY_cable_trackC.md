# 통합 리포트 — MVTec `cable` 불량 판정 경량 모델 (P1~P3)

- SOP: SOP-MVTEC-CABLE-001 (`docs/SOP_cable_defect_lightweight.md`)
- 생성일: 2026-09-22
- 범위: P1 데이터 준비 · P2 Track C 기준선 · P3 A1(해상도) ablation
- 미포함: Track A(YOLO) · Track B(비지도) — GPU 미확보로 연기(§2.2 대안 (c))

## 0. 한 문단 요약

MobileNetV3-Small 이진 분류(Track C)로 `cable` OK/NG 판정을 5-fold 교차검증했다. 입력 해상도 640 / 384 / 256px 전 조건에서 **미검율 0%**, AUROC 0.9992~0.9997로 차이가 없었다. DoD 정확도를 만족하는 최경량은 **256px** (0.16 GFLOPs, e2e p50 26.95ms, 파라미터 1.52M). 다만 모든 수치가 검증셋 기준이고 임계값 τ도 같은 검증셋에서 정해 **낙관적으로 편향**돼 있으며, fold당 불량 표본이 16장 수준이라 신뢰구간이 넓다. 또한 **τ가 fold/환경 간 이식되지 않는 문제**를 확인해 배포 시 재보정을 의무화했다.

---

## 1. 데이터

### 1.1 복원

초기 로컬 사본은 부분 다운로드본이었다(불량 33장, test 91장, 5개 클래스 이미지 누락). Google Drive 원본 zip에서 전량 복원했다. 원인은 Drive의 대용량 폴더 다운로드가 아카이브를 임의 분할하는데 일부 파트만 압축 해제한 것이었다.

| 항목 | 복원 전 | 복원 후 | MVTec AD 공식 |
|---|---|---|---|
| train/good | 224 | 224 | 224 |
| test 총계 | 91 | 150 | 150 |
| test 불량 클래스 | 3종 | 8종 | 8종 |
| 불량 이미지 | 33 | 92 | 92 |
| ground_truth 마스크 | 57 | 92 | 92 |

무결성 검증(§4.1) 결과: 손상 0건, 해상도 1024×1024 전부 일치, 마스크 {0,255} 이진 비이진 0건, test↔mask 인덱스 정합 OK. → `reports/data_integrity_cable.md`

> 백업 보존: `mvtec/cable_partial_bak_20260921/` (복원 전 부분본). 삭제 금지.
> `cable` 외 11개 카테고리는 여전히 부분 결손 상태다 — 확장 시 동일 절차 필요(SOP §3.3).

### 1.2 결함 크기 (마스크 92장 전수)

| 클래스 | n | 면적% min/med/max | bbox 폭%(med) | bbox 높이%(med) |
|---|---|---|---|---|
| bent_wire | 13 | 1.23 / 3.60 / 11.37 | 28.5 | 35.7 |
| cable_swap | 12 | 4.95 / 5.82 / 6.47 | 31.6 | 32.3 |
| combined | 11 | 2.28 / 4.77 / 14.17 | 45.3 | 48.0 |
| cut_inner_insulation | 14 | 1.89 / 3.10 / 9.89 | 30.6 | 30.7 |
| cut_outer_insulation | 10 | 0.44 / 1.25 / 3.39 | 28.5 | 18.0 |
| missing_cable | 12 | 6.60 / 7.40 / 22.58 | 32.7 | 32.0 |
| missing_wire | 10 | 1.45 / 1.80 / 3.88 | 16.8 | 17.1 |
| poke_insulation | 10 | 0.14 / 2.82 / 4.87 | 22.2 | 24.4 |

bbox 변 길이 중앙값이 이미지 변의 17~48%로 **대형 객체**에 해당한다. 이것이 해상도를 크게 낮출 수 있었던 근거다(§3 결과로 검증됨).

## 2. 방법

### 2.1 라벨

마스크 → 연결요소(8-연결) → 외접 bbox. 최소 면적 임계 0.020%(=209 px). 불량 92장에서 **bbox 151개** 생성, 노이즈 제거 0건. 정상 282장은 0바이트 빈 라벨(배경 샘플).

| 클래스 | class_id | 이미지 | bbox | 이미지당 |
|---|---|---|---|---|
| bent_wire | 0 | 13 | 18 | 1.38 |
| cable_swap | 1 | 12 | 12 | 1.00 |
| combined | 2 | 11 | 41 | 3.73 |
| cut_inner_insulation | 3 | 14 | 20 | 1.43 |
| cut_outer_insulation | 4 | 10 | 17 | 1.70 |
| missing_cable | 5 | 12 | 12 | 1.00 |
| missing_wire | 6 | 10 | 12 | 1.20 |
| poke_insulation | 7 | 10 | 19 | 1.90 |
| **합계** | | **92** | **151** | |

다중 클래스(8종)와 이진(OK/NG) 라벨을 모두 생성했다 — 판정 출력 형태(§2.1)가 미확정이므로 어느 쪽이든 즉시 학습 가능하게 했다. 육안 검증 24장 통과(`reports/label_check/`).

### 2.2 분할

| 분할 | 용도 | 구성 |
|---|---|---|
| A | 비지도 이상탐지 (문헌 비교) | train 224 정상 / test 150 (불량 92) — MVTec 원본 유지, 변경 금지 |
| B 홀드아웃 | 최종 DoD 판정 전용 | 불량 16(클래스당 2) + 정상 49 = 65 |
| B 5-fold | 모델 개발·비교 | fold당 train 60D+186N / val 16D+47N |

클래스 층화 적용. 누수 검증 통과(train∩val=∅, 홀드아웃 누수 0, 커버리지 374/374). **홀드아웃은 아직 한 번도 사용하지 않았다** — P5 STOP-GATE 2 전용.

### 2.3 모델 · 학습

- mobilenet_v3_small, ImageNet 사전학습, 분류 헤드만 2클래스로 교체
- AdamW lr 0.001 + cosine, warmup 3, wd 0.0005, batch 16, epochs 50 / patience 15, seed 42
- best checkpoint 선택 기준: val AUROC
- 증강: 수평·수직 플립, 90° 회전, 스케일 ±10%, 이동 ±5%, 밝기·대비 ±20%, 약한 가우시안 노이즈
- **증강 금지: Hue 변형** — 절연체 색상이 `cable_swap` 판별의 직접 근거라 색을 흔들면 라벨이 거짓이 된다

> CPU 전용 환경(GPU 미검출)이라 SOP 기본값 epochs 300 / patience 50을 **50 / 15**로 축소했다. 사전학습 가중치로 10 epoch 내 수렴함을 타이밍 실험에서 확인한 뒤 내린 결정이며 SOP §8.1에 근거를 기록했다.
> `num_workers>0`은 Windows spawn 오버헤드로 2.6배 느려 `num_workers=0` 고정.

## 3. 결과 — A1 해상도 ablation

| 입력 | AUROC | 미검율 % | 과검율 % | F1 | GFLOPs | e2e p50/p95 ms | 모델만 p50 ms | 학습시간 |
|---|---|---|---|---|---|---|---|---|
| 640 | 0.9994 ± 0.0012 | 0.00 ± 0.00 | 0.87 ± 1.74 | 0.9875 ± 0.0250 | 0.988 | 74.88/93.98 | 54.35 | 3.10h |
| 384 | 0.9997 ± 0.0006 | 0.00 ± 0.00 | 0.43 ± 0.87 | 0.9935 ± 0.0129 | 0.357 | 54.61/75.19 | 28.01 | 1.15h |
| **256** | 0.9992 ± 0.0011 | 0.00 ± 0.00 | 1.30 ± 1.73 | 0.9814 ± 0.0249 | 0.16 | 26.95/34.54 | 26.6 | 0.81h |

파라미터는 해상도 무관 1.52M / FP32 6.08 MB로 동일(전역 평균 풀링). 이득은 연산량·지연시간에서만 나온다.

**혼동행렬 (5-fold 합산)**

| 입력 | TP | FN | FP | TN |
|---|---|---|---|---|
| 640 | 76 | **0** | 2 | 231 |
| 384 | 76 | **0** | 1 | 232 |
| 256 | 76 | **0** | 3 | 230 |

**클래스별 recall (§7.3)** — ⚠ = §3.2에서 해상도 민감 예상 클래스

| 클래스 | n | 640px | 384px | 256px |
|---|---|---|---|---|
| bent_wire | 11 | 1.000 | 1.000 | 1.000 |
| cable_swap | 10 | 1.000 | 1.000 | 1.000 |
| combined | 9 | 1.000 | 1.000 | 1.000 |
| cut_inner_insulation | 12 | 1.000 | 1.000 | 1.000 |
| cut_outer_insulation ⚠ | 8 | 1.000 | 1.000 | 1.000 |
| missing_cable | 10 | 1.000 | 1.000 | 1.000 |
| missing_wire ⚠ | 8 | 1.000 | 1.000 | 1.000 |
| poke_insulation ⚠ | 8 | 1.000 | 1.000 | 1.000 |

![A1 pareto](figs/ablation_A1_pareto.png)

## 4. 주요 발견

### 4.1 해상도 민감 가설 기각 (256px까지)

§3.2에서 소형 결함 3종(`poke_insulation` 최소 면적 0.14%, `cut_outer_insulation` 0.44%, `missing_wire` bbox 최소 17%)이 저해상도에서 소실될 것으로 예상했으나, **256px에서도 전 클래스 recall 1.000**이었다. 256px면 0.14% 결함이 약 92px로 줄어드는데도 유지된다. `cable`의 결함 다수가 국소 텍스처가 아니라 **케이블 배치·구성의 전역 변화**로 나타나기 때문으로 보인다.

### 4.2 AUROC는 변별력 없음 — 실제 차이는 과검율

세 해상도 모두 AUROC 0.999대로 포화했다. 과검율 차이(0.43~1.30%)도 오차막대가 서로 겹쳐 **통계적으로 유의하지 않다** — 5-fold 합산 FP 기준 1건 vs 2건 vs 3건 차이일 뿐이다. §7.4에 따라 '어느 해상도가 낫다'가 아니라 **'세 해상도를 구분할 수 없다'**가 정확한 결론이다.

### 4.3 임계값 τ가 환경 간 이식되지 않는다 (배포 리스크)

| 입력 | 분리된 fold | 겹친 fold | 간격 log10 (min/중앙/max) | τ 범위 |
|---|---|---|---|---|
| 640 | 4 | 1 | 0.25 / 0.44 / 0.81 | 5.5e-06 ~ 0.68 |
| 384 | 4 | 1 | 0.19 / 0.57 / 0.90 | 1.2e-08 ~ 0.97 |
| 256 | 3 | 2 | 0.25 / 0.37 / 0.61 | 1.3e-05 ~ 0.0031 |

정상 점수의 절대 스케일이 fold마다 최대 8자릿수까지 다르고, 완전분리 구간 폭도 0.19~0.90 log10로 좁다. **한 환경에서 정한 절대 확률 임계를 다른 환경에 그대로 쓰면 판정이 무너진다.** 조치:

- `choose_threshold`에 **마진 tie-break** 추가 — 완전분리 시 ROC가 구간 끝에 붙이던 τ를 혼동행렬 유지한 채 구간 중앙(로그 기하평균)으로 이동
- **§7.2 정책 보완** — 배포 시 (a) 현장 정상 표본으로 τ 재보정(권장) / (b) 백분위수 기반 임계 / (c) 온도 스케일링 중 택1, 보정 절차를 배포 아티팩트에 동봉 의무화

해상도를 낮춰도 이 문제는 사라지지 않으며, 256px에서 겹친 fold가 2개로 가장 많고 간격 중앙값도 최소라 **오히려 더 까다롭다**.

### 4.4 전처리가 지연시간 하한을 만든다

모델 추론은 640→256px에서 54.35→26.6ms(0.49×)로 줄지만, e2e는 74.88→26.95ms(0.36×)에 그친다. 1024px 원본 리사이즈가 고정비로 남기 때문이다. **카메라 단에서 저해상도로 직접 취득할 수 있으면 추가 이득이 크다.**

### 4.5 Track A(YOLO) 착수 시 주의 — 라벨 변환 중 발견

1. `bent_wire`·`cut_outer_insulation`의 GT 마스크는 **가늘고 휜 곡선**이라 외접 bbox가 실제 결함 면적보다 훨씬 넓다. 이 클래스에서 **mAP@IoU는 검출 품질을 과대평가한다** — 픽셀 레벨 PRO를 반드시 병행한다.
2. `combined`(id 2)는 결함 유형이 아니라 **'복수 결함 존재'를 뜻하는 메타 라벨**이다. 이미지당 bbox 3.73개로 최다. 8클래스 검출에서 타 클래스와 혼동될 소지가 있다.
3. `cable_swap` 마스크는 케이블 단면 **전체**를 덮는다(이미지당 bbox 정확히 1개). 국소 결함이 아니므로 패치 단위 이상탐지(Track B PatchCore/PaDiM)에 불리할 것으로 예상된다 — 미검증.

## 5. DoD 대조 (§1.3) — 잠정

| 항목 | 기준 | 결과 | 판정 |
|---|---|---|---|
| 미검율 (과검 5% 고정) | ≤ 5% | 0.00 % | 충족 |
| 이미지 AUROC | ≥ 0.95 | 0.9992 | 충족 |
| 모델 크기 (INT8 환산) | ≤ 10 MB | 1.52 MB (FP32 6.08 MB) | 충족 |
| 추론 지연 | ≤ 100 ms | 26.95 ms (개발 CPU) | 충족 |
| 피크 메모리 | ≤ 256 MB RSS | 미측정 | — |
| 재현성 | ±1%p | 분할·시드 고정, 체크포인트 재산출로 AUROC 1e-6 이내 일치 확인 | 충족 |

> 기준: `EXP_C_mnv3s_256_fp32_all_20260921`. **모두 검증셋 기준의 잠정치다.** 홀드아웃 최종 판정(P5 STOP-GATE 2)과 타깃 하드웨어 벤치(§9.3)를 거쳐야 확정된다.

## 6. 한계 — 수치를 과신하면 안 되는 이유

1. **τ와 지표를 같은 val fold에서 산출했다.** τ를 검증셋에서 정하는 것은 §7.2 정책대로이나, 그 τ로 같은 검증셋의 미검율·과검율을 계산하면 낙관적으로 편향된다. 편향 없는 추정치는 홀드아웃에서만 얻을 수 있다.
2. **표본이 작다.** fold당 val 불량 16~16장. 미검 0건이어도 rule of three 기준 미검율 95% 상한은 fold 단위 약 20%다. 5-fold 합산(불량 76장)해도 약 3.9%이며, fold 간 학습셋이 겹쳐 완전 독립 시행이 아니라 이 값도 낙관적이다.
3. **클래스별 recall 1.000은 클래스당 n=8~12에서 나온 값이다.** '완벽히 잡는다'가 아니라 '현 표본에서 놓친 사례가 없다'로 읽어야 한다.
4. **Track A/B 미수행.** §1.2 연구질문 1(어느 태스크가 유리한가)은 Track C 단독으로 답할 수 없다.
5. **지연시간·메모리는 개발 머신 CPU 값이다.** 타깃 하드웨어에서 §9.3 프로토콜(워밍업 20회 후 200회, 열 스로틀링 10분 연속 부하 포함)로 재측정해야 배포 판단이 된다.
6. **데이터가 MVTec 공개 데이터다.** 실제 산업 라인 데이터로의 전이는 별도 검증이 필요하며, MVTec AD는 CC BY-NC-SA 4.0(비상업)이다.

## 7. 다음 단계

### 7.1 의사결정 대기 (§2.1) — 현재 최대 병목

| 항목 | 필요 시점 | 영향 |
|---|---|---|
| **타깃 하드웨어** | 즉시 | §9.2 export 경로, §9.3 벤치, 해상도 최종 선택 |
| 택트타임 예산 | 즉시 | 해상도·모델 선택 |
| 미검:과검 비용비 | P4 전 | §7.2 임계값 정책 |
| 판정 출력 형태 | Track A 착수 전 | 본선 트랙 확정 (multi/binary 라벨은 둘 다 준비됨) |
| GPU 확보 | Track A/B 착수 전 | 미확보 시 YOLO 5-fold 비현실적 |

### 7.2 기술 작업 후보

| 우선 | 작업 | 근거 | 예상 비용 |
|---|---|---|---|
| 1 | A6 양자화 (INT8 PTQ) | 6.08 → 1.52 MB, 캘리브레이션에 불량 포함 필수(§9.1-3) | 낮음 |
| 2 | ONNX export + 수치 동등성 검증 | §9.2 — 판정 일치율 100% 확인 | 낮음 |
| 3 | A2 백본 축소 (ShuffleNetV2 0.5×, 1.4M) | 현 정확도에 여유가 커 더 줄일 여지 | 중간 |
| 4 | A3 증강 강도 / A5 클래스 가중치 | 과검율 개선 여지 확인 | 중간 |
| 5 | Track B (EfficientAD-S) | 문헌 비교 가능한 유일 트랙, 불량 데이터 불필요 | GPU 필요 |
| 6 | Track A (YOLO11n) | 위치 출력이 필요할 때만 | GPU 필요 |

> **홀드아웃은 건드리지 않는다.** 위 작업 전부 5-fold val로 판단하고, 홀드아웃은 최종 후보 1개가 확정된 뒤 P5에서 1회만 사용한다(§4.3).

## 8. 재현 절차

```powershell
# 0) 환경
<python3.12> -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt

# 1) P1 — 무결성 검증 → 라벨 변환 → 분할 → 스모크 체크
powershell -ExecutionPolicy Bypass -File src\data\run_p1.ps1

# 2) 이미지 캐시 (CPU 학습 병목 제거)
.venv\Scripts\python.exe src\data\build_cache.py

# 3) P2/P3 — Track C 해상도별 5-fold
.venv\Scripts\python.exe src\train\train_cls.py --fold all --res 640 --epochs 50 --patience 15 --workers 0
.venv\Scripts\python.exe src\train\train_cls.py --fold all --res 384 --epochs 50 --patience 15 --workers 0
.venv\Scripts\python.exe src\train\train_cls.py --fold all --res 256 --epochs 50 --patience 15 --workers 0

# 4) 체크포인트로 점수 재산출 (τ 마진 tie-break 적용, 점수 원본 저장)
.venv\Scripts\python.exe src\eval\score_folds.py EXP_C_mnv3s_640_fp32_all_20260921
.venv\Scripts\python.exe src\eval\score_folds.py EXP_C_mnv3s_384_fp32_all_20260921
.venv\Scripts\python.exe src\eval\score_folds.py EXP_C_mnv3s_256_fp32_all_20260921

# 5) 리포트
.venv\Scripts\python.exe src\eval\report_track_c.py
.venv\Scripts\python.exe src\eval\report_ablation_res.py
.venv\Scripts\python.exe src\eval\report_summary.py
```

시드 42 전역 고정, 분할은 JSON으로 고정 저장(실행마다 재추첨하지 않음). 실험별 `config.yaml` / `env.txt` / `git_rev.txt` / `metrics.json` / `scores_fold*.json` 보존(§11).

## 9. 산출물 인덱스

| 경로 | 내용 |
|---|---|
| `docs/SOP_cable_defect_lightweight.md` | SOP 본문 (실행 이력·이탈 사항 포함) |
| `reports/data_integrity_cable.md` / `.csv` | 무결성 검증 |
| `reports/label_conversion_cable.md` | 라벨 변환 |
| `reports/label_check/` | 육안 검증 오버레이 24장 |
| `reports/splits_cable.md` | 분할 설계 |
| `reports/track_c_results.md` | Track C 상세 |
| `reports/ablation_A1_resolution.md` | A1 해상도 ablation |
| `reports/figs/ablation_A1_pareto.png` | 파레토 그림 |
| `reports/P1_completion.md` | P1 완료 리포트 |
| **`reports/SUMMARY_cable_trackC.md`** | **본 통합 리포트** |
| `runs/EXP_C_mnv3s_640_fp32_all_20260921/` | 640px 실험 일체 |
| `runs/EXP_C_mnv3s_384_fp32_all_20260921/` | 384px 실험 일체 |
| `runs/EXP_C_mnv3s_256_fp32_all_20260921/` | 256px 실험 일체 |
| `data/classes.yaml` | 클래스 ID 매핑 (고정) |
| `data/splits/` | 분할 정의 (A / 홀드아웃 / 5-fold) |
| `data/yolo/{multi,binary}/` | YOLO 포맷 라벨 (Track A 대비) |
| `requirements.lock.txt` | 패키지 잠금 |

