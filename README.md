# MVTec `cable` 산업 제품 양/불 판정 경량 모델 연구

소형 하드웨어(엣지)에서 실시간 동작하는 케이블 단면 양/불(OK/NG) 판정 모델을 확보하기 위한
연구 저장소다. 최종 목표는 **실제 산업 제품 검사**이며 MVTec AD `cable`은 대리 데이터다.

전 과정은 SOP 문서 [`docs/SOP_cable_defect_lightweight.md`](docs/SOP_cable_defect_lightweight.md)를
따른다. 실행 이력과 SOP 이탈 사항은 그 문서의 변경 이력에 기록된다.

---

## 현재 결과 요약

MobileNetV3-Small 이진 분류, ImageNet 사전학습, 5-fold 교차검증.

| 입력 | AUROC | 미검율 | 과검율 | GFLOPs | e2e p50 |
|---|---|---|---|---|---|
| 640 | 0.9994 ± 0.0012 | 0.00 % | 0.87 % | 0.988 | 74.9 ms |
| 384 | 0.9997 ± 0.0006 | 0.00 % | 0.43 % | 0.357 | 54.6 ms |
| 256 | 0.9992 ± 0.0011 | 0.00 % | 1.30 % | 0.160 | 26.9 ms |

5-fold 합산 혼동행렬 TP 76 / FN 0 / FP 1~3 / TN 230~232. 8개 불량 클래스 recall 전부 1.000.
세 해상도는 통계적으로 구분되지 않는다(차이가 표준편차 이내).

지연시간은 **개발 머신 CPU** 값이며 타깃 하드웨어 수치가 아니다.

### 정확도는 포화, 진짜 문제는 다른 곳에 있다

합성 섭동 강건성 벤치마크(Stage 0) 결과, 실제 병목은 정확도가 아니라 환경 변동 강건성이다.

| 축 | 최대 강도 | AUROC (256px) | 재보정 미검율 | 과검율 |
|---|---|---|---|---|
| 회전 | ±80° | 0.9973 | 1.2 % | 12.1 % |
| 이동 | ±10% | 0.9915 | 5.2 % | 9.0 % |
| 스케일 | ±20% | 0.9983 | 1.3 % | 9.5 % |
| 밝기 | ±50% | 0.9903 | 6.6 % | 12.9 % |
| **색바램** | 30% | **0.8924** | **39.2 %** | 65.7 % |
| **황변** | 30% | **0.9018** | **37.4 %** | 81.5 % |
| **정반사** | 면적 10% | **0.8273** | **56.3 %** | **100.0 %** |

- 기하 축은 실측 변동의 10배까지 밀어도 무너지지 않는다
- 색·반사 축은 재보정으로도 복구되지 않는다 — 표현 자체가 무너진다
- 정반사는 면적 1%만으로 AUROC 0.964까지 떨어진다

또한 임계값 τ가 fold마다 8자릿수 차이를 보여(정상 점수 중앙값 2e-11 ~ 2e-4)
**한 환경에서 정한 절대 임계를 다른 환경에 이식할 수 없다**. 배포 시 현장 재보정이 필요하다.

---

## 저장소 구성

```
configs/            경로·시드·파라미터 설정
data/               classes.yaml, splits/(5-fold + 홀드아웃), labels_index.csv
docs/               SOP, 연구 후보 문서
reports/            검증·실험·분석 리포트 (아래 참조)
runs/EXP_*/         실험별 config / env / git_rev / metrics / scores / robustness
src/data/           무결성 검증, 라벨 변환, 분할, 캐시, 전처리, 환경변수 측정
src/train/          Track C 학습
src/eval/           지표, 점수 재산출, 강건성 벤치, 지연 측정, 리포트 생성
```

### 주요 리포트

**먼저 읽을 문서: [`reports/PROGRESS.md`](reports/PROGRESS.md)** — 전체 진행 현황의 최신본이다.
다른 리포트는 특정 시점·특정 단계의 기록이므로, 서로 결론이 다를 때는 `PROGRESS.md`가 우선한다.

| 문서 | 범위 | 비고 |
|---|---|---|
| **[`PROGRESS.md`](reports/PROGRESS.md)** | **전체 진행 (최신본)** | P0~Stage 1 + 문서 분석 + 미결사항 |
| [`SUMMARY_cable_trackC.md`](reports/SUMMARY_cable_trackC.md) | P1~P3 통합 | **P1~P3 시점 기준.** Stage 0 이후 결과 미포함 |
| [`dataset_variation_cable.md`](reports/dataset_variation_cable.md) | 데이터셋 환경 변수 실측 | 회전·조명·색바램 실측치 |
| [`robustness_cable.md`](reports/robustness_cable.md) | Stage 0 강건성 벤치 | 7축 × 강도 × 3해상도 |
| [`stage1_decision_criteria.md`](reports/stage1_decision_criteria.md) | Stage 1 판정 기준 | **결과 확인 전 사전 등록** |
| [`ablation_A1_resolution.md`](reports/ablation_A1_resolution.md) | 입력 해상도 ablation | 640/384/256 |
| [`track_c_results.md`](reports/track_c_results.md) | Track C 상세 | fold별 지표·점수 분포 |
| [`연구후보_분석.md`](reports/연구후보_분석.md) | 제안 방법론 2건 분석 | `code_artifact.md` / `sota.md` |
| [`detection_참조_분석.md`](reports/detection_참조_분석.md) | 참조 문헌 10건 분석 | 적용 가능성 + 서지 검증 |
| [`P1_completion.md`](reports/P1_completion.md) 외 | P1 단계별 검증 | 무결성·라벨·분할 |

> **주의**: `SUMMARY_cable_trackC.md` §7의 "256px 채택" 권고는 이후 Stage 0 강건성 결과
> (섭동 하에서 640px가 일관되게 우수)로 **보류 상태**다. 배포 해상도는 Stage 1 완료 후 확정한다.

---

## 재현

데이터셋과 파생물(캐시·YOLO 라벨·가중치)은 저장소에 포함하지 않는다. 아래 순서로 재생성한다.

```powershell
# 0) 환경 (Python 3.12 기준)
<python3.12> -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt

# 1) 데이터셋 배치 — MVTec AD 에서 cable 을 받아 mvtec/cable/ 에 둔다
#    https://www.mvtec.com/company/research/datasets/mvtec-ad
#    기대 수량: train/good 224, test 150(불량 92), ground_truth 92

# 2) P1 — 무결성 검증 → 라벨 변환 → 분할 → 스모크 체크
powershell -ExecutionPolicy Bypass -File src\data\run_p1.ps1

# 3) 이미지 캐시 (CPU 학습 병목 제거)
.venv\Scripts\python.exe src\data\build_cache.py

# 4) Track C 학습 (해상도별 5-fold)
.venv\Scripts\python.exe src\train\train_cls.py --fold all --res 256 --epochs 50 --patience 15 --workers 0

# 5) 점수 재산출 (τ 마진 tie-break, 점수 원본 저장)
.venv\Scripts\python.exe src\eval\score_folds.py EXP_C_mnv3s_256_fp32_all_<YYYYMMDD>

# 6) 강건성 벤치 · 지연 측정 · 리포트
.venv\Scripts\python.exe src\eval\robustness_bench.py EXP_C_mnv3s_256_fp32_all_<YYYYMMDD>
.venv\Scripts\python.exe src\eval\bench_latency.py
.venv\Scripts\python.exe src\eval\report_summary.py
```

시드 42 전역 고정. 분할은 JSON으로 고정 저장하며 실행마다 재추첨하지 않는다.
실험별로 `config.yaml` / `env.txt` / `git_rev.txt` / `metrics.json` / `scores_fold*.json` 을 남긴다.

---

## 결과 해석 시 주의

1. 모든 수치가 **검증 fold 기준**이고 임계값 τ도 같은 검증셋에서 정해 **낙관적으로 편향**돼 있다.
   편향 없는 값은 홀드아웃에서만 얻으며, **홀드아웃은 아직 사용하지 않았다**
2. fold당 검증 불량이 15~16장이다. 미검 0건이어도 rule of three 기준 95% 상한이
   fold 단위 약 20%, 5-fold 합산(76장)해도 약 3.9%다
3. **합성 섭동은 실제 라인 변동의 대리다.** 섭동 모형이 현장과 다르면 결론도 달라진다
4. 지연시간·메모리는 개발 머신 CPU 값이다. 타깃 하드웨어에서 열 스로틀링을 포함해 재측정해야 한다
5. Track A(YOLO 검출)·Track B(비지도 이상탐지)는 GPU 미확보로 미수행이다.
   "어느 태스크가 유리한가"는 현재 답할 수 없다

---

## 데이터셋 라이선스

MVTec AD 는 **CC BY-NC-SA 4.0**(비상업적 연구 목적)으로 배포된다.
본 저장소는 데이터셋 이미지와 그 파생 이미지를 포함하지 않는다.
상업적 이용은 별도 라이선스 확인이 필요하다.

> Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019).
> MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. *CVPR*.
