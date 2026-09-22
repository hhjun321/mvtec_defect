# 환경 변수 대응형 케이블 결함 검출 방법론

## 1. 개요 (Overview)
MVTec AD 데이터셋의 `cable` 클래스는 피복의 미세 손상(Structural) 및 가닥 배선 오류(Logical/Wire Swap)가 공존하는 비정형 객체입니다. 본 문서에서는 물체의 모양과 방향이 비교적 일정한 환경에서, 미세 회전/이동, 빛 굴절/반사, 색바램 등의 환경 변수에 강건하게 대응하기 위한 **템플릿 정렬 및 속성 분리 기반 결함 검출 흐름도**를 정립합니다.

---

## 2. 단계별 파이프라인 (Step-by-Step Pipeline)

```
[Input Image]
      │
      ▼
[1. 전처리: 공간 정렬 및 환경 정규화]
 (Affine Alignment + HSV/LAB CLAHE + Chromatic Normalization)
      │
      ▼
[2. 영역 매핑: 고정 Template Mask 프로젝션]
 (Reference Mask기반 피복/개별 가닥 영역 분리)
      │
      ▼
[3. 특징 추출: 조명/색상 불변 Feature Embedding]
 (DINOv2 Feature + Relative Chromatic Order Feature)
      │
      ├─────────────────────────────────────────┐
      ▼                                         ▼
[4-A. Track A: 표면/구조 결함 검출]      [4-B. Track B: 배열/논리 결함 검출]
 (DINOv2 Cosine Similarity Map)          (Relative Color Sequence Order Matching)
      │                                         │
      └────────────────────┬────────────────────┘
                           ▼
             [5. Multi-Aspect Anomaly Fusion]
```

### 2.1. 전처리: 공간 정렬 및 환경 변수 정규화 (Preprocessing)
* **미세 위치/각도 오차 보정:**
  * 정상 기준 이미지(Canonical Reference Template)를 설정합니다.
  * 입력 이미지 간 **ORB/ECC (Enhanced Correlation Coefficient)** 기반 Affine Transformation 매트릭스를 추정하여 좌표계를 완벽히 정렬합니다.
* **빛 굴절 및 명암 불균일 대응:**
  * RGB 공간을 **HSV 또는 LAB 색상 공간**으로 변환하여 조명/밝기 성분($V$ 또는 $L$)을 분리합니다.
  * 밝기 채널에 **CLAHE (Contrast Limited Adaptive Histogram Equalization)**를 적용하여 빛 반사 및 하이라이트 영향을 최소화합니다.
* **색바램 대응:**
  * 절대적인 RGB 값 대신 **Chromaticity Normalization ($r = \frac{R}{R+G+B}, g = \frac{G}{R+G+B}$)**을 적용하여 조도 변화 및 색조 변색에 강건한 특성을 확보합니다.

### 2.2. 영역 매핑: 정렬 기반 고정 Mask 프로젝션 (Region Mapping)
* 이미지가 정렬되었으므로, 매 프레임 Mask를 생성하지 않고 미리 정의된 **정밀 Reference Mask**를 프로젝션합니다.
* 영역 구획:
  1. 외부 피복 Mask ($M_{jacket}$)
  2. 개별 내부 가닥 선 Mask ($M_{wire\_1}, M_{wire\_2}, \dots, M_{wire\_k}$)

### 2.3. 특징 추출 (Feature Extraction)
* **DINOv2 Deep Feature:** 픽셀 레벨 조명 변화에 불변하며 본질적인 질감 및 구조 정보를 유지합니다.
* **상대적 색상 스펙트럼 (Relative Chromatic Vector):** Mask 영역별 평균 색상 벡터의 상대적 관계를 인코딩합니다.

### 2.4. 2-Track 결함 판정 (Dual-Track Anomaly Decision)
* **Track A (표면/구조 결함 - Structural):**
  * 정렬된 좌표 대 좌표의 DINOv2 패치 특징 거리(L2 / Cosine Distance) 연산.
  * 피복 Cut, Wire Poke, 찢어짐 등 국소 텍스처 이상을 즉시 검출.
* **Track B (배열/논리 결함 - Logical):**
  * 각 가닥 Mask 영역 $M_{wire\_i}$ 간 상대적 색상 순서를 비교 검증.
  * 전반적인 색바램이 발생하더라도 가닥 간 **상대적 색상 순서(Relative Color Sequence Order)**는 유지되므로 Wire Swap 오탐을 방지할 수 있습니다.

---

## 3. 환경 변수별 대처 요약

| 변수 환경 | 발생 가능한 문제점 | 본 방법론의 해결책 |
| :--- | :--- | :--- |
| **미세 각도/위치 오차** | 패치 비교 시 경계면 False Positive 속출 | ECC/Affine Alignment로 템플릿 좌표 일치화 |
| **빛 굴절/조명 반사** | 하이라이트 부위를 결함으로 오인 | HSV/LAB 분리 및 CLAHE, DINOv2 Deep Feature 활용 |
| **색바램 (Color Fading)** | 기준 정상 색상값과 미매칭 문제 | Absolute RGB 대신 Relative Chromaticity Order 검증 |

---

## 4. 관련 레퍼런스 (References)

1. **MVTec AD Dataset & Benchmarks**
   * Bergmann, P., Bormann, M., Steger, C., & Kubik, S. (2019). MVTec AD—A comprehensive real-world dataset for unsupervised anomaly detection. In *CVPR* (pp. 9592-9600).
2. **Image Alignment & Registration (ECC)**
   * Evangelidis, G. D., & Psarakis, E. Z. (2008). Parametric image alignment using enhanced correlation coefficient maximization. *IEEE TPAMI*, 30(10), 1858-1865.
3. **Vision Foundation Model for Feature Extraction**
   * Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*.
4. **Color Normalization & Invariance**
   * Finlayson, G. D., & Schaefer, G. (2001). Convex position color constancy. *IEEE TPAMI*, 23(2), 120-129.