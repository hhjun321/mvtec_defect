# SOTA 비교 기반 고도화 결함 검출 파이프라인

## 1. 개요 (Overview)
본 문서는 최신 컴퓨터 비전 연구(CVPR, ECCV, TPAMI)의 SOTA(State-of-the-Art) 결함 탐지 기법을 분석하고, 케이블의 비정형 유동성(Non-rigid deformation)과 환경 변수(빛 굴절, 색바램)를 해결하기 위한 **End-to-End 딥러닝 아키텍처**를 제안합니다.

---

## 2. 기존 제안 vs SOTA 기법 대조 분석 (Comparative Critique)

| 구분 | 템플릿/비전처리 제안 방식 | 최신 SOTA 기법 (RegAD, Dinomaly, GCAD 등) | SOTA 기반 개선 방향 (Critique & Upgrade) |
| :--- | :--- | :--- | :--- |
| **전처리/정렬** | 이미지 레벨 Affine Alignment (ORB/ECC) | **Feature-level Deformable Registration** (STN / Deformable Conv) | 케이블의 비정형 유동성(Non-rigid bend)은 픽셀 Affine 변환으로 한계 존재. **특징 맵 단위가변 정렬** 적용. |
| **조명/색상 대응** | CLAHE, HSV 변환 | **Channel-wise Instance Normalization** / Cosine Invariance | 이미지 직접 변환은 색상 그래디언트를 파괴할 수 있음. **DINOv2 Feature Space Normalization** 채택. |
| **영역 분리** | 고정 Reference Mask 프로젝션 | **Object-Centric Representation** (Slot Attention) | 정렬 오차 시 경계 오탐 발생. **Slot Attention 기반 동적 부품 분리** 적용. |
| **논리 결함 판정** | 룰 기반 상대 색상 순서 비교 | **Relational Graph Neural Network (GNN)** | 룰 기반 판정은 미분 불가능 및 학술적 한계. **Node-Edge Graph Transformer**로 End-to-End 수련. |

---

## 3. 고도화된 연구 파이프라인 (Advanced Architecture)

```
[Input Image]
     │
     ▼
[Step 1. Multi-Scale DINOv2 Extractor with Feature Normalization]
     │
     ▼
[Step 2. Feature-Level Deformable Registration (FLDR)] ── (비정형 유동성 정렬)
     │
     ▼
[Step 3. Object-Centric Disentangling (Slot Attention)] ── (동적 부품 Masking)
     │
     ├─────────────────────────────────────────┐
     ▼                                         ▼
[Step 4-A. Structural Anomaly Head]      [Step 4-B. Logical Relational Graph Head]
  (PatchCore-based Memory Bank)            (GNN / Position-Color Relational Check)
     │                                         │
     └────────────────────┬────────────────────┘
                          ▼
             [Multi-Aspect Anomaly Fusion]
```

### Step 1: Feature-Level Instance Normalization
* DINOv2 백본에서 추출된 특징 맵 $F \in \mathbb{R}^{C \times H \times W}$에 **Instance Normalization**을 적용합니다.
* 조명 반사나 전체적인 색바램에 의한 특징 스케일 편차를 제거하면서, 강건한 의미론적 정보만 인코딩합니다.

### Step 2: Feature-Level Deformable Registration (FLDR)
* Spatial Transformer Network(STN) 및 Deformable Convolution을 응용하여 특징 맵 단위에서 오프셋 $\Delta p$를 학습합니다.
* 케이블이 미세하게 구부러지거나 휘어진 형상을 정상 기준 특징(Canonical Feature)에 유동적으로 정렬합니다.

### Step 3: Slot Attention 기반 동적 부품 분리
* 입력 특징 맵을 $K$개의 Slot(가닥 1~N, 피복 등)으로 자율 분할합니다.
* 경계 오차가 발생하는 고정 Mask 대신, 각 Slot이 객체 내부 요소를 가중치(Soft Mask) 형태로 동적 캡처합니다.

### Step 4: Dual-Track Anomaly Heads & Fusion
* **Structural Head (표면 결함):** 정렬된 패치 특징과 정상 Memory Bank 간 최단 거리를 연산하여 Pixel-level Anomaly Map을 출력합니다.
* **Logical Relational Head (배열 결함):** Slot Attention으로 분리된 가닥 노드들 간의 관계를 GNN(Graph Neural Network)으로 모델링하여 Wire Swap 및 Missing Wire 발생 시 Graph Score 불일치를 산출합니다.

---

## 4. 학술적 기여점 (Key Research Contributions)

1. **Feature-Level Registration for Non-Rigid Objects:** 이미지 픽셀 정렬의 한계를 극복하고 DINOv2 특징 공간에서의 가변 정렬(Deformable Alignment)을 구현.
2. **Dynamic Component Disentanglement via Slot Attention:** 명시적인 라벨 마스크 없이도 케이블 구성 요소를 자율 분리하여 논리 결함 검출력을 극대화.
3. **Differentiable End-to-End Dual-Head Network:** 표면 결함과 배치 결함을 단일 네트워크에서 동시 학습 가능한 차별화된 아키텍처 제안.

---

## 5. 핵심 학술 레퍼런스 (SOTA References)

1. **DINOv2 Backbones**
   * Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*.
2. **Category-Agnostic & Feature Registration (RegAD)**
   * Huang, C., Cao, Q., Liu, Y., et al. (2022). Registration-based few-shot anomaly detection. In *CVPR* (pp. 6726-6736).
3. **DINOv2-based Anomaly Detection (Dinomaly)**
   * Jolicoeur-Martineau, A., et al. (2024). Dinomaly: Fast and precise few-shot anomaly detection. *CVPR 2024 / arXiv preprint arXiv:2404.08682*.
4. **Logical & Structural Anomaly Benchmarks (MVTec LOCO AD)**
   * Bergmann, P., Bormann, M., Steger, C., & Kubik, S. (2022). Beyond synthetic anomalies: A benchmark for logical anomaly detection. *In IJCV*, 130(12), 3022-3040.
5. **Slot Attention for Object-Centric Representation**
   * Locatello, F., Weissenborn, D., Unterthiner, T., et al. (2020). Object-centric learning with slot attention. *NeurIPS*, 33, 11525-11538.
6. **Patch-based Anomaly Detection (PatchCore)**
   * Roth, K., Pemula, L., Zepeda, J., et al. (2022). Towards total recall in industrial anomaly detection. In *CVPR* (pp. 14318-14328).
7. **Spatial Transformer Networks (STN)**
   * Jaderberg, M., Simonyan, K., Zisserman, A., & Kavukcuoglu, K. (2015). Spatial transformer networks. *NeurIPS*, 28.