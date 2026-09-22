const pptxgen = require("pptxgenjs");

const P = {
  dark:  "2B3A42",   // 슬레이트 (산업 검사)
  dark2: "1C272C",
  steel: "8FA3AD",
  light: "F4F6F7",
  white: "FFFFFF",
  ok:    "2E9E6B",   // 통과 / 강건
  ng:    "C8453C",   // 붕괴 / 미결
  warn:  "D99A2B",   // 진행중 / 보류
  muted: "6B7C85",
};
const F = { head: "Cambria", body: "Calibri" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5
pres.author = "hojun.han";
pres.title = "MVTec cable 양/불 판정 경량 모델 — 연구 흐름도";

const W = 13.33, H = 7.5, M = 0.6;

function titleBar(slide, t, sub) {
  slide.addText(t, {
    x: M, y: 0.38, w: W - 2 * M, h: 0.62, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 30, bold: true, color: P.dark,
  });
  if (sub) slide.addText(sub, {
    x: M, y: 1.02, w: W - 2 * M, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 13, color: P.muted,
  });
}

function badge(slide, x, y, text, color) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w: 0.82, h: 0.26, fill: { color }, rectRadius: 0.13, line: { color, width: 0 },
  });
  slide.addText(text, {
    x, y, w: 0.82, h: 0.26, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 9.5, bold: true, color: P.white,
    align: "center", valign: "middle",
  });
}

function arrowRight(slide, x, y, w) {
  slide.addShape(pres.ShapeType.rightArrow, {
    x, y, w, h: 0.22, fill: { color: P.steel }, line: { color: P.steel, width: 0 },
  });
}
function arrowDown(slide, x, y, h) {
  slide.addShape(pres.ShapeType.downArrow, {
    x, y, w: 0.22, h, fill: { color: P.steel }, line: { color: P.steel, width: 0 },
  });
}

/* ---------------------------------------------------------- 1. 타이틀 */
{
  const s = pres.addSlide();
  s.background = { color: P.dark };
  s.addText("MVTec cable", {
    x: M, y: 2.25, w: 9.5, h: 0.62, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 20, color: P.steel, charSpacing: 3,
  });
  s.addText("산업 제품 양/불 판정\n경량 모델 연구 흐름도", {
    x: M, y: 2.8, w: 9.6, h: 1.9, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 40, bold: true, color: P.white, lineSpacing: 46,
  });
  s.addText("SOP-MVTEC-CABLE-001  ·  2026-09-22", {
    x: M, y: 4.85, w: 9.5, h: 0.32, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 13, color: P.steel,
  });

  // 핵심 3 지표
  const stats = [
    ["0.00 %", "미검율 (무섭동)", P.ok],
    ["0.9992", "AUROC", P.ok],
    ["18.8 ms", "지연 (단일 스레드)", P.ok],
  ];
  stats.forEach(([v, k, c], i) => {
    const x = M + i * 2.35;
    s.addText(v, {
      x, y: 5.55, w: 2.1, h: 0.55, isTextBox: true, margin: 0,
      fontFace: F.head, fontSize: 28, bold: true, color: c,
    });
    s.addText(k, {
      x, y: 6.08, w: 2.1, h: 0.3, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 11, color: P.steel,
    });
  });
  s.addText("정확도는 포화 — 남은 과제는 환경 변동 강건성과 임계값 이식성", {
    x: 7.6, y: 5.62, w: 5.1, h: 0.9, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 13, italic: true, color: P.white,
  });
  s.addNotes("MobileNetV3-Small 5-fold 결과. 미검율 0%는 검증셋 기준이며 홀드아웃은 미사용.");
}

/* ------------------------------------------------- 2. 전체 연구 흐름 */
{
  const s = pres.addSlide();
  s.background = { color: P.white };
  titleBar(s, "전체 연구 흐름", "SOP 단계 · 각 단계는 게이트 통과 후에만 다음으로 진행");

  const steps = [
    ["P0", "데이터 복원", "불량 33 → 92장", "완료", P.ok],
    ["P1", "데이터 준비", "bbox 151 · 5-fold", "완료", P.ok],
    ["P2", "기준선 학습", "미검율 0.00 %", "완료", P.ok],
    ["P3", "해상도 ablation", "640/384/256", "완료", P.ok],
    ["S0", "강건성 벤치", "색·반사 붕괴", "완료", P.ok],
    ["S1", "처방 A/B/C", "C 채택", "완료", P.ok],
    ["P4~6", "양자화·배포", "타깃 HW 대기", "대기", P.muted],
  ];

  const bw = 1.62, gap = 0.19, y = 2.0;
  const totalW = steps.length * bw + (steps.length - 1) * gap;
  const x0 = (W - totalW) / 2;

  steps.forEach(([tag, name, val, st, c], i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: bw, h: 1.95, fill: { color: P.light }, rectRadius: 0.08,
      line: { color: P.steel, width: 0.75 },
      shadow: { type: "outer", angle: 90, blur: 6, offset: 1, opacity: 0.12, color: "000000" },
    });
    s.addText(tag, {
      x, y: y + 0.16, w: bw, h: 0.32, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 11, bold: true, color: P.steel, align: "center",
    });
    s.addText(name, {
      x: x + 0.08, y: y + 0.5, w: bw - 0.16, h: 0.6, isTextBox: true, margin: 0,
      fontFace: F.head, fontSize: 13.5, bold: true, color: P.dark, align: "center",
    });
    s.addText(val, {
      x: x + 0.06, y: y + 1.08, w: bw - 0.12, h: 0.4, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 10, color: P.muted, align: "center",
    });
    badge(s, x + (bw - 0.82) / 2, y + 1.55, st, c);
    if (i < steps.length - 1) arrowRight(s, x + bw + 0.01, y + 0.88, gap - 0.02);
  });

  // 하단 — 단계별 산출 문서
  s.addText("게이트", {
    x: M, y: 4.35, w: 1.2, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 11, bold: true, color: P.steel,
  });
  const gates = [
    "STOP-GATE 1  데이터 전량 복원 · 무결성 통과",
    "P1 게이트  라벨 육안 검증 24장 통과",
    "P2 게이트  최소 1개 트랙 AUROC ≥ 0.90",
    "STOP-GATE 2  홀드아웃 최종 판정 (미사용 — P5 전용)",
  ];
  gates.forEach((g, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = M + col * 6.2, y2 = 4.72 + row * 0.44;
    s.addShape(pres.ShapeType.ellipse, {
      x, y: y2 + 0.06, w: 0.16, h: 0.16,
      fill: { color: i === 3 ? P.muted : P.ok }, line: { width: 0 },
    });
    s.addText(g, {
      x: x + 0.28, y: y2, w: 5.8, h: 0.3, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 11.5, color: P.dark,
    });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 5.85, w: W - 2 * M, h: 0.92, fill: { color: P.dark }, rectRadius: 0.08,
    line: { width: 0 },
  });
  s.addText("GPU 미확보로 Track A(YOLO 검출)·Track B(비지도 이상탐지)는 연기 — Track C(경량 분류)를 선행", {
    x: M + 0.35, y: 5.98, w: W - 2 * M - 0.7, h: 0.32, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 13, bold: true, color: P.white,
  });
  s.addText("따라서 “어느 태스크가 유리한가”(연구질문 1)는 현재 답할 수 없다", {
    x: M + 0.35, y: 6.32, w: W - 2 * M - 0.7, h: 0.32, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 11.5, color: P.steel,
  });
  s.addNotes("SOP §12 단계별 게이트. STOP-GATE 2는 홀드아웃 전용이며 아직 사용하지 않았다.");
}

/* --------------------------------------------- 3. 데이터 준비 흐름 */
{
  const s = pres.addSlide();
  s.background = { color: P.white };
  titleBar(s, "데이터 준비 흐름 (P1)", "전 단계가 스크립트로 재실행 가능 — src/data/run_p1.ps1");

  const boxes = [
    ["원본 데이터셋", ["MVTec AD cable", "train/good 224", "test 150 (불량 92)", "mask 92"]],
    ["① 무결성 검증", ["해상도·채널·이진 마스크", "인덱스 정합 양방향", "공식 수량 대조", "→ 문제 0건"]],
    ["② 라벨 변환", ["마스크 → 연결요소 → bbox", "노이즈 임계 209 px", "multi(8종) + binary", "→ bbox 151개"]],
    ["③ 분할 확정", ["분할 A: 원본 유지(비지도)", "홀드아웃 65장 (판정 전용)", "5-fold 클래스 층화", "→ 누수 0건"]],
    ["④ 스모크 체크", ["좌표 범위·클래스 ID", "train ∩ val = ∅", "Ultralytics 로딩", "→ PASS"]],
  ];

  const bw = 2.36, gap = 0.22, y = 1.75;
  const totalW = boxes.length * bw + (boxes.length - 1) * gap;
  const x0 = (W - totalW) / 2;

  boxes.forEach(([t, lines], i) => {
    const x = x0 + i * (bw + gap);
    const isSrc = i === 0;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: bw, h: 2.62, rectRadius: 0.08,
      fill: { color: isSrc ? P.dark : P.light },
      line: { color: isSrc ? P.dark : P.steel, width: 0.75 },
    });
    s.addText(t, {
      x: x + 0.14, y: y + 0.18, w: bw - 0.28, h: 0.4, isTextBox: true, margin: 0,
      fontFace: F.head, fontSize: 14, bold: true,
      color: isSrc ? P.white : P.dark,
    });
    s.addText(lines.map((l, k) => ({
      text: l, options: { bullet: { indent: 8 }, breakLine: k < lines.length - 1 },
    })), {
      x: x + 0.14, y: y + 0.68, w: bw - 0.24, h: 1.8, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 10.5, color: isSrc ? P.steel : P.muted,
      paraSpaceAfter: 5,
    });
    if (i < boxes.length - 1) arrowRight(s, x + bw + 0.01, y + 1.2, gap - 0.02);
  });

  s.addText("육안 검증에서 확인한 것 — 이후 설계에 직접 반영", {
    x: M, y: 4.95, w: 8, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 15, bold: true, color: P.dark,
  });
  const finds = [
    ["cable_swap 마스크가 케이블 단면 전체를 덮음", "국소 텍스처 결함 아님 → 구조적 결함"],
    ["bent_wire·cut_outer 마스크가 가늘고 휜 곡선", "mAP@IoU가 검출 품질을 과대평가 → PRO 병행"],
    ["combined은 결함 유형이 아니라 메타 라벨", "8클래스 검출 시 혼동 원인 → binary 병행 준비"],
  ];
  finds.forEach(([a, b], i) => {
    const x = M + i * 4.1;
    s.addShape(pres.ShapeType.ellipse, {
      x, y: 5.45, w: 0.3, h: 0.3, fill: { color: P.warn }, line: { width: 0 },
    });
    s.addText(String(i + 1), {
      x, y: 5.45, w: 0.3, h: 0.3, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 11, bold: true, color: P.white,
      align: "center", valign: "middle",
    });
    s.addText(a, {
      x: x + 0.42, y: 5.39, w: 3.5, h: 0.5, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 11.5, bold: true, color: P.dark,
    });
    s.addText(b, {
      x: x + 0.42, y: 5.94, w: 3.5, h: 0.6, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 10.5, color: P.muted,
    });
  });
  s.addNotes("홀드아웃 65장은 P5 최종 판정 전용이며 아직 한 번도 사용하지 않았다.");
}

/* ------------------------------------- 4. 학습·평가 흐름 + 결과 */
{
  const s = pres.addSlide();
  s.background = { color: P.white };
  titleBar(s, "학습 · 평가 흐름과 결과", "Track C — MobileNetV3-Small, ImageNet 사전학습, 5-fold 교차검증");

  // 좌: 흐름
  const flow = [
    ["이미지 캐시", "1024 PNG 디코딩 병목 제거"],
    ["5-fold 학습", "조기종료 · best = val AUROC"],
    ["점수 재산출", "체크포인트로 val 점수 저장"],
    ["임계값 τ 결정", "과검 5% 하에서 미검 최소화"],
    ["지표 산출", "미검·과검·클래스별 분해"],
  ];
  const fy = 1.78, fh = 0.72, fgap = 0.22;
  flow.forEach(([t, d], i) => {
    const y = fy + i * (fh + fgap);
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y, w: 4.5, h: fh, rectRadius: 0.07,
      fill: { color: P.light }, line: { color: P.steel, width: 0.75 },
    });
    s.addText(t, {
      x: M + 0.18, y: y + 0.08, w: 4.1, h: 0.3, isTextBox: true, margin: 0,
      fontFace: F.head, fontSize: 13, bold: true, color: P.dark,
    });
    s.addText(d, {
      x: M + 0.18, y: y + 0.38, w: 4.1, h: 0.28, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 10, color: P.muted,
    });
    if (i < flow.length - 1) arrowDown(s, M + 2.14, y + fh + 0.01, fgap - 0.02);
  });

  // 우: 결과 표
  const tx = 5.6;
  s.addText("입력 해상도별 결과 (5-fold 평균 ± 표준편차)", {
    x: tx, y: 1.78, w: 7.1, h: 0.32, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 14, bold: true, color: P.dark,
  });
  s.addTable(
    [
      [
        { text: "입력", options: { bold: true, color: P.white, fill: { color: P.dark } } },
        { text: "AUROC", options: { bold: true, color: P.white, fill: { color: P.dark } } },
        { text: "미검율", options: { bold: true, color: P.white, fill: { color: P.dark } } },
        { text: "GFLOPs", options: { bold: true, color: P.white, fill: { color: P.dark } } },
        { text: "지연 p50", options: { bold: true, color: P.white, fill: { color: P.dark } } },
      ],
      ["640", "0.9994 ± 0.0012", "0.00 %", "0.988", "74.9 ms"],
      ["384", "0.9997 ± 0.0006", "0.00 %", "0.357", "54.6 ms"],
      ["256", "0.9992 ± 0.0011", "0.00 %", "0.160", "26.9 ms"],
    ],
    {
      x: tx, y: 2.18, w: 7.1, colW: [0.9, 2.0, 1.15, 1.25, 1.8],
      rowH: 0.42, fontFace: F.body, fontSize: 11.5, color: P.dark,
      border: { type: "solid", color: "D5DDE1", pt: 0.75 },
      align: "center", valign: "middle",
    }
  );
  s.addText("세 해상도는 통계적으로 구분되지 않는다 — 차이가 표준편차 이내", {
    x: tx, y: 4.0, w: 7.1, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 11, italic: true, color: P.muted,
  });

  // 큰 수치
  const kpis = [
    ["76 / 0", "5-fold 합산 TP / FN", P.ok],
    ["1.000", "8클래스 recall 전부", P.ok],
    ["1.52 M", "파라미터", P.dark],
  ];
  kpis.forEach(([v, k, c], i) => {
    const x = tx + i * 2.4;
    s.addText(v, {
      x, y: 4.5, w: 2.3, h: 0.6, isTextBox: true, margin: 0,
      fontFace: F.head, fontSize: 26, bold: true, color: c,
    });
    s.addText(k, {
      x, y: 5.08, w: 2.3, h: 0.5, isTextBox: true, margin: 0,
      fontFace: F.body, fontSize: 10.5, color: P.muted,
    });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: tx, y: 5.72, w: 7.1, h: 1.0, rectRadius: 0.08,
    fill: { color: "FBEEEC" }, line: { color: P.ng, width: 0.75 },
  });
  s.addText("발견 — 임계값 τ가 환경 간 이식되지 않는다", {
    x: tx + 0.25, y: 5.84, w: 6.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 12, bold: true, color: P.ng,
  });
  s.addText("정상 점수 중앙값이 fold마다 2e-11 ~ 2e-4로 8자릿수 차이. 배포 시 현장 재보정 필수", {
    x: tx + 0.25, y: 6.16, w: 6.6, h: 0.44, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 10.5, color: P.dark,
  });
  s.addNotes("모든 수치는 val fold 기준이며 τ도 같은 val에서 정해 낙관적으로 편향돼 있다.");
}

/* ------------------------------- 5. 강건성 검증 흐름 (Stage 0 → 1) */
{
  const s = pres.addSlide();
  s.background = { color: P.white };
  titleBar(s, "강건성 검증 흐름 (Stage 0 → Stage 1)",
    "MVTec에는 환경 변동이 없다 — 조명 4.5 %, 색바램 0.9 %. 합성 섭동으로 대리 측정");

  // Stage 0 결과: 강건 vs 붕괴
  s.addText("Stage 0 — 7축 합성 섭동, 실측값의 1~10배", {
    x: M, y: 1.72, w: 6.0, h: 0.32, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 14, bold: true, color: P.dark,
  });

  const robust = [["회전 ±80°", "0.9973"], ["이동 ±10 %", "0.9915"], ["스케일 ±20 %", "0.9983"], ["밝기 ±50 %", "0.9903"]];
  const broken = [["색바램 30 %", "0.8924"], ["황변 30 %", "0.9018"], ["정반사 면적 10 %", "0.8273"]];

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 2.14, w: 5.9, h: 1.72, rectRadius: 0.08,
    fill: { color: "EAF5EF" }, line: { color: P.ok, width: 0.75 },
  });
  s.addText("강건 — 정렬 모듈 불필요", {
    x: M + 0.22, y: 2.24, w: 5.4, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 12, bold: true, color: P.ok,
  });
  robust.forEach(([k, v], i) => {
    const x = M + 0.22 + (i % 2) * 2.85, y = 2.6 + Math.floor(i / 2) * 0.55;
    s.addText(k, { x, y, w: 1.75, h: 0.28, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 10.5, color: P.dark });
    s.addText(v, { x: x + 1.75, y: y - 0.04, w: 0.95, h: 0.32, isTextBox: true, margin: 0, fontFace: F.head, fontSize: 13, bold: true, color: P.ok });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.83, y: 2.14, w: 5.9, h: 1.72, rectRadius: 0.08,
    fill: { color: "FBEEEC" }, line: { color: P.ng, width: 0.75 },
  });
  s.addText("붕괴 — 재보정으로도 복구 불가", {
    x: 7.05, y: 2.24, w: 5.4, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 12, bold: true, color: P.ng,
  });
  broken.forEach(([k, v], i) => {
    const y = 2.6 + i * 0.4;
    s.addText(k, { x: 7.05, y, w: 3.2, h: 0.28, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 10.5, color: P.dark });
    s.addText(v, { x: 10.3, y: y - 0.04, w: 1.0, h: 0.32, isTextBox: true, margin: 0, fontFace: F.head, fontSize: 13, bold: true, color: P.ng });
  });
  s.addText("정반사는 면적 1 %만으로 0.964 · 10 %면 과검율 100 % (판정기 기능 정지)", {
    x: M, y: 3.94, w: 12.1, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 11, italic: true, color: P.muted,
  });

  // Stage 1 처방 → 판정
  s.addText("Stage 1 — 값싼 처방 3종 (256px 동일 분할·시드)", {
    x: M, y: 4.38, w: 7.0, h: 0.32, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 14, bold: true, color: P.dark,
  });
  const tries = [
    ["A", "robust 증강", "추론 비용 0", P.ok],
    ["B", "전처리 gray+spec", "전처리 시간 추가", P.warn],
    ["C", "A + B 동시", "가장 비쌈", P.warn],
  ];
  tries.forEach(([tag, t, d, c], i) => {
    const x = M + i * 2.6;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 4.8, w: 2.4, h: 1.05, rectRadius: 0.07,
      fill: { color: P.light }, line: { color: P.steel, width: 0.75 },
    });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.16, y: 4.94, w: 0.34, h: 0.34, fill: { color: c }, line: { width: 0 } });
    s.addText(tag, { x: x + 0.16, y: 4.94, w: 0.34, h: 0.34, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 12, bold: true, color: P.white, align: "center", valign: "middle" });
    s.addText(t, { x: x + 0.58, y: 4.94, w: 1.75, h: 0.34, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 11.5, bold: true, color: P.dark, valign: "middle" });
    s.addText(d, { x: x + 0.16, y: 5.36, w: 2.1, h: 0.4, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 10, color: P.muted });
  });
  arrowRight(s, 8.34, 5.22, 0.40);

  s.addShape(pres.ShapeType.roundRect, {
    x: 8.86, y: 4.8, w: 3.87, h: 1.05, rectRadius: 0.07,
    fill: { color: P.dark }, line: { width: 0 },
  });
  s.addText("사전 등록 판정 기준", {
    x: 9.06, y: 4.92, w: 3.47, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 11.5, bold: true, color: P.white,
  });
  s.addText("붕괴 3축 전부에서 다음을 충족해야 합격\nAUROC ≥ 0.95 · 재보정 미검율 ≤ 5 % · 과검율 < 50 %", {
    x: 9.06, y: 5.22, w: 3.47, h: 0.56, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 9.5, color: P.steel, lineSpacing: 13,
  });

  s.addText("판정 — 사전 등록 기준으로는 전원 불합격(정반사 축).  색 2축은 C가 완전 해결 → C 채택, 정반사는 독립 과제로 분리", {
    x: M, y: 6.28, w: 12.1, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 11.5, bold: true, color: P.dark,
  });
  s.addText("C: 색바램 0.9974 · 황변 0.9977 · 정반사 0.9228 (미해결)   |   무섭동 0.9980, 단일 스레드 18.75 ms", {
    x: M, y: 6.64, w: 12.1, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 10.5, italic: true, color: P.muted,
  });
  s.addNotes("합성 섭동은 실제 라인 변동의 대리다. 섭동 모형이 현장과 다르면 결론도 달라진다.");
}

/* --------------------------------------- 6. 현재 상태 · 다음 단계 */
{
  const s = pres.addSlide();
  s.background = { color: P.dark };
  s.addText("현재 상태와 다음 단계", {
    x: M, y: 0.45, w: W - 2 * M, h: 0.6, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 30, bold: true, color: P.white,
  });
  s.addText("기술이 아니라 의사결정이 병목이다", {
    x: M, y: 1.06, w: W - 2 * M, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 13, color: P.steel,
  });

  s.addText("의사결정 대기", {
    x: M, y: 1.75, w: 5.6, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 16, bold: true, color: P.white,
  });
  const decide = [
    ["타깃 하드웨어", "export 경로 · 벤치 · 정반사 대응 방식을 모두 막고 있음", P.ng],
    ["택트타임 예산", "해상도 · 모델 선택", P.warn],
    ["미검 : 과검 비용비", "임계값 정책", P.warn],
    ["판정 출력 형태", "본선 트랙 확정 (라벨은 둘 다 준비됨)", P.warn],
    ["GPU 확보", "Track A · B 착수 조건", P.warn],
  ];
  decide.forEach(([t, d, c], i) => {
    const y = 2.2 + i * 0.78;
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.08, w: 0.2, h: 0.2, fill: { color: c }, line: { width: 0 } });
    s.addText(t, { x: M + 0.35, y, w: 5.2, h: 0.3, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 13, bold: true, color: P.white });
    s.addText(d, { x: M + 0.35, y: y + 0.3, w: 5.2, h: 0.4, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 10.5, color: P.steel });
  });

  s.addText("기술 작업 후보", {
    x: 6.9, y: 1.75, w: 5.8, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F.head, fontSize: 16, bold: true, color: P.white,
  });
  const tasks = [
    ["1", "정반사 대응 (미해결)", "조명·편광 필터 우선 · 안 되면 Stage 2"],
    ["2", "CableInspect-AD 확보", "실측 현장 데이터 · 불량 2,639장(29배)"],
    ["3", "INT8 양자화 + ONNX export", "6.08 → 1.52 MB · 수치 동등성 검증"],
    ["4", "타깃 하드웨어 벤치", "열 스로틀링 포함 §9.3 프로토콜"],
  ];
  tasks.forEach(([n, t, d], i) => {
    const y = 2.2 + i * 0.95;
    s.addShape(pres.ShapeType.roundRect, {
      x: 6.9, y, w: 5.83, h: 0.82, rectRadius: 0.07,
      fill: { color: P.dark2 }, line: { color: "3D4E57", width: 0.75 },
    });
    s.addShape(pres.ShapeType.ellipse, { x: 7.1, y: y + 0.24, w: 0.34, h: 0.34, fill: { color: i === 0 ? P.ok : P.steel }, line: { width: 0 } });
    s.addText(n, { x: 7.1, y: y + 0.24, w: 0.34, h: 0.34, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 12, bold: true, color: P.dark, align: "center", valign: "middle" });
    s.addText(t, { x: 7.55, y: y + 0.11, w: 5.0, h: 0.3, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 12.5, bold: true, color: P.white });
    s.addText(d, { x: 7.55, y: y + 0.42, w: 5.0, h: 0.32, isTextBox: true, margin: 0, fontFace: F.body, fontSize: 10, color: P.steel });
  });

  s.addText("홀드아웃 65장은 아직 사용하지 않았다 — 최종 후보 확정 후 P5에서 1회만 사용한다", {
    x: M, y: 6.55, w: W - 2 * M, h: 0.34, isTextBox: true, margin: 0,
    fontFace: F.body, fontSize: 12, italic: true, color: P.steel,
  });
  s.addNotes("CableInspect-AD는 대리 데이터·표본 부족·임계값 이식성 세 한계를 동시에 해소할 수 있다.");
}

const out = "D:/project/mvtec_defect/docs/연구_흐름도.pptx";
pres.writeFile({ fileName: out }).then(() => console.log("wrote " + out));
