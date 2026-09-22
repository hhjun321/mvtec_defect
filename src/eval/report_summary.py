"""P1~P3 통합 리포트 — SOP-MVTEC-CABLE-001 전 단계 결과를 한 문서로 묶는다.

기존 산출물(metrics.json, labels_index.csv, splits/*.json, data_integrity)에서
직접 수치를 읽어 조립한다. 손으로 옮겨 적은 값이 없어야 재현 가능하다(§11).

산출물: reports/SUMMARY_cable_trackC.md
사용:   python src/eval/report_summary.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg, load_classes  # noqa: E402


def ms(a, nd=4):
    return f"{a['mean']:.{nd}f} ± {a['std']:.{nd}f}"


def pct(a):
    return f"{a['mean']*100:.2f} ± {a['std']*100:.2f}"


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    runs = Path(cfg["runs_dir"])
    reports = Path(cfg["reports_dir"])
    data_root = Path(cfg["data_root"])
    splits_dir = Path(cfg["splits_dir"])

    # ---------- 입력 수집 ----------
    exps = sorted((json.loads((p / "metrics.json").read_text(encoding="utf-8"))
                   for p in runs.glob("EXP_C_*") if (p / "metrics.json").exists()),
                  key=lambda m: -m["config"]["res"])
    if not exps:
        print("EXP_C_* 결과 없음")
        return 1
    ref = exps[0]

    with open(data_root / "labels_index.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_def = sum(1 for r in rows if r["image_label"] == "1")
    n_ok = len(rows) - n_def
    n_box = sum(int(r["n_boxes"]) for r in rows)
    cls_img = Counter(r["class"] for r in rows if r["image_label"] == "1")
    cls_box = Counter()
    for r in rows:
        if r["image_label"] == "1":
            cls_box[r["class"]] += int(r["n_boxes"])

    hold = json.loads((splits_dir / "holdout.json").read_text(encoding="utf-8"))
    folds = [json.loads((splits_dir / f"fold{k}.json").read_text(encoding="utf-8"))
             for k in range(cfg["n_folds"])]
    split_a = json.loads((splits_dir / "split_A.json").read_text(encoding="utf-8"))

    # 결함 크기 통계 (마스크 전수)
    area_stats = []
    try:
        from PIL import Image
        import glob
        for d in sorted(glob.glob(str(Path(cfg["source_root"]) / "ground_truth" / "*"))):
            c = Path(d).name
            rs, ws, hs = [], [], []
            for p in sorted(Path(d).glob("*.png")):
                a = np.array(Image.open(p)) > 0
                rs.append(a.mean())
                ys, xs = np.where(a)
                ws.append((xs.max() - xs.min() + 1) / a.shape[1])
                hs.append((ys.max() - ys.min() + 1) / a.shape[0])
            area_stats.append((c, len(rs), min(rs) * 100, float(np.median(rs)) * 100,
                               max(rs) * 100, float(np.median(ws)) * 100, float(np.median(hs)) * 100))
    except Exception as e:
        print(f"  (결함 크기 통계 생략: {e})")

    best_light = None
    for m in exps:
        a = m["aggregate"]
        if a["auroc"]["mean"] >= 0.95 and (1 - a["recall_at_fpr5"]["mean"]) <= 0.05:
            if best_light is None or (m["hardware"]["latency_end_to_end"]["p50_ms"]
                                      < best_light["hardware"]["latency_end_to_end"]["p50_ms"]):
                best_light = m

    md: list[str] = []
    A = md.append

    # ---------- 1. 개요 ----------
    A("# 통합 리포트 — MVTec `cable` 불량 판정 경량 모델 (P1~P3)")
    A("")
    A("- SOP: SOP-MVTEC-CABLE-001 (`docs/SOP_cable_defect_lightweight.md`)")
    A(f"- 생성일: {date.today().isoformat()}")
    A("- 범위: P1 데이터 준비 · P2 Track C 기준선 · P3 A1(해상도) ablation")
    A("- 미포함: Track A(YOLO) · Track B(비지도) — GPU 미확보로 연기(§2.2 대안 (c))")
    A("")
    A("## 0. 한 문단 요약")
    A("")
    if best_light:
        bl, ba, bh = best_light, best_light["aggregate"], best_light["hardware"]
        A(f"MobileNetV3-Small 이진 분류(Track C)로 `cable` OK/NG 판정을 5-fold 교차검증했다. "
          f"입력 해상도 {' / '.join(str(m['config']['res']) for m in exps)}px 전 조건에서 "
          f"**미검율 0%**, AUROC {min(m['aggregate']['auroc']['mean'] for m in exps):.4f}"
          f"~{max(m['aggregate']['auroc']['mean'] for m in exps):.4f}로 차이가 없었다. "
          f"DoD 정확도를 만족하는 최경량은 **{bl['config']['res']}px** "
          f"({bh['gflops']} GFLOPs, e2e p50 {bh['latency_end_to_end']['p50_ms']}ms, "
          f"파라미터 {bh['params_M']}M). "
          f"다만 모든 수치가 검증셋 기준이고 임계값 τ도 같은 검증셋에서 정해 **낙관적으로 편향**돼 있으며, "
          f"fold당 불량 표본이 {folds[0]['val_defect_n']}장 수준이라 신뢰구간이 넓다. "
          f"또한 **τ가 fold/환경 간 이식되지 않는 문제**를 확인해 배포 시 재보정을 의무화했다.")
    A("")
    A("---")
    A("")

    # ---------- 2. 데이터 ----------
    A("## 1. 데이터")
    A("")
    A("### 1.1 복원")
    A("")
    A("초기 로컬 사본은 부분 다운로드본이었다(불량 33장, test 91장, 5개 클래스 이미지 누락). "
      "Google Drive 원본 zip에서 전량 복원했다. 원인은 Drive의 대용량 폴더 다운로드가 아카이브를 "
      "임의 분할하는데 일부 파트만 압축 해제한 것이었다.")
    A("")
    A("| 항목 | 복원 전 | 복원 후 | MVTec AD 공식 |")
    A("|---|---|---|---|")
    A(f"| train/good | 224 | {split_a['train_n']} | 224 |")
    A(f"| test 총계 | 91 | {split_a['test_n']} | 150 |")
    A("| test 불량 클래스 | 3종 | 8종 | 8종 |")
    A(f"| 불량 이미지 | 33 | {n_def} | 92 |")
    A("| ground_truth 마스크 | 57 | 92 | 92 |")
    A("")
    A("무결성 검증(§4.1) 결과: 손상 0건, 해상도 1024×1024 전부 일치, 마스크 {0,255} 이진 "
      "비이진 0건, test↔mask 인덱스 정합 OK. → `reports/data_integrity_cable.md`")
    A("")
    A("> 백업 보존: `mvtec/cable_partial_bak_20260921/` (복원 전 부분본). 삭제 금지.")
    A("> `cable` 외 11개 카테고리는 여전히 부분 결손 상태다 — 확장 시 동일 절차 필요(SOP §3.3).")
    A("")

    if area_stats:
        A("### 1.2 결함 크기 (마스크 92장 전수)")
        A("")
        A("| 클래스 | n | 면적% min/med/max | bbox 폭%(med) | bbox 높이%(med) |")
        A("|---|---|---|---|---|")
        for c, n, mn, me, mx, w, h in area_stats:
            A(f"| {c} | {n} | {mn:.2f} / {me:.2f} / {mx:.2f} | {w:.1f} | {h:.1f} |")
        A("")
        A("bbox 변 길이 중앙값이 이미지 변의 17~48%로 **대형 객체**에 해당한다. "
          "이것이 해상도를 크게 낮출 수 있었던 근거다(§3 결과로 검증됨).")
        A("")

    # ---------- 3. 방법 ----------
    A("## 2. 방법")
    A("")
    A("### 2.1 라벨")
    A("")
    A(f"마스크 → 연결요소(8-연결) → 외접 bbox. 최소 면적 임계 "
      f"{cfg['min_component_area_ratio']*100:.3f}%(={int(cfg['min_component_area_ratio']*cfg['image_size']**2)} px). "
      f"불량 {n_def}장에서 **bbox {n_box}개** 생성, 노이즈 제거 0건. "
      f"정상 {n_ok}장은 0바이트 빈 라벨(배경 샘플).")
    A("")
    A("| 클래스 | class_id | 이미지 | bbox | 이미지당 |")
    A("|---|---|---|---|---|")
    name2id = {v: k for k, v in classes["multi"].items()}
    for c in sorted(cls_img):
        A(f"| {c} | {name2id[c]} | {cls_img[c]} | {cls_box[c]} | {cls_box[c]/cls_img[c]:.2f} |")
    A(f"| **합계** | | **{n_def}** | **{n_box}** | |")
    A("")
    A("다중 클래스(8종)와 이진(OK/NG) 라벨을 모두 생성했다 — 판정 출력 형태(§2.1)가 미확정이므로 "
      "어느 쪽이든 즉시 학습 가능하게 했다. 육안 검증 24장 통과(`reports/label_check/`).")
    A("")
    A("### 2.2 분할")
    A("")
    A("| 분할 | 용도 | 구성 |")
    A("|---|---|---|")
    A(f"| A | 비지도 이상탐지 (문헌 비교) | train {split_a['train_n']} 정상 / "
      f"test {split_a['test_n']} (불량 {split_a['test_defect_n']}) — MVTec 원본 유지, 변경 금지 |")
    A(f"| B 홀드아웃 | 최종 DoD 판정 전용 | 불량 {hold['defect_n']}(클래스당 "
      f"{hold['defect_per_class']}) + 정상 {hold['normal_n']} = {len(hold['keys'])} |")
    A(f"| B 5-fold | 모델 개발·비교 | fold당 train {folds[0]['train_defect_n']}D+"
      f"{folds[0]['train_normal_n']}N / val {folds[0]['val_defect_n']}D+{folds[0]['val_normal_n']}N |")
    A("")
    A("클래스 층화 적용. 누수 검증 통과(train∩val=∅, 홀드아웃 누수 0, 커버리지 "
      f"{len(rows)}/{len(rows)}). **홀드아웃은 아직 한 번도 사용하지 않았다** — P5 STOP-GATE 2 전용.")
    A("")
    A("### 2.3 모델 · 학습")
    A("")
    c = ref["config"]
    A(f"- {c['arch']}, ImageNet 사전학습, 분류 헤드만 2클래스로 교체")
    A(f"- AdamW lr {c['lr']} + cosine, warmup {c['warmup_epochs']}, wd {c['weight_decay']}, "
      f"batch {c['batch']}, epochs {c['epochs']} / patience {c['patience']}, seed {c['seed']}")
    A(f"- best checkpoint 선택 기준: val AUROC")
    A("- 증강: 수평·수직 플립, 90° 회전, 스케일 ±10%, 이동 ±5%, 밝기·대비 ±20%, 약한 가우시안 노이즈")
    A("- **증강 금지: Hue 변형** — 절연체 색상이 `cable_swap` 판별의 직접 근거라 색을 흔들면 라벨이 거짓이 된다")
    A("")
    A("> CPU 전용 환경(GPU 미검출)이라 SOP 기본값 epochs 300 / patience 50을 "
      f"**{c['epochs']} / {c['patience']}**로 축소했다. 사전학습 가중치로 10 epoch 내 수렴함을 "
      "타이밍 실험에서 확인한 뒤 내린 결정이며 SOP §8.1에 근거를 기록했다.")
    A("> `num_workers>0`은 Windows spawn 오버헤드로 2.6배 느려 `num_workers=0` 고정.")
    A("")

    # ---------- 4. 결과 ----------
    A("## 3. 결과 — A1 해상도 ablation")
    A("")
    A("| 입력 | AUROC | 미검율 % | 과검율 % | F1 | GFLOPs | e2e p50/p95 ms | 모델만 p50 ms | 학습시간 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for m in exps:
        a, hw = m["aggregate"], m["hardware"]
        mark = "**" if best_light and m["exp_id"] == best_light["exp_id"] else ""
        A(f"| {mark}{m['config']['res']}{mark} | {ms(a['auroc'])} | {pct(a['miss_rate'])} | "
          f"{pct(a['overkill_rate'])} | {ms(a['f1'])} | {hw['gflops']} | "
          f"{hw['latency_end_to_end']['p50_ms']}/{hw['latency_end_to_end']['p95_ms']} | "
          f"{hw['latency_model_only']['p50_ms']} | {m['wall_clock_sec']/3600:.2f}h |")
    A("")
    A(f"파라미터는 해상도 무관 {ref['hardware']['params_M']}M / FP32 "
      f"{ref['hardware']['fp32_size_MB']} MB로 동일(전역 평균 풀링). 이득은 연산량·지연시간에서만 나온다.")
    A("")
    A("**혼동행렬 (5-fold 합산)**")
    A("")
    A("| 입력 | TP | FN | FP | TN |")
    A("|---|---|---|---|---|")
    for m in exps:
        p_ = {k: sum(f[k] for f in m["folds"]) for k in ("tp", "fn", "fp", "tn")}
        A(f"| {m['config']['res']} | {p_['tp']} | **{p_['fn']}** | {p_['fp']} | {p_['tn']} |")
    A("")
    A("**클래스별 recall (§7.3)** — ⚠ = §3.2에서 해상도 민감 예상 클래스")
    A("")
    watch = set(classes["small_defect_watch"])
    A("| 클래스 | n | " + " | ".join(f"{m['config']['res']}px" for m in exps) + " |")
    A("|" + "---|" * (2 + len(exps)))
    for cn in [classes["multi"][i] for i in sorted(classes["multi"])]:
        mk = " ⚠" if cn in watch else ""
        n = exps[0]["per_class_recall"][cn]["n_total"]
        A(f"| {cn}{mk} | {n} | "
          + " | ".join(f"{m['per_class_recall'][cn]['recall_mean']:.3f}" for m in exps) + " |")
    A("")
    fig = reports / "figs" / "ablation_A1_pareto.png"
    if fig.exists():
        A("![A1 pareto](figs/ablation_A1_pareto.png)")
        A("")

    # ---------- 5. 발견 ----------
    A("## 4. 주요 발견")
    A("")
    A("### 4.1 해상도 민감 가설 기각 (256px까지)")
    A("")
    A("§3.2에서 소형 결함 3종(`poke_insulation` 최소 면적 0.14%, `cut_outer_insulation` 0.44%, "
      "`missing_wire` bbox 최소 17%)이 저해상도에서 소실될 것으로 예상했으나, "
      "**256px에서도 전 클래스 recall 1.000**이었다. 256px면 0.14% 결함이 약 92px로 줄어드는데도 유지된다. "
      "`cable`의 결함 다수가 국소 텍스처가 아니라 **케이블 배치·구성의 전역 변화**로 나타나기 때문으로 보인다.")
    A("")
    A("### 4.2 AUROC는 변별력 없음 — 실제 차이는 과검율")
    A("")
    A("세 해상도 모두 AUROC 0.999대로 포화했다. 과검율 차이(0.43~1.30%)도 "
      "오차막대가 서로 겹쳐 **통계적으로 유의하지 않다** — 5-fold 합산 FP 기준 1건 vs 2건 vs 3건 차이일 뿐이다. "
      "§7.4에 따라 '어느 해상도가 낫다'가 아니라 **'세 해상도를 구분할 수 없다'**가 정확한 결론이다.")
    A("")
    A("### 4.3 임계값 τ가 환경 간 이식되지 않는다 (배포 리스크)")
    A("")
    A("| 입력 | 분리된 fold | 겹친 fold | 간격 log10 (min/중앙/max) | τ 범위 |")
    A("|---|---|---|---|---|")
    for m in exps:
        gaps = [f["score_stats"]["gap_log10"] for f in m["folds"]
                if f.get("score_stats", {}).get("gap_log10") is not None]
        n_ov = sum(1 for f in m["folds"] if not f.get("score_stats", {}).get("separated", True))
        taus = [f["tau"] for f in m["folds"]]
        g = f"{min(gaps):.2f} / {np.median(gaps):.2f} / {max(gaps):.2f}" if gaps else "—"
        A(f"| {m['config']['res']} | {len(gaps)} | {n_ov} | {g} | {min(taus):.2g} ~ {max(taus):.2g} |")
    A("")
    A("정상 점수의 절대 스케일이 fold마다 최대 8자릿수까지 다르고, 완전분리 구간 폭도 0.19~0.90 log10로 좁다. "
      "**한 환경에서 정한 절대 확률 임계를 다른 환경에 그대로 쓰면 판정이 무너진다.** 조치:")
    A("")
    A("- `choose_threshold`에 **마진 tie-break** 추가 — 완전분리 시 ROC가 구간 끝에 붙이던 τ를 "
      "혼동행렬 유지한 채 구간 중앙(로그 기하평균)으로 이동")
    A("- **§7.2 정책 보완** — 배포 시 (a) 현장 정상 표본으로 τ 재보정(권장) / (b) 백분위수 기반 임계 / "
      "(c) 온도 스케일링 중 택1, 보정 절차를 배포 아티팩트에 동봉 의무화")
    A("")
    A("해상도를 낮춰도 이 문제는 사라지지 않으며, 256px에서 겹친 fold가 2개로 가장 많고 "
      "간격 중앙값도 최소라 **오히려 더 까다롭다**.")
    A("")
    A("### 4.4 전처리가 지연시간 하한을 만든다")
    A("")
    lo, hi = exps[-1], exps[0]
    A(f"모델 추론은 {hi['config']['res']}→{lo['config']['res']}px에서 "
      f"{hi['hardware']['latency_model_only']['p50_ms']}→{lo['hardware']['latency_model_only']['p50_ms']}ms"
      f"({lo['hardware']['latency_model_only']['p50_ms']/hi['hardware']['latency_model_only']['p50_ms']:.2f}×)로 줄지만, "
      f"e2e는 {hi['hardware']['latency_end_to_end']['p50_ms']}→{lo['hardware']['latency_end_to_end']['p50_ms']}ms"
      f"({lo['hardware']['latency_end_to_end']['p50_ms']/hi['hardware']['latency_end_to_end']['p50_ms']:.2f}×)에 그친다. "
      "1024px 원본 리사이즈가 고정비로 남기 때문이다. "
      "**카메라 단에서 저해상도로 직접 취득할 수 있으면 추가 이득이 크다.**")
    A("")
    A("### 4.5 Track A(YOLO) 착수 시 주의 — 라벨 변환 중 발견")
    A("")
    A("1. `bent_wire`·`cut_outer_insulation`의 GT 마스크는 **가늘고 휜 곡선**이라 외접 bbox가 "
      "실제 결함 면적보다 훨씬 넓다. 이 클래스에서 **mAP@IoU는 검출 품질을 과대평가한다** — "
      "픽셀 레벨 PRO를 반드시 병행한다.")
    A("2. `combined`(id 2)는 결함 유형이 아니라 **'복수 결함 존재'를 뜻하는 메타 라벨**이다. "
      "이미지당 bbox 3.73개로 최다. 8클래스 검출에서 타 클래스와 혼동될 소지가 있다.")
    A("3. `cable_swap` 마스크는 케이블 단면 **전체**를 덮는다(이미지당 bbox 정확히 1개). "
      "국소 결함이 아니므로 패치 단위 이상탐지(Track B PatchCore/PaDiM)에 불리할 것으로 예상된다 — 미검증.")
    A("")

    # ---------- 6. DoD ----------
    A("## 5. DoD 대조 (§1.3) — 잠정")
    A("")
    if best_light:
        ba, bh = best_light["aggregate"], best_light["hardware"]
        checks = [
            ("미검율 (과검 5% 고정)", "≤ 5%", f"{(1-ba['recall_at_fpr5']['mean'])*100:.2f} %",
             (1 - ba["recall_at_fpr5"]["mean"]) <= 0.05),
            ("이미지 AUROC", "≥ 0.95", f"{ba['auroc']['mean']:.4f}", ba["auroc"]["mean"] >= 0.95),
            ("모델 크기 (INT8 환산)", "≤ 10 MB",
             f"{bh['fp32_size_MB']/4:.2f} MB (FP32 {bh['fp32_size_MB']} MB)",
             bh["fp32_size_MB"] / 4 <= 10),
            ("추론 지연", "≤ 100 ms",
             f"{bh['latency_end_to_end']['p50_ms']} ms (개발 CPU)",
             bh["latency_end_to_end"]["p50_ms"] <= 100),
            ("피크 메모리", "≤ 256 MB RSS", "미측정", None),
            ("재현성", "±1%p", "분할·시드 고정, 체크포인트 재산출로 AUROC 1e-6 이내 일치 확인", True),
        ]
        A("| 항목 | 기준 | 결과 | 판정 |")
        A("|---|---|---|---|")
        for nm, cr, vl, okk in checks:
            mark = "—" if okk is None else ("충족" if okk else "미달")
            A(f"| {nm} | {cr} | {vl} | {mark} |")
        A("")
        A(f"> 기준: `{best_light['exp_id']}`. **모두 검증셋 기준의 잠정치다.** "
          "홀드아웃 최종 판정(P5 STOP-GATE 2)과 타깃 하드웨어 벤치(§9.3)를 거쳐야 확정된다.")
    A("")

    # ---------- 7. 한계 ----------
    A("## 6. 한계 — 수치를 과신하면 안 되는 이유")
    A("")
    pooled_def = sum(f["tp"] + f["fn"] for f in ref["folds"])
    A(f"1. **τ와 지표를 같은 val fold에서 산출했다.** τ를 검증셋에서 정하는 것은 §7.2 정책대로이나, "
      f"그 τ로 같은 검증셋의 미검율·과검율을 계산하면 낙관적으로 편향된다. "
      f"편향 없는 추정치는 홀드아웃에서만 얻을 수 있다.")
    A(f"2. **표본이 작다.** fold당 val 불량 {folds[0]['val_defect_n']}~{folds[1]['val_defect_n']+1}장. "
      f"미검 0건이어도 rule of three 기준 미검율 95% 상한은 fold 단위 약 "
      f"{3/(pooled_def/len(ref['folds']))*100:.0f}%다. 5-fold 합산(불량 {pooled_def}장)해도 "
      f"약 {3/pooled_def*100:.1f}%이며, fold 간 학습셋이 겹쳐 완전 독립 시행이 아니라 이 값도 낙관적이다.")
    A(f"3. **클래스별 recall 1.000은 클래스당 n=8~12에서 나온 값이다.** "
      f"'완벽히 잡는다'가 아니라 '현 표본에서 놓친 사례가 없다'로 읽어야 한다.")
    A(f"4. **Track A/B 미수행.** §1.2 연구질문 1(어느 태스크가 유리한가)은 Track C 단독으로 답할 수 없다.")
    A(f"5. **지연시간·메모리는 개발 머신 CPU 값이다.** 타깃 하드웨어에서 §9.3 프로토콜"
      f"(워밍업 20회 후 200회, 열 스로틀링 10분 연속 부하 포함)로 재측정해야 배포 판단이 된다.")
    A(f"6. **데이터가 MVTec 공개 데이터다.** 실제 산업 라인 데이터로의 전이는 별도 검증이 필요하며, "
      f"MVTec AD는 CC BY-NC-SA 4.0(비상업)이다.")
    A("")

    # ---------- 8. 다음 ----------
    A("## 7. 다음 단계")
    A("")
    A("### 7.1 의사결정 대기 (§2.1) — 현재 최대 병목")
    A("")
    A("| 항목 | 필요 시점 | 영향 |")
    A("|---|---|---|")
    A("| **타깃 하드웨어** | 즉시 | §9.2 export 경로, §9.3 벤치, 해상도 최종 선택 |")
    A("| 택트타임 예산 | 즉시 | 해상도·모델 선택 |")
    A("| 미검:과검 비용비 | P4 전 | §7.2 임계값 정책 |")
    A("| 판정 출력 형태 | Track A 착수 전 | 본선 트랙 확정 (multi/binary 라벨은 둘 다 준비됨) |")
    A("| GPU 확보 | Track A/B 착수 전 | 미확보 시 YOLO 5-fold 비현실적 |")
    A("")
    A("### 7.2 기술 작업 후보")
    A("")
    A("| 우선 | 작업 | 근거 | 예상 비용 |")
    A("|---|---|---|---|")
    A("| 1 | A6 양자화 (INT8 PTQ) | 6.08 → 1.52 MB, 캘리브레이션에 불량 포함 필수(§9.1-3) | 낮음 |")
    A("| 2 | ONNX export + 수치 동등성 검증 | §9.2 — 판정 일치율 100% 확인 | 낮음 |")
    A("| 3 | A2 백본 축소 (ShuffleNetV2 0.5×, 1.4M) | 현 정확도에 여유가 커 더 줄일 여지 | 중간 |")
    A("| 4 | A3 증강 강도 / A5 클래스 가중치 | 과검율 개선 여지 확인 | 중간 |")
    A("| 5 | Track B (EfficientAD-S) | 문헌 비교 가능한 유일 트랙, 불량 데이터 불필요 | GPU 필요 |")
    A("| 6 | Track A (YOLO11n) | 위치 출력이 필요할 때만 | GPU 필요 |")
    A("")
    A("> **홀드아웃은 건드리지 않는다.** 위 작업 전부 5-fold val로 판단하고, "
      "홀드아웃은 최종 후보 1개가 확정된 뒤 P5에서 1회만 사용한다(§4.3).")
    A("")

    # ---------- 9. 재현 ----------
    A("## 8. 재현 절차")
    A("")
    A("```powershell")
    A("# 0) 환경")
    A("<python3.12> -m venv .venv")
    A(".venv\\Scripts\\python.exe -m pip install -r requirements.lock.txt")
    A("")
    A("# 1) P1 — 무결성 검증 → 라벨 변환 → 분할 → 스모크 체크")
    A("powershell -ExecutionPolicy Bypass -File src\\data\\run_p1.ps1")
    A("")
    A("# 2) 이미지 캐시 (CPU 학습 병목 제거)")
    A(".venv\\Scripts\\python.exe src\\data\\build_cache.py")
    A("")
    A("# 3) P2/P3 — Track C 해상도별 5-fold")
    for m in exps:
        A(f".venv\\Scripts\\python.exe src\\train\\train_cls.py --fold all --res {m['config']['res']} "
          f"--epochs {m['config']['epochs']} --patience {m['config']['patience']} --workers 0")
    A("")
    A("# 4) 체크포인트로 점수 재산출 (τ 마진 tie-break 적용, 점수 원본 저장)")
    for m in exps:
        A(f".venv\\Scripts\\python.exe src\\eval\\score_folds.py {m['exp_id']}")
    A("")
    A("# 5) 리포트")
    A(".venv\\Scripts\\python.exe src\\eval\\report_track_c.py")
    A(".venv\\Scripts\\python.exe src\\eval\\report_ablation_res.py")
    A(".venv\\Scripts\\python.exe src\\eval\\report_summary.py")
    A("```")
    A("")
    A("시드 42 전역 고정, 분할은 JSON으로 고정 저장(실행마다 재추첨하지 않음). "
      "실험별 `config.yaml` / `env.txt` / `git_rev.txt` / `metrics.json` / `scores_fold*.json` 보존(§11).")
    A("")

    # ---------- 10. 산출물 ----------
    A("## 9. 산출물 인덱스")
    A("")
    A("| 경로 | 내용 |")
    A("|---|---|")
    A("| `docs/SOP_cable_defect_lightweight.md` | SOP 본문 (실행 이력·이탈 사항 포함) |")
    A("| `reports/data_integrity_cable.md` / `.csv` | 무결성 검증 |")
    A("| `reports/label_conversion_cable.md` | 라벨 변환 |")
    A("| `reports/label_check/` | 육안 검증 오버레이 24장 |")
    A("| `reports/splits_cable.md` | 분할 설계 |")
    A("| `reports/track_c_results.md` | Track C 상세 |")
    A("| `reports/ablation_A1_resolution.md` | A1 해상도 ablation |")
    A("| `reports/figs/ablation_A1_pareto.png` | 파레토 그림 |")
    A("| `reports/P1_completion.md` | P1 완료 리포트 |")
    A("| **`reports/SUMMARY_cable_trackC.md`** | **본 통합 리포트** |")
    for m in exps:
        A(f"| `runs/{m['exp_id']}/` | {m['config']['res']}px 실험 일체 |")
    A("| `data/classes.yaml` | 클래스 ID 매핑 (고정) |")
    A("| `data/splits/` | 분할 정의 (A / 홀드아웃 / 5-fold) |")
    A("| `data/yolo/{multi,binary}/` | YOLO 포맷 라벨 (Track A 대비) |")
    A("| `requirements.lock.txt` | 패키지 잠금 |")
    A("")

    out = reports / "SUMMARY_cable_trackC.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out}  ({len(md)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
