"""Track C 결과 리포트 생성 — SOP-MVTEC-CABLE-001 §7.1, §7.3, §7.4

정확도 지표와 하드웨어 지표를 항상 같은 표에 기재한다(§7.1 보고 규칙).
5-fold 평균 ± 표준편차로만 보고한다(§7.4).

사용: python src/eval/report_track_c.py [EXP_ID ...]
      인자 없으면 runs/ 의 EXP_C_* 전부.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg  # noqa: E402


def fmt(a, pct=False, nd=4):
    if a is None:
        return "—"
    m, s = a["mean"], a["std"]
    if pct:
        return f"{m*100:.2f} ± {s*100:.2f}"
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main() -> int:
    cfg = load_cfg()
    runs = Path(cfg["runs_dir"])
    exp_ids = sys.argv[1:] or sorted(p.name for p in runs.glob("EXP_C_*") if (p / "metrics.json").exists())
    if not exp_ids:
        print("EXP_C_* 실험 없음")
        return 1

    results = []
    for e in exp_ids:
        mp = runs / e / "metrics.json"
        if not mp.exists():
            print(f"건너뜀 (metrics.json 없음): {e}")
            continue
        results.append(json.loads(mp.read_text(encoding="utf-8")))

    md = [
        "# Track C 결과 리포트 — 지도학습 이진 분류 (OK/NG)",
        "",
        "- SOP: SOP-MVTEC-CABLE-001 §6 Track C, §7, §8.1",
        f"- 생성일: {date.today().isoformat()}",
        f"- 실험 수: {len(results)}",
        "",
        "> §7.1 보고 규칙: 정확도 지표와 하드웨어 지표를 같은 표에 기재한다.",
        "> §7.4: 5-fold 평균 ± 표준편차. 단일 fold로 우열을 주장하지 않는다.",
        "",
        "## 1. 종합 (정확도 + 하드웨어)",
        "",
        "| 실험 | arch | 입력 | AUROC | 미검율 % | 과검율 % | Recall@FPR5 | params(M) | GFLOPs | e2e p50/p95 ms | 모델 MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        a, hw, c = r["aggregate"], r["hardware"], r["config"]
        lat = hw["latency_end_to_end"]
        md.append(
            f"| `{r['exp_id']}` | {c['arch']} | {c['res']} | {fmt(a['auroc'])} | "
            f"{fmt(a['miss_rate'], pct=True)} | {fmt(a['overkill_rate'], pct=True)} | "
            f"{fmt(a['recall_at_fpr5'])} | {hw['params_M']} | {hw['gflops']} | "
            f"{lat['p50_ms']}/{lat['p95_ms']} | {hw['fp32_size_MB']} |"
        )

    md += [
        "",
        f"> 지연시간은 개발 머신 CPU 측정치다. **타깃 하드웨어 수치가 아니다** — §9.3 벤치마크로 별도 측정한다.",
        f"> 측정 범위: 1024px 원본 리사이즈 + 정규화 + 추론 (전처리 포함).",
        "",
        "## 2. 실험별 상세",
        "",
    ]

    for r in results:
        a, c, hw = r["aggregate"], r["config"], r["hardware"]
        md += [
            f"### `{r['exp_id']}`",
            "",
            f"- arch: {c['arch']} / 입력 {c['res']}px / FP32 / 사전학습 {c['pretrained']}",
            f"- epochs {c['epochs']} (patience {c['patience']}), batch {c['batch']}, lr {c['lr']}, "
            f"aug `{c['aug']}`, class_weight {c['class_weight']}",
            f"- seed {c['seed']}, {c['n_folds']}-fold, 총 소요 {r['wall_clock_sec']:.0f}s",
            "",
            "**이미지 레벨 지표 (5-fold 평균 ± 표준편차)**",
            "",
            "| 지표 | 값 |",
            "|---|---|",
            f"| AUROC | {fmt(a['auroc'])} |",
            f"| AUPRC | {fmt(a['auprc'])} |",
            f"| **미검율 (escape)** | **{fmt(a['miss_rate'], pct=True)} %** |",
            f"| 과검율 (over-kill) | {fmt(a['overkill_rate'], pct=True)} % |",
            f"| Recall @ FPR=5% | {fmt(a['recall_at_fpr5'])} |",
            f"| F1 | {fmt(a['f1'])} |",
            f"| Balanced Acc | {fmt(a['balanced_acc'])} |",
            f"| 임계값 τ | {fmt(a['tau'])} |",
            "",
            "**fold별**",
            "",
            "| fold | best epoch | AUROC | 미검율 % | 과검율 % | TP/FN/FP/TN | τ |",
            "|---|---|---|---|---|---|---|",
        ]
        for f_ in r["folds"]:
            md.append(
                f"| {f_['fold']} | {f_['best_epoch']} | {f_['auroc']:.4f} | "
                f"{f_['miss_rate']*100:.2f} | {f_['overkill_rate']*100:.2f} | "
                f"{f_['tp']}/{f_['fn']}/{f_['fp']}/{f_['tn']} | {f_['tau']:.4f} |"
            )

        md += [
            "",
            "**클래스별 recall 분해 (§7.3 — 필수 항목)**",
            "",
            "| 클래스 | 총 n | recall 평균 ± 표준편차 | fold별 |",
            "|---|---|---|---|",
        ]
        weak = []
        overall = a["recall"]["mean"] if a["recall"] else 0.0
        for cls, v in r["per_class_recall"].items():
            if v["recall_mean"] is None:
                continue
            per = ", ".join(f"{x:.2f}" for x in v["per_fold"])
            mark = ""
            if v["recall_mean"] < overall - 0.15:
                mark = " ⚠"
                weak.append((cls, v["recall_mean"]))
            md.append(f"| {cls}{mark} | {v['n_total']} | "
                      f"{v['recall_mean']:.3f} ± {v['recall_std']:.3f} | {per} |")

        if any("score_stats" in f_ for f_ in r["folds"]):
            md += [
                "",
                "**점수 분포 · 임계값 이식성 (배포 관점)**",
                "",
                "| fold | OK 중앙 | OK 최대 | NG 최소 | NG 중앙 | 분리 간격(log10) | τ |",
                "|---|---|---|---|---|---|---|",
            ]
            for f_ in r["folds"]:
                st = f_.get("score_stats")
                if not st:
                    continue
                g = st["gap_log10"]
                md.append(
                    f"| {f_['fold']} | {st['OK_median']:.3g} | {st['OK_max']:.3g} | "
                    f"{st['NG_min']:.3g} | {st['NG_median']:.3g} | "
                    f"{(f'{g:.2f}' if g is not None else '**겹침**')} | {f_['tau']:.3g} |"
                )

        md += ["", "**해석**", ""]
        if weak:
            md.append(f"- ⚠ 전체 recall({overall:.3f}) 대비 현저히 낮은 클래스: "
                      + ", ".join(f"`{c}` ({v:.3f})" for c, v in weak))
            md.append("  → §7.3에 따라 별도 대응 검토: 해상도 상향 / 클래스 가중치 / 별도 모델")
        else:
            md.append(f"- 전체 recall({overall:.3f}) 대비 현저히 낮은 클래스 없음")
        md.append(f"- fold당 val 불량이 15~16장이므로 클래스별 n은 1~3장이다. "
                  f"클래스별 recall의 신뢰구간은 매우 넓다 — 경향 파악용으로만 사용한다.")
        md += ["", "---", ""]

    md += [
        "## 3. DoD 대조 (§1.3)",
        "",
        "| 항목 | 기준 | 최고 실험 결과 | 판정 |",
        "|---|---|---|---|",
    ]
    best = max(results, key=lambda r: r["aggregate"]["auroc"]["mean"])
    ba, bhw = best["aggregate"], best["hardware"]
    checks = [
        ("미검율 (과검 5% 고정)", "≤ 5%", f"{(1-ba['recall_at_fpr5']['mean'])*100:.2f} %",
         (1 - ba["recall_at_fpr5"]["mean"]) <= 0.05),
        ("이미지 AUROC", "≥ 0.95", f"{ba['auroc']['mean']:.4f}", ba["auroc"]["mean"] >= 0.95),
        ("모델 크기 (INT8 환산)", "≤ 10 MB", f"{bhw['fp32_size_MB']/4:.2f} MB (FP32 {bhw['fp32_size_MB']} MB)",
         bhw["fp32_size_MB"] / 4 <= 10),
        ("추론 지연", "≤ 100 ms", f"{bhw['latency_end_to_end']['p50_ms']} ms (개발 CPU)",
         bhw["latency_end_to_end"]["p50_ms"] <= 100),
    ]
    for name, crit, val, ok in checks:
        md.append(f"| {name} | {crit} | {val} | {'✅' if ok else '❌'} |")
    md += [
        "",
        f"> 기준 실험: `{best['exp_id']}`. **검증셋(val fold) 기준이며 홀드아웃 최종 판정이 아니다.** "
        "홀드아웃은 P5 STOP-GATE 2에서 1회만 사용한다(§4.3).",
        "> 지연시간·메모리는 개발 머신 CPU 값이므로 타깃 하드웨어 확정 후 §9.3으로 재측정해야 최종 판정이 된다.",
        "",
        "## 4. 통계적 한계 · 결과 해석 시 주의",
        "",
    ]

    pooled = {k: sum(f_[k] for f_ in best["folds"]) for k in ("tp", "fn", "fp", "tn")}
    n_def = pooled["tp"] + pooled["fn"]
    n_norm = pooled["fp"] + pooled["tn"]
    rule_of_three_fold = 3 / (n_def / len(best["folds"])) * 100
    rule_of_three_pooled = 3 / n_def * 100
    md += [
        f"1. **τ와 지표를 같은 val fold에서 산출했다.** §7.2 정책상 τ는 검증셋에서 정하는 것이 맞으나, "
        f"그 τ로 같은 검증셋의 미검율·과검율을 계산하면 **낙관적으로 편향**된다. 편향 없는 추정치는 "
        f"홀드아웃(P5 STOP-GATE 2)에서만 얻을 수 있다.",
        f"2. **표본이 작다.** fold당 val 불량은 15~16장이다. 미검 0건이어도 rule of three 기준 "
        f"미검율 95% 상한은 fold 단위로 **약 {rule_of_three_fold:.0f}%**다. "
        f"5-fold를 합치면 불량 {n_def}장 · 미검 {pooled['fn']}건으로 상한이 **약 {rule_of_three_pooled:.1f}%**까지 "
        f"내려가지만, fold 간 학습셋이 겹쳐 완전히 독립적인 시행이 아니므로 이 값도 낙관적이다.",
        f"3. **클래스별 recall 1.000은 클래스당 n=8~12에서 나온 값이다.** 클래스 단위 신뢰구간은 매우 넓다. "
        f"'모든 클래스를 완벽히 잡는다'가 아니라 '현 표본에서 놓친 사례가 없다'로 읽어야 한다.",
        f"4. **임계값 τ가 fold 간 이식되지 않는다.** 점수 분포의 절대 스케일이 fold마다 최대 8자릿수까지 "
        f"차이나고(OK 중앙값 2e-11 ~ 2e-4), 완전분리 구간의 폭도 0.25~0.81 log10로 좁다. "
        f"한 fold에서 정한 τ를 다른 fold/배포 환경에 그대로 쓰면 판정이 무너진다. "
        f"→ **배포 시 현장 정상 표본으로 τ를 재보정하거나, 절대 확률 대신 백분위수 기반 임계를 쓰고, "
        f"온도 스케일링 등 확률 보정을 검토한다.** (§7.2 정책 보완 필요)",
        f"5. **`combined` 클래스 주의.** 결함 유형이 아니라 '복수 결함 존재' 메타 라벨이다(§4.2 관찰). "
        f"이진 판정에서는 문제되지 않지만 다중 클래스로 확장할 때 혼동 원인이 된다.",
    ]

    out = Path(cfg["reports_dir"]) / "track_c_results.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
