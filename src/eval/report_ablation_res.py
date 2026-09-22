"""A1 해상도 ablation 리포트 — SOP-MVTEC-CABLE-001 §8.2 A1, §9.1-1

입력 해상도만 바꾼 Track C 실험들을 묶어 정확도-지연시간 파레토를 정리한다.
한 번에 하나의 변수만 바뀐 실험끼리만 비교한다(§8.2).

산출물: reports/ablation_A1_resolution.md
        reports/figs/ablation_A1_pareto.png

사용: python src/eval/report_ablation_res.py [EXP_ID ...]
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg, load_classes  # noqa: E402

# 비교 가능 조건: 해상도 외 모든 설정이 같아야 한다 (§8.2 "한 번에 하나의 변수만")
COMPARE_KEYS = ["arch", "epochs", "patience", "batch", "lr", "weight_decay",
                "warmup_epochs", "aug", "pretrained", "class_weight", "seed", "n_folds"]


def ms(a):
    return f"{a['mean']:.4f} ± {a['std']:.4f}"


def pct(a):
    return f"{a['mean']*100:.2f} ± {a['std']*100:.2f}"


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    runs = Path(cfg["runs_dir"])
    ids = sys.argv[1:] or sorted(p.name for p in runs.glob("EXP_C_*")
                                 if (p / "metrics.json").exists())
    res_list = []
    for e in ids:
        m = json.loads((runs / e / "metrics.json").read_text(encoding="utf-8"))
        res_list.append(m)
    res_list.sort(key=lambda m: -m["config"]["res"])
    if len(res_list) < 2:
        print("비교할 실험이 2개 미만")
        return 1

    # --- 비교 가능성 검증 (§8.2) ---
    base = res_list[0]["config"]
    mismatch = []
    for m in res_list[1:]:
        for k in COMPARE_KEYS:
            if m["config"].get(k) != base.get(k):
                mismatch.append(f"{m['exp_id']}: {k} = {m['config'].get(k)} (기준 {base.get(k)})")

    md = [
        "# A1 Ablation — 입력 해상도 (Track C)",
        "",
        "- SOP: SOP-MVTEC-CABLE-001 §8.2 A1, §9.1-1 (경량화 1축)",
        f"- 생성일: {date.today().isoformat()}",
        f"- 고정 조건: arch `{base['arch']}` / epochs {base['epochs']}(patience {base['patience']}) / "
        f"batch {base['batch']} / lr {base['lr']} / aug `{base['aug']}` / "
        f"pretrained {base['pretrained']} / seed {base['seed']} / {base['n_folds']}-fold (동일 분할)",
        "",
    ]
    if mismatch:
        md += ["> ⚠ **비교 조건 불일치 — 아래 항목이 해상도 외에 달라졌다. 해석 시 교란 요인이다.**", ""]
        md += [f"> - {x}" for x in mismatch] + [""]
    else:
        md += ["> 해상도 외 모든 설정이 동일하다. 단일 변수 비교 조건 충족(§8.2).", ""]

    # --- 종합 표 ---
    md += [
        "## 1. 정확도 · 하드웨어 종합 (§7.1 — 같은 표에 기재)",
        "",
        "| 입력 | AUROC | 미검율 % | 과검율 % | Recall@FPR5 | F1 | GFLOPs | e2e p50 ms | e2e p95 ms | 모델만 p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in res_list:
        a, hw = m["aggregate"], m["hardware"]
        md.append(
            f"| **{m['config']['res']}** | {ms(a['auroc'])} | {pct(a['miss_rate'])} | "
            f"{pct(a['overkill_rate'])} | {ms(a['recall_at_fpr5'])} | {ms(a['f1'])} | "
            f"{hw['gflops']} | {hw['latency_end_to_end']['p50_ms']} | "
            f"{hw['latency_end_to_end']['p95_ms']} | {hw['latency_model_only']['p50_ms']} |"
        )

    ref = res_list[0]
    md += [
        "",
        f"파라미터 수는 해상도와 무관하게 {ref['hardware']['params_M']}M / FP32 {ref['hardware']['fp32_size_MB']} MB로 동일하다 "
        "(전역 평균 풀링 구조). 해상도 축소의 이득은 **연산량과 지연시간**에서 나온다.",
        "",
        "### 기준(최고 해상도) 대비 상대치",
        "",
        "| 입력 | GFLOPs 비 | e2e p50 비 | AUROC 차 | 미검율 차 %p |",
        "|---|---|---|---|---|",
    ]
    r0 = ref["hardware"]
    a0 = ref["aggregate"]
    for m in res_list:
        hw, a = m["hardware"], m["aggregate"]
        md.append(
            f"| {m['config']['res']} | {hw['gflops']/r0['gflops']:.3f}× | "
            f"{hw['latency_end_to_end']['p50_ms']/r0['latency_end_to_end']['p50_ms']:.3f}× | "
            f"{a['auroc']['mean']-a0['auroc']['mean']:+.4f} | "
            f"{(a['miss_rate']['mean']-a0['miss_rate']['mean'])*100:+.2f} |"
        )

    # --- 클래스별 recall (소형 결함 감시) ---
    watch = set(classes["small_defect_watch"])
    cls_names = [classes["multi"][i] for i in sorted(classes["multi"])]
    md += [
        "",
        "## 2. 클래스별 recall (§7.3) — 해상도 민감도",
        "",
        "⚠ = §3.2에서 해상도 축소에 민감할 것으로 지목한 클래스",
        "",
        "| 클래스 | n | " + " | ".join(f"{m['config']['res']}px" for m in res_list) + " |",
        "|" + "---|" * (2 + len(res_list)),
    ]
    for c in cls_names:
        mark = " ⚠" if c in watch else ""
        n = res_list[0]["per_class_recall"][c]["n_total"]
        cells = []
        for m in res_list:
            v = m["per_class_recall"][c]
            cells.append(f"{v['recall_mean']:.3f} ± {v['recall_std']:.3f}")
        md.append(f"| {c}{mark} | {n} | " + " | ".join(cells) + " |")

    # --- 점수 분리폭 ---
    if all(any("score_stats" in f for f in m["folds"]) for m in res_list):
        md += [
            "",
            "## 3. 점수 분리폭 · 임계값 안정성",
            "",
            "완전분리된 fold의 간격(log10)과 겹친 fold 수. 간격이 좁을수록 배포 시 임계 보정이 까다롭다(§7.2 보완).",
            "",
            "| 입력 | 분리된 fold | 겹친 fold | 간격 log10 (min / 중앙 / max) | τ 범위 |",
            "|---|---|---|---|---|",
        ]
        for m in res_list:
            gaps = [f["score_stats"]["gap_log10"] for f in m["folds"]
                    if f["score_stats"]["gap_log10"] is not None]
            n_ov = sum(1 for f in m["folds"] if not f["score_stats"]["separated"])
            taus = [f["tau"] for f in m["folds"]]
            g = (f"{min(gaps):.2f} / {np.median(gaps):.2f} / {max(gaps):.2f}" if gaps else "—")
            md.append(f"| {m['config']['res']} | {len(gaps)} | {n_ov} | {g} | "
                      f"{min(taus):.2g} ~ {max(taus):.2g} |")

    # --- 파레토 그림 ---
    fig_rel = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        figs = Path(cfg["reports_dir"]) / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        x = [m["hardware"]["latency_end_to_end"]["p50_ms"] for m in res_list]
        gf = [m["hardware"]["gflops"] for m in res_list]
        y = [m["aggregate"]["auroc"]["mean"] for m in res_list]
        yerr = [m["aggregate"]["auroc"]["std"] for m in res_list]
        ok = [m["aggregate"]["overkill_rate"]["mean"] * 100 for m in res_list]
        okerr = [m["aggregate"]["overkill_rate"]["std"] * 100 for m in res_list]
        miss = [m["aggregate"]["miss_rate"]["mean"] * 100 for m in res_list]
        lbl = [f"{m['config']['res']}px" for m in res_list]

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=140)

        ax = axes[0]
        ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, lw=1.6, ms=7, color="#2563eb")
        for xi, yi, li, g in zip(x, y, lbl, gf):
            ax.annotate(f"{li}\n{g:.2f} GFLOPs", (xi, yi), textcoords="offset points",
                        xytext=(6, -26), fontsize=8)
        ax.axhline(0.95, ls="--", lw=1, color="#9ca3af")
        ax.annotate("DoD AUROC 0.95", (min(x), 0.9515), fontsize=8, color="#6b7280")
        ax.set_ylim(0.945, 1.003)
        ax.set_xlabel("end-to-end latency p50 (ms, dev CPU)")
        ax.set_ylabel("image-level AUROC (5-fold mean ± std)")
        ax.set_title("AUROC — 세 해상도 모두 포화, 변별력 없음")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.errorbar(x, ok, yerr=okerr, fmt="s-", capsize=4, lw=1.6, ms=7,
                    color="#ea580c", label="과검율 (over-kill)")
        ax.plot(x, miss, "^--", lw=1.4, ms=7, color="#16a34a", label="미검율 (escape)")
        for xi, yi, li in zip(x, ok, lbl):
            ax.annotate(li, (xi, yi), textcoords="offset points", xytext=(7, 7), fontsize=9)
        ax.axhline(5.0, ls="--", lw=1, color="#9ca3af")
        ax.annotate("DoD 미검율 5%", (min(x), 5.15), fontsize=8, color="#6b7280")
        ax.set_xlabel("end-to-end latency p50 (ms, dev CPU)")
        ax.set_ylabel("rate (%, 5-fold mean ± std)")
        ax.set_title("실제 변별 지점은 과검율 (미검율은 전 조건 0%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.suptitle("Track C (MobileNetV3-Small) — 입력 해상도 A1 ablation", fontsize=11)
        fig.tight_layout()
        out_png = figs / "ablation_A1_pareto.png"
        fig.savefig(out_png)
        plt.close(fig)
        fig_rel = out_png.relative_to(Path(cfg["reports_dir"]))
        md += ["", "## 4. 정확도-지연시간 파레토", "", f"![A1 pareto]({fig_rel.as_posix()})", ""]
    except Exception as e:
        md += ["", f"> 파레토 그림 생성 실패: {e}", ""]

    # --- 결론 ---
    dod_ok = [m for m in res_list
              if m["aggregate"]["auroc"]["mean"] >= 0.95
              and (1 - m["aggregate"]["recall_at_fpr5"]["mean"]) <= 0.05]
    pick = min(dod_ok, key=lambda m: m["hardware"]["latency_end_to_end"]["p50_ms"]) if dod_ok else None

    md += ["## 5. 결론", ""]
    if pick:
        pa, ph = pick["aggregate"], pick["hardware"]
        md += [
            f"- **DoD(§1.3) 정확도 기준을 만족하는 가장 가벼운 설정: {pick['config']['res']}px** "
            f"— AUROC {pa['auroc']['mean']:.4f}, 미검율 {pa['miss_rate']['mean']*100:.2f}%, "
            f"{ph['gflops']} GFLOPs, e2e p50 {ph['latency_end_to_end']['p50_ms']}ms",
            f"- 최고 해상도({ref['config']['res']}px) 대비 연산량 "
            f"{ph['gflops']/r0['gflops']:.2f}×, 지연 {ph['latency_end_to_end']['p50_ms']/r0['latency_end_to_end']['p50_ms']:.2f}×",
        ]
    else:
        md += ["- DoD 정확도 기준을 만족하는 해상도가 없다. 상위 해상도 또는 다른 백본 검토 필요."]

    md += [
        "",
        "### 해석 시 주의 (Track C 리포트 §4와 동일하게 적용)",
        "",
        "1. 모든 수치는 **val fold 기준**이며 τ도 같은 val에서 정했다 → 낙관적 편향. 편향 없는 값은 홀드아웃(P5)에서만 얻는다.",
        "2. fold당 val 불량 15~16장. 미검 0건이어도 rule of three 기준 fold 단위 95% 상한은 약 20%다.",
        "3. 클래스별 recall은 클래스당 n=8~12에서 나온 값이다. 해상도 간 미세한 차이를 유의미한 차이로 읽지 않는다(§7.4).",
        "4. 지연시간은 **개발 머신 CPU** 값이다. 타깃 하드웨어 확정 후 §9.3 프로토콜(열 스로틀링 포함)로 재측정해야 배포 판단이 된다.",
        "5. 해상도를 낮춰 정확도가 유지되더라도, **놓치는 결함의 종류**가 바뀔 수 있다. §2 클래스별 표에서 "
        "소형 결함 감시 클래스(⚠)의 recall을 개별로 확인한다.",
    ]

    out = Path(cfg["reports_dir"]) / "ablation_A1_resolution.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out}")
    if fig_rel:
        print(f"-> {Path(cfg['reports_dir']) / fig_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
