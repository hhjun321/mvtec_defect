"""Stage 1 전후 비교 리포트 — 처방이 실제로 강건성을 올렸는가.

기준선(전처리 없음 / 증강 medium) 대비 Stage 1 변형들을 비교한다.
  - 무섭동 정확도가 희생됐는가
  - 섭동 축별 강건성이 올랐는가
  - 추가 지연시간이 DoD를 깨는가

산출물: reports/stage1_comparison.md, reports/figs/stage1_*.png
사용:   python src/eval/report_stage1.py            (256px 실험 전부)
        python src/eval/report_stage1.py EXP_A EXP_B ...
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg, load_classes  # noqa: E402

AXES = ["rot", "trans", "scale", "bright", "fade", "yellow", "spec"]
AXIS_KO = {"rot": "회전", "trans": "이동", "scale": "스케일", "bright": "밝기",
           "fade": "색바램", "yellow": "황변", "spec": "정반사"}


def variant_name(m: dict) -> str:
    c = m["config"]
    prep = c.get("prep", "") or "—"
    aug = c.get("aug", "?")
    return f"prep={prep} / aug={aug}"


def worst(rob: dict, axis: str, key: str):
    """해당 축의 가장 강한 섭동에서의 값."""
    cs = [c for c in rob["conditions"] if c["axis"] == axis]
    if not cs:
        return None
    c = max(cs, key=lambda x: x["level"])
    return c["agg"][key][0], c["label"]


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    runs = Path(cfg["runs_dir"])
    ids = sys.argv[1:]
    if not ids:
        ids = sorted(p.name for p in runs.glob("EXP_C_*")
                     if (p / "metrics.json").exists() and (p / "robustness.json").exists()
                     and json.loads((p / "metrics.json").read_text(encoding="utf-8"))["config"]["res"] == 256)
    exps = []
    for e in ids:
        m = json.loads((runs / e / "metrics.json").read_text(encoding="utf-8"))
        rpath = runs / e / "robustness.json"
        if not rpath.exists():
            print(f"건너뜀 (robustness.json 없음): {e}")
            continue
        exps.append((e, m, json.loads(rpath.read_text(encoding="utf-8"))))
    if not exps:
        print("비교할 실험 없음")
        return 1

    # 기준선 = prep 없음 + aug medium
    base = next((x for x in exps if not x[1]["config"].get("prep")
                 and x[1]["config"].get("aug") == "medium"), exps[0])

    md: list[str] = []
    A = md.append
    A("# Stage 1 전후 비교 — 값싼 처방의 강건성 효과")
    A("")
    A("- SOP: SOP-MVTEC-CABLE-001 / `reports/연구후보_분석.md` §6 Stage 1")
    A(f"- 생성일: {date.today().isoformat()}")
    A(f"- 기준선: `{base[0]}` ({variant_name(base[1])})")
    A(f"- 입력 해상도: {base[1]['config']['res']}px (동일), 분할·시드 동일")
    A("")
    A("## 0. 변형 정의")
    A("")
    A("| 실험 | 전처리 | 증강 | 비고 |")
    A("|---|---|---|---|")
    for e, m, _ in exps:
        c = m["config"]
        note = "**기준선**" if e == base[0] else ""
        A(f"| `{e}` | `{c.get('prep') or '—'}` | `{c.get('aug')}` | {note} |")
    A("")
    A("> 증강 범위는 시험 섭동보다 좁게 설정했다(채널 게인 ±15% vs 시험 30%, "
      "정반사 ≤3% vs 시험 10%). 증강과 시험 분포가 같으면 강건성 측정이 의미를 잃기 때문이다.")
    A("")

    # ---------- 1. 무섭동 정확도 + 지연 ----------
    A("## 1. 무섭동 정확도 · 지연시간 (§7.1 — 같은 표)")
    A("")
    A("| 실험 | AUROC | 미검율 % | 과검율 % | e2e p50 ms | e2e p95 ms | 모델만 p50 ms | 전처리 비용 ms |")
    A("|---|---|---|---|---|---|---|---|")
    for e, m, _ in exps:
        a, hw = m["aggregate"], m["hardware"]
        prep_cost = (hw["latency_end_to_end"]["p50_ms"]
                     - base[1]["hardware"]["latency_end_to_end"]["p50_ms"])
        A(f"| `{e.split('_20260')[0]}…` | {a['auroc']['mean']:.4f} ± {a['auroc']['std']:.4f} | "
          f"{a['miss_rate']['mean']*100:.2f} ± {a['miss_rate']['std']*100:.2f} | "
          f"{a['overkill_rate']['mean']*100:.2f} ± {a['overkill_rate']['std']*100:.2f} | "
          f"{hw['latency_end_to_end']['p50_ms']:.1f} | {hw['latency_end_to_end']['p95_ms']:.1f} | "
          f"{hw['latency_model_only']['p50_ms']:.1f} | "
          f"{prep_cost:+.1f} |")
    A("")
    A(f"> DoD 지연 상한 100 ms(§1.3) 대조. 전처리 비용은 기준선 e2e p50 대비 증분이다.")
    A("")

    # ---------- 2. 축별 강건성 (AUROC) ----------
    A("## 2. 축별 강건성 — 최대 섭동에서의 AUROC")
    A("")
    A("| 실험 | 무섭동 | " + " | ".join(AXIS_KO[a] for a in AXES) + " |")
    A("|" + "---|" * (2 + len(AXES)))
    for e, m, rob in exps:
        clean = next(c for c in rob["conditions"] if c["axis"] == "clean")
        cells = []
        for ax in AXES:
            w = worst(rob, ax, "auroc")
            cells.append(f"{w[0]:.4f}" if w else "—")
        A(f"| `{e.split('_20260')[0]}…` | {clean['agg']['auroc'][0]:.4f} | " + " | ".join(cells) + " |")
    A("")
    A("최대 섭동 강도: " + ", ".join(
        f"{AXIS_KO[a]} {worst(base[2], a, 'auroc')[1]}" for a in AXES if worst(base[2], a, "auroc")))
    A("")
    A("### 기준선 대비 AUROC 변화")
    A("")
    A("| 실험 | 무섭동 | " + " | ".join(AXIS_KO[a] for a in AXES) + " |")
    A("|" + "---|" * (2 + len(AXES)))
    b_clean = next(c for c in base[2]["conditions"] if c["axis"] == "clean")["agg"]["auroc"][0]
    for e, m, rob in exps:
        if e == base[0]:
            continue
        clean = next(c for c in rob["conditions"] if c["axis"] == "clean")["agg"]["auroc"][0]
        cells = []
        for ax in AXES:
            w, bw = worst(rob, ax, "auroc"), worst(base[2], ax, "auroc")
            cells.append(f"{w[0]-bw[0]:+.4f}" if (w and bw) else "—")
        A(f"| `{e.split('_20260')[0]}…` | {clean-b_clean:+.4f} | " + " | ".join(cells) + " |")
    A("")

    # ---------- 3. 재보정 미검율 ----------
    A("## 3. 최대 섭동에서의 미검율 (재보정 τ) — 표현 붕괴 여부")
    A("")
    A("재보정 τ로도 미검율이 높으면 임계값 문제가 아니라 **표현 자체가 무너진 것**이다.")
    A("")
    A("| 실험 | " + " | ".join(AXIS_KO[a] for a in AXES) + " |")
    A("|" + "---|" * (1 + len(AXES)))
    for e, m, rob in exps:
        cells = []
        for ax in AXES:
            w = worst(rob, ax, "miss_refit")
            cells.append(f"{w[0]*100:.1f}" if w else "—")
        A(f"| `{e.split('_20260')[0]}…` | " + " | ".join(cells) + " |")
    A("")

    # ---------- 4. 과검율 ----------
    A("## 4. 최대 섭동에서의 과검율 (고정 τ) — 판정기 기능 정지 여부")
    A("")
    A("과검율 100%는 전 샘플을 NG로 판정한다는 뜻이며 판정기가 기능을 멈춘 상태다.")
    A("")
    A("| 실험 | 무섭동 | " + " | ".join(AXIS_KO[a] for a in AXES) + " |")
    A("|" + "---|" * (2 + len(AXES)))
    for e, m, rob in exps:
        clean = next(c for c in rob["conditions"] if c["axis"] == "clean")
        cells = []
        for ax in AXES:
            w = worst(rob, ax, "overkill_fixed")
            cells.append(f"{w[0]*100:.1f}" if w else "—")
        A(f"| `{e.split('_20260')[0]}…` | {clean['agg']['overkill_fixed'][0]*100:.1f} | "
          + " | ".join(cells) + " |")
    A("")

    # ---------- 5. 전처리 실패율 ----------
    any_prep = [x for x in exps if x[1]["config"].get("prep")]
    if any_prep:
        A("## 5. 전처리 검출 실패율")
        A("")
        A("| 실험 | 조건 | 원판 검출 실패 % | 회전 추정 실패 % |")
        A("|---|---|---|---|")
        for e, m, rob in any_prep:
            for c in rob["conditions"]:
                ps = [f["prep_stats"] for f in c["folds"] if f.get("prep_stats")]
                if not ps:
                    continue
                df = np.mean([p["disk_fail_rate"] for p in ps]) * 100
                rf = np.mean([p["rot_fail_rate"] for p in ps]) * 100
                if df > 0.5 or rf > 0.5 or c["axis"] == "clean":
                    A(f"| `{e.split('_20260')[0]}…` | {c['axis']} {c['label']} | {df:.1f} | {rf:.1f} |")
        A("")
        A("> 검출 실패 시 항등 변환으로 후퇴하므로 판정이 중단되지는 않으나, 해당 샘플은 전처리 효과를 받지 못한다.")
        A("")

    # ---------- 그림 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        figs = Path(cfg["reports_dir"]) / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(2, 4, figsize=(18, 8), dpi=130)
        axs = axs.ravel()
        for i, ax_name in enumerate(AXES):
            ax = axs[i]
            for e, m, rob in exps:
                cs = sorted([c for c in rob["conditions"] if c["axis"] == ax_name],
                            key=lambda c: c["level"])
                clean = next(c for c in rob["conditions"] if c["axis"] == "clean")
                xs = [0] + [c["level"] for c in cs]
                ys = [clean["agg"]["auroc"][0]] + [c["agg"]["auroc"][0] for c in cs]
                lbl = ("기준선" if e == base[0] else
                       f"{m['config'].get('prep') or '—'}/{m['config'].get('aug')}")
                ax.plot(xs, ys, "o-", lw=1.6, ms=5, label=lbl)
            ax.axhline(0.95, ls="--", lw=1, color="#9ca3af")
            ax.set_title(AXIS_KO[ax_name], fontsize=11)
            ax.set_xlabel("섭동 강도")
            ax.set_ylabel("AUROC")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        axs[-1].axis("off")
        axs[-1].text(0.02, 0.5, "점선 = DoD AUROC 0.95\n\n위로 갈수록 강건",
                     fontsize=11, va="center")
        fig.suptitle("Stage 1 전후 — 섭동 축별 AUROC", fontsize=13)
        fig.tight_layout()
        fig.savefig(figs / "stage1_auroc.png")
        plt.close(fig)
        A("## 저하 곡선 비교")
        A("")
        A("![stage1](figs/stage1_auroc.png)")
        A("")
    except Exception as ex:
        A(f"> 그림 생성 실패: {ex}")
        A("")

    # ---------- 판정 ----------
    A("## 6. 판정")
    A("")
    for e, m, rob in exps:
        if e == base[0]:
            continue
        gains, losses = [], []
        for ax in AXES:
            w, bw = worst(rob, ax, "auroc"), worst(base[2], ax, "auroc")
            if not (w and bw):
                continue
            d = w[0] - bw[0]
            if d >= 0.01:
                gains.append(f"{AXIS_KO[ax]} {d:+.3f}")
            elif d <= -0.01:
                losses.append(f"{AXIS_KO[ax]} {d:+.3f}")
        clean_d = (next(c for c in rob["conditions"] if c["axis"] == "clean")["agg"]["auroc"][0]
                   - b_clean)
        lat_d = (m["hardware"]["latency_end_to_end"]["p50_ms"]
                 - base[1]["hardware"]["latency_end_to_end"]["p50_ms"])
        A(f"### `{e}`")
        A("")
        A(f"- 개선: {', '.join(gains) if gains else '없음 (|ΔAUROC| < 0.01)'}")
        A(f"- 악화: {', '.join(losses) if losses else '없음'}")
        A(f"- 무섭동 AUROC 변화: {clean_d:+.4f}")
        A(f"- e2e p50 지연 변화: {lat_d:+.1f} ms "
          f"(→ {m['hardware']['latency_end_to_end']['p50_ms']:.1f} ms, DoD 상한 100 ms)")
        A("")

    A("### 해석 시 주의")
    A("")
    A("1. 모든 수치는 **val fold 기준**이며 τ도 같은 val에서 정했다 — 낙관적 편향. 홀드아웃은 아직 사용하지 않았다.")
    A("2. fold당 val 불량 15~16장. 축별 AUROC 차이가 0.01 미만이면 유의하다고 보지 않는다(§7.4).")
    A("3. 합성 섭동은 실제 라인 변동의 **대리**다. 섭동 모형이 현장과 다르면 결론도 달라진다.")
    A("4. robust 증강을 쓴 변형은 시험 섭동과 **같은 축**을 학습에서 겪었다(범위는 더 좁게). "
      "따라서 그 축의 개선은 '분포 내 강건성'이며, 완전히 새로운 변동에 대한 일반화는 별도 문제다.")
    A("5. 지연시간은 개발 머신 CPU 값이다. 타깃 하드웨어에서 §9.3으로 재측정해야 한다.")

    out = Path(cfg["reports_dir"]) / "stage1_comparison.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
