"""Stage 0 — 합성 섭동 강건성 벤치마크 (`reports/연구후보_분석.md` §6)

목적: `docs/연구후보/` 두 제안 문서의 전제 — "MVTec 촬영 조건을 벗어나면 현 모델이
무너진다" — 를 정량 검증한다. 실제 라인 배포(산업 제품 양/불 판정)가 목표이므로,
MVTec에 존재하지 않는 환경 변동을 합성해 현 Track C 기준선의 저하 곡선을 만든다.

섭동 강도는 `reports/dataset_variation_cable.md` 실측값을 1배로 두고 확장한다.

핵심 설계
  - **τ는 무섭동 val에서 정한 값을 고정 사용**한다. 배포 시 임계값은 정상 기준
    데이터로 정해지고 현장 변동에 따라 자동으로 바뀌지 않기 때문이다.
  - 동시에 **섭동 데이터로 τ를 재보정한 경우**도 계산한다. 두 값을 비교하면
    "표현 자체가 무너진 것"과 "임계값만 밀린 것"을 분리할 수 있다.
  - AUROC(임계값 무관)도 함께 본다.

입력 소스: 640px 캐시를 '센서 영상'으로 간주해 섭동을 가한 뒤 모델 입력 해상도로
리사이즈한다. 해상도별로 동일한 섭동 파이프라인을 쓰기 위함이다.

산출물: reports/robustness_cable.md, reports/figs/robustness_*.png
        runs/<EXP_ID>/robustness.json

사용: python src/eval/robustness_bench.py [EXP_ID ...]
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg, load_classes  # noqa: E402
from data.preprocess import Preprocessor  # noqa: E402
from eval.metrics import choose_threshold, image_level_metrics, per_class_recall  # noqa: E402
from train.train_cls import IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402

def seed_for(key: str, axis: str, label: str) -> int:
    """이미지·조건별 결정적 시드.

    내장 hash()는 문자열에 대해 프로세스마다 값이 달라진다(PYTHONHASHSEED 무작위화).
    해상도별 실행 간 '같은 이미지에 같은 섭동'을 보장해야 비교가 성립하므로
    blake2b로 고정한다.
    """
    h = hashlib.blake2b(f"{key}|{axis}|{label}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 ** 32)


SRC_RES = 640          # 섭동을 가하는 '센서 영상' 해상도
BORDER = cv2.BORDER_REFLECT_101

# (축, 라벨, 강도) — 강도 1배 = 실측값 (dataset_variation_cable.md)
CONDITIONS: list[tuple[str, str, float]] = [
    ("clean", "무섭동", 0.0),
    # 회전: 실측 원형 표준편차 8.0°, 범위 -20~+24°
    ("rot", "±10°", 10), ("rot", "±20°", 20), ("rot", "±40°", 40), ("rot", "±80°", 80),
    # 이동: 실측 중심 이탈 중앙값 = 이미지 폭의 약 0.9%
    ("trans", "±1%", 0.01), ("trans", "±2.5%", 0.025),
    ("trans", "±5%", 0.05), ("trans", "±10%", 0.10),
    # 스케일: 실측 변동계수 1.84%
    ("scale", "±2%", 0.02), ("scale", "±5%", 0.05),
    ("scale", "±10%", 0.10), ("scale", "±20%", 0.20),
    # 밝기/감마: 실측 이미지 간 평균 L 상대 편차 4.5%
    ("bright", "±5%", 0.05), ("bright", "±15%", 0.15),
    ("bright", "±30%", 0.30), ("bright", "±50%", 0.50),
    # 색바램(무작위 채널 게인): 실측 색도 상대 편차 0.9%
    ("fade", "1%", 0.01), ("fade", "5%", 0.05),
    ("fade", "15%", 0.15), ("fade", "30%", 0.30),
    # 황변(방향성 색바램) — 절연체 노화 모사
    ("yellow", "5%", 0.05), ("yellow", "15%", 0.15), ("yellow", "30%", 0.30),
    # 정반사 하이라이트 합성
    ("spec", "면적1%", 0.01), ("spec", "면적5%", 0.05), ("spec", "면적10%", 0.10),
]


def perturb(img: np.ndarray, axis: str, lv: float, rng: np.random.Generator) -> np.ndarray:
    """img: (H,W,3) uint8 RGB. 반환 동일 규격."""
    if axis == "clean" or lv == 0:
        return img
    h, w = img.shape[:2]

    if axis in ("rot", "trans", "scale"):
        ang = float(rng.uniform(-lv, lv)) if axis == "rot" else 0.0
        sc = 1.0 + float(rng.uniform(-lv, lv)) if axis == "scale" else 1.0
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        if axis == "trans":
            M[0, 2] += float(rng.uniform(-lv, lv)) * w
            M[1, 2] += float(rng.uniform(-lv, lv)) * h
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=BORDER)

    f = img.astype(np.float32)

    if axis == "bright":
        gain = 1.0 + float(rng.uniform(-lv, lv))
        gamma = 1.0 + float(rng.uniform(-lv, lv))
        f = 255.0 * np.power(np.clip(f * gain, 0, 255) / 255.0, gamma)
        return np.clip(f, 0, 255).astype(np.uint8)

    if axis == "fade":
        # 채널별 무작위 게인 → 휘도는 보존하고 색도만 이동
        g = 1.0 + rng.uniform(-lv, lv, 3).astype(np.float32)
        g = g / (g.mean() + 1e-8)
        f = f * g[None, None, :]
        return np.clip(f, 0, 255).astype(np.uint8)

    if axis == "yellow":
        # 황변: B 감쇠, R·G 소폭 증가 + 전체 채도 저하(바램)
        g = np.array([1.0 + 0.3 * lv, 1.0 + 0.1 * lv, 1.0 - lv], dtype=np.float32)
        g = g / (g.mean() + 1e-8)
        f = f * g[None, None, :]
        grayv = f.mean(axis=2, keepdims=True)
        f = f * (1.0 - 0.5 * lv) + grayv * (0.5 * lv)      # 채도 저하
        return np.clip(f, 0, 255).astype(np.uint8)

    if axis == "spec":
        # 타원형 정반사 블롭을 목표 면적 비율만큼 합성
        target = lv * h * w
        mask = np.zeros((h, w), np.float32)
        placed = 0.0
        guard = 0
        while placed < target and guard < 60:
            guard += 1
            ax = int(rng.uniform(0.02, 0.09) * w)
            by = int(rng.uniform(0.02, 0.09) * h)
            cxp = int(rng.uniform(0.2, 0.8) * w)
            cyp = int(rng.uniform(0.2, 0.8) * h)
            blob = np.zeros((h, w), np.float32)
            cv2.ellipse(blob, (cxp, cyp), (ax, by), float(rng.uniform(0, 180)),
                        0, 360, 1.0, -1)
            placed += float(blob.sum())
            mask = np.maximum(mask, blob)
        k = max(3, (int(0.03 * w) // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
        return np.clip(f * (1 - mask) + 255.0 * mask, 0, 255).astype(np.uint8)

    raise ValueError(axis)


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    runs = Path(cfg["runs_dir"])
    exp_ids = sys.argv[1:] or sorted(
        (p.name for p in runs.glob("EXP_C_*") if (p / "metrics.json").exists()),
        key=lambda n: -int(n.split("_")[3]))

    cache = Path(cfg["data_root"]) / "cache"
    keys = json.loads((cache / f"keys_{SRC_RES}.json").read_text(encoding="utf-8"))
    key2row = {k: i for i, k in enumerate(keys)}
    src = np.load(cache / f"img_{SRC_RES}.npy", mmap_mode="r")
    with open(Path(cfg["data_root"]) / "labels_index.csv", encoding="utf-8") as f:
        meta = {r["key"]: r for r in csv.DictReader(f)}

    all_results = {}
    for exp_id in exp_ids:
        exp_root = runs / exp_id
        m = json.loads((exp_root / "metrics.json").read_text(encoding="utf-8"))
        R = m["config"]["res"]
        torch.set_num_threads(m["config"]["threads"])
        print(f"\n[{exp_id}] 입력 {R}px, 조건 {len(CONDITIONS)}개 × {cfg['n_folds']} fold")

        prep_spec = m["config"].get("prep", "")
        src_r = m["config"].get("src_res", R)
        if src_r != SRC_RES:
            print(f"  경고: 학습 센서 해상도 {src_r} != 벤치 소스 {SRC_RES}")

        models, folds_info = [], []
        for fm in m["folds"]:
            k = fm["fold"]
            net = build_model(m["config"]["arch"], pretrained=False)
            ck = torch.load(exp_root / "weights" / f"fold{k}_best.pt",
                            map_location="cpu", weights_only=False)
            net.load_state_dict(ck["model"])
            net.eval()
            split = json.loads((Path(cfg["splits_dir"]) / f"fold{k}.json").read_text(encoding="utf-8"))
            va = split["val"]
            prep = None
            if prep_spec:
                prep = Preprocessor(prep_spec)
                rp = exp_root / f"ref_profile_fold{k}.npy"
                if prep.use_rot:
                    if not rp.exists():
                        raise FileNotFoundError(f"회전 기준 프로파일 없음: {rp}")
                    prep.ref_profile = np.load(rp)
            models.append(net)
            folds_info.append({
                "fold": k, "val": va, "tau_clean": fm["tau"], "prep": prep,
                "y": np.array([int(meta[x]["image_label"]) for x in va]),
                "cls": [meta[x]["class"] for x in va],
            })

        cond_rows = []
        for ci, (axis, label, lv) in enumerate(CONDITIONS):
            per_fold = []
            for net, fi in zip(models, folds_info):
                scores = []
                batch, bkeys = [], []

                def flush():
                    if not batch:
                        return
                    t = torch.from_numpy(np.stack(batch))
                    with torch.no_grad():
                        scores.append(torch.softmax(net(t), 1)[:, 1].numpy())
                    batch.clear()

                for key in fi["val"]:
                    img = np.asarray(src[key2row[key]])
                    rng = np.random.default_rng(seed_for(key, axis, label))
                    img = perturb(img, axis, lv, rng)
                    # 학습과 동일한 전처리를 추론에도 적용한다 (§4.4 전처리 불일치 금지)
                    if fi["prep"] is not None and fi["prep"].enabled:
                        img = fi["prep"](img)
                    if R != img.shape[0]:
                        img = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
                    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                    batch.append(np.ascontiguousarray(x.transpose(2, 0, 1)))
                    bkeys.append(key)
                    if len(batch) >= m["config"]["batch"]:
                        flush()
                flush()
                sc = np.concatenate(scores)
                y = fi["y"]

                prep_stats = fi["prep"].stats.as_dict() if fi["prep"] is not None else None
                if fi["prep"] is not None:
                    from data.preprocess import PrepStats
                    fi["prep"].stats = PrepStats()      # 조건마다 초기화
                fixed = image_level_metrics(sc, y, fi["tau_clean"])
                tau_re, _ = choose_threshold(sc, y)
                refit = image_level_metrics(sc, y, tau_re)
                per_fold.append({
                    "fold": fi["fold"],
                    "auroc": fixed["auroc"],
                    "miss_fixed": fixed["miss_rate"], "overkill_fixed": fixed["overkill_rate"],
                    "miss_refit": refit["miss_rate"], "overkill_refit": refit["overkill_rate"],
                    "recall_at_fpr5": fixed["recall_at_fpr5"],
                    "per_class": per_class_recall(sc, y, fi["cls"], fi["tau_clean"]),
                    "prep_stats": prep_stats,
                })

            agg = {k: (float(np.mean([f[k] for f in per_fold])),
                       float(np.std([f[k] for f in per_fold])))
                   for k in ("auroc", "miss_fixed", "overkill_fixed",
                             "miss_refit", "overkill_refit", "recall_at_fpr5")}
            cls_agg = {}
            for c in [classes["multi"][i] for i in sorted(classes["multi"])]:
                vals = [f["per_class"][c]["recall"] for f in per_fold if c in f["per_class"]]
                cls_agg[c] = float(np.mean(vals)) if vals else None
            cond_rows.append({"axis": axis, "label": label, "level": lv,
                              "agg": agg, "per_class": cls_agg, "folds": per_fold})
            print(f"  [{ci+1:2d}/{len(CONDITIONS)}] {axis:7s} {label:8s} "
                  f"AUROC {agg['auroc'][0]:.4f}  미검(고정τ) {agg['miss_fixed'][0]*100:6.2f}%  "
                  f"미검(재보정) {agg['miss_refit'][0]*100:6.2f}%  "
                  f"과검(고정τ) {agg['overkill_fixed'][0]*100:6.2f}%")

        (exp_root / "robustness.json").write_text(
            json.dumps({"exp_id": exp_id, "res": R, "src_res": SRC_RES,
                        "prep": prep_spec, "aug": m["config"].get("aug"),
                        "conditions": cond_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        all_results[exp_id] = {"res": R, "prep": prep_spec,
                               "aug": m["config"].get("aug"), "conditions": cond_rows}

    write_report(cfg, classes, all_results)
    return 0


def write_report(cfg, classes, all_results: dict) -> None:
    """이번 실행분 + 이전에 저장된 robustness.json을 모두 합쳐 하나의 리포트로 낸다."""
    reports = Path(cfg["reports_dir"])
    runs = Path(cfg["runs_dir"])
    merged = {}
    for p in sorted(runs.glob("EXP_C_*/robustness.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        merged[d["exp_id"]] = {"res": d["res"], "prep": d.get("prep", ""),
                               "aug": d.get("aug"), "conditions": d["conditions"]}
    merged.update(all_results)
    all_results = dict(sorted(merged.items(), key=lambda kv: -kv[1]["res"]))
    md: list[str] = []
    A = md.append
    A("# Stage 0 — 합성 섭동 강건성 벤치마크")
    A("")
    A("- 목적: `docs/연구후보/` 두 제안 문서의 전제(\"MVTec 조건을 벗어나면 현 모델이 무너진다\") 정량 검증")
    A("- 배경: 최종 목표가 **실제 산업 제품 양/불 판정**이므로 MVTec은 대리 데이터다. "
      "촬영 조건 변동에 대한 강건성은 원 데이터로 측정할 수 없어 합성 섭동으로 대체한다.")
    A(f"- 생성일: {date.today().isoformat()}")
    A(f"- 섭동 적용 해상도: {SRC_RES}px(센서 영상 가정) → 모델 입력 해상도로 리사이즈")
    A("- 기준 강도(1배): `reports/dataset_variation_cable.md` 실측값")
    A("")
    A("## 방법 — 임계값 처리")
    A("")
    A("| 표기 | 의미 |")
    A("|---|---|")
    A("| **고정 τ** | 무섭동 val에서 정한 τ를 그대로 사용. **배포 현실에 해당** — 임계값은 정상 기준 데이터로 정해지고 현장 변동에 따라 자동으로 바뀌지 않는다. |")
    A("| 재보정 τ | 섭동 데이터에서 τ를 다시 정함. 고정 τ와의 차이가 크면 **표현은 살아 있고 임계값만 밀린 것** — 현장 재보정으로 복구 가능하다는 뜻. |")
    A("| AUROC | 임계값 무관 순위 품질. 이것이 무너지면 **표현 자체가 무너진 것**이라 재보정으로도 복구 불가. |")
    A("")

    for exp_id, res in all_results.items():
        A(f"## `{exp_id}` (입력 {res['res']}px)")
        A("")
        if res.get("prep"):
            A(f"- 전처리: `{res['prep']}` / 증강: `{res.get('aug', '?')}`")
            A("")
        A("| 축 | 강도 | AUROC | 미검율 % (고정 τ) | 미검율 % (재보정 τ) | 과검율 % (고정 τ) | Recall@FPR5 |")
        A("|---|---|---|---|---|---|---|")
        for c in res["conditions"]:
            g = c["agg"]
            A(f"| {c['axis']} | {c['label']} | {g['auroc'][0]:.4f} ± {g['auroc'][1]:.4f} | "
              f"{g['miss_fixed'][0]*100:.2f} ± {g['miss_fixed'][1]*100:.2f} | "
              f"{g['miss_refit'][0]*100:.2f} ± {g['miss_refit'][1]*100:.2f} | "
              f"{g['overkill_fixed'][0]*100:.2f} ± {g['overkill_fixed'][1]*100:.2f} | "
              f"{g['recall_at_fpr5'][0]:.4f} |")
        A("")

    # 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        figs = reports / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        axes_order = ["rot", "trans", "scale", "bright", "fade", "yellow", "spec"]
        titles = {"rot": "회전", "trans": "이동", "scale": "스케일", "bright": "밝기/감마",
                  "fade": "색바램(무작위)", "yellow": "황변", "spec": "정반사"}
        fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), dpi=130)
        axs = axs.ravel()
        for i, ax_name in enumerate(axes_order):
            ax = axs[i]
            for exp_id, res in all_results.items():
                cs = [c for c in res["conditions"] if c["axis"] == ax_name]
                clean = next(c for c in res["conditions"] if c["axis"] == "clean")
                xs = [0] + [c["level"] for c in cs]
                miss = [clean["agg"]["miss_fixed"][0] * 100] + [c["agg"]["miss_fixed"][0] * 100 for c in cs]
                ax.plot(xs, miss, "o-", lw=1.6, ms=5, label=f"{res['res']}px")
            ax.axhline(5, ls="--", lw=1, color="#9ca3af")
            ax.set_title(titles[ax_name], fontsize=10)
            ax.set_xlabel("섭동 강도")
            ax.set_ylabel("미검율 % (고정 τ)")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        axs[-1].axis("off")
        axs[-1].text(0.02, 0.5, "점선 = DoD 미검율 5%\n\n고정 τ 기준\n(배포 현실)",
                     fontsize=10, va="center")
        fig.suptitle("Stage 0 — 섭동 축별 미검율 저하 곡선 (Track C)", fontsize=12)
        fig.tight_layout()
        fig.savefig(figs / "robustness_miss.png")
        plt.close(fig)
        A("## 저하 곡선")
        A("")
        A("![robustness](figs/robustness_miss.png)")
        A("")
    except Exception as e:
        A(f"> 그림 생성 실패: {e}")
        A("")

    out = reports / "robustness_cable.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
