"""SOP-DATA-03 — 데이터 분할 설계 (SOP-MVTEC-CABLE-001 §4.3)

분할 A: 비지도 이상탐지용. MVTec 원본 분할 유지 (문헌 비교 가능). 변경 금지.
분할 B: 지도학습용. 불량 이미지를 클래스 층화 5-fold + 독립 홀드아웃으로 재분할.

산출물:
  data/splits/split_A.json
  data/splits/holdout.json
  data/splits/fold{0..4}.json
  data/splits/yolo/{multi,binary}/fold{k}_{train,val}.txt, holdout.txt
  data/splits/yolo/{multi,binary}/dataset_fold{k}.yaml
  reports/splits_cable.md
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import env_stamp, load_cfg, load_classes, set_seed, write_json  # noqa: E402


def sha1_of(items) -> str:
    return hashlib.sha1("\n".join(sorted(items)).encode()).hexdigest()[:12]


def sha1_of_parts(**parts) -> str:
    """분할 식별용 해시. 부분집합 경계를 구분하므로 train/val 합집합이 같아도 달라진다."""
    blob = "".join(f"[{name}]\n" + "\n".join(sorted(keys)) + "\n"
                   for name, keys in sorted(parts.items()))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    seed = cfg["seed"]
    set_seed(seed)

    data_root = Path(cfg["data_root"])
    splits_dir = Path(cfg["splits_dir"])
    yolo_root = Path(cfg["yolo_root"])
    reports = Path(cfg["reports_dir"])
    n_folds = cfg["n_folds"]
    n_hold = cfg["holdout_per_defect_class"]

    with open(data_root / "labels_index.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    defects = [r for r in rows if r["image_label"] == "1"]
    normals = [r for r in rows if r["image_label"] == "0"]

    # ---------------- 분할 A: 원본 유지 (비지도 이상탐지) -------------------
    split_a = {
        "name": "A_unsupervised_original",
        "purpose": "비지도 이상탐지. MVTec 원본 분할 유지 — 문헌값 비교 가능. 변경 금지.",
        "seed": seed,
        "train": sorted(r["key"] for r in rows if r["orig_split"] == "train"),
        "test": sorted(r["key"] for r in rows if r["orig_split"] == "test"),
    }
    split_a["train_n"] = len(split_a["train"])
    split_a["test_n"] = len(split_a["test"])
    split_a["test_defect_n"] = sum(1 for r in rows if r["orig_split"] == "test" and r["image_label"] == "1")
    split_a["sha1"] = sha1_of_parts(train=split_a["train"], test=split_a["test"])
    write_json(splits_dir / "split_A.json", split_a)

    # ---------------- 분할 B: 홀드아웃 분리 ---------------------------------
    # 불량: 클래스당 n_hold 장. 정상: 전체 불량:정상 비율 유지.
    rng_state = sorted(defects, key=lambda r: r["key"])
    by_cls = defaultdict(list)
    for r in rng_state:
        by_cls[r["class"]].append(r)

    import random
    rng = random.Random(seed)
    hold_def = []
    for cls in sorted(by_cls):
        pool = sorted(by_cls[cls], key=lambda r: r["key"])
        hold_def += rng.sample(pool, n_hold)
    hold_def_keys = {r["key"] for r in hold_def}

    ratio = len(normals) / len(defects)
    n_hold_norm = round(len(hold_def) * ratio)
    norm_sorted = sorted(normals, key=lambda r: r["key"])
    # 정상도 출처(train/good, test/good) 비율 유지
    by_src = defaultdict(list)
    for r in norm_sorted:
        by_src[r["orig_split"]].append(r)
    hold_norm = []
    for src in sorted(by_src):
        k = round(n_hold_norm * len(by_src[src]) / len(norm_sorted))
        hold_norm += rng.sample(sorted(by_src[src], key=lambda r: r["key"]), k)
    hold_norm_keys = {r["key"] for r in hold_norm}

    holdout = {
        "name": "B_holdout",
        "purpose": "최종 DoD 판정 전용(§1.3). 모델 선택·튜닝·임계값 결정에 절대 사용 금지. 1회만 사용.",
        "seed": seed,
        "defect_per_class": n_hold,
        "keys": sorted(hold_def_keys | hold_norm_keys),
        "defect_keys": sorted(hold_def_keys),
        "normal_keys": sorted(hold_norm_keys),
        "defect_n": len(hold_def_keys),
        "normal_n": len(hold_norm_keys),
        "class_counts": dict(Counter(r["class"] for r in hold_def)),
    }
    holdout["sha1"] = sha1_of(holdout["keys"])
    write_json(splits_dir / "holdout.json", holdout)

    # ---------------- 분할 B: 5-fold 층화 교차검증 --------------------------
    cv_def = [r for r in rng_state if r["key"] not in hold_def_keys]
    cv_norm = [r for r in norm_sorted if r["key"] not in hold_norm_keys]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    def_folds = list(skf.split(cv_def, [r["class"] for r in cv_def]))
    norm_folds = list(skf.split(cv_norm, [r["orig_split"] for r in cv_norm]))

    fold_summaries = []
    for k in range(n_folds):
        dtr, dva = def_folds[k]
        ntr, nva = norm_folds[k]
        train_keys = sorted([cv_def[i]["key"] for i in dtr] + [cv_norm[i]["key"] for i in ntr])
        val_keys = sorted([cv_def[i]["key"] for i in dva] + [cv_norm[i]["key"] for i in nva])
        assert not set(train_keys) & set(val_keys), "train/val 누수"
        assert not (set(train_keys) | set(val_keys)) & set(holdout["keys"]), "홀드아웃 누수"

        val_cls = Counter(cv_def[i]["class"] for i in dva)
        fold = {
            "name": f"B_fold{k}",
            "seed": seed,
            "fold": k,
            "n_folds": n_folds,
            "train": train_keys,
            "val": val_keys,
            "train_defect_n": len(dtr),
            "train_normal_n": len(ntr),
            "val_defect_n": len(dva),
            "val_normal_n": len(nva),
            "val_class_counts": dict(sorted(val_cls.items())),
        }
        fold["sha1"] = sha1_of_parts(train=train_keys, val=val_keys)
        write_json(splits_dir / f"fold{k}.json", fold)
        fold_summaries.append(fold)

    # ---------------- Ultralytics 데이터셋 파일 ------------------------------
    for variant in ("multi", "binary"):
        vdir = splits_dir / "yolo" / variant
        vdir.mkdir(parents=True, exist_ok=True)
        img_dir = yolo_root / variant / "images"

        def write_list(path: Path, keys):
            path.write_text(
                "\n".join(str((img_dir / f"{k}.png").as_posix()) for k in keys) + "\n",
                encoding="utf-8",
            )

        write_list(vdir / "holdout.txt", holdout["keys"])
        write_list(vdir / "splitA_train.txt", split_a["train"])
        write_list(vdir / "splitA_test.txt", split_a["test"])

        names = classes["multi"] if variant == "multi" else classes["binary"]
        for k, fold in enumerate(fold_summaries):
            write_list(vdir / f"fold{k}_train.txt", fold["train"])
            write_list(vdir / f"fold{k}_val.txt", fold["val"])
            yaml_lines = [
                f"# SOP-MVTEC-CABLE-001 §4.3 — 분할 B fold {k} ({variant})",
                f"# 자동 생성. 직접 수정 금지 — src/data/make_splits.py 로 재생성.",
                f"path: {(yolo_root / variant).as_posix()}",
                f"train: {(vdir / f'fold{k}_train.txt').as_posix()}",
                f"val: {(vdir / f'fold{k}_val.txt').as_posix()}",
                f"test: {(vdir / 'holdout.txt').as_posix()}",
                "names:",
            ]
            yaml_lines += [f"  {cid}: {nm}" for cid, nm in sorted(names.items())]
            (vdir / f"dataset_fold{k}.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # ---------------- 리포트 -------------------------------------------------
    env = env_stamp()
    md = [
        "# 데이터 분할 리포트 — MVTec AD `cable`",
        "",
        "- SOP: SOP-MVTEC-CABLE-001 §4.3 (SOP-DATA-03)",
        f"- 실행일: {date.today().isoformat()}",
        f"- git: `{env['git_rev']}`",
        f"- 시드: {seed} (분할은 JSON으로 고정 저장, 실행 시마다 재추첨하지 않음)",
        "",
        "## 분할 A — 비지도 이상탐지용 (원본 유지, 변경 금지)",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| train (정상만) | {split_a['train_n']} |",
        f"| test | {split_a['test_n']} (불량 {split_a['test_defect_n']} / 정상 {split_a['test_n'] - split_a['test_defect_n']}) |",
        f"| sha1 | `{split_a['sha1']}` |",
        "",
        "## 홀드아웃 (분할 B, 최종 판정 전용)",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 불량 | {holdout['defect_n']} (클래스당 {n_hold}) |",
        f"| 정상 | {holdout['normal_n']} |",
        f"| 합계 | {len(holdout['keys'])} |",
        f"| sha1 | `{holdout['sha1']}` |",
        "",
        "> **홀드아웃은 §1.3 DoD 최종 판정에만 1회 사용한다.** 모델 선택, 하이퍼파라미터 튜닝, "
        "임계값 τ 결정(§7.2)에 사용하면 그 결과는 무효다.",
        "",
        "## 5-fold 교차검증 (분할 B)",
        "",
        "| fold | train 불량 | train 정상 | val 불량 | val 정상 | sha1 |",
        "|---|---|---|---|---|---|",
    ]
    for f_ in fold_summaries:
        md.append(
            f"| {f_['fold']} | {f_['train_defect_n']} | {f_['train_normal_n']} | "
            f"{f_['val_defect_n']} | {f_['val_normal_n']} | `{f_['sha1']}` |"
        )

    cls_names = [classes["multi"][i] for i in sorted(classes["multi"])]
    md += ["", "### fold별 val 불량 클래스 분포", "",
           "| fold | " + " | ".join(cls_names) + " |",
           "|" + "---|" * (1 + len(cls_names))]
    for f_ in fold_summaries:
        cnts = f_["val_class_counts"]
        md.append("| " + str(f_["fold"]) + " | "
                  + " | ".join(str(cnts.get(n, 0)) for n in cls_names) + " |")

    md += [
        "",
        "## 누수 방지 검증",
        "",
        "- 모든 fold에서 train ∩ val = ∅ (assert 통과)",
        "- 모든 fold에서 (train ∪ val) ∩ holdout = ∅ (assert 통과)",
        "- 분할 단위 = 이미지 파일. 파생 크롭·증강본은 원본과 같은 분할에 속한다.",
        "",
        "## 생성 파일",
        "",
        "```",
        "data/splits/split_A.json          분할 A (비지도)",
        "data/splits/holdout.json          홀드아웃",
        "data/splits/fold{0..4}.json       5-fold",
        "data/splits/yolo/multi/           8클래스 검출용 Ultralytics 설정",
        "data/splits/yolo/binary/          OK/NG 단일클래스 검출용 Ultralytics 설정",
        "  fold{k}_train.txt / fold{k}_val.txt / holdout.txt",
        "  splitA_train.txt / splitA_test.txt",
        "  dataset_fold{k}.yaml            → yolo train data=... 에 지정",
        "```",
        "",
        "## 통계적 한계 (해석 시 필수 고려)",
        "",
        f"- fold당 val 불량이 약 {sum(f['val_defect_n'] for f in fold_summaries) // n_folds}장이므로, "
        f"recall 1건 차이가 약 {100 / (sum(f['val_defect_n'] for f in fold_summaries) / n_folds):.1f}%p로 나타난다.",
        f"- 홀드아웃 불량 {holdout['defect_n']}장 기준 recall의 95% 신뢰구간은 ±20%p 수준이다. "
        "홀드아웃 단독 수치로 모델 우열을 주장하지 않는다.",
        "- 보고는 5-fold 평균 ± 표준편차로 한다(§7.4).",
    ]
    (reports / "splits_cable.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[SOP-DATA-03] 분할 A: train {split_a['train_n']} / test {split_a['test_n']}")
    print(f"[SOP-DATA-03] 홀드아웃: 불량 {holdout['defect_n']} / 정상 {holdout['normal_n']}")
    for f_ in fold_summaries:
        print(f"  fold{f_['fold']}: train {f_['train_defect_n']}D+{f_['train_normal_n']}N / "
              f"val {f_['val_defect_n']}D+{f_['val_normal_n']}N")
    print(f"  -> {splits_dir}")
    print(f"  -> {reports / 'splits_cable.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
