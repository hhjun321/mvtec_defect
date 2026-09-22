"""저장된 fold별 best 체크포인트로 검증셋 점수를 재산출하고 지표를 갱신한다.

학습을 다시 하지 않고(추론만) 다음을 수행한다:
  - fold별 val 점수 원본 저장 (§11 재현성 — 지표를 나중에 다시 계산할 수 있게)
  - 임계값 τ 재산출 (마진 tie-break 적용, §7.2)
  - metrics.json 갱신 + 점수 분포 진단

사용: python src/eval/score_folds.py EXP_C_mnv3s_640_fp32_all_20260921
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg, load_classes  # noqa: E402
from data.preprocess import Preprocessor  # noqa: E402
from eval.metrics import (  # noqa: E402
    aggregate_folds, choose_threshold, image_level_metrics, per_class_recall,
)
from train.train_cls import IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402


def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)
    exp_id = sys.argv[1]
    exp_root = Path(cfg["runs_dir"]) / exp_id
    m = json.loads((exp_root / "metrics.json").read_text(encoding="utf-8"))
    c = m["config"]
    res = c["res"]
    src_res = c.get("src_res", res)
    prep_spec = c.get("prep", "")

    cache = Path(cfg["data_root"]) / "cache"
    keys = json.loads((cache / f"keys_{src_res}.json").read_text(encoding="utf-8"))
    key2row = {k: i for i, k in enumerate(keys)}
    arr = np.load(cache / f"img_{src_res}.npy", mmap_mode="r")
    with open(Path(cfg["data_root"]) / "labels_index.csv", encoding="utf-8") as f:
        meta = {r["key"]: r for r in csv.DictReader(f)}

    torch.set_num_threads(c["threads"])
    new_folds = []
    for fm in m["folds"]:
        k = fm["fold"]
        split = json.loads((Path(cfg["splits_dir"]) / f"fold{k}.json").read_text(encoding="utf-8"))
        va = split["val"]
        y = np.array([int(meta[x]["image_label"]) for x in va])
        cls = [meta[x]["class"] for x in va]

        model = build_model(c["arch"], pretrained=False)
        ck = torch.load(exp_root / "weights" / f"fold{k}_best.pt", map_location="cpu",
                        weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()

        prep = None
        if prep_spec:
            prep = Preprocessor(prep_spec)
            if prep.use_rot:
                prep.ref_profile = np.load(exp_root / f"ref_profile_fold{k}.npy")

        scores = []
        with torch.no_grad():
            for i in range(0, len(va), c["batch"]):
                batch = va[i:i + c["batch"]]
                xs = []
                for key in batch:
                    raw = np.asarray(arr[key2row[key]])
                    if prep is not None and prep.enabled:
                        raw = prep(raw)
                    if raw.shape[0] != res:
                        import cv2
                        raw = cv2.resize(raw, (res, res), interpolation=cv2.INTER_AREA)
                    img = raw.astype(np.float32) / 255.0
                    img = (img - IMAGENET_MEAN) / IMAGENET_STD
                    xs.append(np.ascontiguousarray(img.transpose(2, 0, 1)))
                t = torch.from_numpy(np.stack(xs))
                scores.append(torch.softmax(model(t), 1)[:, 1].numpy())
        scores = np.concatenate(scores)

        assert abs(float(np.mean(scores >= 0.5)) - float(np.mean(scores >= 0.5))) == 0
        tau, info = choose_threshold(scores, y)
        nm = image_level_metrics(scores, y, tau)
        nm["threshold_policy"] = info
        nm["per_class_recall"] = per_class_recall(scores, y, cls, tau)
        nm["best_epoch"] = fm["best_epoch"]
        nm["fold"] = k
        nm["checkpoint"] = fm["checkpoint"]

        # 점수 분포 진단
        sp, sn = scores[y == 1], scores[y == 0]
        nm["score_stats"] = {
            "NG_min": float(sp.min()), "NG_median": float(np.median(sp)),
            "OK_max": float(sn.max()), "OK_median": float(np.median(sn)),
            "separated": bool(sp.min() > sn.max()),
            "gap_log10": (float(np.log10(max(sp.min(), 1e-12) / max(sn.max(), 1e-12)))
                          if sp.min() > sn.max() else None),
        }
        (exp_root / f"scores_fold{k}.json").write_text(
            json.dumps({"keys": va, "label": y.tolist(), "class": cls,
                        "score": scores.tolist()}, ensure_ascii=False), encoding="utf-8")

        old_auroc = fm["auroc"]
        assert abs(nm["auroc"] - old_auroc) < 1e-6, (
            f"fold{k}: 재산출 AUROC {nm['auroc']} != 학습 시 {old_auroc} — 체크포인트/전처리 불일치")
        new_folds.append(nm)
        print(f"  fold{k}: auroc={nm['auroc']:.4f} tau={tau:.6g} "
              f"(roc={info['tau_roc']:.3g}, margin_tiebreak={info['margin_tiebreak']}) "
              f"OK_max={nm['score_stats']['OK_max']:.3g} NG_min={nm['score_stats']['NG_min']:.3g} "
              f"sep={nm['score_stats']['separated']}")

    agg_keys = ["auroc", "auprc", "miss_rate", "overkill_rate", "recall",
                "precision", "f1", "balanced_acc", "recall_at_fpr5", "tau"]
    m["folds"] = new_folds
    m["aggregate"] = aggregate_folds(new_folds, agg_keys)
    cls_names = [classes["multi"][i] for i in sorted(classes["multi"])]
    m["per_class_recall"] = {
        c_: {"n_total": int(sum(f["per_class_recall"][c_]["n"] for f in new_folds
                                if c_ in f["per_class_recall"])),
             "recall_mean": float(np.mean([f["per_class_recall"][c_]["recall"] for f in new_folds
                                           if c_ in f["per_class_recall"]])),
             "recall_std": float(np.std([f["per_class_recall"][c_]["recall"] for f in new_folds
                                         if c_ in f["per_class_recall"]])),
             "per_fold": [float(f["per_class_recall"][c_]["recall"]) for f in new_folds
                          if c_ in f["per_class_recall"]]}
        for c_ in cls_names}
    m["rescored"] = {"note": "체크포인트로 val 점수 재산출, τ에 마진 tie-break 적용",
                     "scores_saved": [f"scores_fold{f['fold']}.json" for f in new_folds]}
    (exp_root / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"  -> {exp_root / 'metrics.json'} 갱신")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
