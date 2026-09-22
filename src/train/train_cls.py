"""Track C — 지도학습 이진 분류 (OK/NG) 학습 (SOP-MVTEC-CABLE-001 §6 Track C, §8.1)

기본값: MobileNetV3-Small, ImageNet 사전학습, 640px, FP32, 분할 B 5-fold.

산출물 (실험 1건당, §11):
  runs/<EXP_ID>/config.yaml   설정 스냅샷 (실값)
  runs/<EXP_ID>/env.txt       pip freeze + 하드웨어
  runs/<EXP_ID>/git_rev.txt   커밋 해시
  runs/<EXP_ID>/weights/best.pt
  runs/<EXP_ID>/metrics.json  §7.1 전 지표 + 임계값 tau
  runs/<EXP_ID>/logs/train.csv

사용:
  python src/train/train_cls.py --fold 0
  python src/train/train_cls.py --fold all --res 640 --arch mobilenet_v3_small
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import env_stamp, git_rev, load_cfg, load_classes, set_seed  # noqa: E402
from data.preprocess import Preprocessor  # noqa: E402
from eval.metrics import (  # noqa: E402
    aggregate_folds, choose_threshold, image_level_metrics, measure_latency,
    model_complexity, per_class_recall,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------- 데이터 ---------------------------------

class CableClsDataset(Dataset):
    """§4.4 전처리·증강.

    증강 허용: 수평/수직 플립, 90도 회전, 스케일 +-10%, 평행이동 +-5%,
              밝기/대비 +-20%, 약한 가우시안 노이즈
    증강 금지: 강한 색상(Hue) 변형 — 절연체 색상이 cable_swap 판별 근거이므로
              색을 흔들면 라벨이 거짓이 된다 (§4.4)
    """

    def __init__(self, arr, row_idx, labels, classes, train: bool, aug_strength: str = "medium",
                 prep=None, out_res: int | None = None):
        self.arr = arr
        self.row_idx = row_idx
        self.labels = labels
        self.classes = classes
        self.train = train
        self.aug = aug_strength
        self.prep = prep                 # Preprocessor 또는 None
        self.out_res = out_res           # 전처리 후 리사이즈할 모델 입력 해상도

    def __len__(self):
        return len(self.row_idx)

    def _augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import cv2
        if self.aug == "none":
            return img
        # weak = 절반, medium = 기본, robust = Stage 0 섭동 축을 학습에서 미리 겪게 한다
        s = {"weak": 0.5, "medium": 1.0, "robust": 1.0}[self.aug]
        robust = self.aug == "robust"

        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1]
        k = int(rng.integers(0, 4))
        if k:
            img = np.rot90(img, k)
        img = np.ascontiguousarray(img)

        # 스케일/평행이동 (+ robust 에서는 연속 회전)
        h, w = img.shape[:2]
        rs, ts, rot_deg = (0.20, 0.10, 40.0) if robust else (0.10, 0.05, 0.0)
        scale = 1.0 + float(rng.uniform(-rs, rs)) * s
        tx = float(rng.uniform(-ts, ts)) * s * w
        ty = float(rng.uniform(-ts, ts)) * s * h
        ang = float(rng.uniform(-rot_deg, rot_deg)) if robust else 0.0
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

        # 밝기/대비 (채널 공통 — Hue 변형 아님)
        br = 0.30 if robust else 0.20
        alpha = 1.0 + float(rng.uniform(-br, br)) * s       # 대비
        beta = float(rng.uniform(-br, br)) * s * 255.0      # 밝기
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
        if robust:
            # 채널 게인(색바램) — 휘도 보존. Hue 자체를 돌리는 변형이 아니라
            # 조명 색온도/절연체 퇴색을 모사하는 선형 게인이다 (§4.4 금지 항목 아님)
            g = 1.0 + rng.uniform(-0.15, 0.15, 3).astype(np.float32)
            img = np.clip(img * (g / g.mean())[None, None, :], 0, 255)

        if rng.random() < 0.3:
            img = np.clip(img + rng.normal(0, 4.0 * s, img.shape).astype(np.float32), 0, 255)

        if robust and rng.random() < 0.5:
            # 정반사 하이라이트 — 시험 범위(면적 최대 10%)보다 좁게(최대 3%) 둔다.
            # 증강과 시험 섭동이 같은 분포면 강건성 측정이 의미를 잃기 때문이다.
            hh, ww = img.shape[:2]
            target = float(rng.uniform(0.002, 0.03)) * hh * ww
            blobs = np.zeros((hh, ww), np.float32)
            placed, guard = 0.0, 0
            while placed < target and guard < 20:
                guard += 1
                b = np.zeros((hh, ww), np.float32)
                cv2.ellipse(b, (int(rng.uniform(0.2, 0.8) * ww), int(rng.uniform(0.2, 0.8) * hh)),
                            (int(rng.uniform(0.02, 0.07) * ww), int(rng.uniform(0.02, 0.07) * hh)),
                            float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
                placed += float(b.sum())
                blobs = np.maximum(blobs, b)
            kk = max(3, (int(0.03 * ww) // 2) * 2 + 1)
            blobs = cv2.GaussianBlur(blobs, (kk, kk), 0)[..., None]
            img = img * (1 - blobs) + 255.0 * blobs

        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, i):
        import cv2
        img = np.asarray(self.arr[self.row_idx[i]])
        if self.train:
            rng = np.random.default_rng()
            img = self._augment(img, rng)
        if self.prep is not None and self.prep.enabled:
            img = self.prep(img)
        if self.out_res is not None and img.shape[0] != self.out_res:
            img = cv2.resize(img, (self.out_res, self.out_res), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return x, int(self.labels[i])


# --------------------------------- 모델 -----------------------------------

def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    import torchvision.models as tvm
    if arch == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                                   if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    elif arch == "shufflenet_v2_x0_5":
        m = tvm.shufflenet_v2_x0_5(weights=tvm.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
                                   if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1
                                   if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    else:
        raise ValueError(f"미지원 arch: {arch}")
    return m


# --------------------------------- 학습 -----------------------------------

def run_fold(cfg, args, fold: int, exp_root: Path) -> dict:
    set_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    data_root = Path(cfg["data_root"])
    cache = data_root / "cache"
    src_res = args.src_res
    keys = json.loads((cache / f"keys_{src_res}.json").read_text(encoding="utf-8"))
    key2row = {k: i for i, k in enumerate(keys)}
    arr = np.load(cache / f"img_{src_res}.npy", mmap_mode="r")

    with open(data_root / "labels_index.csv", encoding="utf-8") as f:
        meta = {r["key"]: r for r in csv.DictReader(f)}

    split = json.loads((Path(cfg["splits_dir"]) / f"fold{fold}.json").read_text(encoding="utf-8"))
    tr_keys, va_keys = split["train"], split["val"]

    def pack(ks):
        return (np.array([key2row[k] for k in ks]),
                np.array([int(meta[k]["image_label"]) for k in ks]),
                [meta[k]["class"] for k in ks])

    tr_rows, tr_y, _ = pack(tr_keys)
    va_rows, va_y, va_cls = pack(va_keys)

    # 전처리기 — 회전 기준 프로파일은 **학습 분할의 정상 이미지로만** 만든다(누수 방지)
    prep = Preprocessor(args.prep) if args.prep else None
    if prep is not None and prep.use_rot:
        ref_rows = [r for r, y in zip(tr_rows, tr_y) if y == 0]
        prep.fit_reference([np.asarray(arr[r]) for r in ref_rows[:80]])

    out_res = args.res if args.res != src_res else None
    ds_tr = CableClsDataset(arr, tr_rows, tr_y, None, train=True, aug_strength=args.aug,
                            prep=prep, out_res=out_res)
    ds_va = CableClsDataset(arr, va_rows, va_y, None, train=False,
                            prep=prep, out_res=out_res)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       drop_last=False, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       persistent_workers=args.workers > 0)

    model = build_model(args.arch, pretrained=args.pretrained)
    torch.set_num_threads(args.threads)

    if args.class_weight:
        w = torch.tensor([len(tr_y) / (2 * (tr_y == 0).sum()),
                          len(tr_y) / (2 * (tr_y == 1).sum())], dtype=torch.float32)
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps = max(1, len(dl_tr))
    warm = args.warmup_epochs * steps
    total = args.epochs * steps

    def lr_at(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    log_path = exp_root / "logs" / f"fold{fold}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8", newline="")
    logw = csv.writer(logf)
    logw.writerow(["epoch", "lr", "train_loss", "val_loss", "val_auroc", "sec"])

    best = {"auroc": -1.0, "epoch": -1}
    bad = 0
    wdir = exp_root / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    ckpt = wdir / f"fold{fold}_best.pt"

    for ep in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        tr_loss = tot / len(ds_tr)

        model.eval()
        vs, vl = [], 0.0
        with torch.no_grad():
            for x, y in dl_va:
                logits = model(x)
                vl += crit(logits, y).item() * len(y)
                vs.append(torch.softmax(logits, 1)[:, 1].numpy())
        va_scores = np.concatenate(vs)
        va_loss = vl / len(ds_va)
        from sklearn.metrics import roc_auc_score
        va_auroc = float(roc_auc_score(va_y, va_scores))
        dt = time.perf_counter() - t0

        logw.writerow([ep, f"{opt.param_groups[0]['lr']:.6g}", f"{tr_loss:.5f}",
                       f"{va_loss:.5f}", f"{va_auroc:.5f}", f"{dt:.1f}"])
        logf.flush()

        if va_auroc > best["auroc"] + 1e-6:
            best = {"auroc": va_auroc, "epoch": ep, "scores": va_scores.copy()}
            torch.save({"model": model.state_dict(), "epoch": ep, "arch": args.arch,
                        "res": args.res, "auroc": va_auroc}, ckpt)
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"    fold{fold} ep{ep:3d} tr={tr_loss:.4f} va={va_loss:.4f} "
                  f"auroc={va_auroc:.4f} best={best['auroc']:.4f}@{best['epoch']} {dt:.1f}s")
        if bad >= args.patience:
            print(f"    fold{fold} early stop @ep{ep} (patience {args.patience}, "
                  f"best {best['auroc']:.4f}@{best['epoch']})")
            break
    logf.close()

    # ---- 임계값 결정 + 지표 (검증셋에서만, §7.2-4) ----
    scores = best["scores"]
    tau, tau_info = choose_threshold(scores, va_y)
    m = image_level_metrics(scores, va_y, tau)
    m["threshold_policy"] = tau_info
    m["per_class_recall"] = per_class_recall(scores, va_y, va_cls, tau)
    m["best_epoch"] = best["epoch"]
    m["fold"] = fold
    m["checkpoint"] = str(ckpt.relative_to(exp_root))
    if prep is not None:
        m["prep_stats"] = prep.stats.as_dict()
        if prep.use_rot and prep.ref_profile is not None:
            np.save(exp_root / f"ref_profile_fold{fold}.npy", prep.ref_profile)
    return m


# --------------------------------- 메인 -----------------------------------

def main() -> int:
    cfg = load_cfg()
    classes = load_classes(cfg)

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_small")
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--fold", default="all")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--aug", default="medium", choices=["none", "weak", "medium", "robust"])
    ap.add_argument("--prep", default="", help="전처리 spec (예: disk, disk+rot, disk+rot+gray+chrom)")
    ap.add_argument("--src-res", type=int, default=0,
                    help="센서 영상 해상도. 0이면 prep 사용 시 640, 아니면 --res 와 동일")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--class-weight", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.pretrained = bool(args.pretrained)
    args.class_weight = bool(args.class_weight)
    if args.threads <= 0:
        import os
        args.threads = max(1, (os.cpu_count() or 4))
    if args.src_res <= 0:
        args.src_res = 640 if args.prep else args.res
    if args.src_res < args.res:
        raise SystemExit(f"--src-res({args.src_res}) 가 --res({args.res}) 보다 작을 수 없다")

    quant = "fp32"
    stamp = date.today().strftime("%Y%m%d")
    short = {"mobilenet_v3_small": "mnv3s", "mobilenet_v3_large": "mnv3l",
             "shufflenet_v2_x0_5": "sfn05"}.get(args.arch, args.arch)
    folds = list(range(cfg["n_folds"])) if args.fold == "all" else [int(args.fold)]
    fold_tag = "all" if args.fold == "all" else f"f{args.fold}"
    variant = ""
    if args.prep:
        variant += "_prep-" + args.prep.replace("+", "")
    if args.aug != "medium":
        variant += f"_aug-{args.aug}"
    exp_id = (f"EXP_C_{short}_{args.res}_{quant}_{fold_tag}_{stamp}"
              + variant + (f"_{args.tag}" if args.tag else ""))
    exp_root = Path(cfg["runs_dir"]) / exp_id
    exp_root.mkdir(parents=True, exist_ok=True)

    # ---- §11 재현성 기록 ----
    snap = {**vars(args), "seed": cfg["seed"], "n_folds": cfg["n_folds"],
            "track": "C_binary_classification", "exp_id": exp_id,
            "splits": {f"fold{k}": json.loads(
                (Path(cfg["splits_dir"]) / f"fold{k}.json").read_text(encoding="utf-8"))["sha1"]
                for k in folds}}
    (exp_root / "config.yaml").write_text(
        yaml.safe_dump(snap, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (exp_root / "git_rev.txt").write_text(git_rev() + "\n", encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    env = env_stamp()
    (exp_root / "env.txt").write_text(
        f"python: {env['python']}\nexecutable: {env['executable']}\n"
        f"platform: {env['platform']}\ntorch: {torch.__version__}\n"
        f"cuda_available: {torch.cuda.is_available()}\nthreads: {args.threads}\n\n{freeze}",
        encoding="utf-8")

    print(f"[Track C] {exp_id}")
    print(f"  arch={args.arch} res={args.res} epochs={args.epochs} patience={args.patience} "
          f"batch={args.batch} aug={args.aug} pretrained={args.pretrained} threads={args.threads}")

    t_start = time.perf_counter()
    fold_metrics = [run_fold(cfg, args, k, exp_root) for k in folds]

    # ---- §7.1 하드웨어 지표 (정확도와 같은 표에 보고) ----
    model = build_model(args.arch, pretrained=False)
    model.eval()
    hw = model_complexity(model, args.res)
    x1 = torch.zeros(1, 3, args.res, args.res)
    with torch.no_grad():
        hw["latency_model_only"] = measure_latency(lambda: model(x1), warmup=10, iters=50)

    import cv2
    # 실제 센서 영상에 가까운 입력으로 측정한다(0 배열은 원판 검출이 자명해져 전처리 비용을 과소평가)
    _keys = json.loads((Path(cfg["data_root"]) / "cache" / f"keys_{args.src_res}.json")
                       .read_text(encoding="utf-8"))
    _arr = np.load(Path(cfg["data_root"]) / "cache" / f"img_{args.src_res}.npy", mmap_mode="r")
    raw_src = np.asarray(_arr[0])
    raw = cv2.resize(raw_src, (cfg["image_size"], cfg["image_size"]), interpolation=cv2.INTER_LINEAR)
    lat_prep = Preprocessor(args.prep) if args.prep else None
    if lat_prep is not None and lat_prep.use_rot:
        lat_prep.fit_reference([np.asarray(_arr[i]) for i in range(20)])

    def e2e():
        img = cv2.resize(raw, (args.src_res, args.src_res), interpolation=cv2.INTER_AREA)
        if lat_prep is not None and lat_prep.enabled:
            img = lat_prep(img)
        if args.src_res != args.res:
            img = cv2.resize(img, (args.res, args.res), interpolation=cv2.INTER_AREA)
        z = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1)))[None]
        with torch.no_grad():
            logits = model(t)
        return torch.softmax(logits, 1)[0, 1].item()

    hw["latency_end_to_end"] = measure_latency(e2e, warmup=10, iters=50)
    hw["note"] = ("end_to_end = 1024px 원본 → 센서 해상도 리사이즈 + 전처리 + 모델 입력 리사이즈 "
                  "+ 정규화 + 추론. "
                  "개발 머신(CPU) 측정치이며 타깃 하드웨어 수치가 아니다(§9.3).")

    agg_keys = ["auroc", "auprc", "miss_rate", "overkill_rate", "recall",
                "precision", "f1", "balanced_acc", "recall_at_fpr5", "tau"]
    agg = aggregate_folds(fold_metrics, agg_keys)

    # 클래스별 recall 집계 (§7.3)
    cls_names = [classes["multi"][i] for i in sorted(classes["multi"])]
    cls_agg = {}
    for c in cls_names:
        vals = [m["per_class_recall"][c]["recall"] for m in fold_metrics
                if c in m["per_class_recall"]]
        ns = [m["per_class_recall"][c]["n"] for m in fold_metrics if c in m["per_class_recall"]]
        cls_agg[c] = {"n_total": int(sum(ns)),
                      "recall_mean": float(np.mean(vals)) if vals else None,
                      "recall_std": float(np.std(vals)) if vals else None,
                      "per_fold": [float(v) for v in vals]}

    out = {
        "exp_id": exp_id, "track": "C", "date": date.today().isoformat(),
        "config": snap, "hardware": hw,
        "folds": fold_metrics, "aggregate": agg, "per_class_recall": cls_agg,
        "wall_clock_sec": round(time.perf_counter() - t_start, 1),
    }
    (exp_root / "metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[Track C] {exp_id} 완료 ({out['wall_clock_sec']:.0f}s)")
    print(f"  params {hw['params_M']}M / GFLOPs {hw['gflops']} / "
          f"latency e2e p50 {hw['latency_end_to_end']['p50_ms']}ms p95 "
          f"{hw['latency_end_to_end']['p95_ms']}ms")
    for k in ["auroc", "miss_rate", "overkill_rate", "recall_at_fpr5"]:
        a = agg[k]
        print(f"  {k:16s} {a['mean']:.4f} +- {a['std']:.4f}")
    print("  클래스별 recall:")
    for c, v in cls_agg.items():
        if v["recall_mean"] is not None:
            print(f"    {c:24s} n={v['n_total']:2d}  {v['recall_mean']:.3f} +- {v['recall_std']:.3f}")
    print(f"  -> {exp_root / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
