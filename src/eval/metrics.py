"""평가 지표 — SOP-MVTEC-CABLE-001 §7.1, §7.2, §7.3

이미지 레벨 판정 지표 + 임계값 결정 + 클래스별 분해 + 하드웨어 지표.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# §7.2 기본 정책: 과검율 5% 고정 하에서 미검율 최소화
DEFAULT_MAX_FPR = 0.05


def choose_threshold(scores: np.ndarray, labels: np.ndarray,
                     max_fpr: float = DEFAULT_MAX_FPR,
                     margin_tiebreak: bool = True) -> tuple[float, dict]:
    """검증셋에서만 호출할 것 (§7.2-4).

    과검율(FPR) <= max_fpr 제약 하에서 재현율(recall)을 최대화하는 임계값.
    제약을 만족하는 점이 없으면 FPR이 가장 낮은 점을 쓴다.

    margin_tiebreak: 동일한 (FPR, TPR)을 주는 임계값이 여러 개일 때 — 특히 두
    분포가 완전분리되어 점수 사이에 빈 구간이 생길 때 — ROC가 돌려주는 값은
    그 구간의 한쪽 끝에 붙는다. 같은 혼동행렬을 유지하면서 양쪽 점수 군집의
    중앙(로그 스케일 기하평균)으로 옮겨 배포 시 안정성을 확보한다.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    fpr, tpr, thr = roc_curve(labels, scores)
    ok = fpr <= max_fpr + 1e-12
    if ok.any():
        idx = int(np.argmax(np.where(ok, tpr, -np.inf)))
    else:
        idx = int(np.argmin(fpr))
    tau = float(thr[idx])
    if not np.isfinite(tau):  # roc_curve 첫 원소는 +inf
        finite = thr[np.isfinite(thr)]
        tau = float(finite.max()) if len(finite) else 0.5

    info = {"policy": f"FPR<={max_fpr}", "fpr_at_tau": float(fpr[idx]),
            "tpr_at_tau": float(tpr[idx]), "tau_roc": tau, "margin_tiebreak": False}

    if margin_tiebreak:
        pred = scores >= tau
        below = scores[~pred]          # 음성 판정된 점수들
        above = scores[pred]           # 양성 판정된 점수들
        if len(below) and len(above):
            lo, hi = float(below.max()), float(above.min())
            if hi > lo:
                eps = 1e-12
                tau_m = float(np.sqrt(max(lo, eps) * max(hi, eps)))  # 로그 스케일 중앙
                # 혼동행렬이 바뀌지 않는지 확인
                if np.array_equal(scores >= tau_m, pred):
                    info.update(margin_tiebreak=True, tau_roc=tau,
                                gap_low=lo, gap_high=hi,
                                gap_log10_width=float(np.log10(max(hi, eps) / max(lo, eps))))
                    tau = tau_m
    return tau, info


def recall_at_fpr(scores: np.ndarray, labels: np.ndarray,
                  max_fpr: float = DEFAULT_MAX_FPR) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= max_fpr + 1e-12
    return float(np.max(tpr[ok])) if ok.any() else 0.0


def image_level_metrics(scores: np.ndarray, labels: np.ndarray, tau: float) -> dict:
    """labels: 1=NG(불량), 0=OK(정상). scores: 높을수록 NG."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (scores >= tau).astype(int)

    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())

    n_pos, n_neg = tp + fn, fp + tn
    return {
        "n": int(len(labels)), "n_NG": n_pos, "n_OK": n_neg,
        "tau": float(tau),
        "auroc": float(roc_auc_score(labels, scores)) if 0 < n_pos < len(labels) else float("nan"),
        "auprc": float(average_precision_score(labels, scores)) if n_pos else float("nan"),
        "miss_rate": fn / n_pos if n_pos else float("nan"),        # 미검율 — 최우선
        "overkill_rate": fp / n_neg if n_neg else float("nan"),    # 과검율
        "recall": tp / n_pos if n_pos else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(labels, pred)),
        "recall_at_fpr5": recall_at_fpr(scores, labels, DEFAULT_MAX_FPR),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def per_class_recall(scores: np.ndarray, labels: np.ndarray,
                     class_names: list[str], tau: float) -> dict:
    """§7.3 — 불량 클래스별 recall 분해. 전체 평균만으로는 가설 검증 불가."""
    pred = (np.asarray(scores) >= tau).astype(int)
    agg = defaultdict(lambda: {"n": 0, "detected": 0})
    for p, y, c in zip(pred, labels, class_names):
        if y != 1:
            continue
        agg[c]["n"] += 1
        agg[c]["detected"] += int(p == 1)
    return {c: {"n": v["n"], "detected": v["detected"],
                "recall": v["detected"] / v["n"] if v["n"] else float("nan")}
            for c, v in sorted(agg.items())}


# --------------------------- 하드웨어 지표 ---------------------------------

def model_complexity(model, input_size: int) -> dict:
    """파라미터 수 / GFLOPs / FP32 기준 크기."""
    import torch
    n_params = sum(p.numel() for p in model.parameters())
    gflops = float("nan")
    try:
        from thop import profile
        was_training = model.training
        model.eval()
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, input_size, input_size),), verbose=False)
        gflops = float(macs) * 2 / 1e9
        if was_training:
            model.train()
    except Exception:
        pass
    return {
        "params": int(n_params),
        "params_M": round(n_params / 1e6, 3),
        "gflops": round(gflops, 3) if gflops == gflops else None,
        "fp32_size_MB": round(n_params * 4 / 1e6, 2),
    }


def measure_latency(fn, warmup: int = 20, iters: int = 200) -> dict:
    """§7.1 / §9.3 — p50·p95 보고. 평균만 보고하지 않는다."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return {
        "warmup": warmup, "iters": iters,
        "p50_ms": round(float(np.percentile(ts, 50)), 2),
        "p95_ms": round(float(np.percentile(ts, 95)), 2),
        "mean_ms": round(float(ts.mean()), 2),
        "min_ms": round(float(ts.min()), 2),
    }


def aggregate_folds(fold_metrics: list[dict], keys: list[str]) -> dict:
    """§7.4 — 5-fold 평균 ± 표준편차. 단일 fold로 우열 주장 금지."""
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None and m[k] == m[k]]
        if not vals:
            out[k] = None
            continue
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)),
                  "values": [float(v) for v in vals]}
    return out
