"""지연시간 정밀 측정 — SOP-MVTEC-CABLE-001 §7.1, §9.3

파이프라인 단계별로 분해 측정한다. 전처리(Stage 1 처방)가 DoD 지연 예산을
얼마나 잡아먹는지 판단하려면 모델 추론과 분리해서 봐야 한다.

§1.3 DoD는 **단일 스레드** 기준 100 ms이므로 1-스레드와 전-스레드를 모두 잰다.
§9.3에 따라 워밍업 20회 후 200회 측정, p50·p95를 보고한다(평균만 보고하지 않음).

측정 전 시스템 부하를 확인해 경합 여부를 기록한다 — 경합 중 측정치는 무효다.

산출물: reports/latency_cable.md
사용:   python src/eval/bench_latency.py [--iters 200] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import load_cfg  # noqa: E402
from data.preprocess import Preprocessor  # noqa: E402
from eval.metrics import model_complexity  # noqa: E402
from train.train_cls import IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402

SRC_RES = 640
CONFIGS = [
    # (모델 입력, 전처리 spec)
    (256, ""), (256, "gray"), (256, "spec"), (256, "gray+spec"), (256, "disk+rot"),
    (384, ""), (384, "gray+spec"),
    (640, ""), (640, "gray+spec"),
]


def cpu_load_pct(sample_s: float = 1.0) -> float:
    """측정 직전 시스템 CPU 사용률 — 경합 여부 판정용."""
    try:
        import psutil
        return float(psutil.cpu_percent(interval=sample_s))
    except Exception:
        return float("nan")


def timeit(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2),
            "min": round(float(a.min()), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.iters, args.warmup = 40, 10

    cfg = load_cfg()
    cache = Path(cfg["data_root"]) / "cache"
    arr = np.load(cache / f"img_{SRC_RES}.npy", mmap_mode="r")
    sample = np.asarray(arr[0])
    raw1024 = cv2.resize(sample, (cfg["image_size"], cfg["image_size"]),
                         interpolation=cv2.INTER_LINEAR)

    load_before = cpu_load_pct(1.0)
    n_cpu = torch.get_num_threads()
    print(f"측정 시작 — 직전 CPU 사용률 {load_before:.1f} %, 가용 스레드 {n_cpu}")
    if load_before > 25:
        print("  ⚠ 경합 감지: 다른 작업이 CPU를 쓰고 있다. 측정치가 부풀려진다.")

    rows = []
    for res, spec in CONFIGS:
        prep = Preprocessor(spec) if spec else None
        if prep is not None and prep.use_rot:
            prep.fit_reference([np.asarray(arr[i]) for i in range(30)])

        model = build_model("mobilenet_v3_small", pretrained=False).eval()
        hw = model_complexity(model, res)

        def stage_sensor():
            return cv2.resize(raw1024, (SRC_RES, SRC_RES), interpolation=cv2.INTER_AREA)

        sensor_img = stage_sensor()

        def stage_prep():
            return prep(sensor_img) if prep is not None else sensor_img

        prepped = stage_prep()

        def stage_resize():
            return (cv2.resize(prepped, (res, res), interpolation=cv2.INTER_AREA)
                    if prepped.shape[0] != res else prepped)

        small = stage_resize()

        def stage_norm():
            z = (small.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1)))[None]

        tensor = stage_norm()

        for threads in (1, n_cpu):
            torch.set_num_threads(threads)

            def stage_model():
                with torch.no_grad():
                    return torch.softmax(model(tensor), 1)[0, 1].item()

            def e2e():
                img = cv2.resize(raw1024, (SRC_RES, SRC_RES), interpolation=cv2.INTER_AREA)
                if prep is not None:
                    img = prep(img)
                if img.shape[0] != res:
                    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
                z = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                t = torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1)))[None]
                with torch.no_grad():
                    return torch.softmax(model(t), 1)[0, 1].item()

            r = {
                "res": res, "prep_spec": spec or "—", "threads": threads,
                "gflops": hw["gflops"], "params_M": hw["params_M"],
                "sensor_resize": timeit(stage_sensor, args.warmup, args.iters),
                "prep_ms": timeit(stage_prep, args.warmup, args.iters) if prep else None,
                "input_resize": timeit(stage_resize, args.warmup, args.iters),
                "normalize": timeit(stage_norm, args.warmup, args.iters),
                "model": timeit(stage_model, args.warmup, args.iters),
                "e2e": timeit(e2e, args.warmup, args.iters),
            }
            rows.append(r)
            pm = r["prep_ms"]["p50"] if r["prep_ms"] else 0.0
            print(f"  res={res:3d} prep={spec or '—':10s} threads={threads:2d}  "
                  f"e2e p50 {r['e2e']['p50']:7.2f} / p95 {r['e2e']['p95']:7.2f} ms  "
                  f"(전처리 {pm:6.2f}, 모델 {r['model']['p50']:6.2f})")

    load_after = cpu_load_pct(1.0)
    torch.set_num_threads(n_cpu)

    # ---------------- 리포트 ----------------
    md: list[str] = []
    A = md.append
    A("# 지연시간 정밀 측정 — Track C 파이프라인")
    A("")
    A("- SOP: SOP-MVTEC-CABLE-001 §7.1, §9.3")
    A(f"- 생성일: {date.today().isoformat()}")
    A(f"- 워밍업 {args.warmup}회 후 {args.iters}회 측정, p50·p95 보고 (§9.3-2)")
    A(f"- 측정 직전/직후 시스템 CPU 사용률: {load_before:.1f} % / {load_after:.1f} %")
    A(f"- 개발 머신 CPU, 가용 스레드 {n_cpu}")
    A("")
    if max(load_before, load_after) > 25:
        A("> ⚠ **경합 상태에서 측정됐다. 수치가 부풀려져 있으니 유휴 상태에서 재측정할 것.**")
    else:
        A("> 유휴 상태에서 측정됐다(CPU 사용률 25% 미만).")
    A("")
    A("> **이것은 개발 머신 값이며 타깃 하드웨어 수치가 아니다.** "
      "§1.3 DoD(단일 스레드 100 ms)의 최종 판정은 타깃 하드웨어에서 "
      "§9.3 전체 프로토콜(열 스로틀링 10분 연속 부하 포함)로 해야 한다.")
    A("")
    A("## 1. 단일 스레드 (DoD 기준 조건)")
    A("")
    A("| 입력 | 전처리 | GFLOPs | 센서리사이즈 | 전처리 | 입력리사이즈 | 정규화 | 모델 | **e2e p50** | e2e p95 | DoD 100ms |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in [x for x in rows if x["threads"] == 1]:
        pm = r["prep_ms"]["p50"] if r["prep_ms"] else 0.0
        A(f"| {r['res']} | `{r['prep_spec']}` | {r['gflops']} | {r['sensor_resize']['p50']} | "
          f"{pm:.2f} | {r['input_resize']['p50']} | {r['normalize']['p50']} | {r['model']['p50']} | "
          f"**{r['e2e']['p50']}** | {r['e2e']['p95']} | "
          f"{'충족' if r['e2e']['p50'] <= 100 else '초과'} |")
    A("")
    A("## 2. 전 스레드")
    A("")
    A("| 입력 | 전처리 | 전처리 ms | 모델 ms | e2e p50 | e2e p95 |")
    A("|---|---|---|---|---|---|")
    for r in [x for x in rows if x["threads"] != 1]:
        pm = r["prep_ms"]["p50"] if r["prep_ms"] else 0.0
        A(f"| {r['res']} | `{r['prep_spec']}` | {pm:.2f} | "
          f"{r['model']['p50']} | {r['e2e']['p50']} | {r['e2e']['p95']} |")
    A("")
    A("## 3. 전처리 비용 증분")
    A("")
    A("| 입력 | 스레드 | 전처리 없음 e2e | gray+spec e2e | 증분 | 증분 비율 |")
    A("|---|---|---|---|---|---|")
    for res in (256, 384, 640):
        for th in (1, n_cpu):
            b = next((x for x in rows if x["res"] == res and x["prep_spec"] == "—" and x["threads"] == th), None)
            p = next((x for x in rows if x["res"] == res and x["prep_spec"] == "gray+spec" and x["threads"] == th), None)
            if b and p:
                d = p["e2e"]["p50"] - b["e2e"]["p50"]
                A(f"| {res} | {th} | {b['e2e']['p50']} | {p['e2e']['p50']} | {d:+.2f} ms | "
                  f"{d / b['e2e']['p50'] * 100:+.1f} % |")
    A("")
    A("## 4. 관찰")
    A("")
    one = [x for x in rows if x["threads"] == 1]
    if one:
        worst = max(one, key=lambda x: x["e2e"]["p50"])
        best = min(one, key=lambda x: x["e2e"]["p50"])
        A(f"- 단일 스레드 최저 지연: {best['res']}px / `{best['prep_spec']}` → {best['e2e']['p50']} ms")
        A(f"- 단일 스레드 최대 지연: {worst['res']}px / `{worst['prep_spec']}` → {worst['e2e']['p50']} ms")
        over = [x for x in one if x["e2e"]["p50"] > 100]
        if over:
            A(f"- **DoD 100 ms 초과 설정**: "
              + ", ".join(f"{x['res']}px/`{x['prep_spec']}`({x['e2e']['p50']}ms)" for x in over))
        else:
            A("- 단일 스레드에서 모든 설정이 DoD 100 ms 이내 (개발 머신 기준)")
    A("- 센서 리사이즈(1024→640)는 전 설정 공통 고정비다. 카메라에서 저해상도로 직접 "
      "취득하면 이 비용이 사라진다.")
    A("")

    out = Path(cfg["reports_dir"]) / "latency_cable.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    (Path(cfg["reports_dir"]) / "latency_cable.json").write_text(
        json.dumps({"cpu_load_before": load_before, "cpu_load_after": load_after,
                    "threads_available": n_cpu, "iters": args.iters,
                    "warmup": args.warmup, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
