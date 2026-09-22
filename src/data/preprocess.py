"""Stage 1 처방 — 값싼 전처리 모듈 (`reports/연구후보_분석.md` §6 Stage 1)

`code_artifact.md`의 아이디어 중 연산 예산을 깨지 않는 것만 구현한다.
전부 고전 CV이며 추가 GFLOPs는 0에 가깝다(측정치는 §latency 참조).

구성 요소 (spec 문자열로 조합, 예: "disk+rot+chrom")

  disk   원판 정규화 — 외피 원판을 검출해 중심을 이미지 중앙으로, 반지름을
         목표 비율로 맞춘다. 이동·스케일 변동을 기하적으로 제거한다.
         (ECC Affine 대신 쓴다 — 배경 화강암 텍스처에 물릴 위험이 없고 더 싸다)
         **한계**: 섭동으로 원판이 화면 밖으로 잘리면 외접원 추정이 편향돼
         복원이 부분적이다. 이동 ±10%에서 중심 이탈 55.7px→32.0px 수준.

  rot    회전 정렬 — 원판 정규화 후 각도별 밝기 프로파일을 기준 템플릿과
         1D 상호상관해 회전각을 추정·보정한다. 그레이스케일 기반이라
         색바램에 영향받지 않는다. 가닥이 3개라 120° 주기 모호성이 있어
         탐색 범위를 ±60°로 제한한다(실측 회전 범위는 ±24°).

  gray   그레이월드 정규화 — 채널별 평균으로 나눠 색 캐스트 제거.
         색바램/황변(채널 게인)의 정확한 역연산이다.

  chrom  색도 정규화 — 화소별 c/(R+G+B). **조도(밝기)** 변동에 불변.
         출력 색도가 입력 색도와 항등이므로 채널 게인은 제거하지 못한다 —
         색바램 대응은 gray 가 담당한다.
         휘도 정보를 완전히 버리지 않도록 원 휘도를 별도 채널 스케일로 복원한다.

  spec   정반사 억제 — **넓은** 포화 덩어리만 제거한다(연결요소 면적 기준).
         이 데이터의 포화 화소 6.28%는 대부분 구리 소선의 고유 반사이므로
         단순 임계 인페인팅은 정상 특징을 지운다. 크기로 둘을 가르고,
         채움은 마스크 인식 저해상도 블러로 처리해 비용을 낮춘다.
         Stage 0에서 정반사가 최대 취약점(면적1%에 AUROC 0.964, 10%에 과검 100%)
         으로 드러나 우선 처방 대상이 되었다.

모든 단계는 검출 실패 시 항등 변환으로 안전하게 후퇴하며, 실패 건수를 집계한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

TARGET_DISK_FRAC = 0.87   # 정규화 후 원판 지름 / 이미지 변
ROT_SEARCH_DEG = 60       # 회전 탐색 범위 (가닥 3개 → 120° 주기 모호성 회피)
ANG_BINS = 360
SPEC_THR = 250


@dataclass
class PrepStats:
    n: int = 0
    disk_fail: int = 0
    rot_fail: int = 0

    def as_dict(self) -> dict:
        return {"n": self.n, "disk_fail": self.disk_fail, "rot_fail": self.rot_fail,
                "disk_fail_rate": self.disk_fail / self.n if self.n else 0.0,
                "rot_fail_rate": self.rot_fail / self.n if self.n else 0.0}


def detect_disk(img_rgb: np.ndarray):
    """외피 원판 → (cx, cy, r). 실패 시 None."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[k, cv2.CC_STAT_AREA]
    h, w = gray.shape
    if area < 0.05 * h * w:          # 너무 작으면 검출 실패로 본다
        return None
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    if r < 0.15 * min(h, w) or r > 1.2 * min(h, w):
        return None
    return float(cx), float(cy), float(r)


def disk_normalize(img: np.ndarray, disk=None, target_frac: float = TARGET_DISK_FRAC):
    """원판 중심을 이미지 중앙으로, 지름을 target_frac 비율로 맞춘다."""
    h, w = img.shape[:2]
    if disk is None:
        disk = detect_disk(img)
    if disk is None:
        return img, False
    cx, cy, r = disk
    target_r = target_frac * min(h, w) / 2.0
    s = target_r / max(r, 1e-6)
    M = np.array([[s, 0, w / 2 - s * cx],
                  [0, s, h / 2 - s * cy]], dtype=np.float32)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return out, True


def angular_profile(img: np.ndarray, bins: int = ANG_BINS) -> np.ndarray:
    """원판 정규화된 이미지의 각도별 평균 밝기 프로파일 (그레이스케일)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    R = int(TARGET_DISK_FRAC * min(h, w) / 2)
    polar = cv2.warpPolar(gray, (R, bins), (w / 2, h / 2), R, cv2.WARP_POLAR_LINEAR)
    # 중심부(구리 소선 무작위성)를 빼고 0.35R~0.92R 링만 사용.
    # 대안(0.10-0.60, 0.20-0.75, HSV 채도)을 모두 시험했고 이 조합이 최소 오차였다.
    band = polar[:, int(0.35 * R):int(0.92 * R)]
    p = band.mean(axis=1)
    return p - p.mean()


def estimate_rotation(prof: np.ndarray, ref_prof: np.ndarray,
                      search_deg: int = ROT_SEARCH_DEG) -> float | None:
    """1D 원형 상호상관으로 회전각(도) 추정. 탐색 범위 밖이면 None."""
    n = len(prof)
    # 평탄 상호상관을 쓴다. 위상 백색화(phase correlation)를 쓰면 저진폭 고주파가
    # 증폭돼 추정이 무너진다 — 실측 복원 오차 16.2° vs 2.9° (scratchpad/rot_test.py)
    corr = np.fft.irfft(np.fft.rfft(prof) * np.conj(np.fft.rfft(ref_prof)), n)
    span = int(search_deg * n / 360)
    idx = np.concatenate([np.arange(0, span + 1), np.arange(n - span, n)])
    k = idx[int(np.argmax(corr[idx]))]
    if corr[k] <= 0:
        return None
    deg = k * 360.0 / n
    if deg > 180:
        deg -= 360.0
    return float(deg)


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def grayworld_normalize(img: np.ndarray) -> np.ndarray:
    """채널별 평균으로 나눠 색 캐스트를 제거한다 (gray-world 가정).

    색바램/황변은 채널별 게인 g_c 로 모형화되는데, 이 연산이 그 **정확한 역연산**이다.
    색도 정규화(c/(R+G+B))는 출력의 색도가 입력 색도와 항등이므로 채널 게인을
    제거하지 못한다 — 조도(밝기) 변동에만 유효하다. 둘은 상호 보완적이다.
    """
    # 평균은 축소본에서 구하고(통계량이므로 충분), 적용은 채널별 256엔트리 LUT로 한다.
    # float32 전체 변환보다 훨씬 싸다.
    smallv = img[::4, ::4] if min(img.shape[:2]) >= 256 else img
    mu = smallv.reshape(-1, 3).mean(0).astype(np.float32) + 1e-6
    gain = float(mu.mean()) / mu
    ramp = np.arange(256, dtype=np.float32)
    lut = np.stack([np.clip(ramp * g, 0, 255) for g in gain], axis=1).astype(np.uint8)
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = cv2.LUT(img[..., c], lut[:, c])
    return out


def chromaticity_normalize(img: np.ndarray, keep_luma: bool = True) -> np.ndarray:
    """화소별 c/(R+G+B). 조도·색바램 불변. keep_luma면 전역 휘도를 되살린다."""
    f = img.astype(np.float32) + 1.0
    s = f.sum(axis=2, keepdims=True)
    c = f / s                                   # 0~1, 합=1
    out = c * 3.0 * 85.0                        # 평균 밝기 ~85로 스케일
    if keep_luma:
        luma = f.mean(axis=2, keepdims=True)
        g = float(np.mean(luma)) / 128.0
        out = out * (0.5 + 0.5 * g)
    return np.clip(out, 0, 255).astype(np.uint8)


SPEC_STD_THR = 2.0            # 포화영역 국소 표준편차 임계 (이하 = 매끄러운 외란 하이라이트)
SPEC_STD_WIN = 9              # 국소 표준편차 창 크기
SPEC_MIN_AREA_FRAC = 0.0015   # 이보다 작은 덩어리는 무시
SPEC_DILATE_FRAC = 0.012      # 흐릿한 가장자리(rim)까지 덮기 위한 팽창 반경


def _local_std(gray: np.ndarray, k: int = SPEC_STD_WIN) -> np.ndarray:
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k))
    m2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def specular_suppress(img: np.ndarray, thr: int = SPEC_THR,
                      std_thr: float = SPEC_STD_THR,
                      min_area_frac: float = SPEC_MIN_AREA_FRAC) -> np.ndarray:
    """**매끄러운** 정반사 하이라이트만 제거한다.

    설계 근거 (실측):
      - 이 데이터의 포화 화소 평균 6.28%는 대부분 **구리 소선의 고유 반사**다.
        단순 임계 + 인페인팅은 이를 지워 무섭동 이미지를 25.75dB까지 훼손했고
        복원 효과도 없었다(면적1%에서 -1.37dB).
      - 연결요소 **면적**으로 가르는 방법도 실패했다 — 구리 반사가 소선 코어 안에서
        큰 덩어리로 뭉치기 때문(무섭동 훼손 21.22dB로 오히려 악화).
      - 유효한 판별자는 **국소 표준편차**다. 외란 하이라이트 내부는 평탄하고
        (중앙값 0.36) 구리 반사는 소선 단위로 거칠다(중앙값 6.12) — 17배 차이.
      - 채움은 마스크 인식 저해상도 블러로 처리한다(인페인팅보다 훨씬 싸다).
    """
    h, w = img.shape[:2]
    # 마스크는 절반 해상도에서 구한다 — 결과가 사실상 같으면서 비용이 1/4이다.
    # (덩어리는 크고 매끄러워 축소해도 형태가 보존된다)
    ds = 2 if min(h, w) >= 384 else 1
    small_img = img[::ds, ::ds] if ds > 1 else img
    sh, sw = small_img.shape[:2]

    gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
    sat = small_img.max(axis=2) >= thr
    if not sat.any():
        return img
    smooth = _local_std(gray, max(3, SPEC_STD_WIN // ds | 1)) <= std_thr
    mask = (sat & smooth).astype(np.uint8)
    if mask.sum() < min_area_frac * sh * sw:
        return img

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = min_area_frac * sh * sw
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    if not keep.any():
        return img
    mask = keep[lab].astype(np.uint8)

    k = max(3, int(SPEC_DILATE_FRAC * min(sh, sw)) | 1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if ds > 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        k = max(3, int(SPEC_DILATE_FRAC * min(h, w)) | 1)

    small = max(8, min(h, w) // 32)
    valid = (1 - mask).astype(np.float32)[..., None]
    num = cv2.resize(img.astype(np.float32) * valid, (small, small), interpolation=cv2.INTER_AREA)
    den = cv2.resize(valid, (small, small), interpolation=cv2.INTER_AREA)
    if den.ndim == 2:
        den = den[..., None]
    fill = cv2.resize(num / np.maximum(den, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)

    soft = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)[..., None]
    out = img.astype(np.float32) * (1 - soft) + fill * soft
    return np.clip(out, 0, 255).astype(np.uint8)


class Preprocessor:
    """spec 문자열로 구성 요소를 조합한다. 예: "disk", "disk+rot", "disk+rot+chrom"."""

    def __init__(self, spec: str = "", ref_profile: np.ndarray | None = None):
        self.spec = (spec or "").strip().lower()
        parts = [p for p in self.spec.split("+") if p]
        unknown = set(parts) - {"disk", "rot", "gray", "chrom", "spec", "none"}
        if unknown:
            raise ValueError(f"미지원 전처리 요소: {sorted(unknown)}")
        self.use_disk = "disk" in parts
        self.use_rot = "rot" in parts
        self.use_gray = "gray" in parts
        self.use_chrom = "chrom" in parts
        self.use_spec = "spec" in parts
        self.ref_profile = ref_profile
        self.stats = PrepStats()
        if self.use_rot and not self.use_disk:
            raise ValueError("rot 은 disk 정규화를 전제로 한다 (spec에 disk 포함 필요)")

    @property
    def enabled(self) -> bool:
        return self.use_disk or self.use_gray or self.use_chrom or self.use_spec

    def fit_reference(self, images) -> None:
        """정상 이미지들로 회전 기준 프로파일을 만든다 (학습 분할에서만 호출)."""
        profs = []
        for im in images:
            norm, ok = disk_normalize(im)
            if ok:
                profs.append(angular_profile(norm))
        if not profs:
            raise RuntimeError("기준 프로파일 생성 실패 — 원판 검출 0건")
        self.ref_profile = np.mean(np.stack(profs), axis=0)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        self.stats.n += 1
        if self.use_disk:
            img, ok = disk_normalize(img)
            if not ok:
                self.stats.disk_fail += 1
            elif self.use_rot and self.ref_profile is not None:
                deg = estimate_rotation(angular_profile(img), self.ref_profile)
                if deg is None:
                    self.stats.rot_fail += 1
                else:
                    img = rotate(img, deg)
        if self.use_spec:
            img = specular_suppress(img)
        if self.use_gray:
            img = grayworld_normalize(img)
        if self.use_chrom:
            img = chromaticity_normalize(img)
        return img
