"""공통 유틸 — SOP-MVTEC-CABLE-001 §5, §11"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"


def load_cfg(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes(cfg: dict) -> dict:
    with open(cfg["classes_file"], encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        rev = out.stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unavailable"


def env_stamp() -> dict:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "git_rev": git_rev(),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def image_key(split: str, cls: str, stem: str) -> str:
    """전역 고유 키. train/good/000 과 test/good/000 충돌 방지."""
    return f"{split}__{cls}__{stem}"
