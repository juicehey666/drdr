"""Randomness and runtime provenance helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import canonical_yaml, config_sha256


def seed_everything(seed: int, *, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def create_run_directory(
    project_root: Path, config: Mapping[str, Any], run_id: str
) -> Path:
    output_root = project_root / str(config["output"]["root"])
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    return run_dir


def write_run_provenance(
    *,
    project_root: Path,
    run_dir: Path,
    config: Mapping[str, Any],
    argv: list[str],
    device: torch.device,
) -> None:
    (run_dir / "config.resolved.yaml").write_text(
        canonical_yaml(config), encoding="utf-8"
    )
    metadata = {
        "config_sha256": config_sha256(config),
        "command": argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "selected_device": str(device),
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_value(project_root, "status", "--porcelain")),
        "pid": os.getpid(),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
