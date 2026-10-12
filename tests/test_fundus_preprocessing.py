from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from dr_grading.preprocessing.fundus import (
    MANIFEST_COLUMNS,
    detect_fundus_bbox,
    load_config,
    preprocess_fundus_image,
    run_preprocessing,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/data/fundus_512_v1.yaml"


def _circle_image(size: tuple[int, int] = (800, 600), radius: int = 270) -> Image.Image:
    image = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    cx, cy = size[0] // 2, size[1] // 2
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(120, 45, 25),
    )
    return image


def test_protocol_config_is_locked() -> None:
    config = load_config(CONFIG_PATH)
    assert config["version"] == "fundus_512_v1"
    assert config["detection"]["luma_threshold_8bit"] == 10
    assert config["detection"]["bbox_expansion_fraction"] == 0.02
    assert config["output"]["size"] == 512
    assert config["output"]["jpeg_quality"] == 95
    assert config["output"]["jpeg_subsampling"] == 0
    assert config["normalization"]["applied_offline"] is False


def test_detects_centered_fundus_without_cutting_circle() -> None:
    config = load_config(CONFIG_PATH)
    image = _circle_image()
    decision = detect_fundus_bbox(image, config["detection"])
    assert not decision.fallback
    x0, y0, x1, y1 = decision.bbox
    # Original circle bounds are x=[130,670], y=[30,570].
    assert x0 <= 130 and x1 >= 671
    assert y0 <= 30 and y1 >= 571
    assert decision.foreground_fraction >= 0.30


def test_small_foreground_triggers_protocol_fallback() -> None:
    config = load_config(CONFIG_PATH)
    image = _circle_image(radius=60)
    decision = detect_fundus_bbox(image, config["detection"])
    assert decision.fallback
    assert decision.bbox == (0, 0, image.width, image.height)
    assert "foreground_fraction_below_0.30" in decision.fallback_reason


def test_preprocess_produces_rgb_512_without_changing_source(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    source = tmp_path / "source.jpg"
    _circle_image().save(source, format="JPEG", quality=95)
    before = source.read_bytes()
    output, decision, original_size = preprocess_fundus_image(
        source, config["detection"], config["output"]
    )
    assert output.mode == "RGB"
    assert output.size == (512, 512)
    assert original_size == (800, 600)
    assert not decision.fallback
    assert source.read_bytes() == before


def test_non_rgb_source_is_rejected_instead_of_silently_converted(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    source = tmp_path / "grayscale.png"
    Image.new("L", (600, 600), 100).save(source)
    with pytest.raises(ValueError, match="must decode as RGB"):
        preprocess_fundus_image(source, config["detection"], config["output"])


def test_square_padding_uses_black_background(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    source = tmp_path / "wide.jpg"
    Image.new("RGB", (800, 400), (80, 30, 20)).save(source, quality=95)
    output, decision, _ = preprocess_fundus_image(
        source, config["detection"], config["output"]
    )
    assert decision.fallback  # full non-square frame violates bbox aspect rule
    array = np.asarray(output)
    assert tuple(array[0, 0]) == (0, 0, 0)
    assert tuple(array[-1, -1]) == (0, 0, 0)


def test_dry_run_writes_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    output_root = tmp_path / "dry_run"
    summary = run_preprocessing(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        output_root=output_root,
        datasets=("brset", "mbrset"),
        limit_per_dataset=1,
    )
    assert summary["record_count"] == 2
    assert summary["ok_count"] == 2
    assert summary["error_count"] == 0
    manifest = output_root / "preprocessing_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == MANIFEST_COLUMNS
    assert {row["dataset"] for row in rows} == {"brset", "mbrset"}
    assert all(row["status"] == "ok" for row in rows)
    assert all(len(row["source_sha256"]) == 64 for row in rows)
    assert all(len(row["output_sha256"]) == 64 for row in rows)
    assert all(Path(row["output_path"]).is_absolute() for row in rows)
    assert all(Path(row["output_path"]).is_file() for row in rows)
    # Project outputs are relative; isolated temp-test outputs are absolute so
    # that the manifest never records a misleading relative path.
    assert all((output_root / row["dataset"]).is_dir() for row in rows)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_preprocessing(
            project_root=PROJECT_ROOT,
            config_path=CONFIG_PATH,
            output_root=output_root,
            datasets=("brset",),
            limit_per_dataset=1,
        )
