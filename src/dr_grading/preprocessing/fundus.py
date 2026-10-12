"""Implementation of the deterministic ``fundus_512_v1`` pipeline.

The raw image is never modified.  Foreground detection is performed on a
downsampled preview, while cropping and square padding use the original RGB
pixels.  Training-time augmentation and backbone normalization intentionally
do not belong in this offline pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps, __version__ as pillow_version


MANIFEST_COLUMNS = [
    "dataset",
    "dataset_version",
    "image_id",
    "patient_id",
    "laterality",
    "camera",
    "icdr",
    "quality",
    "original_width",
    "original_height",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "foreground_fraction",
    "crop_fallback",
    "fallback_reason",
    "output_path",
    "source_sha256",
    "output_sha256",
    "status",
    "error",
]


@dataclass(frozen=True)
class CropDecision:
    """Crop coordinates use PIL's half-open ``(x0, y0, x1, y1)`` convention."""

    bbox: tuple[int, int, int, int]
    foreground_fraction: float
    fallback: bool
    fallback_reason: str
    center_covered: bool


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    dataset_version: str
    image_id: str
    patient_id: str
    laterality: str
    camera: str
    icdr: str
    quality: str
    source_path: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "fundus_512_v1":
        raise ValueError("This implementation only accepts version=fundus_512_v1")

    detection = config["detection"]
    output = config["output"]
    expected = {
        "preview_long_side": 512,
        "luma_threshold_8bit": 10,
        "bbox_expansion_fraction": 0.02,
        "min_foreground_fraction": 0.30,
        "min_bbox_width_fraction": 0.50,
        "min_bbox_height_fraction": 0.50,
        "bbox_aspect_ratio_min": 0.75,
        "bbox_aspect_ratio_max": 1.33,
    }
    for key, expected_value in expected.items():
        if detection.get(key) != expected_value:
            raise ValueError(
                f"Protocol parameter {key} must be {expected_value!r}, "
                f"got {detection.get(key)!r}"
            )
    if output.get("size") != 512:
        raise ValueError("Protocol output.size must be 512")
    if output.get("interpolation") != "bicubic":
        raise ValueError("Protocol output.interpolation must be bicubic")
    if output.get("format") != "JPEG":
        raise ValueError("Protocol output.format must be JPEG")
    if output.get("jpeg_quality") != 95 or output.get("jpeg_subsampling") != 0:
        raise ValueError("Protocol requires JPEG quality=95 and subsampling=0")
    if output.get("background_rgb") != [0, 0, 0]:
        raise ValueError("Protocol requires pure-black square padding")
    if detection.get("require_center_coverage") is not True:
        raise ValueError("Protocol requires center coverage")
    if config["normalization"].get("applied_offline") is not False:
        raise ValueError("Offline normalization is forbidden by the protocol")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _preview_rgb(image: Image.Image, long_side: int) -> np.ndarray:
    width, height = image.size
    scale = min(1.0, long_side / max(width, height))
    preview_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    preview = image.resize(preview_size, Image.Resampling.BILINEAR)
    return np.asarray(preview, dtype=np.uint8)


def _select_component(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, bool]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=connectivity
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), False

    center_y = mask.shape[0] // 2
    center_x = mask.shape[1] // 2
    center_label = int(labels[center_y, center_x])
    if center_label > 0:
        chosen = center_label
        center_covered = True
    else:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        chosen = 1 + int(np.argmax(component_areas))
        center_covered = False
    return labels == chosen, center_covered


def detect_fundus_bbox(
    image: Image.Image, detection: Mapping[str, Any]
) -> CropDecision:
    """Detect and validate a fundus crop on an EXIF-corrected RGB image."""

    if image.mode != "RGB":
        raise ValueError(f"Expected RGB image, got {image.mode}")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")

    preview = _preview_rgb(image, int(detection["preview_long_side"]))
    luma = (
        0.299 * preview[..., 0].astype(np.float32)
        + 0.587 * preview[..., 1].astype(np.float32)
        + 0.114 * preview[..., 2].astype(np.float32)
    )
    mask = (luma > float(detection["luma_threshold_8bit"])).astype(np.uint8)

    kernel_size = int(detection["closing_kernel_size"])
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=int(detection["closing_iterations"]),
    )
    # Flood-fill the inverse background from the padded border; remaining
    # inverse pixels are holes and are merged into the foreground.
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    filled = (mask.astype(bool) | holes).astype(np.uint8)

    component, center_covered = _select_component(
        filled, int(detection["connectivity"])
    )
    foreground_fraction = float(component.mean())
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return CropDecision(
            bbox=(0, 0, width, height),
            foreground_fraction=0.0,
            fallback=True,
            fallback_reason="no_foreground_component",
            center_covered=False,
        )

    px0, py0 = int(xs.min()), int(ys.min())
    px1, py1 = int(xs.max()) + 1, int(ys.max()) + 1
    expansion = float(detection["bbox_expansion_fraction"])
    pad_x = int(math.ceil((px1 - px0) * expansion))
    pad_y = int(math.ceil((py1 - py0) * expansion))
    px0 = max(0, px0 - pad_x)
    py0 = max(0, py0 - pad_y)
    px1 = min(preview.shape[1], px1 + pad_x)
    py1 = min(preview.shape[0], py1 + pad_y)

    scale_x = width / preview.shape[1]
    scale_y = height / preview.shape[0]
    x0 = max(0, int(math.floor(px0 * scale_x)))
    y0 = max(0, int(math.floor(py0 * scale_y)))
    x1 = min(width, int(math.ceil(px1 * scale_x)))
    y1 = min(height, int(math.ceil(py1 * scale_y)))

    bbox_width_fraction = (x1 - x0) / width
    bbox_height_fraction = (y1 - y0) / height
    aspect_ratio = (x1 - x0) / max(1, y1 - y0)
    reasons: list[str] = []
    if foreground_fraction < float(detection["min_foreground_fraction"]):
        reasons.append("foreground_fraction_below_0.30")
    if bbox_width_fraction < float(detection["min_bbox_width_fraction"]):
        reasons.append("bbox_width_fraction_below_0.50")
    if bbox_height_fraction < float(detection["min_bbox_height_fraction"]):
        reasons.append("bbox_height_fraction_below_0.50")
    if not (
        float(detection["bbox_aspect_ratio_min"])
        <= aspect_ratio
        <= float(detection["bbox_aspect_ratio_max"])
    ):
        reasons.append("bbox_aspect_ratio_outside_0.75_1.33")
    if bool(detection["require_center_coverage"]) and not center_covered:
        reasons.append("center_not_covered")

    if reasons:
        return CropDecision(
            bbox=(0, 0, width, height),
            foreground_fraction=foreground_fraction,
            fallback=True,
            fallback_reason=";".join(reasons),
            center_covered=center_covered,
        )
    return CropDecision(
        bbox=(x0, y0, x1, y1),
        foreground_fraction=foreground_fraction,
        fallback=False,
        fallback_reason="",
        center_covered=center_covered,
    )


def _square_crop_with_padding(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    background_rgb: Sequence[int],
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop_width, crop_height = x1 - x0, y1 - y0
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"Invalid crop bbox: {bbox}")
    side = max(crop_width, crop_height)
    square_x0 = x0 - (side - crop_width) // 2
    square_y0 = y0 - (side - crop_height) // 2
    square_x1 = square_x0 + side
    square_y1 = square_y0 + side

    src_x0 = max(0, square_x0)
    src_y0 = max(0, square_y0)
    src_x1 = min(image.width, square_x1)
    src_y1 = min(image.height, square_y1)
    patch = image.crop((src_x0, src_y0, src_x1, src_y1))
    canvas = Image.new("RGB", (side, side), tuple(int(v) for v in background_rgb))
    canvas.paste(patch, (src_x0 - square_x0, src_y0 - square_y0))
    return canvas


def preprocess_fundus_image(
    source_path: Path,
    detection: Mapping[str, Any],
    output: Mapping[str, Any],
) -> tuple[Image.Image, CropDecision, tuple[int, int]]:
    """Read, EXIF-correct, crop, square-pad and resize one source image."""

    with Image.open(source_path) as raw:
        corrected = ImageOps.exif_transpose(raw)
        if corrected.mode != "RGB":
            raise ValueError(
                f"Source image must decode as RGB; got mode={corrected.mode!r}: "
                f"{source_path}"
            )
        rgb = corrected.copy()
        original_size = rgb.size
        decision = detect_fundus_bbox(rgb, detection)
        square = _square_crop_with_padding(
            rgb, decision.bbox, output["background_rgb"]
        )
        resized = square.resize(
            (int(output["size"]), int(output["size"])),
            Image.Resampling.BICUBIC,
        )
        return resized, decision, original_size


def _clean_csv_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _portable_output_path(path: Path, project_root: Path) -> str:
    """Prefer a project-relative path, while supporting isolated temp tests."""

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def load_dataset_records(
    project_root: Path, dataset: str, spec: Mapping[str, Any]
) -> list[ImageRecord]:
    metadata_path = project_root / str(spec["metadata_csv"])
    image_dir = project_root / str(spec["image_dir"])
    records: list[ImageRecord] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            str(spec["image_column"]),
            str(spec["patient_column"]),
            str(spec["laterality_column"]),
            str(spec["icdr_column"]),
            str(spec["quality_column"]),
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")
        for row in reader:
            raw_name = _clean_csv_value(row[str(spec["image_column"])])
            extension = str(spec.get("image_extension", ""))
            filename = raw_name if raw_name.lower().endswith(extension.lower()) else raw_name + extension
            camera = (
                _clean_csv_value(row[str(spec["camera_column"])])
                if "camera_column" in spec
                else str(spec["camera_constant"])
            )
            records.append(
                ImageRecord(
                    dataset=dataset,
                    dataset_version=str(spec["dataset_version"]),
                    image_id=raw_name,
                    patient_id=_clean_csv_value(row[str(spec["patient_column"])]),
                    laterality=_clean_csv_value(row[str(spec["laterality_column"])]),
                    camera=camera,
                    icdr=_clean_csv_value(row[str(spec["icdr_column"])]),
                    quality=_clean_csv_value(row[str(spec["quality_column"])]),
                    source_path=image_dir / filename,
                )
            )
    return records


def _balanced_prefix(records: Sequence[ImageRecord], limit: int) -> list[ImageRecord]:
    """Deterministic small-run sample balanced across camera/laterality groups."""

    if limit <= 0 or limit >= len(records):
        return list(records)
    groups: dict[tuple[str, str], list[ImageRecord]] = {}
    for record in records:
        key = (record.camera, record.laterality)
        groups.setdefault(key, []).append(record)
    selected: list[ImageRecord] = []
    group_lists = [groups[key] for key in sorted(groups)]
    index = 0
    while len(selected) < limit:
        made_progress = False
        for group in group_lists:
            if index < len(group):
                selected.append(group[index])
                made_progress = True
                if len(selected) == limit:
                    break
        if not made_progress:
            break
        index += 1
    return selected


def _save_jpeg(image: Image.Image, destination: Path, output: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        image.save(
            temporary_path,
            format="JPEG",
            quality=int(output["jpeg_quality"]),
            subsampling=int(output["jpeg_subsampling"]),
            optimize=bool(output["jpeg_optimize"]),
        )
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def run_preprocessing(
    *,
    project_root: Path,
    config_path: Path,
    output_root: Path | None = None,
    datasets: Sequence[str] | None = None,
    limit_per_dataset: int = 0,
) -> dict[str, Any]:
    """Run preprocessing and return paths plus summary.

    Existing outputs are never overwritten.  Any per-image error is captured in
    the manifest, and the caller receives ``error_count`` for a non-zero exit.
    """

    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = load_config(config_path)
    configured_root = project_root / str(config["output"]["root"])
    resolved_output_root = (output_root or configured_root).resolve()
    if resolved_output_root.exists() and any(resolved_output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {resolved_output_root}"
        )
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    selected_datasets = list(datasets or config["datasets"].keys())
    unknown = set(selected_datasets).difference(config["datasets"])
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    rows: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        records = load_dataset_records(project_root, dataset, config["datasets"][dataset])
        records = _balanced_prefix(records, limit_per_dataset)
        for record in records:
            output_name = Path(record.source_path.name).with_suffix(
                str(config["output"]["extension"])
            )
            destination = resolved_output_root / dataset / output_name
            base_row: dict[str, Any] = {
                "dataset": record.dataset,
                "dataset_version": record.dataset_version,
                "image_id": record.image_id,
                "patient_id": record.patient_id,
                "laterality": record.laterality,
                "camera": record.camera,
                "icdr": record.icdr,
                "quality": record.quality,
                "original_width": "",
                "original_height": "",
                "crop_x0": "",
                "crop_y0": "",
                "crop_x1": "",
                "crop_y1": "",
                "foreground_fraction": "",
                "crop_fallback": "",
                "fallback_reason": "",
                "output_path": _portable_output_path(destination, project_root),
                "source_sha256": "",
                "output_sha256": "",
                "status": "error",
                "error": "",
            }
            try:
                if not record.source_path.is_file():
                    raise FileNotFoundError(f"Missing source image: {record.source_path}")
                source_sha = sha256_file(record.source_path)
                base_row["source_sha256"] = source_sha
                processed, decision, original_size = preprocess_fundus_image(
                    record.source_path, config["detection"], config["output"]
                )
                _save_jpeg(processed, destination, config["output"])
                with Image.open(destination) as check:
                    check.load()
                    if check.mode != "RGB" or check.size != (512, 512):
                        raise ValueError(
                            f"Invalid output image mode/size: {check.mode}, {check.size}"
                        )
                x0, y0, x1, y1 = decision.bbox
                base_row.update(
                    {
                        "original_width": original_size[0],
                        "original_height": original_size[1],
                        "crop_x0": x0,
                        "crop_y0": y0,
                        "crop_x1": x1,
                        "crop_y1": y1,
                        "foreground_fraction": f"{decision.foreground_fraction:.8f}",
                        "crop_fallback": str(decision.fallback).lower(),
                        "fallback_reason": decision.fallback_reason,
                        "output_sha256": sha256_file(destination),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # captured, reported, and causes non-zero CLI exit
                base_row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(base_row)

    manifest_path = resolved_output_root / "preprocessing_manifest.csv"
    _atomic_write_csv(manifest_path, rows)
    manifest_sha = sha256_file(manifest_path)

    resolved_config_path = resolved_output_root / "resolved_config.yaml"
    with resolved_config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    ok_count = sum(row["status"] == "ok" for row in rows)
    error_count = len(rows) - ok_count
    fallback_count = sum(
        row["status"] == "ok" and row["crop_fallback"] == "true" for row in rows
    )
    summary = {
        "pipeline_version": config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "output_root": str(resolved_output_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "selected_datasets": selected_datasets,
        "limit_per_dataset": limit_per_dataset,
        "record_count": len(rows),
        "ok_count": ok_count,
        "error_count": error_count,
        "fallback_count": fallback_count,
        "python_version": platform.python_version(),
        "pillow_version": pillow_version,
        "opencv_version": cv2.__version__,
    }
    metadata_path = resolved_output_root / "run_metadata.json"
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary

