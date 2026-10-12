#!/usr/bin/env python3
"""Fit train-only clinical statistics and smoke-test patient DataLoaders."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dr_grading.data.patient_dataset import (  # noqa: E402
    PatientDataModule,
    load_dataloader_config,
    sha256_file,
    validate_patient_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare clinical normalization and smoke-test all patient-level splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/patient_dataloader_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_dataloader_config(config_path)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (PROJECT_ROOT / str(config["output"]["root"])).resolve()
    )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataloader artifacts: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    module = PatientDataModule(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        validate_image_paths=True,
    )
    smoke: dict[str, object] = {
        "version": config["version"],
        "seed": seed,
        "batch_size": int(config["loader"]["smoke_batch_size"]),
        "num_workers": int(config["loader"]["smoke_num_workers"]),
        "splits": {},
    }
    for split in config["inputs"]["allowed_splits"]:
        loader = module.dataloader(
            str(split),
            batch_size=int(config["loader"]["smoke_batch_size"]),
            num_workers=int(config["loader"]["smoke_num_workers"]),
            training=str(split) == "train",
            pin_memory=False,
        )
        batch = next(iter(loader))
        validate_patient_batch(
            batch, expected_batch_size=int(config["loader"]["smoke_batch_size"])
        )
        smoke["splits"][str(split)] = {
            "patient_uids": batch["patient_uid"],
            "left_images_shape": list(batch["left_images"].shape),
            "right_images_shape": list(batch["right_images"].shape),
            "left_view_counts": batch["left_view_mask"].sum(dim=1).tolist(),
            "right_view_counts": batch["right_view_mask"].sum(dim=1).tolist(),
            "clinical_features_shape": list(batch["clinical_features"].shape),
            "clinical_observed_counts": batch["clinical_mask"].sum(dim=1).tolist(),
            "patient_grades": batch["patient_grade"].tolist(),
            "patient_rdr": batch["patient_rdr"].tolist(),
            "left_tensor_sha256": _tensor_sha256(batch["left_images"]),
            "right_tensor_sha256": _tensor_sha256(batch["right_images"]),
        }

    view_summary = {
        str(split): {
            "patients": int(len(group)),
            "max_left_views": int(group["left_image_count"].max()),
            "max_right_views": int(group["right_image_count"].max()),
            "left_view_count_distribution": {
                str(int(key)): int(value)
                for key, value in group["left_image_count"].value_counts().sort_index().items()
            },
            "right_view_count_distribution": {
                str(int(key)): int(value)
                for key, value in group["right_image_count"].value_counts().sort_index().items()
            },
        }
        for split, group in module.patients.groupby("split", sort=False)
    }

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp.", dir=output_root.parent)
    )
    try:
        clinical_path = temporary / "clinical_normalization.json"
        clinical_path.write_text(
            json.dumps(
                module.clinical_normalizer.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        smoke_path = temporary / "dataloader_smoke_report.json"
        smoke_path.write_text(
            json.dumps(smoke, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        views_path = temporary / "view_count_summary.json"
        views_path.write_text(
            json.dumps(view_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_hashes = {
            path.name: sha256_file(path) for path in (clinical_path, smoke_path, views_path)
        }
        metadata = {
            "version": config["version"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "config_sha256": sha256_file(config_path),
            "patient_manifest_sha256": module.patient_manifest_sha256,
            "image_manifest_sha256": module.image_manifest_sha256,
            "clinical_fit_dataset": module.clinical_normalizer.fit_dataset,
            "clinical_fit_split": module.clinical_normalizer.fit_split,
            "clinical_fit_patient_count": module.clinical_normalizer.fit_patient_count,
            "artifact_sha256": artifact_hashes,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
        }
        metadata_path = temporary / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksums = {**artifact_hashes, "run_metadata.json": sha256_file(metadata_path)}
        (temporary / "checksums.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    summary = {
        "version": config["version"],
        "output_root": str(output_root),
        "clinical_normalization": str(output_root / "clinical_normalization.json"),
        "smoke_report": str(output_root / "dataloader_smoke_report.json"),
        "clinical_fit_patient_count": module.clinical_normalizer.fit_patient_count,
        "clinical_means": module.clinical_normalizer.means,
        "clinical_standard_deviations": module.clinical_normalizer.standard_deviations,
        "splits_smoke_tested": list(config["inputs"]["allowed_splits"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
