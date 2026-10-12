from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from dr_grading.data.patient_dataset import (
    ClinicalNormalizer,
    PatientFundusDataset,
    build_image_transform,
    load_dataloader_config,
    patient_collate_fn,
    validate_patient_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/data/patient_dataloader_v1.yaml"


def _patient_rows() -> pd.DataFrame:
    rows = [
        {
            "dataset": "brset",
            "patient_id": "1",
            "patient_uid": "brset:1",
            "split": "train",
            "patient_grade": 2,
            "patient_rDR": 1,
            "left_eye_grade": 2,
            "right_eye_grade": 1,
            "camera_group": "canon",
            "sex": "female",
            "age": 50.0,
            "diabetes_duration_years": 10.0,
            "insulin_use": 0.0,
            "age_observed": True,
            "sex_observed": True,
            "diabetes_duration_observed": True,
            "insulin_observed": True,
            "metadata_mask": "1111",
            "inter_eye_discordance": True,
        },
        {
            "dataset": "brset",
            "patient_id": "2",
            "patient_uid": "brset:2",
            "split": "train",
            "patient_grade": 0,
            "patient_rDR": 0,
            "left_eye_grade": 0,
            "right_eye_grade": 0,
            "camera_group": "nikon",
            "sex": "male",
            "age": 70.0,
            "diabetes_duration_years": float("nan"),
            "insulin_use": 1.0,
            "age_observed": True,
            "sex_observed": True,
            "diabetes_duration_observed": False,
            "insulin_observed": True,
            "metadata_mask": "1101",
            "inter_eye_discordance": False,
        },
        {
            "dataset": "mbrset",
            "patient_id": "3",
            "patient_uid": "mbrset:3",
            "split": "external_test",
            "patient_grade": 4,
            "patient_rDR": 1,
            "left_eye_grade": 4,
            "right_eye_grade": 4,
            "camera_group": "phelcom",
            "sex": "unknown",
            "age": float("nan"),
            "diabetes_duration_years": float("nan"),
            "insulin_use": float("nan"),
            "age_observed": False,
            "sex_observed": False,
            "diabetes_duration_observed": False,
            "insulin_observed": False,
            "metadata_mask": "0000",
            "inter_eye_discordance": False,
        },
    ]
    return pd.DataFrame(rows)


def _normalizer(patients: pd.DataFrame) -> ClinicalNormalizer:
    config = load_dataloader_config(CONFIG_PATH)
    return ClinicalNormalizer.fit(
        patients,
        config["clinical"],
        patient_manifest_sha256="synthetic",
    )


def test_dataloader_config_is_protocol_locked() -> None:
    config = load_dataloader_config(CONFIG_PATH)
    assert config["version"] == "patient_dataloader_v1"
    assert config["image"]["size"] == 512
    assert config["image"]["preserve_all_views"] is True
    assert config["clinical"]["fit_split"] == "train"
    assert config["clinical"]["fit_dataset"] == "brset"
    assert config["train_augmentation"]["random_resized_crop"] is False


def test_clinical_normalizer_fits_train_only_and_masks_missing() -> None:
    patients = _patient_rows()
    normalizer = _normalizer(patients)
    assert normalizer.fit_patient_count == 2
    assert normalizer.means == {
        "age": 60.0,
        "sex": 0.5,
        "diabetes_duration_years": 10.0,
        "insulin_use": 0.5,
    }
    features, mask = normalizer.transform_frame(patients.iloc[[2]])
    assert features.shape == (1, 4)
    assert torch.equal(features, torch.zeros_like(features))
    assert mask.tolist() == [[False, False, False, False]]


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), color).save(path, format="JPEG", quality=95)


def test_patient_dataset_and_collate_preserve_all_eye_views(tmp_path: Path) -> None:
    patients = _patient_rows().iloc[:2].copy()
    image_specs = [
        ("brset:1", "left", "a", "a.jpg", (120, 10, 10)),
        ("brset:1", "left", "b", "b.jpg", (130, 10, 10)),
        ("brset:1", "right", "c", "c.jpg", (10, 120, 10)),
        ("brset:2", "left", "d", "d.jpg", (10, 10, 120)),
        ("brset:2", "right", "e", "e.jpg", (10, 130, 10)),
        ("brset:2", "right", "f", "f.jpg", (10, 140, 10)),
    ]
    image_rows = []
    for patient_uid, laterality, image_id, relative, color in image_specs:
        _write_image(tmp_path / relative, color)
        image_rows.append(
            {
                "patient_uid": patient_uid,
                "laterality": laterality,
                "image_id": image_id,
                "processed_path": relative,
            }
        )
    images = pd.DataFrame(image_rows)
    config = load_dataloader_config(CONFIG_PATH)
    dataset = PatientFundusDataset(
        project_root=tmp_path,
        patients=patients,
        images=images,
        clinical_normalizer=_normalizer(patients),
        image_transform=build_image_transform(config, training=False),
    )
    first = dataset[0]
    second = dataset[1]
    assert first["left_images"].shape == (2, 3, 512, 512)
    assert first["right_images"].shape == (1, 3, 512, 512)
    assert first["left_image_ids"] == ["a", "b"]
    batch = patient_collate_fn([first, second], padded_value=0.0)
    validate_patient_batch(batch, expected_batch_size=2)
    assert batch["left_images"].shape == (2, 2, 3, 512, 512)
    assert batch["right_images"].shape == (2, 2, 3, 512, 512)
    assert batch["left_view_mask"].tolist() == [[True, True], [True, False]]
    assert batch["right_view_mask"].tolist() == [[True, False], [True, True]]
    assert torch.equal(batch["left_images"][1, 1], torch.zeros(3, 512, 512))
    assert torch.equal(batch["right_images"][0, 1], torch.zeros(3, 512, 512))


def test_evaluation_transform_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "image.jpg"
    _write_image(path, (50, 100, 150))
    config = load_dataloader_config(CONFIG_PATH)
    transform = build_image_transform(config, training=False)
    with Image.open(path) as image:
        first = transform(image)
    with Image.open(path) as image:
        second = transform(image)
    assert first.shape == (3, 512, 512)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_random_training_transform_keeps_shape_and_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "image.jpg"
    _write_image(path, (50, 100, 150))
    config = load_dataloader_config(CONFIG_PATH)
    transform = build_image_transform(config, training=True)
    torch.manual_seed(20260929)
    with Image.open(path) as image:
        tensor = transform(image)
    assert tensor.shape == (3, 512, 512)
    assert torch.isfinite(tensor).all()
