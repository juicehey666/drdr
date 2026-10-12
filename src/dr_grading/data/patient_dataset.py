"""Patient-level PyTorch input pipeline for bilateral fundus grading.

Every item is one patient. Left and right eyes remain separate, all available
views are retained, and the collator pads only the view dimension while
returning boolean masks. Clinical preprocessing is fitted exclusively on the
BRSET training patients defined by the frozen split manifest.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional


CLINICAL_FEATURE_ORDER = ["age", "sex", "diabetes_duration_years", "insulin_use"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataloader_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "patient_dataloader_v1":
        raise ValueError("This implementation requires version=patient_dataloader_v1")
    if int(config.get("seed", -1)) != 20260929:
        raise ValueError("Protocol requires seed=20260929")
    image = config["image"]
    if image.get("size") != 512 or image.get("mode") != "RGB":
        raise ValueError("fundus_512_v1 inputs must be RGB 512x512")
    if image.get("preserve_all_views") is not True:
        raise ValueError("All same-eye views must be preserved")
    if image.get("normalization") != "imagenet_convnext":
        raise ValueError("patient_dataloader_v1 is locked to ConvNeXt ImageNet normalization")
    if image.get("mean") != [0.485, 0.456, 0.406] or image.get("std") != [
        0.229,
        0.224,
        0.225,
    ]:
        raise ValueError("ImageNet normalization parameters differ from protocol")
    augmentation = config["train_augmentation"]
    if augmentation.get("version") != "aug_moderate_v1":
        raise ValueError("Training augmentation must be aug_moderate_v1")
    for forbidden in ("vertical_flip", "random_resized_crop", "mixup", "cutmix", "random_erasing"):
        if augmentation.get(forbidden) is not False:
            raise ValueError(f"Protocol forbids {forbidden}")
    clinical = config["clinical"]
    if clinical.get("feature_order") != CLINICAL_FEATURE_ORDER:
        raise ValueError("Clinical feature order is protocol locked")
    if clinical.get("fit_split") != "train" or clinical.get("fit_dataset") != "brset":
        raise ValueError("Clinical preprocessing must be fitted on BRSET train only")
    if clinical.get("imputation") != "observed_train_mean":
        raise ValueError("Clinical imputation must use observed BRSET-train means")
    if clinical.get("standardization") != "observed_train_zscore":
        raise ValueError("Clinical features must use observed BRSET-train z-scores")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        values = sorted(series.loc[mapped.isna()].astype(str).unique())
        raise ValueError(f"Invalid boolean values: {values}")
    return mapped.astype(bool)


def _numeric_feature(frame: pd.DataFrame, feature: str, sex_mapping: Mapping[str, float]) -> pd.Series:
    if feature == "sex":
        result = frame[feature].astype(str).map({str(k): float(v) for k, v in sex_mapping.items()})
        return result.astype(float)
    return pd.to_numeric(frame[feature], errors="coerce").astype(float)


@dataclass(frozen=True)
class ClinicalNormalizer:
    """Observed-only train statistics and deterministic missing-value handling."""

    feature_order: tuple[str, ...]
    observed_columns: dict[str, str]
    sex_mapping: dict[str, float]
    means: dict[str, float]
    standard_deviations: dict[str, float]
    observed_counts: dict[str, int]
    fit_dataset: str
    fit_split: str
    fit_patient_count: int
    patient_manifest_sha256: str

    @classmethod
    def fit(
        cls,
        patients: pd.DataFrame,
        clinical_config: Mapping[str, Any],
        *,
        patient_manifest_sha256: str,
    ) -> "ClinicalNormalizer":
        fit = patients.loc[
            patients["dataset"].eq(str(clinical_config["fit_dataset"]))
            & patients["split"].eq(str(clinical_config["fit_split"]))
        ].copy()
        if fit.empty:
            raise ValueError("No patients are available for fitting clinical normalization")
        features = tuple(str(value) for value in clinical_config["feature_order"])
        observed_columns = {
            str(key): str(value) for key, value in clinical_config["observed_columns"].items()
        }
        sex_mapping = {
            str(key): float(value) for key, value in clinical_config["sex_mapping"].items()
        }
        means: dict[str, float] = {}
        standard_deviations: dict[str, float] = {}
        observed_counts: dict[str, int] = {}
        minimum_std = float(clinical_config["minimum_standard_deviation"])
        ddof = int(clinical_config["standard_deviation_ddof"])
        for feature in features:
            observed = _as_bool(fit[observed_columns[feature]])
            numeric = _numeric_feature(fit, feature, sex_mapping)
            usable = numeric.loc[observed & numeric.notna()]
            if usable.empty:
                raise ValueError(f"No observed BRSET-train values for clinical feature {feature}")
            mean = float(usable.mean())
            std = float(usable.std(ddof=ddof))
            if not np.isfinite(std) or std < minimum_std:
                std = 1.0
            means[feature] = mean
            standard_deviations[feature] = std
            observed_counts[feature] = int(len(usable))
        return cls(
            feature_order=features,
            observed_columns=observed_columns,
            sex_mapping=sex_mapping,
            means=means,
            standard_deviations=standard_deviations,
            observed_counts=observed_counts,
            fit_dataset=str(clinical_config["fit_dataset"]),
            fit_split=str(clinical_config["fit_split"]),
            fit_patient_count=int(len(fit)),
            patient_manifest_sha256=patient_manifest_sha256,
        )

    def transform_frame(self, patients: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        values: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for feature in self.feature_order:
            observed = _as_bool(patients[self.observed_columns[feature]])
            numeric = _numeric_feature(patients, feature, self.sex_mapping)
            valid = observed & numeric.notna()
            imputed = numeric.where(valid, self.means[feature])
            standardised = (imputed - self.means[feature]) / self.standard_deviations[feature]
            values.append(standardised.to_numpy(dtype=np.float32))
            masks.append(valid.to_numpy(dtype=bool))
        feature_tensor = torch.from_numpy(np.stack(values, axis=1))
        mask_tensor = torch.from_numpy(np.stack(masks, axis=1))
        if not torch.isfinite(feature_tensor).all():
            raise ValueError("Clinical normalization produced non-finite values")
        return feature_tensor, mask_tensor

    def transform_row(self, patient: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        frame = patient.to_frame().T
        features, mask = self.transform_frame(frame)
        return features[0], mask[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "clinical_4_v1",
            "feature_order": list(self.feature_order),
            "observed_columns": self.observed_columns,
            "sex_mapping": self.sex_mapping,
            "means": self.means,
            "standard_deviations": self.standard_deviations,
            "observed_counts": self.observed_counts,
            "fit_dataset": self.fit_dataset,
            "fit_split": self.fit_split,
            "fit_patient_count": self.fit_patient_count,
            "patient_manifest_sha256": self.patient_manifest_sha256,
            "missing_imputation": "observed_train_mean_then_zero_after_zscore",
            "standard_deviation_ddof": 0,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClinicalNormalizer":
        if data.get("version") != "clinical_4_v1":
            raise ValueError("Unsupported clinical normalizer version")
        return cls(
            feature_order=tuple(str(value) for value in data["feature_order"]),
            observed_columns={str(k): str(v) for k, v in data["observed_columns"].items()},
            sex_mapping={str(k): float(v) for k, v in data["sex_mapping"].items()},
            means={str(k): float(v) for k, v in data["means"].items()},
            standard_deviations={
                str(k): float(v) for k, v in data["standard_deviations"].items()
            },
            observed_counts={str(k): int(v) for k, v in data["observed_counts"].items()},
            fit_dataset=str(data["fit_dataset"]),
            fit_split=str(data["fit_split"]),
            fit_patient_count=int(data["fit_patient_count"]),
            patient_manifest_sha256=str(data["patient_manifest_sha256"]),
        )

    @classmethod
    def load(cls, path: Path) -> "ClinicalNormalizer":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class RandomGamma:
    """Apply a gamma sampled uniformly from a closed interval."""

    def __init__(self, gamma_range: Sequence[float]) -> None:
        if len(gamma_range) != 2 or gamma_range[0] <= 0 or gamma_range[1] < gamma_range[0]:
            raise ValueError(f"Invalid gamma range: {gamma_range}")
        self.minimum = float(gamma_range[0])
        self.maximum = float(gamma_range[1])

    def __call__(self, image: Image.Image) -> Image.Image:
        gamma = float(torch.empty(1).uniform_(self.minimum, self.maximum).item())
        return vision_functional.adjust_gamma(image, gamma=gamma, gain=1.0)


def build_image_transform(config: Mapping[str, Any], *, training: bool) -> transforms.Compose:
    operations: list[Any] = []
    if training:
        aug = config["train_augmentation"]
        operations.extend(
            [
                transforms.RandomHorizontalFlip(
                    p=float(aug["horizontal_flip"]["probability"])
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=tuple(float(v) for v in aug["affine"]["rotation_degrees"]),
                            translate=(
                                float(aug["affine"]["max_translation_fraction"]),
                                float(aug["affine"]["max_translation_fraction"]),
                            ),
                            scale=tuple(float(v) for v in aug["affine"]["scale"]),
                            interpolation=InterpolationMode.BILINEAR,
                            fill=tuple(int(v) for v in aug["affine"]["fill_rgb"]),
                        )
                    ],
                    p=float(aug["affine"]["probability"]),
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=float(aug["color_jitter"]["brightness"]),
                            contrast=float(aug["color_jitter"]["contrast"]),
                            saturation=float(aug["color_jitter"]["saturation"]),
                            hue=float(aug["color_jitter"]["hue"]),
                        )
                    ],
                    p=float(aug["color_jitter"]["probability"]),
                ),
                transforms.RandomApply(
                    [RandomGamma(aug["gamma"]["range"])],
                    p=float(aug["gamma"]["probability"]),
                ),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=int(aug["gaussian_blur"]["kernel_size"]),
                            sigma=tuple(float(v) for v in aug["gaussian_blur"]["sigma"]),
                        )
                    ],
                    p=float(aug["gaussian_blur"]["probability"]),
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[float(value) for value in config["image"]["mean"]],
                std=[float(value) for value in config["image"]["std"]],
            ),
        ]
    )
    return transforms.Compose(operations)


class PatientFundusDataset(Dataset[dict[str, Any]]):
    """One item per patient with separate variable-length left/right views."""

    def __init__(
        self,
        *,
        project_root: Path,
        patients: pd.DataFrame,
        images: pd.DataFrame,
        clinical_normalizer: ClinicalNormalizer,
        image_transform: Any,
        expected_image_size: int = 512,
    ) -> None:
        self.project_root = project_root.resolve()
        self.patients = patients.reset_index(drop=True).copy()
        self.images = images.copy()
        self.clinical_normalizer = clinical_normalizer
        self.image_transform = image_transform
        self.expected_image_size = int(expected_image_size)
        if self.patients["patient_uid"].duplicated().any():
            raise ValueError("Patient dataset contains duplicate patient_uid rows")
        image_uids = set(self.images["patient_uid"])
        missing = set(self.patients["patient_uid"]).difference(image_uids)
        if missing:
            raise ValueError(f"Patients have no image records: {sorted(missing)[:5]}")
        self.image_groups = {
            str(patient_uid): group.sort_values(["laterality", "image_id"], kind="mergesort")
            for patient_uid, group in self.images.groupby("patient_uid", sort=False)
        }

    def __len__(self) -> int:
        return len(self.patients)

    def _load_views(self, records: pd.DataFrame) -> tuple[torch.Tensor, list[str], list[str]]:
        tensors: list[torch.Tensor] = []
        image_ids: list[str] = []
        paths: list[str] = []
        for record in records.itertuples(index=False):
            path = self.project_root / str(record.processed_path)
            with Image.open(path) as source:
                source.load()
                if source.mode != "RGB":
                    raise ValueError(f"Expected RGB image, got {source.mode}: {path}")
                if source.size != (self.expected_image_size, self.expected_image_size):
                    raise ValueError(
                        f"Expected {self.expected_image_size}x{self.expected_image_size}, "
                        f"got {source.size}: {path}"
                    )
                tensor = self.image_transform(source)
            if tensor.shape != (3, self.expected_image_size, self.expected_image_size):
                raise ValueError(f"Unexpected transformed tensor shape {tuple(tensor.shape)}: {path}")
            tensors.append(tensor)
            image_ids.append(str(record.image_id))
            paths.append(str(record.processed_path))
        if not tensors:
            raise ValueError("A required eye has no images")
        return torch.stack(tensors), image_ids, paths

    def __getitem__(self, index: int) -> dict[str, Any]:
        patient = self.patients.iloc[index]
        records = self.image_groups[str(patient["patient_uid"])]
        left_records = records.loc[records["laterality"].eq("left")]
        right_records = records.loc[records["laterality"].eq("right")]
        left_images, left_ids, left_paths = self._load_views(left_records)
        right_images, right_ids, right_paths = self._load_views(right_records)
        clinical_features, clinical_mask = self.clinical_normalizer.transform_row(patient)
        return {
            "patient_uid": str(patient["patient_uid"]),
            "patient_id": str(patient["patient_id"]),
            "dataset": str(patient["dataset"]),
            "split": str(patient["split"]),
            "left_images": left_images,
            "right_images": right_images,
            "left_image_ids": left_ids,
            "right_image_ids": right_ids,
            "left_paths": left_paths,
            "right_paths": right_paths,
            "clinical_features": clinical_features,
            "clinical_mask": clinical_mask,
            "patient_grade": torch.tensor(int(patient["patient_grade"]), dtype=torch.long),
            "patient_rdr": torch.tensor(float(patient["patient_rDR"]), dtype=torch.float32),
            "left_eye_grade": torch.tensor(int(patient["left_eye_grade"]), dtype=torch.long),
            "right_eye_grade": torch.tensor(int(patient["right_eye_grade"]), dtype=torch.long),
            "inter_eye_discordance": torch.tensor(
                bool(patient["inter_eye_discordance"]), dtype=torch.bool
            ),
            "camera_group": str(patient["camera_group"]),
            "metadata_mask": str(patient["metadata_mask"]),
        }


def _pad_views(
    samples: Sequence[dict[str, Any]],
    key: str,
    *,
    padded_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(int(sample[key].shape[0]) for sample in samples)
    first = samples[0][key]
    batch = first.new_full((len(samples), maximum, *first.shape[1:]), float(padded_value))
    mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = int(sample[key].shape[0])
        batch[index, :count] = sample[key]
        mask[index, :count] = True
    return batch, mask


def patient_collate_fn(
    samples: Sequence[dict[str, Any]],
    *,
    padded_value: float = 0.0,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty patient batch")
    left_images, left_mask = _pad_views(samples, "left_images", padded_value=padded_value)
    right_images, right_mask = _pad_views(samples, "right_images", padded_value=padded_value)
    return {
        "patient_uid": [sample["patient_uid"] for sample in samples],
        "patient_id": [sample["patient_id"] for sample in samples],
        "dataset": [sample["dataset"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "left_images": left_images,
        "right_images": right_images,
        "left_view_mask": left_mask,
        "right_view_mask": right_mask,
        "left_image_ids": [sample["left_image_ids"] for sample in samples],
        "right_image_ids": [sample["right_image_ids"] for sample in samples],
        "left_paths": [sample["left_paths"] for sample in samples],
        "right_paths": [sample["right_paths"] for sample in samples],
        "clinical_features": torch.stack([sample["clinical_features"] for sample in samples]),
        "clinical_mask": torch.stack([sample["clinical_mask"] for sample in samples]),
        "patient_grade": torch.stack([sample["patient_grade"] for sample in samples]),
        "patient_rdr": torch.stack([sample["patient_rdr"] for sample in samples]),
        "left_eye_grade": torch.stack([sample["left_eye_grade"] for sample in samples]),
        "right_eye_grade": torch.stack([sample["right_eye_grade"] for sample in samples]),
        "inter_eye_discordance": torch.stack(
            [sample["inter_eye_discordance"] for sample in samples]
        ),
        "camera_group": [sample["camera_group"] for sample in samples],
        "metadata_mask": [sample["metadata_mask"] for sample in samples],
    }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PatientDataModule:
    """Validated manifest reader and factory for patient-level datasets/loaders."""

    def __init__(
        self,
        *,
        project_root: Path,
        config_path: Path,
        clinical_normalizer_path: Path | None = None,
        validate_image_paths: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config_path = config_path.resolve()
        self.config = load_dataloader_config(self.config_path)
        inputs = self.config["inputs"]
        self.patient_manifest_path = self.project_root / str(inputs["patient_manifest"])
        self.image_manifest_path = self.project_root / str(inputs["image_manifest"])
        patient_hash = sha256_file(self.patient_manifest_path)
        image_hash = sha256_file(self.image_manifest_path)
        if patient_hash != str(inputs["patient_manifest_sha256"]):
            raise ValueError("Frozen patient manifest SHA256 does not match dataloader config")
        if image_hash != str(inputs["image_manifest_sha256"]):
            raise ValueError("Frozen image manifest SHA256 does not match dataloader config")
        self.patient_manifest_sha256 = patient_hash
        self.image_manifest_sha256 = image_hash
        self.patients = pd.read_csv(
            self.patient_manifest_path,
            dtype={"patient_id": str, "patient_uid": str, "metadata_mask": str},
        )
        self.images = pd.read_csv(
            self.image_manifest_path,
            dtype={"patient_id": str, "patient_uid": str, "image_id": str},
        )
        self._validate_manifests(validate_image_paths=validate_image_paths)
        if clinical_normalizer_path is None:
            self.clinical_normalizer = ClinicalNormalizer.fit(
                self.patients,
                self.config["clinical"],
                patient_manifest_sha256=self.patient_manifest_sha256,
            )
        else:
            self.clinical_normalizer = ClinicalNormalizer.load(clinical_normalizer_path)
            if self.clinical_normalizer.patient_manifest_sha256 != self.patient_manifest_sha256:
                raise ValueError("Clinical normalizer was fitted on a different patient manifest")
            if list(self.clinical_normalizer.feature_order) != CLINICAL_FEATURE_ORDER:
                raise ValueError("Clinical normalizer feature order differs from protocol")

    def _validate_manifests(self, *, validate_image_paths: bool) -> None:
        allowed_splits = set(str(value) for value in self.config["inputs"]["allowed_splits"])
        if set(self.patients["split"]) != allowed_splits:
            raise ValueError("Patient manifest split names differ from configuration")
        if self.patients["patient_uid"].duplicated().any():
            raise ValueError("Patient manifest contains duplicate patient_uid rows")
        if self.images["image_id"].duplicated().any():
            raise ValueError("Image manifest contains duplicate image_id rows")
        if not self.images["patient_uid"].isin(set(self.patients["patient_uid"])).all():
            raise ValueError("Image manifest contains unknown patients")
        merged = self.images.merge(
            self.patients[["patient_uid", "split", "patient_grade", "patient_rDR"]],
            on="patient_uid",
            how="left",
            suffixes=("_image", "_patient"),
            validate="many_to_one",
        )
        for column in ("split", "patient_grade", "patient_rDR"):
            if not merged[f"{column}_image"].eq(merged[f"{column}_patient"]).all():
                raise ValueError(f"Patient/image manifest mismatch for {column}")
        counts = self.images.groupby(["patient_uid", "laterality"]).size().unstack(fill_value=0)
        for side in ("left", "right"):
            if side not in counts:
                raise ValueError(f"Image manifest has no {side} eye records")
        expected = self.patients.set_index("patient_uid")[["left_image_count", "right_image_count"]]
        actual = counts[["left", "right"]].rename(
            columns={"left": "left_image_count", "right": "right_image_count"}
        )
        actual = actual.reindex(expected.index).fillna(0).astype(int)
        if not actual.eq(expected.astype(int)).all().all():
            raise ValueError("Per-patient eye image counts disagree between manifests")
        if not self.patients.loc[self.patients["split"].eq("external_test"), "dataset"].eq(
            "mbrset"
        ).all():
            raise ValueError("External test contains a non-mBRSET patient")
        if not self.patients.loc[~self.patients["split"].eq("external_test"), "dataset"].eq(
            "brset"
        ).all():
            raise ValueError("Development split contains a non-BRSET patient")
        if validate_image_paths:
            missing_paths = [
                str(path)
                for value in self.images["processed_path"]
                if not (path := self.project_root / str(value)).is_file()
            ]
            if missing_paths:
                raise FileNotFoundError(f"Processed images are missing: {missing_paths[:5]}")

    def dataset(self, split: str, *, training: bool | None = None) -> PatientFundusDataset:
        if split not in set(self.config["inputs"]["allowed_splits"]):
            raise ValueError(f"Unknown split: {split}")
        if training is None:
            training = split == "train"
        if training and split != "train":
            raise ValueError("Random augmentation is permitted only for the train split")
        patients = self.patients.loc[self.patients["split"].eq(split)].copy()
        patient_uids = set(patients["patient_uid"])
        images = self.images.loc[self.images["patient_uid"].isin(patient_uids)].copy()
        transform = build_image_transform(self.config, training=bool(training))
        return PatientFundusDataset(
            project_root=self.project_root,
            patients=patients,
            images=images,
            clinical_normalizer=self.clinical_normalizer,
            image_transform=transform,
            expected_image_size=int(self.config["image"]["size"]),
        )

    def dataloader(
        self,
        split: str,
        *,
        batch_size: int,
        num_workers: int,
        training: bool | None = None,
        pin_memory: bool | None = None,
    ) -> DataLoader[dict[str, Any]]:
        dataset = self.dataset(split, training=training)
        is_training = split == "train" if training is None else bool(training)
        if pin_memory is None:
            pin_memory = bool(self.config["loader"]["pin_memory"])
        generator = torch.Generator()
        generator.manual_seed(int(self.config["seed"]))
        kwargs: dict[str, Any] = {}
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(self.config["loader"]["persistent_workers"])
            kwargs["prefetch_factor"] = int(self.config["loader"]["prefetch_factor"])
        padded_value = float(self.config["image"]["padded_tensor_value"])
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=(
                bool(self.config["loader"]["train_shuffle"])
                if is_training
                else bool(self.config["loader"]["evaluation_shuffle"])
            ),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=bool(self.config["loader"]["drop_last"]),
            collate_fn=partial(patient_collate_fn, padded_value=padded_value),
            worker_init_fn=seed_worker,
            generator=generator,
            **kwargs,
        )


def validate_patient_batch(batch: Mapping[str, Any], *, expected_batch_size: int | None = None) -> None:
    batch_size = len(batch["patient_uid"])
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise AssertionError(f"Expected batch size {expected_batch_size}, got {batch_size}")
    for side in ("left", "right"):
        images = batch[f"{side}_images"]
        mask = batch[f"{side}_view_mask"]
        if images.ndim != 5 or images.shape[0] != batch_size or images.shape[2:] != (3, 512, 512):
            raise AssertionError(f"Invalid {side} image batch shape: {tuple(images.shape)}")
        if mask.shape != images.shape[:2] or mask.dtype != torch.bool or not mask.any(dim=1).all():
            raise AssertionError(f"Invalid {side} view mask")
        if not torch.isfinite(images).all():
            raise AssertionError(f"Non-finite {side} image tensor")
    if batch["clinical_features"].shape != (batch_size, len(CLINICAL_FEATURE_ORDER)):
        raise AssertionError("Invalid clinical feature tensor shape")
    if batch["clinical_mask"].shape != batch["clinical_features"].shape:
        raise AssertionError("Clinical mask shape differs from clinical features")
    if batch["clinical_mask"].dtype != torch.bool:
        raise AssertionError("Clinical mask must be boolean")
    if not torch.isfinite(batch["clinical_features"]).all():
        raise AssertionError("Clinical features contain non-finite values")
    if batch["patient_grade"].dtype != torch.long:
        raise AssertionError("Patient grade must be int64")
    if batch["patient_rdr"].dtype != torch.float32:
        raise AssertionError("Patient rDR target must be float32")

