"""Build the frozen patient-level BRSET--mBRSET analysis cohort.

The module deliberately separates outcome construction from clinical metadata:
only gradable, quality-accepted ICDR labels determine the main-cohort truth.
Clinical variables are used for split balancing and later modelling, never for
inclusion labels or target generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


PATIENT_MANIFEST_COLUMNS = [
    "dataset",
    "patient_id",
    "patient_uid",
    "split",
    "patient_grade",
    "patient_rDR",
    "patient_any_dr",
    "patient_vtDR",
    "left_eye_grade",
    "right_eye_grade",
    "left_image_count",
    "right_image_count",
    "total_image_count",
    "camera_group",
    "sex",
    "age",
    "age_top_coded",
    "age_bin",
    "diabetes_duration_years",
    "insulin_use",
    "age_observed",
    "sex_observed",
    "diabetes_duration_observed",
    "insulin_observed",
    "metadata_mask",
    "metadata_observed_count",
    "age_conflict",
    "sex_conflict",
    "diabetes_duration_conflict",
    "insulin_conflict",
    "inter_eye_discordance",
    "within_eye_label_conflict",
    "mean_foreground_fraction",
    "mean_synthetic_padding_fraction",
    "max_synthetic_padding_fraction",
    "any_crop_fallback",
]

IMAGE_MANIFEST_COLUMNS = [
    "dataset",
    "patient_id",
    "patient_uid",
    "split",
    "image_id",
    "laterality",
    "image_icdr",
    "eye_grade",
    "patient_grade",
    "patient_rDR",
    "camera",
    "processed_path",
    "foreground_fraction",
    "synthetic_padding_fraction",
    "crop_fallback",
    "source_sha256",
    "output_sha256",
]

SPLIT_ORDER = ["train", "validation", "internal_test", "external_test"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_cohort_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "patient_split_v1":
        raise ValueError("This implementation requires version=patient_split_v1")
    if int(config.get("seed", -1)) != 20260929:
        raise ValueError("Protocol requires seed=20260929")
    ratios = config["split"]["ratios"]
    if list(ratios) != ["train", "validation", "internal_test"]:
        raise ValueError("Split ratio order must be train/validation/internal_test")
    if not math.isclose(sum(float(value) for value in ratios.values()), 1.0):
        raise ValueError("Development split ratios must sum to 1")
    if config["cohort"]["required_lateralities"] != ["left", "right"]:
        raise ValueError("Protocol requires both left and right eyes")
    if config["cohort"]["eye_grade_aggregation"] != "max":
        raise ValueError("Protocol requires max eye-grade aggregation")
    if config["cohort"]["patient_grade_aggregation"] != "max":
        raise ValueError("Protocol requires max patient-grade aggregation")
    if config["cohort"]["clinical_variable_order"] != [
        "age",
        "sex",
        "diabetes_duration",
        "insulin",
    ]:
        raise ValueError("Clinical metadata-mask order is protocol locked")
    if config["cohort"]["conflicting_patient_metadata_policy"] != "set_missing":
        raise ValueError("Conflicting patient metadata must be set missing")


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _stable_unit(seed: int, *parts: object) -> float:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    integer = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return integer / float(2**64)


def _normalise_identifier(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty patient or image identifier")
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _parse_numeric(
    series: pd.Series,
    *,
    top_code_mapping: Mapping[str, object] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw = series.astype(str).str.strip()
    top_codes = {str(key): float(value) for key, value in (top_code_mapping or {}).items()}
    top_coded = raw.isin(top_codes)
    replaced = raw.str.replace(",", ".", regex=False).replace(top_codes)
    numeric = pd.to_numeric(replaced.replace("", np.nan), errors="coerce")
    invalid = raw.ne("") & numeric.isna()
    return numeric.astype(float), top_coded.astype(bool), invalid.astype(bool)


def _mapped_values(series: pd.Series, mapping: Mapping[str, object]) -> tuple[pd.Series, pd.Series]:
    raw = series.astype(str).str.strip()
    normalised_mapping = {str(key): value for key, value in mapping.items()}
    mapped = raw.map(normalised_mapping)
    invalid = raw.ne("") & mapped.isna()
    return mapped, invalid.astype(bool)


def _collapse_observed(values: Iterable[object]) -> tuple[object, bool, bool]:
    observed: list[object] = []
    for value in values:
        if pd.isna(value) or value == "":
            continue
        if value not in observed:
            observed.append(value)
    if not observed:
        return np.nan, False, False
    if len(observed) > 1:
        return np.nan, False, True
    return observed[0], True, False


def _age_bin(value: object) -> str:
    if pd.isna(value):
        return "missing"
    age = float(value)
    if age < 50:
        return "<50"
    if age < 65:
        return "50-64"
    if age < 75:
        return "65-74"
    return ">=75"


def _synthetic_padding_fraction(row: pd.Series) -> float:
    x0, y0, x1, y1 = (int(row[key]) for key in ("crop_x0", "crop_y0", "crop_x1", "crop_y1"))
    width, height = x1 - x0, y1 - y0
    side = max(width, height)
    square_x0 = x0 - (side - width) // 2
    square_y0 = y0 - (side - height) // 2
    square_x1 = square_x0 + side
    square_y1 = square_y0 + side
    copied_width = max(0, min(int(row["original_width"]), square_x1) - max(0, square_x0))
    copied_height = max(0, min(int(row["original_height"]), square_y1) - max(0, square_y0))
    return float(1.0 - (copied_width * copied_height) / float(side * side))


def _standardise_dataset(
    dataset: str,
    spec: Mapping[str, Any],
    project_root: Path,
    preprocessing: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    metadata_path = project_root / str(spec["metadata_csv"])
    raw = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    required = {
        spec["image_column"],
        spec["patient_column"],
        spec["laterality_column"],
        spec["grade_column"],
        spec["quality_column"],
        spec["age_column"],
        spec["sex_column"],
        spec["diabetes_duration_column"],
        spec["insulin_column"],
    }
    if "camera_column" in spec:
        required.add(spec["camera_column"])
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{dataset} metadata is missing columns: {sorted(missing)}")

    frame = pd.DataFrame(index=raw.index)
    frame["dataset"] = dataset
    frame["image_id"] = raw[spec["image_column"]].map(_normalise_identifier)
    frame["patient_id"] = raw[spec["patient_column"]].map(_normalise_identifier)
    frame["patient_uid"] = dataset + ":" + frame["patient_id"]

    laterality_mapping = {str(key): str(value) for key, value in spec["laterality_mapping"].items()}
    frame["laterality"] = raw[spec["laterality_column"]].str.strip().map(laterality_mapping)
    if frame["laterality"].isna().any():
        bad = sorted(raw.loc[frame["laterality"].isna(), spec["laterality_column"]].unique())
        raise ValueError(f"{dataset} has unknown laterality values: {bad}")

    if "camera_column" in spec:
        frame["camera"] = raw[spec["camera_column"]].str.strip()
    else:
        frame["camera"] = str(spec["camera_constant"])
    frame["quality"] = raw[spec["quality_column"]].str.strip()
    frame["quality_ok"] = frame["quality"].eq(str(spec["accepted_quality"]))
    frame["image_icdr"] = pd.to_numeric(
        raw[spec["grade_column"]].replace("", np.nan), errors="coerce"
    )
    invalid_grade = raw[spec["grade_column"]].str.strip().ne("") & frame["image_icdr"].isna()
    valid_grade_values = frame["image_icdr"].dropna()
    if invalid_grade.any() or not valid_grade_values.isin(range(5)).all():
        raise ValueError(f"{dataset} contains invalid ICDR values")

    age, age_top_coded, age_invalid = _parse_numeric(
        raw[spec["age_column"]], top_code_mapping=spec.get("age_top_code_mapping")
    )
    duration, _, duration_invalid = _parse_numeric(raw[spec["diabetes_duration_column"]])
    sex, sex_invalid = _mapped_values(raw[spec["sex_column"]], spec["sex_mapping"])
    insulin, insulin_invalid = _mapped_values(raw[spec["insulin_column"]], spec["insulin_mapping"])
    frame["age_value"] = age.where(age.between(0, 120))
    frame["age_top_coded_raw"] = age_top_coded
    frame["sex_value"] = sex
    frame["diabetes_duration_value"] = duration.where(duration.between(0, 120))
    frame["insulin_value"] = pd.to_numeric(insulin, errors="coerce")

    if frame["image_id"].duplicated().any():
        raise ValueError(f"{dataset} image identifiers are not unique")

    pp = preprocessing.loc[preprocessing["dataset"].eq(dataset)].copy()
    if len(pp) != len(frame):
        raise ValueError(
            f"{dataset} preprocessing/raw row mismatch: {len(pp)} versus {len(frame)}"
        )
    pp_columns = [
        "image_id",
        "patient_id",
        "original_width",
        "original_height",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "foreground_fraction",
        "crop_fallback",
        "output_path",
        "source_sha256",
        "output_sha256",
        "status",
    ]
    pp = pp[pp_columns].rename(columns={"patient_id": "preprocessing_patient_id"})
    merged = frame.merge(pp, on="image_id", how="left", validate="one_to_one")
    if merged["output_path"].isna().any() or not merged["status"].eq("ok").all():
        raise ValueError(f"{dataset} has missing or failed preprocessing records")
    if not merged["patient_id"].eq(merged["preprocessing_patient_id"].map(_normalise_identifier)).all():
        raise ValueError(f"{dataset} patient identifiers disagree with preprocessing manifest")
    numeric_geometry = [
        "original_width",
        "original_height",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "foreground_fraction",
    ]
    for column in numeric_geometry:
        merged[column] = pd.to_numeric(merged[column], errors="raise")
    merged["crop_fallback"] = (
        merged["crop_fallback"].astype(str).str.lower().map({"true": True, "false": False})
    )
    if merged["crop_fallback"].isna().any():
        raise ValueError(f"{dataset} has invalid crop_fallback values")
    merged["synthetic_padding_fraction"] = merged.apply(_synthetic_padding_fraction, axis=1)

    audit = {
        "raw_rows": int(len(raw)),
        "raw_patients": int(frame["patient_id"].nunique()),
        "invalid_age_values": int(age_invalid.sum()),
        "invalid_sex_values": int(sex_invalid.sum()),
        "invalid_diabetes_duration_values": int(duration_invalid.sum()),
        "invalid_insulin_values": int(insulin_invalid.sum()),
    }
    return merged, audit


def _patient_metadata(images: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient_id, group in images.groupby("patient_id", sort=True):
        age, age_observed, age_conflict = _collapse_observed(group["age_value"])
        sex, sex_observed, sex_conflict = _collapse_observed(group["sex_value"])
        duration, duration_observed, duration_conflict = _collapse_observed(
            group["diabetes_duration_value"]
        )
        insulin, insulin_observed, insulin_conflict = _collapse_observed(group["insulin_value"])
        age_top_coded = bool(age_observed and group.loc[group["age_value"].eq(age), "age_top_coded_raw"].any())
        mask_bits = [age_observed, sex_observed, duration_observed, insulin_observed]
        rows.append(
            {
                "patient_id": str(patient_id),
                "age": age,
                "age_top_coded": age_top_coded,
                "age_bin": _age_bin(age),
                "sex": sex if sex_observed else "unknown",
                "diabetes_duration_years": duration,
                "insulin_use": insulin,
                "age_observed": age_observed,
                "sex_observed": sex_observed,
                "diabetes_duration_observed": duration_observed,
                "insulin_observed": insulin_observed,
                "metadata_mask": "".join("1" if value else "0" for value in mask_bits),
                "metadata_observed_count": int(sum(mask_bits)),
                "age_conflict": age_conflict,
                "sex_conflict": sex_conflict,
                "diabetes_duration_conflict": duration_conflict,
                "insulin_conflict": insulin_conflict,
            }
        )
    return pd.DataFrame(rows)


def _camera_group(cameras: pd.Series) -> str:
    values = sorted(set(cameras.astype(str)))
    if len(values) > 1:
        return "mixed"
    mapping = {
        "Canon CR": "canon",
        "NIKON NF5050": "nikon",
        "Phelcom Eyer": "phelcom",
    }
    if values[0] not in mapping:
        raise ValueError(f"Unknown camera: {values[0]}")
    return mapping[values[0]]


def _bilateral_truth(images: pd.DataFrame, *, quality_required: bool) -> pd.DataFrame:
    eligible = images.loc[images["image_icdr"].notna()].copy()
    if quality_required:
        eligible = eligible.loc[eligible["quality_ok"]].copy()
    eyes = (
        eligible.groupby(["patient_id", "laterality"], sort=True)
        .agg(
            eye_grade=("image_icdr", "max"),
            eye_image_count=("image_id", "size"),
            eye_label_nunique=("image_icdr", "nunique"),
        )
        .reset_index()
    )
    grades = eyes.pivot(index="patient_id", columns="laterality", values="eye_grade")
    counts = eyes.pivot(index="patient_id", columns="laterality", values="eye_image_count")
    conflicts = eyes.pivot(index="patient_id", columns="laterality", values="eye_label_nunique")
    for laterality in ("left", "right"):
        if laterality not in grades:
            grades[laterality] = np.nan
            counts[laterality] = np.nan
            conflicts[laterality] = np.nan
    complete = grades.dropna(subset=["left", "right"]).index
    result = pd.DataFrame(index=complete)
    result["left_eye_grade"] = grades.loc[complete, "left"].astype(int)
    result["right_eye_grade"] = grades.loc[complete, "right"].astype(int)
    result["left_image_count"] = counts.loc[complete, "left"].astype(int)
    result["right_image_count"] = counts.loc[complete, "right"].astype(int)
    result["within_eye_label_conflict"] = (
        conflicts.loc[complete, ["left", "right"]].fillna(0).gt(1).any(axis=1)
    )
    result.index = result.index.astype(str)
    result.index.name = "patient_id"
    return result.reset_index()


def _build_dataset_cohort(
    dataset: str,
    images: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata = _patient_metadata(images)
    main_truth = _bilateral_truth(images, quality_required=True)
    sensitivity_truth = _bilateral_truth(images, quality_required=False)
    main_ids = set(main_truth["patient_id"])
    eligible_images = images.loc[
        images["patient_id"].isin(main_ids)
        & images["quality_ok"]
        & images["image_icdr"].notna()
    ].copy()

    patient = main_truth.merge(metadata, on="patient_id", how="left", validate="one_to_one")
    patient["dataset"] = dataset
    patient["patient_uid"] = dataset + ":" + patient["patient_id"]
    patient["patient_grade"] = patient[["left_eye_grade", "right_eye_grade"]].max(axis=1).astype(int)
    patient["patient_rDR"] = (
        patient["patient_grade"] >= int(config["cohort"]["referable_dr_min_grade"])
    ).astype(int)
    patient["patient_any_dr"] = (
        patient["patient_grade"] >= int(config["cohort"]["any_dr_min_grade"])
    ).astype(int)
    patient["patient_vtDR"] = (
        patient["patient_grade"] >= int(config["cohort"]["vision_threatening_dr_min_grade"])
    ).astype(int)
    patient["total_image_count"] = patient["left_image_count"] + patient["right_image_count"]
    patient["inter_eye_discordance"] = patient["left_eye_grade"].ne(patient["right_eye_grade"])

    camera = eligible_images.groupby("patient_id")["camera"].agg(_camera_group).rename("camera_group")
    audit = eligible_images.groupby("patient_id").agg(
        mean_foreground_fraction=("foreground_fraction", "mean"),
        mean_synthetic_padding_fraction=("synthetic_padding_fraction", "mean"),
        max_synthetic_padding_fraction=("synthetic_padding_fraction", "max"),
        any_crop_fallback=("crop_fallback", "max"),
    )
    patient = patient.merge(camera, on="patient_id", how="left", validate="one_to_one")
    patient = patient.merge(audit, on="patient_id", how="left", validate="one_to_one")

    eye_grade_lookup = (
        eligible_images.groupby(["patient_id", "laterality"])["image_icdr"].max().rename("eye_grade")
    )
    image = eligible_images.merge(
        eye_grade_lookup.reset_index(), on=["patient_id", "laterality"], how="left", validate="many_to_one"
    )
    image = image.merge(
        patient[["patient_id", "patient_uid", "patient_grade", "patient_rDR"]],
        on="patient_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_patient"),
    )
    image = image.rename(columns={"output_path": "processed_path"})
    image["eye_grade"] = image["eye_grade"].astype(int)
    image["image_icdr"] = image["image_icdr"].astype(int)

    # Explain exclusions without using clinical metadata in the decision.
    labelled = images.loc[images["image_icdr"].notna()]
    all_patients = pd.DataFrame({"patient_id": sorted(images["patient_id"].unique())})
    for prefix, subset in (("labelled", labelled), ("quality", labelled.loc[labelled["quality_ok"]])):
        counts = subset.groupby(["patient_id", "laterality"])["image_id"].size().unstack(fill_value=0)
        for laterality in ("left", "right"):
            if laterality not in counts:
                counts[laterality] = 0
        counts = counts[["left", "right"]].rename(
            columns={side: f"{prefix}_{side}_images" for side in ("left", "right")}
        )
        all_patients = all_patients.merge(counts, on="patient_id", how="left")
    count_columns = [column for column in all_patients if column.endswith("_images")]
    all_patients[count_columns] = all_patients[count_columns].fillna(0).astype(int)
    all_patients["included_main"] = all_patients["patient_id"].isin(main_ids)

    def exclusion_reason(row: pd.Series) -> str:
        if bool(row["included_main"]):
            return "included"
        reasons: list[str] = []
        if row["labelled_left_images"] == 0:
            reasons.append("no_gradable_left_eye")
        if row["labelled_right_images"] == 0:
            reasons.append("no_gradable_right_eye")
        if row["labelled_left_images"] > 0 and row["quality_left_images"] == 0:
            reasons.append("no_quality_accepted_left_eye")
        if row["labelled_right_images"] > 0 and row["quality_right_images"] == 0:
            reasons.append("no_quality_accepted_right_eye")
        return ";".join(reasons) or "not_eligible"

    all_patients["exclusion_reason"] = all_patients.apply(exclusion_reason, axis=1)
    excluded = all_patients.loc[~all_patients["included_main"]].copy()
    excluded.insert(0, "dataset", dataset)
    excluded.insert(2, "patient_uid", dataset + ":" + excluded["patient_id"])

    sensitivity_grade = sensitivity_truth[["left_eye_grade", "right_eye_grade"]].max(axis=1).astype(int)
    summary = {
        "main_patient_count": int(len(patient)),
        "main_grade_counts": {
            str(int(key)): int(value)
            for key, value in patient["patient_grade"].value_counts().sort_index().items()
        },
        "main_referable_dr_count": int(patient["patient_rDR"].sum()),
        "main_image_count": int(len(image)),
        "sensitivity_patient_count": int(len(sensitivity_truth)),
        "sensitivity_referable_dr_count": int(
            (sensitivity_grade >= int(config["cohort"]["referable_dr_min_grade"])).sum()
        ),
        "excluded_patient_count": int(len(excluded)),
        "within_eye_label_conflict_patients": int(patient["within_eye_label_conflict"].sum()),
        "inter_eye_discordant_patients": int(patient["inter_eye_discordance"].sum()),
        "metadata_conflicts": {
            column.replace("_conflict", ""): int(patient[column].sum())
            for column in [
                "age_conflict",
                "sex_conflict",
                "diabetes_duration_conflict",
                "insulin_conflict",
            ]
        },
    }
    return patient, image, excluded, summary


def _target_split_sizes(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    expected = {name: total * float(ratio) for name, ratio in ratios.items()}
    sizes = {name: int(math.floor(value)) for name, value in expected.items()}
    remaining = total - sum(sizes.values())
    order = sorted(ratios, key=lambda name: (-(expected[name] - sizes[name]), list(ratios).index(name)))
    for name in order[:remaining]:
        sizes[name] += 1
    return sizes


def _integer_allocation(
    row_counts: Mapping[object, int],
    column_totals: Mapping[str, int],
    *,
    seed: int,
    namespace: str,
) -> pd.DataFrame:
    rows = list(row_counts)
    columns = list(column_totals)
    total = int(sum(row_counts.values()))
    if total != int(sum(column_totals.values())):
        raise ValueError("Allocation margins do not sum to the same total")
    expected = pd.DataFrame(
        [
            [float(row_counts[row]) * float(column_totals[column]) / total for column in columns]
            for row in rows
        ],
        index=rows,
        columns=columns,
    )
    allocation = np.floor(expected).astype(int)
    row_deficit = pd.Series(row_counts, dtype=int) - allocation.sum(axis=1)
    column_deficit = pd.Series(column_totals, dtype=int) - allocation.sum(axis=0)
    while int(row_deficit.sum()) > 0:
        candidates: list[tuple[float, float, object, str]] = []
        for row in rows:
            if row_deficit[row] <= 0:
                continue
            for column in columns:
                if column_deficit[column] <= 0:
                    continue
                residual = float(expected.loc[row, column] - allocation.loc[row, column])
                tie = _stable_unit(seed, namespace, row, column, int(allocation.loc[row, column]))
                candidates.append((residual, tie, row, column))
        if not candidates:
            raise RuntimeError("Unable to complete integer allocation")
        _, _, row, column = max(candidates, key=lambda item: (item[0], item[1]))
        allocation.loc[row, column] += 1
        row_deficit[row] -= 1
        column_deficit[column] -= 1
    if allocation.sum(axis=1).to_dict() != {key: int(value) for key, value in row_counts.items()}:
        raise AssertionError("Allocation row margins are incorrect")
    if allocation.sum(axis=0).to_dict() != {key: int(value) for key, value in column_totals.items()}:
        raise AssertionError("Allocation column margins are incorrect")
    return allocation


def assign_development_splits(
    patients: pd.DataFrame,
    split_config: Mapping[str, Any],
    *,
    seed: int,
) -> pd.Series:
    """Assign deterministic patient splits with exact grade and size margins."""

    ratios = {name: float(value) for name, value in split_config["ratios"].items()}
    split_sizes = _target_split_sizes(len(patients), ratios)
    grade_counts = patients["patient_grade"].value_counts().sort_index().to_dict()
    grade_quota = _integer_allocation(
        grade_counts, split_sizes, seed=seed, namespace="patient_grade"
    )

    primary_quotas: dict[tuple[int, str], dict[str, int]] = {}
    for grade in grade_quota.index:
        subset = patients.loc[patients["patient_grade"].eq(grade)]
        camera_counts = subset["camera_group"].value_counts().sort_index().to_dict()
        camera_quota = _integer_allocation(
            camera_counts,
            {split: int(grade_quota.loc[grade, split]) for split in grade_quota.columns},
            seed=seed,
            namespace=f"camera_within_grade_{grade}",
        )
        for camera in camera_quota.index:
            primary_quotas[(int(grade), str(camera))] = {
                split: int(camera_quota.loc[camera, split]) for split in camera_quota.columns
            }

    secondary = list(split_config["secondary_strata_priority"])
    weights = {key: float(value) for key, value in split_config["secondary_weights"].items()}
    secondary_targets: dict[str, pd.DataFrame] = {}
    for feature in secondary:
        counts = patients[feature].astype(str).value_counts().sort_index().to_dict()
        secondary_targets[feature] = _integer_allocation(
            counts, split_sizes, seed=seed, namespace=f"secondary_{feature}"
        )
    secondary_current = {
        feature: pd.DataFrame(
            0,
            index=secondary_targets[feature].index,
            columns=secondary_targets[feature].columns,
            dtype=int,
        )
        for feature in secondary
    }

    working = patients.copy()
    working[secondary] = working[secondary].astype(str)
    combo_frequency = working[secondary].value_counts().rename("combo_frequency").reset_index()
    working = working.merge(combo_frequency, on=secondary, how="left", validate="many_to_one")
    assignments: dict[str, str] = {}
    primary_sizes = working.groupby(["patient_grade", "camera_group"]).size().sort_values()
    for (grade, camera), _ in primary_sizes.items():
        stratum = working.loc[
            working["patient_grade"].eq(grade) & working["camera_group"].eq(camera)
        ].copy()
        stratum["tie_order"] = stratum["patient_uid"].map(
            lambda uid: _stable_unit(seed, "patient_order", uid)
        )
        stratum = stratum.sort_values(
            ["combo_frequency", "tie_order", "patient_uid"], kind="mergesort"
        )
        remaining = primary_quotas[(int(grade), str(camera))].copy()
        for row in stratum.itertuples(index=False):
            candidates = [split for split in ratios if remaining[split] > 0]
            if not candidates:
                raise RuntimeError(f"No split capacity remains for stratum {(grade, camera)}")
            scores: dict[str, float] = {}
            for split in candidates:
                score = 0.0
                for feature in secondary:
                    category = str(getattr(row, feature))
                    target = int(secondary_targets[feature].loc[category, split])
                    current = int(secondary_current[feature].loc[category, split])
                    score += weights[feature] * (target - current) / max(1, target)
                score += 1e-6 * _stable_unit(seed, "split_tie", row.patient_uid, split)
                scores[split] = score
            chosen = max(candidates, key=lambda split: scores[split])
            assignments[str(row.patient_uid)] = chosen
            remaining[chosen] -= 1
            for feature in secondary:
                category = str(getattr(row, feature))
                secondary_current[feature].loc[category, chosen] += 1
        if any(value != 0 for value in remaining.values()):
            raise AssertionError(f"Unfilled primary quota for {(grade, camera)}: {remaining}")

    result = patients["patient_uid"].map(assignments)
    if result.isna().any() or result.value_counts().to_dict() != split_sizes:
        raise AssertionError("Final patient split sizes are incorrect")
    return result


def _verify_expected_counts(
    dataset: str,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    expected = config["expected_main_cohort"][dataset]
    if int(summary["main_patient_count"]) != int(expected["patient_count"]):
        raise AssertionError(f"{dataset} main patient count differs from protocol")
    expected_grades = {str(key): int(value) for key, value in expected["grade_counts"].items()}
    if summary["main_grade_counts"] != expected_grades:
        raise AssertionError(f"{dataset} grade distribution differs from protocol")
    if int(summary["main_referable_dr_count"]) != int(expected["referable_dr_count"]):
        raise AssertionError(f"{dataset} rDR count differs from protocol")
    sensitivity = config["expected_sensitivity_cohort"][dataset]
    if int(summary["sensitivity_patient_count"]) != int(sensitivity["patient_count"]):
        raise AssertionError(f"{dataset} sensitivity patient count differs from protocol")
    if int(summary["sensitivity_referable_dr_count"]) != int(sensitivity["referable_dr_count"]):
        raise AssertionError(f"{dataset} sensitivity rDR count differs from protocol")


def _split_statistics(patients: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "dataset",
        "patient_grade",
        "patient_rDR",
        "camera_group",
        "sex",
        "age_bin",
        "metadata_mask",
        "inter_eye_discordance",
        "within_eye_label_conflict",
    ]
    rows: list[dict[str, object]] = []
    for split in SPLIT_ORDER:
        subset = patients.loc[patients["split"].eq(split)]
        if subset.empty:
            continue
        for dimension in dimensions:
            counts = subset[dimension].astype(str).value_counts(dropna=False).sort_index()
            for category, count in counts.items():
                rows.append(
                    {
                        "split": split,
                        "dimension": dimension,
                        "category": category,
                        "n": int(count),
                        "percent_within_split": 100.0 * int(count) / len(subset),
                    }
                )
    return pd.DataFrame(rows)


def _cohort_flow(
    dataset_summaries: Mapping[str, Mapping[str, Any]],
    input_audits: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in sorted(dataset_summaries):
        summary = dataset_summaries[dataset]
        audit = input_audits[dataset]
        values = [
            ("raw_images", audit["raw_rows"]),
            ("raw_patients", audit["raw_patients"]),
            ("bilateral_gradable_any_quality_patients", summary["sensitivity_patient_count"]),
            ("main_quality_bilateral_patients", summary["main_patient_count"]),
            ("main_quality_images", summary["main_image_count"]),
            ("main_referable_dr_patients", summary["main_referable_dr_count"]),
            ("excluded_from_main_patients", summary["excluded_patient_count"]),
        ]
        for stage, count in values:
            rows.append({"dataset": dataset, "stage": stage, "count": int(count)})
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.8g")


def _build_report(
    patients: pd.DataFrame,
    images: pd.DataFrame,
    summaries: Mapping[str, Mapping[str, Any]],
    patient_manifest_hash: str,
) -> str:
    split_counts = patients.groupby("split").agg(
        patients=("patient_uid", "size"),
        rdr_positive=("patient_rDR", "sum"),
        rdr_prevalence=("patient_rDR", "mean"),
    ).reindex(SPLIT_ORDER)
    grade_table = pd.crosstab(patients["split"], patients["patient_grade"]).reindex(SPLIT_ORDER).fillna(0).astype(int)
    lines = [
        "# patient_split_v1 患者级队列与冻结划分报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 主分析定义",
        "",
        "- 眼级 ICDR：同眼合格图像取最大值；",
        "- 患者级 ICDR：左右眼眼级 ICDR 取最大值；",
        "- 主终点：患者级 rDR，ICDR >= 2；",
        "- 主队列要求双眼均至少有一张质量合格且 ICDR 非缺失图像；",
        "- 临床变量不参与真值生成；患者内冲突值置为缺失并记录冲突标志。",
        "",
        "## 冻结划分",
        "",
        "| split | patients | rDR positive | prevalence |",
        "|---|---:|---:|---:|",
    ]
    for split, row in split_counts.dropna().iterrows():
        lines.append(
            f"| {split} | {int(row.patients)} | {int(row.rdr_positive)} | {row.rdr_prevalence:.4f} |"
        )
    lines.extend(
        [
            "",
            "患者级 ICDR 分布：",
            "",
            "| split | 0 | 1 | 2 | 3 | 4 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, row in grade_table.iterrows():
        lines.append(
            f"| {split} | " + " | ".join(str(int(row.get(grade, 0))) for grade in range(5)) + " |"
        )
    lines.extend(
        [
            "",
            "## 数据集队列",
            "",
        ]
    )
    for dataset in ("brset", "mbrset"):
        summary = summaries[dataset]
        lines.append(
            f"- {dataset}: 主队列 {summary['main_patient_count']} 人、{summary['main_image_count']} 张图像、"
            f"rDR {summary['main_referable_dr_count']} 人；全质量敏感性队列 "
            f"{summary['sensitivity_patient_count']} 人。"
        )
    lines.extend(
        [
            "",
            "## 防泄漏与使用约束",
            "",
            "- BRSET 患者只出现在 train、validation 或 internal_test 之一；",
            "- mBRSET 全部固定为 external_test；",
            "- mBRSET 不得用于训练、早停、阈值选择、填补参数或模型选择；",
            "- 图像和临床变量通过 patient_uid 继承同一患者划分；",
            f"- 冻结患者清单 SHA256：`{patient_manifest_hash}`。",
            "",
            f"总计：{len(patients)} 名患者，{len(images)} 张主分析图像。",
            "",
        ]
    )
    return "\n".join(lines)


def build_and_freeze_cohort(
    *,
    project_root: Path,
    config_path: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = load_cohort_config(config_path)
    output = (output_root or (project_root / str(config["output"]["root"]))).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    preprocessing_path = project_root / str(config["preprocessing"]["manifest"])
    preprocessing = pd.read_csv(preprocessing_path, dtype=str, keep_default_na=False)
    if set(preprocessing["dataset"]) != set(config["datasets"]):
        raise ValueError("Preprocessing manifest datasets differ from cohort config")

    standardised: dict[str, pd.DataFrame] = {}
    input_audits: dict[str, dict[str, int]] = {}
    for dataset, spec in config["datasets"].items():
        standardised[dataset], input_audits[dataset] = _standardise_dataset(
            dataset, spec, project_root, preprocessing
        )

    patient_frames: list[pd.DataFrame] = []
    image_frames: list[pd.DataFrame] = []
    excluded_frames: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, Any]] = {}
    for dataset in config["datasets"]:
        patient, image, excluded, summary = _build_dataset_cohort(
            dataset, standardised[dataset], config
        )
        _verify_expected_counts(dataset, summary, config)
        patient_frames.append(patient)
        image_frames.append(image)
        excluded_frames.append(excluded)
        summaries[dataset] = summary

    patients = pd.concat(patient_frames, ignore_index=True)
    images = pd.concat(image_frames, ignore_index=True)
    excluded = pd.concat(excluded_frames, ignore_index=True)
    development = str(config["split"]["development_dataset"])
    external = str(config["split"]["external_test_dataset"])
    development_mask = patients["dataset"].eq(development)
    patients.loc[development_mask, "split"] = assign_development_splits(
        patients.loc[development_mask].copy(), config["split"], seed=int(config["seed"])
    ).to_numpy()
    patients.loc[patients["dataset"].eq(external), "split"] = str(
        config["split"]["external_split_name"]
    )
    if patients["split"].isna().any():
        raise AssertionError("Some patients were not assigned a split")

    images = images.drop(columns=["split"], errors="ignore").merge(
        patients[["patient_uid", "split"]], on="patient_uid", how="left", validate="many_to_one"
    )
    if images["split"].isna().any():
        raise AssertionError("Some images were not assigned a patient split")

    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    patients["_split_rank"] = patients["split"].map(split_rank)
    patients["_patient_num"] = pd.to_numeric(patients["patient_id"], errors="coerce")
    patients = patients.sort_values(
        ["_split_rank", "dataset", "_patient_num", "patient_id"], kind="mergesort"
    ).drop(columns=["_split_rank", "_patient_num"])
    images["_split_rank"] = images["split"].map(split_rank)
    images["_patient_num"] = pd.to_numeric(images["patient_id"], errors="coerce")
    images = images.sort_values(
        ["_split_rank", "dataset", "_patient_num", "patient_id", "laterality", "image_id"],
        kind="mergesort",
    ).drop(columns=["_split_rank", "_patient_num"])
    excluded["_patient_num"] = pd.to_numeric(excluded["patient_id"], errors="coerce")
    excluded = excluded.sort_values(["dataset", "_patient_num", "patient_id"], kind="mergesort").drop(
        columns="_patient_num"
    )

    # Leakage checks are computed before writing and saved as a frozen audit.
    development_sets = {
        split: set(patients.loc[patients["split"].eq(split), "patient_uid"])
        for split in ("train", "validation", "internal_test")
    }
    overlaps = {
        f"{left}__{right}": len(development_sets[left].intersection(development_sets[right]))
        for index, left in enumerate(development_sets)
        for right in list(development_sets)[index + 1 :]
    }
    patient_image_split_match = images.merge(
        patients[["patient_uid", "split"]], on="patient_uid", suffixes=("_image", "_patient")
    )
    leakage_audit = {
        "passed": bool(
            all(value == 0 for value in overlaps.values())
            and patients.loc[development_mask, "dataset"].eq(development).all()
            and patients.loc[patients["split"].eq("external_test"), "dataset"].eq(external).all()
            and patient_image_split_match["split_image"].eq(patient_image_split_match["split_patient"]).all()
            and not images["image_id"].duplicated().any()
        ),
        "development_split_patient_overlaps": overlaps,
        "unique_patient_uids": int(patients["patient_uid"].nunique()),
        "patient_rows": int(len(patients)),
        "unique_images": int(images["image_id"].nunique()),
        "image_rows": int(len(images)),
        "all_mbrset_external_test": bool(
            patients.loc[patients["dataset"].eq(external), "split"].eq("external_test").all()
        ),
        "all_image_splits_match_patient": bool(
            patient_image_split_match["split_image"].eq(patient_image_split_match["split_patient"]).all()
        ),
    }
    if not leakage_audit["passed"]:
        raise AssertionError(f"Leakage audit failed: {leakage_audit}")

    patients = patients[PATIENT_MANIFEST_COLUMNS]
    images = images[IMAGE_MANIFEST_COLUMNS]
    statistics = _split_statistics(patients)
    flow = _cohort_flow(summaries, input_audits)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        paths = {
            "patient_split_manifest.csv": temporary / "patient_split_manifest.csv",
            "image_split_manifest.csv": temporary / "image_split_manifest.csv",
            "excluded_patient_audit.csv": temporary / "excluded_patient_audit.csv",
            "cohort_flow.csv": temporary / "cohort_flow.csv",
            "split_statistics.csv": temporary / "split_statistics.csv",
            "leakage_audit.json": temporary / "leakage_audit.json",
        }
        _write_csv(patients, paths["patient_split_manifest.csv"])
        _write_csv(images, paths["image_split_manifest.csv"])
        _write_csv(excluded, paths["excluded_patient_audit.csv"])
        _write_csv(flow, paths["cohort_flow.csv"])
        _write_csv(statistics, paths["split_statistics.csv"])
        paths["leakage_audit.json"].write_text(
            json.dumps(leakage_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        patient_hash = sha256_file(paths["patient_split_manifest.csv"])
        report_path = temporary / "cohort_report.md"
        report_path.write_text(
            _build_report(patients, images, summaries, patient_hash), encoding="utf-8"
        )
        paths["cohort_report.md"] = report_path
        artifact_hashes = {name: sha256_file(path) for name, path in paths.items()}

        input_hashes = {
            _portable_path(config_path, project_root): sha256_file(config_path),
            _portable_path(preprocessing_path, project_root): sha256_file(preprocessing_path),
        }
        for spec in config["datasets"].values():
            path = project_root / str(spec["metadata_csv"])
            input_hashes[_portable_path(path, project_root)] = sha256_file(path)
        metadata = {
            "version": config["version"],
            "frozen": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(config["seed"]),
            "algorithm": "hierarchical_integer_quota_greedy_v1",
            "patient_split_manifest_sha256": patient_hash,
            "input_sha256": input_hashes,
            "artifact_sha256": artifact_hashes,
            "dataset_summaries": summaries,
            "input_audits": input_audits,
            "split_patient_counts": {
                str(key): int(value) for key, value in patients["split"].value_counts().items()
            },
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
        }
        metadata_path = temporary / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_entries = {**artifact_hashes, "run_metadata.json": sha256_file(metadata_path)}
        (temporary / "checksums.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksum_entries.items())),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "version": config["version"],
        "output_root": str(output),
        "patient_manifest": str(output / "patient_split_manifest.csv"),
        "patient_manifest_sha256": patient_hash,
        "patient_count": int(len(patients)),
        "image_count": int(len(images)),
        "split_patient_counts": {
            str(key): int(value) for key, value in patients["split"].value_counts().items()
        },
        "dataset_summaries": summaries,
        "leakage_audit_passed": bool(leakage_audit["passed"]),
    }

