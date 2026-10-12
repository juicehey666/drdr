from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dr_grading.data.cohort import (
    _bilateral_truth,
    _parse_numeric,
    _patient_metadata,
    assign_development_splits,
    load_cohort_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/data/patient_split_v1.yaml"


def test_patient_split_protocol_config_is_locked() -> None:
    config = load_cohort_config(CONFIG_PATH)
    assert config["version"] == "patient_split_v1"
    assert config["seed"] == 20260929
    assert config["split"]["ratios"] == {
        "train": 0.70,
        "validation": 0.10,
        "internal_test": 0.20,
    }
    assert config["cohort"]["required_lateralities"] == ["left", "right"]
    assert config["cohort"]["referable_dr_min_grade"] == 2


def test_bilateral_truth_uses_quality_and_maximum_eye_grade() -> None:
    images = pd.DataFrame(
        [
            {"patient_id": "1", "laterality": "left", "image_id": "a", "image_icdr": 1, "quality_ok": True},
            {"patient_id": "1", "laterality": "left", "image_id": "b", "image_icdr": 3, "quality_ok": True},
            {"patient_id": "1", "laterality": "right", "image_id": "c", "image_icdr": 2, "quality_ok": True},
            {"patient_id": "2", "laterality": "left", "image_id": "d", "image_icdr": 0, "quality_ok": True},
            {"patient_id": "2", "laterality": "right", "image_id": "e", "image_icdr": 4, "quality_ok": False},
        ]
    )
    main = _bilateral_truth(images, quality_required=True).set_index("patient_id")
    sensitivity = _bilateral_truth(images, quality_required=False).set_index("patient_id")
    assert list(main.index) == ["1"]
    assert main.loc["1", "left_eye_grade"] == 3
    assert main.loc["1", "right_eye_grade"] == 2
    assert bool(main.loc["1", "within_eye_label_conflict"])
    assert set(sensitivity.index) == {"1", "2"}


def test_conflicting_patient_metadata_is_set_missing() -> None:
    images = pd.DataFrame(
        {
            "patient_id": ["1", "1", "2", "2"],
            "age_value": [60.0, 61.0, np.nan, 55.0],
            "age_top_coded_raw": [False, False, False, False],
            "sex_value": ["female", "female", "male", "male"],
            "diabetes_duration_value": [10.0, 10.0, np.nan, np.nan],
            "insulin_value": [1.0, 1.0, 0.0, 1.0],
        }
    )
    metadata = _patient_metadata(images).set_index("patient_id")
    assert pd.isna(metadata.loc["1", "age"])
    assert bool(metadata.loc["1", "age_conflict"])
    assert metadata.loc["1", "age_bin"] == "missing"
    assert metadata.loc["1", "metadata_mask"] == "0111"
    assert metadata.loc["2", "age"] == 55.0
    assert not bool(metadata.loc["2", "age_conflict"])
    assert pd.isna(metadata.loc["2", "insulin_use"])
    assert bool(metadata.loc["2", "insulin_conflict"])
    assert metadata.loc["2", "metadata_mask"] == "1100"


def test_numeric_parser_preserves_top_coded_age_and_decimal_comma() -> None:
    values, top_coded, invalid = _parse_numeric(
        pd.Series([">= 90", "10,5", "", "not-a-number"]),
        top_code_mapping={">= 90": 90},
    )
    assert values.iloc[0] == 90.0
    assert values.iloc[1] == 10.5
    assert pd.isna(values.iloc[2])
    assert pd.isna(values.iloc[3])
    assert top_coded.tolist() == [True, False, False, False]
    assert invalid.tolist() == [False, False, False, True]


def _synthetic_patients() -> pd.DataFrame:
    rows = []
    for index in range(100):
        grade = 0 if index < 70 else (1 if index < 80 else (2 if index < 90 else (3 if index < 95 else 4)))
        rows.append(
            {
                "patient_uid": f"brset:{index}",
                "patient_grade": grade,
                "camera_group": "canon" if index % 3 else "nikon",
                "sex": "female" if index % 2 else "male",
                "age_bin": ["<50", "50-64", "65-74", ">=75"][index % 4],
                "metadata_mask": ["1111", "1100", "0100"][index % 3],
                "inter_eye_discordance": bool(index % 5 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_patient_split_is_deterministic_exact_and_patient_level() -> None:
    patients = _synthetic_patients()
    config = load_cohort_config(CONFIG_PATH)
    first = assign_development_splits(patients, config["split"], seed=config["seed"])
    second = assign_development_splits(patients, config["split"], seed=config["seed"])
    assert first.tolist() == second.tolist()
    assert first.value_counts().to_dict() == {
        "train": 70,
        "internal_test": 20,
        "validation": 10,
    }
    assigned = patients.assign(split=first)
    assert assigned.groupby("patient_uid")["split"].nunique().max() == 1
    # Exact grade quotas preserve the rare grades in every split when feasible.
    table = pd.crosstab(assigned["patient_grade"], assigned["split"])
    assert table.loc[4].sum() == 5
    assert (table.loc[2:4, "train"] > 0).all()
    assert (table.loc[2:4, "internal_test"] > 0).all()
