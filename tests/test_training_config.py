from __future__ import annotations

from pathlib import Path

import pytest

from dr_grading.training.config import (
    ConfigError,
    config_sha256,
    load_experiment_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,model_type",
    [
        ("b0_prevalence_v1.yaml", "prevalence_constant"),
        ("b1_clinical_v1.yaml", "clinical_mlp"),
        ("b1_mask_v1.yaml", "clinical_mlp"),
        ("b2_convnext_tiny_v1.yaml", "bilateral_independent_image"),
    ],
)
def test_formal_training_configs_resolve_and_validate(
    name: str, model_type: str
) -> None:
    config = load_experiment_config(PROJECT_ROOT / "configs/experiment" / name)
    assert config["model"]["type"] == model_type
    assert config["data"]["train_split"] == "train"
    assert config["data"]["validation_split"] == "validation"
    assert config["evaluation"]["splits"] == ["validation"]
    assert len(config_sha256(config)) == 64


def test_dotted_override_is_typed_and_does_not_mutate_protocol() -> None:
    path = PROJECT_ROOT / "configs/experiment/b1_clinical_v1.yaml"
    config = load_experiment_config(
        path,
        overrides=["training.batch_size=16", "runtime.deterministic=false"],
    )
    assert config["training"]["batch_size"] == 16
    assert config["runtime"]["deterministic"] is False
    assert config["task"]["positive_grade_threshold"] == 2


def test_training_entry_rejects_held_out_test_access() -> None:
    path = PROJECT_ROOT / "configs/experiment/b1_clinical_v1.yaml"
    with pytest.raises(ConfigError, match="held-out test"):
        load_experiment_config(
            path, overrides=["evaluation.splits=[validation,external_test]"]
        )


def test_training_entry_rejects_primary_endpoint_drift() -> None:
    path = PROJECT_ROOT / "configs/experiment/b1_clinical_v1.yaml"
    with pytest.raises(ConfigError, match="ICDR >= 2"):
        load_experiment_config(path, overrides=["task.positive_grade_threshold=1"])
