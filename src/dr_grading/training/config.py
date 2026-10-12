"""Composable, dependency-light experiment configuration.

YAML files may declare a ``defaults`` list. Entries are resolved relative to
the declaring file and merged from left to right. The declaring file is then
merged last. Command-line overrides use dotted keys, for example
``training.batch_size=4``.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the frozen contract."""


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(value) for value in (*stack, path))
        raise ConfigError(f"Configuration inheritance cycle: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    defaults = data.get("defaults", [])
    if not isinstance(defaults, list):
        raise ConfigError(f"defaults must be a list: {path}")
    merged: dict[str, Any] = {}
    for entry in defaults:
        if not isinstance(entry, str):
            raise ConfigError(f"defaults entries must be paths: {path}")
        parent = (path.parent / entry).resolve()
        merged = _deep_merge(merged, _load_yaml_tree(parent, (*stack, path)))
    local = {key: value for key, value in data.items() if key != "defaults"}
    return _deep_merge(merged, local)


def _set_dotted(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not all(parts):
        raise ConfigError(f"Invalid override key: {dotted_key!r}")
    cursor: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, MutableMapping):
            raise ConfigError(
                f"Cannot descend into non-mapping override key: {dotted_key}"
            )
        cursor = current
    cursor[parts[-1]] = value


def apply_overrides(
    config: Mapping[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for expression in overrides:
        if "=" not in expression:
            raise ConfigError(f"Override must use key=value syntax: {expression!r}")
        key, raw_value = expression.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_dotted(result, key.strip(), value)
    return result


def _require(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        value = value[part]
    return value


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    if _require(config, "schema_version") != "patient_rdr_training_v1":
        raise ConfigError("Unsupported training configuration schema")
    if _require(config, "task.name") != "patient_rdr":
        raise ConfigError("The v1 training entry point supports patient_rdr only")
    if int(_require(config, "task.positive_grade_threshold")) != 2:
        raise ConfigError("The frozen primary task requires ICDR >= 2")
    if _require(config, "data.train_split") != "train":
        raise ConfigError("Training split must remain 'train'")
    if _require(config, "data.validation_split") != "validation":
        raise ConfigError("Validation split must remain 'validation'")
    evaluation_splits = list(_require(config, "evaluation.splits"))
    forbidden = {"internal_test", "external_test"}.intersection(evaluation_splits)
    if forbidden:
        raise ConfigError(
            "The training entry point cannot access held-out test splits: "
            f"{sorted(forbidden)}"
        )
    if set(evaluation_splits) != {"validation"}:
        raise ConfigError("Training-time evaluation is restricted to validation")

    model_type = str(_require(config, "model.type"))
    allowed_models = {
        "prevalence_constant",
        "clinical_mlp",
        "bilateral_independent_image",
    }
    if model_type not in allowed_models:
        raise ConfigError(f"Unsupported model.type: {model_type}")
    if model_type == "clinical_mlp":
        mode = str(_require(config, "model.input_mode"))
        if mode not in {"values_only", "mask_only", "values_and_mask"}:
            raise ConfigError(f"Unsupported clinical input_mode: {mode}")
        if int(_require(config, "model.feature_count")) != 4:
            raise ConfigError("The frozen clinical feature set contains four variables")
        if bool(_require(config, "data.load_images")):
            raise ConfigError("Clinical-only baselines must not perform image I/O")
    if model_type == "prevalence_constant" and bool(
        _require(config, "data.load_images")
    ):
        raise ConfigError("The prevalence baseline must not perform image I/O")
    if model_type == "bilateral_independent_image":
        if not bool(_require(config, "data.load_images")):
            raise ConfigError("The B2 image baseline requires image loading")
        if _require(config, "model.backbone") != "convnext_tiny":
            raise ConfigError("The v1 image baseline currently supports convnext_tiny")
        if _require(config, "model.view_pooling") != "masked_mean":
            raise ConfigError(
                "The frozen B2 baseline requires masked_mean view pooling"
            )
        if _require(config, "model.patient_pooling") != "max_logit":
            raise ConfigError(
                "The frozen B2 baseline requires max_logit patient pooling"
            )

    if int(_require(config, "training.epochs")) < 1:
        raise ConfigError("training.epochs must be positive")
    if int(_require(config, "training.batch_size")) < 1:
        raise ConfigError("training.batch_size must be positive")
    if int(_require(config, "training.gradient_accumulation_steps")) < 1:
        raise ConfigError("gradient_accumulation_steps must be positive")
    if float(_require(config, "optimizer.learning_rate")) <= 0:
        raise ConfigError("optimizer.learning_rate must be positive")
    if _require(config, "selection.metric") != "auprc":
        raise ConfigError("The frozen early-stopping metric is validation AUPRC")
    if _require(config, "selection.mode") != "max":
        raise ConfigError("AUPRC model selection must use mode=max")
    if int(_require(config, "runtime.seed")) < 0:
        raise ConfigError("runtime.seed must be non-negative")
    if _require(config, "runtime.precision") not in {"fp32", "fp16", "bf16"}:
        raise ConfigError("runtime.precision must be fp32, fp16, or bf16")
    if _require(config, "output.existing_policy") != "error":
        raise ConfigError("Run directories must never be overwritten")


def canonical_yaml(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(config), sort_keys=True, allow_unicode=True)


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_yaml(config).encode("utf-8")).hexdigest()


def load_experiment_config(
    path: Path,
    *,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    config = _load_yaml_tree(path)
    config = apply_overrides(config, overrides)
    validate_experiment_config(config)
    return config
