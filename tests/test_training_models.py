from __future__ import annotations

import math

import torch

from dr_grading.models.baselines import ClinicalMLP, PrevalenceConstant
from dr_grading.training.metrics import binary_metrics


def _clinical_batch() -> dict[str, torch.Tensor]:
    return {
        "clinical_features": torch.tensor(
            [[1.0, 0.0, -0.5, 0.2], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
        ),
        "clinical_mask": torch.tensor(
            [[True, True, True, True], [False, True, False, False]], dtype=torch.bool
        ),
        "patient_rdr": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }


def test_prevalence_constant_emits_exact_probability() -> None:
    model = PrevalenceConstant(0.1)
    probabilities = torch.sigmoid(model(_clinical_batch())["patient_logits"])
    assert torch.allclose(probabilities, torch.full((2,), 0.1))


def test_clinical_mlp_supports_values_and_mask_and_mask_only() -> None:
    batch = _clinical_batch()
    for mode in ("values_and_mask", "mask_only"):
        model = ClinicalMLP(
            input_mode=mode,
            feature_count=4,
            hidden_dims=[8],
            dropout=0.0,
        )
        logits = model(batch)["patient_logits"]
        assert logits.shape == (2,)
        assert torch.isfinite(logits).all()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["patient_rdr"]
        )
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_binary_metrics_are_patient_level_and_numerically_correct() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], threshold=0.5)
    assert math.isclose(metrics["auroc"], 1.0)
    assert math.isclose(metrics["auprc"], 1.0)
    assert math.isclose(metrics["sensitivity"], 1.0)
    assert math.isclose(metrics["specificity"], 1.0)
    assert metrics["n"] == 4
