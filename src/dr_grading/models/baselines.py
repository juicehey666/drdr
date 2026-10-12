"""Protocol-aligned B0/B1/B1-mask/B2 patient-level baselines."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class PrevalenceConstant(nn.Module):
    """B0: emit the frozen BRSET-training prevalence for every patient."""

    def __init__(self, prevalence: float) -> None:
        super().__init__()
        if not 0.0 < prevalence < 1.0:
            raise ValueError("Prevalence must lie strictly between 0 and 1")
        logit = math.log(prevalence / (1.0 - prevalence))
        self.register_buffer("constant_logit", torch.tensor(logit, dtype=torch.float32))

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        batch_size = int(batch["patient_rdr"].shape[0])
        return {"patient_logits": self.constant_logit.expand(batch_size)}


class ClinicalMLP(nn.Module):
    """B1/B1-mask with an explicit and auditable input mode."""

    def __init__(
        self,
        *,
        input_mode: str,
        feature_count: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_mode not in {"values_only", "mask_only", "values_and_mask"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        self.input_mode = input_mode
        input_dim = feature_count * (2 if input_mode == "values_and_mask" else 1)
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, int(hidden)),
                    nn.LayerNorm(int(hidden)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(hidden)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def _inputs(self, batch: Mapping[str, Any]) -> torch.Tensor:
        values = batch["clinical_features"]
        mask = batch["clinical_mask"].to(dtype=values.dtype)
        if self.input_mode == "values_only":
            return values
        if self.input_mode == "mask_only":
            return mask
        return torch.cat([values, mask], dim=1)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"patient_logits": self.network(self._inputs(batch)).squeeze(1)}


class ConvNeXtTinyEncoder(nn.Module):
    """Torchvision ConvNeXt-Tiny feature encoder with optional ImageNet weights."""

    output_dim = 768

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = self.avgpool(features)
        features = self.norm(features)
        return torch.flatten(features, 1)


class BilateralIndependentImageModel(nn.Module):
    """B2: shared eye encoder without cross-eye feature interaction."""

    def __init__(
        self, *, pretrained: bool, freeze_encoder: bool, dropout: float
    ) -> None:
        super().__init__()
        self.encoder = ConvNeXtTinyEncoder(pretrained=pretrained)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(768, 1))

    def _eye_features(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, views = images.shape[:2]
        flat_images = images.reshape(batch_size * views, *images.shape[2:])
        flat_mask = mask.reshape(-1)
        encoded_valid = self.encoder(flat_images[flat_mask])
        encoded = encoded_valid.new_zeros((batch_size * views, encoded_valid.shape[1]))
        encoded[flat_mask] = encoded_valid
        encoded = encoded.reshape(batch_size, views, -1)
        weights = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        left = self._eye_features(batch["left_images"], batch["left_view_mask"])
        right = self._eye_features(batch["right_images"], batch["right_view_mask"])
        left_logit = self.head(left).squeeze(1)
        right_logit = self.head(right).squeeze(1)
        patient_logit = torch.maximum(left_logit, right_logit)
        return {
            "patient_logits": patient_logit,
            "left_eye_logits": left_logit,
            "right_eye_logits": right_logit,
        }


def build_model(config: Mapping[str, Any], *, train_prevalence: float) -> nn.Module:
    model_type = str(config["type"])
    if model_type == "prevalence_constant":
        return PrevalenceConstant(train_prevalence)
    if model_type == "clinical_mlp":
        return ClinicalMLP(
            input_mode=str(config["input_mode"]),
            feature_count=int(config["feature_count"]),
            hidden_dims=[int(value) for value in config["hidden_dims"]],
            dropout=float(config["dropout"]),
        )
    if model_type == "bilateral_independent_image":
        return BilateralIndependentImageModel(
            pretrained=bool(config["pretrained"]),
            freeze_encoder=bool(config["freeze_encoder"]),
            dropout=float(config["dropout"]),
        )
    raise ValueError(f"Unsupported model type: {model_type}")
