"""Single-process training engine for protocol-locked patient-level baselines."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dr_grading.data.patient_dataset import PatientDataModule, seed_worker
from dr_grading.models import build_model

from .metrics import binary_metrics


class ClinicalPatientDataset(Dataset[dict[str, Any]]):
    """Patient rows with precomputed clinical tensors and no image I/O."""

    def __init__(self, patients: pd.DataFrame, data_module: PatientDataModule) -> None:
        self.patients = patients.reset_index(drop=True).copy()
        self.features, self.mask = data_module.clinical_normalizer.transform_frame(
            self.patients
        )

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.patients.iloc[index]
        return {
            "patient_uid": str(row["patient_uid"]),
            "clinical_features": self.features[index],
            "clinical_mask": self.mask[index],
            "patient_rdr": torch.tensor(float(row["patient_rDR"]), dtype=torch.float32),
        }


def _clinical_loader(
    data_module: PatientDataModule,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    patients = data_module.patients.loc[data_module.patients["split"].eq(split)].copy()
    dataset = ClinicalPatientDataset(patients, data_module)
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )


def _make_loaders(
    data_module: PatientDataModule,
    config: Mapping[str, Any],
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    data_config = config["data"]
    training = config["training"]
    seed = int(config["runtime"]["seed"])
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    if bool(data_config["load_images"]):
        train_loader = data_module.dataloader(
            str(data_config["train_split"]),
            batch_size=batch_size,
            num_workers=num_workers,
            training=True,
        )
        validation_loader = data_module.dataloader(
            str(data_config["validation_split"]),
            batch_size=int(config["evaluation"]["batch_size"]),
            num_workers=num_workers,
            training=False,
        )
    else:
        train_loader = _clinical_loader(
            data_module,
            str(data_config["train_split"]),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            seed=seed,
        )
        validation_loader = _clinical_loader(
            data_module,
            str(data_config["validation_split"]),
            batch_size=int(config["evaluation"]["batch_size"]),
            num_workers=num_workers,
            shuffle=False,
            seed=seed,
        )
    return train_loader, validation_loader


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _evaluate(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    precision: str,
    loss_fn: nn.Module,
    threshold: float,
    max_batches: int | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    targets: list[float] = []
    probabilities: list[float] = []
    patient_uids: list[str] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _move_batch(raw_batch, device)
            with _autocast_context(device, precision):
                logits = model(batch)["patient_logits"]
                loss = loss_fn(logits, batch["patient_rdr"])
            probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
            targets.extend(batch["patient_rdr"].float().cpu().tolist())
            patient_uids.extend(str(value) for value in raw_batch["patient_uid"])
            losses.append(float(loss.detach().cpu()))
    metrics = binary_metrics(targets, probabilities, threshold=threshold)
    metrics["loss"] = float(sum(losses) / len(losses))
    predictions = pd.DataFrame(
        {
            "patient_uid": patient_uids,
            "target": targets,
            "probability": probabilities,
        }
    )
    return metrics, predictions


def _cosine_warmup_lambda(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
    data_hashes: Mapping[str, str],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_metric": best_metric,
            "config": dict(config),
            "data_hashes": dict(data_hashes),
        },
        path,
    )


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_safe(dict(record)), ensure_ascii=False, allow_nan=False)
            + "\n"
        )


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats into JSON null recursively."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _gpu_memory_stats(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    return {
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_experiment(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run B0/B1/B1-mask/B2 using the same entry point and artifact contract."""

    normalizer_path = project_root / str(config["data"]["clinical_normalizer"])
    data_module = PatientDataModule(
        project_root=project_root,
        config_path=project_root / str(config["data"]["dataloader_config"]),
        clinical_normalizer_path=normalizer_path,
        validate_image_paths=bool(config["data"]["load_images"]),
    )
    train_rows = data_module.patients.loc[data_module.patients["split"].eq("train")]
    train_prevalence = float(train_rows["patient_rDR"].mean())
    positive_count = int(train_rows["patient_rDR"].sum())
    negative_count = int(len(train_rows) - positive_count)
    data_hashes = {
        "patient_manifest_sha256": data_module.patient_manifest_sha256,
        "image_manifest_sha256": data_module.image_manifest_sha256,
    }
    model = build_model(config["model"], train_prevalence=train_prevalence).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader, validation_loader = _make_loaders(data_module, config)
    pos_weight = torch.tensor(negative_count / positive_count, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    precision = str(config["runtime"]["precision"])
    threshold = float(config["evaluation"]["development_threshold"])
    max_batches = 1 if dry_run else config["runtime"].get("max_batches")
    if max_batches is not None:
        max_batches = int(max_batches)

    summary: dict[str, Any] = {
        "status": "running",
        "dry_run": dry_run,
        "model_type": config["model"]["type"],
        "device": str(device),
        "train_prevalence": train_prevalence,
        "train_positive": positive_count,
        "train_negative": negative_count,
        "data_hashes": data_hashes,
    }

    if config["model"]["type"] == "prevalence_constant":
        metrics, predictions = _evaluate(
            model,
            validation_loader,
            device=device,
            precision=precision,
            loss_fn=loss_fn,
            threshold=threshold,
            max_batches=max_batches,
        )
        predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
        summary.update(
            {
                "status": "complete",
                "best_epoch": 0,
                "validation": metrics,
                **_gpu_memory_stats(device),
            }
        )
        summary = _json_safe(summary)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return summary

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["optimizer"]["learning_rate"]),
        betas=tuple(float(v) for v in config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["epsilon"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = 1 if dry_run else int(config["training"]["epochs"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    batches_per_epoch = (
        len(train_loader)
        if max_batches is None
        else min(len(train_loader), max_batches)
    )
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / accumulation))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = round(total_steps * float(config["scheduler"]["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_warmup_lambda(
            step, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and precision == "fp16"
    )
    best_metric = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    patience = int(config["selection"]["patience"])
    metrics_path = run_dir / "metrics.jsonl"
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses: list[float] = []
        micro_in_update = 0
        for batch_index, raw_batch in enumerate(train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _move_batch(raw_batch, device)
            with _autocast_context(device, precision):
                logits = model(batch)["patient_logits"]
                raw_loss = loss_fn(logits, batch["patient_rdr"])
                loss = raw_loss / accumulation
            scaler.scale(loss).backward()
            micro_in_update += 1
            train_losses.append(float(raw_loss.detach().cpu()))
            is_last = batch_index + 1 >= batches_per_epoch
            if micro_in_update == accumulation or is_last:
                if micro_in_update < accumulation:
                    correction = accumulation / micro_in_update
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(config["training"]["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                micro_in_update = 0

        validation_metrics, _ = _evaluate(
            model,
            validation_loader,
            device=device,
            precision=precision,
            loss_fn=loss_fn,
            threshold=threshold,
            max_batches=max_batches,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(sum(train_losses) / len(train_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": validation_metrics,
            "elapsed_seconds": time.time() - start_time,
        }
        _append_jsonl(metrics_path, record)
        selection_value = float(validation_metrics[config["selection"]["metric"]])
        improved = selection_value > best_metric + float(
            config["selection"]["minimum_delta"]
        )
        if improved:
            best_metric = selection_value
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            _save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=best_epoch,
                best_metric=best_metric,
                config=config,
                data_hashes=data_hashes,
            )
        else:
            epochs_without_improvement += 1
        _save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_metric=best_metric,
            config=config,
            data_hashes=data_hashes,
        )
        if not dry_run and epochs_without_improvement >= patience:
            break

    best_checkpoint = torch.load(
        run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"])
    final_metrics, predictions = _evaluate(
        model,
        validation_loader,
        device=device,
        precision=precision,
        loss_fn=loss_fn,
        threshold=threshold,
        max_batches=max_batches,
    )
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    summary.update(
        {
            "status": "complete",
            "best_epoch": best_epoch,
            "best_validation_auprc": best_metric,
            "validation": final_metrics,
            "elapsed_seconds": time.time() - start_time,
            **_gpu_memory_stats(device),
        }
    )
    summary = _json_safe(summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary
