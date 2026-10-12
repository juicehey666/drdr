#!/usr/bin/env python3
"""Generate the audited B2 three-seed summary tables and learning curves."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_STEM = PROJECT_ROOT / "figures" / "fig_b2_three_seed_results_v1"
REPORT_DATA_DIR = PROJECT_ROOT / "outputs" / "reports" / "b2_three_seed_results_v1"

RUNS = {
    20260929: "20261012_b2_convnext_tiny_v1_seed20260929_rerun1",
    20261001: "20261012_b2_convnext_tiny_v1_seed20261001",
    20261003: "20261012_b2_convnext_tiny_v1_seed20261003",
}

EXPECTED_PATIENT_MANIFEST_SHA256 = (
    "1e8b6e135c3cdb7e7d1840d5ac7ccb9dc4039656eb410a26921c53535f8341b3"
)
EXPECTED_IMAGE_MANIFEST_SHA256 = (
    "dacc4a0d5bae150caca3301f01238f6751e93fb3464c415e2122083c982529c7"
)

COLORS = {
    20260929: "#0072B2",
    20261001: "#D55E00",
    20261003: "#009E73",
}
MARKERS = {20260929: "o", 20261001: "s", 20261003: "^"}
LINESTYLES = {20260929: "-", 20261001: "--", 20261003: "-."}


def _load_run(seed: int, run_id: str) -> tuple[dict, list[dict], dict, dict, pd.DataFrame]:
    run_dir = PROJECT_ROOT / "outputs" / "experiments" / run_id
    required = [
        "summary.json",
        "metrics.jsonl",
        "run_metadata.json",
        "config.resolved.yaml",
        "validation_predictions.csv",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_id}: missing required artifacts: {missing}")

    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    predictions = pd.read_csv(run_dir / "validation_predictions.csv")

    if summary["status"] != "complete":
        raise ValueError(f"{run_id}: run status is not complete")
    if config["runtime"]["seed"] != seed:
        raise ValueError(f"{run_id}: seed mismatch")
    if metadata.get("cublas_workspace_config") != ":4096:8":
        raise ValueError(f"{run_id}: strict cuBLAS determinism was not recorded")
    if summary["data_hashes"]["patient_manifest_sha256"] != EXPECTED_PATIENT_MANIFEST_SHA256:
        raise ValueError(f"{run_id}: patient manifest hash mismatch")
    if summary["data_hashes"]["image_manifest_sha256"] != EXPECTED_IMAGE_MANIFEST_SHA256:
        raise ValueError(f"{run_id}: image manifest hash mismatch")
    if len(predictions) != 618 or predictions["patient_uid"].nunique() != 618:
        raise ValueError(f"{run_id}: validation prediction cardinality mismatch")
    if int(predictions["target"].sum()) != 40:
        raise ValueError(f"{run_id}: validation positive count mismatch")
    if predictions["patient_uid"].duplicated().any():
        raise ValueError(f"{run_id}: duplicate patient predictions")
    if not metrics:
        raise ValueError(f"{run_id}: empty metrics log")

    numeric_values = []
    for row in metrics:
        numeric_values.extend(
            [
                row["train_loss"],
                row["learning_rate"],
                row["validation"]["auroc"],
                row["validation"]["auprc"],
                row["validation"]["brier"],
            ]
        )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError(f"{run_id}: non-finite metric detected")

    return summary, metrics, metadata, config, predictions


def _sample_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> int:
    REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict] = []
    epoch_rows: list[dict] = []

    for seed, run_id in RUNS.items():
        summary, metrics, metadata, config, _ = _load_run(seed, run_id)
        validation = summary["validation"]
        seed_rows.append(
            {
                "seed": seed,
                "run_id": run_id,
                "physical_gpu": config["runtime"]["physical_gpu_ids"][0],
                "epochs_completed": len(metrics),
                "best_epoch": summary["best_epoch"],
                "auroc": validation["auroc"],
                "auprc": validation["auprc"],
                "brier": validation["brier"],
                "sensitivity_at_0.5": validation["sensitivity"],
                "specificity_at_0.5": validation["specificity"],
                "ppv_at_0.5": validation["ppv"],
                "npv_at_0.5": validation["npv"],
                "elapsed_minutes": summary["elapsed_seconds"] / 60.0,
                "peak_allocated_gib": summary["peak_gpu_memory_allocated_bytes"] / 2**30,
                "peak_reserved_gib": summary["peak_gpu_memory_reserved_bytes"] / 2**30,
                "config_sha256": metadata["config_sha256"],
            }
        )
        for row in metrics:
            epoch_rows.append(
                {
                    "seed": seed,
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "learning_rate": row["learning_rate"],
                    "validation_auroc": row["validation"]["auroc"],
                    "validation_auprc": row["validation"]["auprc"],
                    "validation_brier": row["validation"]["brier"],
                }
            )

    seed_df = pd.DataFrame(seed_rows).sort_values("seed")
    epoch_df = pd.DataFrame(epoch_rows).sort_values(["seed", "epoch"])
    seed_df.to_csv(REPORT_DATA_DIR / "seed_summary.csv", index=False)
    epoch_df.to_csv(REPORT_DATA_DIR / "epoch_metrics.csv", index=False)

    aggregate = {
        metric: _sample_summary(seed_df[metric].to_numpy(dtype=float))
        for metric in ("auroc", "auprc", "brier")
    }
    aggregate.update(
        {
            "n_seeds": len(seed_df),
            "seeds": seed_df["seed"].astype(int).tolist(),
            "patient_manifest_sha256": EXPECTED_PATIENT_MANIFEST_SHA256,
            "image_manifest_sha256": EXPECTED_IMAGE_MANIFEST_SHA256,
        }
    )
    (REPORT_DATA_DIR / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.7,
            "lines.markersize": 4,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.0), constrained_layout=True)
    panels = [
        (axes[0, 0], "validation_auprc", "Validation AUPRC", "AUPRC"),
        (axes[0, 1], "validation_auroc", "Validation AUROC", "AUROC"),
        (axes[1, 0], "validation_brier", "Validation Brier score", "Brier score"),
        (axes[1, 1], "train_loss", "Training loss", "Weighted BCE loss"),
    ]

    for seed in RUNS:
        group = epoch_df[epoch_df["seed"] == seed]
        best_epoch = int(seed_df.loc[seed_df["seed"] == seed, "best_epoch"].iloc[0])
        for axis, column, _, _ in panels:
            axis.plot(
                group["epoch"],
                group[column],
                label=f"Seed {seed}",
                color=COLORS[seed],
                marker=MARKERS[seed],
                linestyle=LINESTYLES[seed],
                markevery=max(1, len(group) // 7),
                zorder=2,
            )
            best_row = group[group["epoch"] == best_epoch].iloc[0]
            axis.scatter(
                best_epoch,
                best_row[column],
                marker="*",
                s=70,
                color=COLORS[seed],
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )

    for panel_label, (axis, _, title, ylabel) in zip("abcd", panels):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.text(
            -0.16,
            1.06,
            f"({panel_label})",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    axes[0, 0].set_ylim(0.70, 0.93)
    axes[0, 1].set_ylim(0.95, 1.002)
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 1].set_ylim(bottom=0.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
    )
    fig.savefig(
        FIGURE_STEM.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        FIGURE_STEM.with_suffix(".png"),
        dpi=300,
        metadata={"Software": "dr-grading academic-plotting pipeline"},
    )
    plt.close(fig)

    print(
        json.dumps(
            {
                "seed_summary": str(REPORT_DATA_DIR / "seed_summary.csv"),
                "epoch_metrics": str(REPORT_DATA_DIR / "epoch_metrics.csv"),
                "aggregate_summary": str(REPORT_DATA_DIR / "aggregate_summary.json"),
                "figure_pdf": str(FIGURE_STEM.with_suffix(".pdf")),
                "figure_png": str(FIGURE_STEM.with_suffix(".png")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
