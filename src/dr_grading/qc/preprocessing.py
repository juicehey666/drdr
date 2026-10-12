"""Reproducible QC figures and reports for fundus preprocessing manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image


REQUIRED_COLUMNS = {
    "dataset",
    "image_id",
    "patient_id",
    "camera",
    "icdr",
    "original_width",
    "original_height",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "foreground_fraction",
    "crop_fallback",
    "fallback_reason",
    "output_path",
    "status",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _configure_style(config: Mapping[str, Any]) -> None:
    figure = config["figure"]
    plt.rcParams.update(
        {
            "font.family": figure["font_family"],
            "font.serif": figure["font_serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": int(figure["dpi"]),
            "savefig.dpi": int(figure["dpi"]),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
        }
    )


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Manifest is empty")
    failed = frame[frame["status"] != "ok"]
    if not failed.empty:
        raise ValueError(
            f"Manifest contains {len(failed)} non-ok records; resolve them before QC"
        )
    numeric_columns = [
        "original_width",
        "original_height",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "foreground_fraction",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    lowered = frame["crop_fallback"].str.lower()
    if not lowered.isin(["true", "false"]).all():
        raise ValueError("crop_fallback must contain only true/false")
    frame["crop_fallback"] = lowered.eq("true")
    frame["icdr"] = frame["icdr"].replace("", "Missing")
    return frame


def _resolve_source_paths(
    frame: pd.DataFrame,
    project_root: Path,
    preprocessing_config: Mapping[str, Any],
) -> pd.DataFrame:
    resolved = frame.copy()
    source_paths: list[str] = []
    processed_paths: list[str] = []
    for row in resolved.itertuples(index=False):
        spec = preprocessing_config["datasets"][row.dataset]
        image_id = str(row.image_id)
        extension = str(spec.get("image_extension", ""))
        filename = (
            image_id
            if not extension or image_id.lower().endswith(extension.lower())
            else image_id + extension
        )
        source = (project_root / str(spec["image_dir"]) / filename).resolve()
        output = Path(str(row.output_path))
        if not output.is_absolute():
            output = (project_root / output).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing source image: {source}")
        if not output.is_file():
            raise FileNotFoundError(f"Missing processed image: {output}")
        source_paths.append(str(source))
        processed_paths.append(str(output))
    resolved["source_path"] = source_paths
    resolved["processed_path"] = processed_paths
    return resolved


def stratified_round_robin(
    frame: pd.DataFrame,
    *,
    strata: Sequence[str],
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministically cover all strata before taking a second sample."""

    if max_samples <= 0 or max_samples >= len(frame):
        return frame.sort_values(list(strata) + ["image_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    work = frame.copy()
    work["_tie"] = rng.random(len(work))
    group_key: str | list[str] = list(strata)
    groups_by_primary: dict[str, list[pd.DataFrame]] = {}
    for key, group in work.groupby(group_key, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        primary = str(key_tuple[0])
        groups_by_primary.setdefault(primary, []).append(
            group.sort_values(["_tie", "image_id"]).reset_index(drop=True)
        )
    selected: list[pd.Series] = []
    level = 0
    while len(selected) < max_samples:
        progressed = False
        max_group_count = max(len(groups) for groups in groups_by_primary.values())
        for group_index in range(max_group_count):
            for primary in sorted(groups_by_primary):
                primary_groups = groups_by_primary[primary]
                if group_index >= len(primary_groups):
                    continue
                group = primary_groups[group_index]
                if level < len(group):
                    selected.append(group.iloc[level])
                    progressed = True
                    if len(selected) == max_samples:
                        break
            if len(selected) == max_samples:
                break
        if not progressed:
            break
        level += 1
    result = pd.DataFrame(selected).drop(columns=["_tie"])
    return result.reset_index(drop=True)


def grouped_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in ("dataset", "camera", "icdr"):
        for category, group in frame.groupby(dimension, dropna=False, sort=True):
            values = group["foreground_fraction"].astype(float)
            fallback_count = int(group["crop_fallback"].sum())
            rows.append(
                {
                    "dimension": dimension,
                    "category": str(category),
                    "n": len(group),
                    "fallback_count": fallback_count,
                    "fallback_rate": fallback_count / len(group),
                    "foreground_mean": values.mean(),
                    "foreground_std": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "foreground_median": values.median(),
                    "foreground_q1": values.quantile(0.25),
                    "foreground_q3": values.quantile(0.75),
                    "foreground_min": values.min(),
                    "foreground_max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def _save_both(fig: plt.Figure, base_path: Path, dpi: int) -> tuple[Path, Path]:
    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return pdf_path, png_path


def _distribution_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    category: str,
    title: str,
    palette: Sequence[str],
    seed: int,
) -> None:
    categories = sorted(frame[category].astype(str).unique())
    rng = np.random.default_rng(seed)
    for index, value in enumerate(categories):
        values = frame.loc[frame[category].astype(str) == value, "foreground_fraction"].astype(float)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        color = palette[index % len(palette)]
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            color=color,
            marker="o",
            s=28,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        median = float(values.median())
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        ax.vlines(index, q1, q3, color="#333333", linewidth=2.5, zorder=4)
        ax.hlines(median, index - 0.14, index + 0.14, color="#111111", linewidth=1.3, zorder=5)
        ax.text(index, min(1.0, float(values.max()) + 0.012), f"n={len(values)}", ha="center", fontsize=7)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=25 if category == "camera" else 0, ha="right" if category == "camera" else "center")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Foreground fraction")


def plot_foreground_distribution(
    frame: pd.DataFrame, output_dir: Path, config: Mapping[str, Any]
) -> tuple[Path, Path]:
    palette = config["figure"]["palette"]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0), sharey=True)
    _distribution_panel(axes[0], frame, "dataset", "(a) Dataset", palette, int(config["seed"]))
    _distribution_panel(axes[1], frame, "camera", "(b) Camera", palette, int(config["seed"]) + 1)
    axes[1].set_ylabel("")
    fig.suptitle("Fundus foreground coverage after detection", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save_both(
        fig,
        output_dir / "fig_foreground_fraction_distribution",
        int(config["figure"]["dpi"]),
    )


def _fallback_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    dimension: str,
    title: str,
    color: str,
) -> None:
    summary = (
        frame.groupby(dimension, dropna=False)["crop_fallback"]
        .agg(["sum", "count"])
        .reset_index()
    )
    summary[dimension] = summary[dimension].astype(str)
    if dimension == "icdr":
        def sort_key(value: str) -> tuple[int, float | str]:
            try:
                return (0, float(value))
            except ValueError:
                return (1, value)
        order = sorted(summary[dimension], key=sort_key)
        summary = summary.set_index(dimension).loc[order].reset_index()
    else:
        summary = summary.sort_values(dimension)
    rate = summary["sum"] / summary["count"] * 100.0
    y = np.arange(len(summary))
    ax.barh(y, rate, color=color, height=0.62, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(summary[dimension])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fallback rate (%)")
    ax.set_title(title)
    for yi, (fallbacks, total, percent) in enumerate(zip(summary["sum"], summary["count"], rate)):
        x = max(float(percent) + 1.5, 1.5)
        ax.text(x, yi, f"{int(fallbacks)}/{int(total)}", va="center", fontsize=7.5, color="#333333")


def plot_fallback_statistics(
    frame: pd.DataFrame, output_dir: Path, config: Mapping[str, Any]
) -> tuple[Path, Path]:
    palette = config["figure"]["palette"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.15))
    _fallback_panel(axes[0], frame, "dataset", "(a) Dataset", palette[1])
    _fallback_panel(axes[1], frame, "camera", "(b) Camera", palette[2])
    _fallback_panel(axes[2], frame, "icdr", "(c) ICDR grade", palette[4])
    fig.suptitle("Crop fallback by acquisition and disease strata", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save_both(
        fig,
        output_dir / "fig_crop_fallback_statistics",
        int(config["figure"]["dpi"]),
    )


def plot_comparison_pages(
    selected: pd.DataFrame,
    output_dir: Path,
    config: Mapping[str, Any],
) -> tuple[Path, list[Path]]:
    pairs_per_page = int(config["pairs_per_page"])
    page_count = math.ceil(len(selected) / pairs_per_page)
    pdf_path = output_dir / "fig_stratified_before_after.pdf"
    png_paths: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        for page_index in range(page_count):
            page = selected.iloc[
                page_index * pairs_per_page : (page_index + 1) * pairs_per_page
            ]
            fig, axes = plt.subplots(
                len(page), 2, figsize=(7.0, max(2.5, 2.55 * len(page))), squeeze=False
            )
            for row_index, row in enumerate(page.itertuples(index=False)):
                with Image.open(row.source_path) as source_handle:
                    source = np.asarray(source_handle.convert("RGB"))
                with Image.open(row.processed_path) as processed_handle:
                    processed = np.asarray(processed_handle.convert("RGB"))
                left, right = axes[row_index]
                left.imshow(source)
                left.add_patch(
                    Rectangle(
                        (float(row.crop_x0), float(row.crop_y0)),
                        float(row.crop_x1 - row.crop_x0),
                        float(row.crop_y1 - row.crop_y0),
                        fill=False,
                        edgecolor="#D55E00",
                        linewidth=1.5,
                    )
                )
                left.set_title(
                    f"Original + crop box | {row.dataset} | {row.camera}\n"
                    f"ID={row.image_id}, ICDR={row.icdr}, fallback={str(row.crop_fallback).lower()}",
                    fontsize=8.2,
                )
                right.imshow(processed)
                right.set_title(
                    f"fundus_512_v1 | foreground={float(row.foreground_fraction):.3f}",
                    fontsize=8.2,
                )
                left.axis("off")
                right.axis("off")
            fig.suptitle(
                f"Stratified preprocessing QC — page {page_index + 1}/{page_count}",
                fontsize=11,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            pdf.savefig(fig)
            png_path = output_dir / f"fig_stratified_before_after_page_{page_index + 1:02d}.png"
            fig.savefig(png_path, dpi=int(config["figure"]["dpi"]))
            png_paths.append(png_path)
            plt.close(fig)
    return pdf_path, png_paths


def _write_manual_review(selected: pd.DataFrame, path: Path) -> None:
    columns = [
        "dataset",
        "image_id",
        "patient_id",
        "camera",
        "icdr",
        "crop_fallback",
        "foreground_fraction",
        "source_path",
        "processed_path",
    ]
    review = selected[columns].copy()
    review["reviewer"] = ""
    review["complete_field_of_view"] = ""
    review["incorrect_crop"] = ""
    review["obvious_color_change"] = ""
    review["visible_lesion_loss"] = ""
    review["review_status"] = "pending"
    review["notes"] = ""
    review.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _write_report(
    *,
    path: Path,
    manifest_path: Path,
    manifest_sha: str,
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    selected: pd.DataFrame,
    comparison_pngs: Sequence[Path],
) -> None:
    dataset_stats = statistics[statistics["dimension"] == "dataset"].copy()
    dataset_stats["fallback_rate_percent"] = (
        dataset_stats["fallback_rate"] * 100
    ).map(lambda value: f"{value:.2f}")
    dataset_stats["foreground_median"] = dataset_stats["foreground_median"].map(
        lambda value: f"{value:.4f}"
    )
    current_is_small = len(frame) < 300
    page_links = "\n".join(
        f"![Stratified comparison page {index}]({png.name})"
        for index, png in enumerate(comparison_pngs, start=1)
    )
    warning = (
        "> **范围提醒：** 当前 manifest 少于300张，只能作为管线dry-run QC，不能替代协议要求的"
        "BRSET-Canon、BRSET-Nikon、mBRSET各至少100张分层人工检查。\n"
        if current_is_small
        else ""
    )
    text = f"""# fundus_512_v1 预处理 QC 报告

生成时间：{datetime.now(timezone.utc).isoformat()}  
manifest：`{manifest_path}`  
manifest SHA256：`{manifest_sha}`

{warning}
## 1. 总览

- 成功图像：{len(frame)}；
- 数据集：{', '.join(sorted(frame['dataset'].astype(str).unique()))}；
- 相机：{', '.join(sorted(frame['camera'].astype(str).unique()))}；
- crop fallback：{int(frame['crop_fallback'].sum())}/{len(frame)}；
- foreground fraction：中位数 {frame['foreground_fraction'].median():.4f}，范围 {frame['foreground_fraction'].min():.4f}–{frame['foreground_fraction'].max():.4f}；
- 分层人工检查候选：{len(selected)}。

{_markdown_table(dataset_stats, ['category', 'n', 'fallback_count', 'fallback_rate_percent', 'foreground_median'])}

`fallback_rate_percent` 单位为百分比。

## 2. 分层抽样原图—预处理图对照

橙色矩形表示 manifest 记录的裁剪框。右侧是方形填充并缩放至512×512的最终缓存图。

{page_links}

完整多页PDF：[`fig_stratified_before_after.pdf`](fig_stratified_before_after.pdf)  
人工审核表：[`manual_review_form.csv`](manual_review_form.csv)

## 3. Foreground fraction 分布

散点表示单张图像，粗竖线表示四分位距，短横线表示中位数。小样本组不应把四分位距解释为稳定总体分布。

![Foreground fraction distribution](fig_foreground_fraction_distribution.png)

矢量版本：[`fig_foreground_fraction_distribution.pdf`](fig_foreground_fraction_distribution.pdf)

## 4. Crop fallback 统计

每个标签显示 `fallback数量/该组总数`。fallback为0不等于裁剪一定正确，仍必须人工检查完整视场。

![Crop fallback statistics](fig_crop_fallback_statistics.png)

矢量版本：[`fig_crop_fallback_statistics.pdf`](fig_crop_fallback_statistics.pdf)

## 5. 人工审核标准

逐行填写 `manual_review_form.csv`：

1. `complete_field_of_view`：视盘、黄斑及原图可见的视网膜边缘是否完整保留；
2. `incorrect_crop`：裁剪框是否错误包含大面积无关区域或删除有效视网膜；
3. `obvious_color_change`：除JPEG重编码外是否出现明显色彩改变；
4. `visible_lesion_loss`：可见出血、渗出、微小病灶或瘢痕是否因裁剪丢失；
5. `review_status`：填写 `pass` 或 `fail`，失败时必须在 `notes` 说明原因。

通过标准：错误裁剪应接近0；一旦发现有效视网膜被算法新增截断，停止全量预处理并检查统一检测算法。

## 6. 可复现产物

- `qc_selected_samples.csv`：分层抽样清单；
- `qc_group_statistics.csv`：数据集、相机和ICDR分组统计；
- `manual_review_form.csv`：人工审核记录；
- `qc_run_metadata.json`：输入和输出哈希、运行参数。
"""
    path.write_text(text, encoding="utf-8")


def generate_qc_report(
    *,
    project_root: Path,
    manifest_path: Path,
    preprocessing_config_path: Path,
    qc_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    preprocessing_config_path = preprocessing_config_path.resolve()
    qc_config_path = qc_config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty QC directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_config = _read_yaml(preprocessing_config_path)
    qc_config = _read_yaml(qc_config_path)
    if qc_config.get("version") != "fundus_512_v1_qc":
        raise ValueError("Unsupported QC configuration version")
    _configure_style(qc_config)
    frame = load_manifest(manifest_path)
    frame = _resolve_source_paths(frame, project_root, preprocessing_config)
    selected = stratified_round_robin(
        frame,
        strata=qc_config["stratify_by"],
        max_samples=int(qc_config["max_comparison_pairs"]),
        seed=int(qc_config["seed"]),
    )
    statistics = grouped_statistics(frame)

    selected_path = output_dir / "qc_selected_samples.csv"
    stats_path = output_dir / "qc_group_statistics.csv"
    selected.to_csv(selected_path, index=False)
    statistics.to_csv(stats_path, index=False)
    manual_path = output_dir / "manual_review_form.csv"
    _write_manual_review(selected, manual_path)

    foreground_pdf, foreground_png = plot_foreground_distribution(
        frame, output_dir, qc_config
    )
    fallback_pdf, fallback_png = plot_fallback_statistics(
        frame, output_dir, qc_config
    )
    comparison_pdf, comparison_pngs = plot_comparison_pages(
        selected, output_dir, qc_config
    )
    report_path = output_dir / "qc_report.md"
    manifest_sha = sha256_file(manifest_path)
    _write_report(
        path=report_path,
        manifest_path=manifest_path,
        manifest_sha=manifest_sha,
        frame=frame,
        statistics=statistics,
        selected=selected,
        comparison_pngs=comparison_pngs,
    )

    artifact_paths = [
        selected_path,
        stats_path,
        manual_path,
        foreground_pdf,
        foreground_png,
        fallback_pdf,
        fallback_png,
        comparison_pdf,
        *comparison_pngs,
        report_path,
    ]
    metadata = {
        "version": qc_config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "preprocessing_config_sha256": sha256_file(preprocessing_config_path),
        "qc_config_sha256": sha256_file(qc_config_path),
        "record_count": len(frame),
        "selected_count": len(selected),
        "fallback_count": int(frame["crop_fallback"].sum()),
        "foreground_min": float(frame["foreground_fraction"].min()),
        "foreground_median": float(frame["foreground_fraction"].median()),
        "foreground_max": float(frame["foreground_fraction"].max()),
        "artifacts": {
            str(path.relative_to(output_dir)): sha256_file(path) for path in artifact_paths
        },
        "matplotlib_version": matplotlib.__version__,
        "pandas_version": pd.__version__,
    }
    metadata_path = output_dir / "qc_run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**metadata, "output_dir": str(output_dir), "report_path": str(report_path)}

