from __future__ import annotations

import pandas as pd

from dr_grading.qc.preprocessing import grouped_statistics, stratified_round_robin


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "dataset": "brset" if index < 6 else "mbrset",
                "camera": "Canon" if index < 3 else ("Nikon" if index < 6 else "Phelcom"),
                "icdr": str(index % 3),
                "crop_fallback": index in {1, 8},
                "image_id": f"img{index:02d}",
                "foreground_fraction": 0.70 + index / 100,
            }
        )
    return pd.DataFrame(rows)


def test_stratified_round_robin_is_deterministic_and_covers_datasets() -> None:
    frame = _frame()
    kwargs = {
        "strata": ["dataset", "camera", "icdr", "crop_fallback"],
        "max_samples": 6,
        "seed": 20260929,
    }
    first = stratified_round_robin(frame, **kwargs)
    second = stratified_round_robin(frame, **kwargs)
    assert first["image_id"].tolist() == second["image_id"].tolist()
    assert len(first) == 6
    assert set(first["dataset"]) == {"brset", "mbrset"}


def test_grouped_statistics_preserves_counts_and_fallbacks() -> None:
    frame = _frame()
    stats = grouped_statistics(frame)
    dataset = stats[stats["dimension"] == "dataset"].set_index("category")
    assert int(dataset.loc["brset", "n"]) == 6
    assert int(dataset.loc["mbrset", "n"]) == 6
    assert int(dataset["fallback_count"].sum()) == 2
    assert 0.70 <= float(dataset.loc["brset", "foreground_min"]) <= 0.71
