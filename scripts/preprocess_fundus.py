#!/usr/bin/env python3
"""CLI for the versioned fundus_512_v1 preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dr_grading.preprocessing.fundus import run_preprocessing  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BRSET/mBRSET without modifying raw images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/fundus_512_v1.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Fresh output directory. Defaults to config output.root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("brset", "mbrset"),
        default=("brset", "mbrset"),
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=0,
        help="Deterministic balanced prefix for dry-runs; 0 means all records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_preprocessing(
            project_root=PROJECT_ROOT,
            config_path=args.config,
            output_root=args.output_root,
            datasets=args.datasets,
            limit_per_dataset=args.limit_per_dataset,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

