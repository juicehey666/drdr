#!/usr/bin/env python3
"""Build and freeze the patient_split_v1 cohort artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dr_grading.data.cohort import build_and_freeze_cohort  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen BRSET--mBRSET patient-level cohort and split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/patient_split_v1.yaml",
        help="Protocol-locked cohort configuration.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output override. The target must not already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_and_freeze_cohort(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
