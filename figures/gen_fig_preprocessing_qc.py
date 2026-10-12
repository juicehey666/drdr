#!/usr/bin/env python3
"""Generate reproducible QC figures and a report from a preprocessing manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dr_grading.qc.preprocessing import generate_qc_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/fundus_512_v1.yaml",
    )
    parser.add_argument(
        "--qc-config",
        type=Path,
        default=PROJECT_ROOT / "configs/qc/fundus_512_v1_qc.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = generate_qc_report(
            project_root=PROJECT_ROOT,
            manifest_path=args.manifest,
            preprocessing_config_path=args.preprocessing_config,
            qc_config_path=args.qc_config,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

