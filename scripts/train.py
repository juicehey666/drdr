#!/usr/bin/env python3
"""Unified protocol-locked training entry point."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dr_grading.training import load_experiment_config, run_experiment
from dr_grading.training.config import canonical_yaml, config_sha256
from dr_grading.training.reproducibility import (
    create_run_directory,
    resolve_device,
    seed_everything,
    write_run_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a resolved value with dotted key=value syntax",
    )
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one train and one validation batch for one epoch",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    )
    config = load_experiment_config(config_path, overrides=args.overrides)
    if args.check_config:
        print(canonical_yaml(config))
        print(f"config_sha256: {config_sha256(config)}")
        return 0

    seed_everything(
        int(config["runtime"]["seed"]),
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    device = resolve_device(str(config["runtime"]["device"]))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "dryrun" if args.dry_run else str(config["experiment"]["tag"])
    run_id = args.run_id or f"{timestamp}_{config['experiment']['name']}_{suffix}"
    run_dir = create_run_directory(PROJECT_ROOT, config, run_id)
    write_run_provenance(
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        config=config,
        argv=sys.argv,
        device=device,
    )
    try:
        summary = run_experiment(
            project_root=PROJECT_ROOT,
            config=config,
            run_dir=run_dir,
            device=device,
            dry_run=args.dry_run,
        )
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        (run_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    print(
        json.dumps(
            {"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
