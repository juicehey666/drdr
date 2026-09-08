from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 1024**3
        print(f"GPU {index}: {properties.name} ({memory_gib:.1f} GiB)")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the dr-grading environment")


if __name__ == "__main__":
    main()
