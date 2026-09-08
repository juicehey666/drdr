# DR Grading Research Workspace

This repository is the working area for deep-learning research on the `010`
server. Python work uses the existing Conda environment named `dr-grading`.

## Quick start

```bash
cd ~/research/dr-grading-workspace
conda activate dr-grading
python scripts/verify_environment.py
```

For non-interactive commands, use:

```bash
conda run -n dr-grading python <script.py>
```

## Environment baseline

- Python 3.10
- PyTorch 2.1.2 with CUDA 12.1
- Four NVIDIA GeForce RTX 3090 GPUs
- JupyterLab, TensorBoard, pytest, and Ruff

`albumentations` is intentionally not installed in the baseline. Its current
dependency resolution upgrades NumPy and OpenCV in ways that can break the
verified PyTorch stack. Use `torchvision.transforms` until the augmentation
requirements are defined and tested in a cloned environment.

## Layout

- `configs/`: experiment configuration files
- `data/`: local datasets and generated indices (not committed)
- `docs/`: research notes and documentation
- `notebooks/`: exploratory notebooks
- `outputs/`: logs, checkpoints, and experiment outputs (not committed)
- `scripts/`: command-line utilities
- `src/`: reusable source code
- `tests/`: automated tests
