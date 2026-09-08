# Research Workspace Instructions

- Use the existing Conda environment `dr-grading` for every Python, pip, test,
  notebook, and training command.
- Prefer `conda run -n dr-grading <command>` in non-interactive shells. Do not
  install Python packages globally or into the base Conda environment.
- Preserve the verified PyTorch 2.1.2 / CUDA 12.1 stack unless the user
  explicitly requests an upgrade.
- Run `python scripts/verify_environment.py` after dependency or CUDA changes.
- Record dependency changes in `environment.lock.yml` and
  `requirements.lock.txt`.
- Keep datasets, credentials, checkpoints, and generated experiment outputs out
  of Git. Never commit secrets or API keys.
- Ask before starting long training jobs or using multiple GPUs. Report the
  selected GPU IDs, expected runtime, and output directory first.
- Put reusable code in `src/`, experiment settings in `configs/`, small
  reproducible checks in `tests/`, and exploratory work in `notebooks/`.
