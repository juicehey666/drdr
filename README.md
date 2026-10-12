# DR Grading Research Workspace

糖尿病视网膜病变（Diabetic Retinopathy, DR）分级研究工作区。本仓库只保存可复现研究所需的代码、配置、文档和小型测试；数据集、模型权重、凭据和生成的实验产物不进入 Git。

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

添加或调整依赖、CUDA 后运行：

```bash
conda run -n dr-grading python scripts/verify_environment.py
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

## Project layout

```text
configs/                 可版本控制的实验配置
data/                    本地数据及划分文件（内容不提交）
docs/                    数据卡、实验记录、研究方案和论文材料
models/                  预训练权重、检查点及导出模型（不提交）
notebooks/               探索性分析；稳定逻辑应迁移至 src/
outputs/                 每次运行生成的指标、预测、日志和图表（不提交）
scripts/                 数据准备、训练和评估入口脚本
src/dr_grading/          可复用 Python 包
tests/                   单元与集成测试
```

各本地存储目录的具体约定见 [data/README.md](data/README.md)、[models/README.md](models/README.md) 和 [outputs/README.md](outputs/README.md)。数据集与实验应分别使用 [dataset_card.md](docs/templates/dataset_card.md) 和 [experiment_record.md](docs/templates/experiment_record.md) 留下可审计记录。

## Research workflow

1. 将下载的原始数据只读保存至 `data/raw/<dataset_id>/`，第三方预处理数据放到 `data/external/`。
2. 为每个数据集填写数据卡，核对许可、患者级标识、标签定义和类别分布。
3. 使用脚本生成 `data/interim/`、`data/processed/` 和患者级 `data/splits/`，不直接修改原始数据。
4. 将所有可改变实验结果的参数写入 `configs/`，训练结果保存到唯一的 `outputs/experiments/<run_id>/`。
5. 在实验记录中保存 Git commit、配置、随机种子、数据版本、运行环境和核心结果。
6. 稳定的数据处理、模型、训练和评价逻辑放入 `src/dr_grading/`，并在 `tests/` 中加入小型可复现测试。

建议标识格式：数据集使用小写短名（如 `aptos2019`），实验使用 `YYYYMMDD_HHMM_<model>_<task>_<tag>`。公开数据许可仍需逐项核验；目录被 Git 忽略不等于可以绕过数据使用协议。

## Patient-level training entry point

统一训练入口为 `scripts/train.py`。先检查解析后的完整配置：

```bash
conda run -n dr-grading python scripts/train.py \
  --config configs/experiment/b1_clinical_v1.yaml \
  --check-config
```

再执行单批次 CPU dry-run：

```bash
conda run -n dr-grading python scripts/train.py \
  --config configs/experiment/b1_clinical_v1.yaml \
  --set runtime.device=cpu \
  --dry-run
```

训练入口仅允许访问 BRSET `train` 和 `validation`，配置中出现 `internal_test` 或 `external_test` 会直接报错。正式 GPU 训练前先确认 GPU、预计时间与输出目录。完整说明见 [training_entrypoint_v1.md](docs/training_entrypoint_v1.md)。
