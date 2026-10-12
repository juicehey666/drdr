# 患者级统一训练入口 v1

日期：2026-10-12

## 1. 目标

`scripts/train.py` 是 B0、B1、B1-mask 和 B2 的统一训练入口。它复用冻结的
`patient_dataloader_v1`，并把模型、优化器、训练预算、选择指标、运行设备和输出目录全部写入可追溯配置。

训练入口只允许读取 `train` 和 `validation`。`internal_test` 与 `external_test` 被配置校验器硬性禁止，避免在方法开发期间误看测试结果。

## 2. 配置层次

```text
configs/training/patient_rdr_base_v1.yaml  公共任务、优化和运行约束
configs/model/*.yaml                       模型与模型专属训练覆盖项
configs/experiment/*.yaml                  可直接执行的实验组合
```

实验配置通过 `defaults` 从左到右组合，命令行可用 `--set key=value` 临时覆盖。每次运行都会把解析后的完整配置保存到独立运行目录，正式实验不应只依赖命令行覆盖项。

## 3. 已登记实验

| 配置 | 作用 | 是否读取图像 |
|---|---|---:|
| `b0_prevalence_v1.yaml` | 训练集患病率常数 | 否 |
| `b1_clinical_v1.yaml` | 四项临床数值（训练集均值填补） | 否 |
| `b1_mask_v1.yaml` | 仅缺失掩码捷径诊断 | 否 |
| `b2_convnext_tiny_v1.yaml` | 共享 ConvNeXt-Tiny、眼独立预测、患者取最大 logit | 是 |

## 4. 安全检查

只检查配置，不创建运行目录：

```bash
conda run -n dr-grading python scripts/train.py \
  --config configs/experiment/b1_clinical_v1.yaml \
  --check-config
```

执行单个训练 batch 和单个 validation batch：

```bash
conda run -n dr-grading python scripts/train.py \
  --config configs/experiment/b1_clinical_v1.yaml \
  --dry-run \
  --run-id smoke_b1_clinical_v1
```

`--dry-run` 强制只运行一个 epoch、一个训练 batch 和一个 validation batch。B2 dry-run 应额外设置 `model.pretrained=false`，避免仅为接口测试下载权重：

```bash
conda run -n dr-grading python scripts/train.py \
  --config configs/experiment/b2_convnext_tiny_v1.yaml \
  --set model.pretrained=false \
  --set runtime.device=cpu \
  --dry-run
```

## 5. 正式运行产物

```text
outputs/experiments/<run_id>/
  config.resolved.yaml
  run_metadata.json
  metrics.jsonl
  summary.json
  validation_predictions.csv
  checkpoints/best.pt
  checkpoints/last.pt
  logs/
```

运行目录禁止覆盖。元数据包括配置哈希、代码提交、工作区是否有未提交修改、Python/PyTorch/CUDA、所选设备以及冻结数据 manifest 哈希。

## 6. 当前边界

- 当前入口只负责模型开发阶段的 train/validation；测试集评价将使用独立冻结评价入口。
- 当前开发阈值 0.5 只用于训练过程监控，不是论文最终的 90% 敏感度冻结阈值。
- B2 当前实现的是普通眼独立基线，不包含 B3 双眼特征交互或 IAAM-Net。
- 开始任何完整 GPU 训练前，必须先报告 GPU、预计时间和输出目录并获得确认。
