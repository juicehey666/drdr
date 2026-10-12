# Experiment outputs

此目录保存可重新生成的实验产物，内容默认不进入 Git。

```text
experiments/<run_id>/
  config.yaml       本次运行解析后的完整配置
  metrics.json      机器可读的最终指标
  predictions.*     样本级预测（不得包含直接身份标识）
  logs/             训练与评估日志
  checkpoints/      与本次运行相关的权重
  figures/          曲线、混淆矩阵和论文候选图
tensorboard/        TensorBoard 事件文件
```

`run_id` 建议使用 `YYYYMMDD_HHMM_<model>_<task>_<tag>`。禁止覆盖已有运行目录；失败实验也应记录失败原因。计划写入论文的最终数值需要能追溯到配置、代码版本、数据版本和随机种子。
