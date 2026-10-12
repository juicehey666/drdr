# Model storage

模型二进制文件均被 Git 忽略，代码和模型配置应分别保存在 `src/dr_grading/models/` 与 `configs/model/`。

```text
pretrained/   从公开来源获得的初始化权重
checkpoints/  训练过程保存的权重快照
exports/      ONNX、TorchScript 或部署格式模型
```

每个外部权重需记录来源 URL、版本、许可和 SHA-256 校验值。训练检查点应能通过实验 ID 关联到 `outputs/experiments/<run_id>/` 中的配置与指标。不要把凭据或数据路径硬编码进权重元数据。
