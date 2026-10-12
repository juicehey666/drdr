# Configuration conventions

- `data/`：数据路径、标签体系、划分 ID 和预处理参数。
- `model/`：模型架构及初始化设置。
- `training/`：优化、增强、随机种子和运行参数。
- `experiment/`：完整实验组合或覆盖项。

凡是会改变实验结果的参数都应显式写入配置。机器相关路径建议通过环境变量或本地忽略文件注入，不要提交个人绝对路径、凭据或密钥。

当前患者级 rDR 训练配置采用三层组合：

```text
training/patient_rdr_base_v1.yaml  公共任务和训练约束
model/*.yaml                       模型定义及模型专属覆盖项
experiment/*.yaml                  可直接运行的完整实验
```

`experiment/*.yaml` 中的 `defaults` 按顺序合并，实验文件自身最后覆盖。运行时 `--set key=value` 仅用于检查和小规模调试；正式实验的重要变化应固化为新的版本化 YAML。
