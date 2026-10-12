# B2 ConvNeXt-Tiny 单 epoch GPU 试跑报告 v1

日期：2026-10-12  
性质：训练链路、吞吐与显存试跑，不是最终模型结果

## 1. 运行设置

| 项目 | 设置 |
|---|---|
| 物理 GPU | GPU 1，NVIDIA GeForce RTX 3090 24 GB |
| 可见设备 | `CUDA_VISIBLE_DEVICES=1` |
| 图像编码器 | ImageNet 预训练 ConvNeXt-Tiny |
| 输入 | 患者级双眼图像，512×512 |
| 每卡 batch size | 4 名患者 |
| 梯度累积 | 4，等效 batch size 16 |
| 精度 | bf16 |
| epoch | 1 |
| 随机种子 | 20260929 |
| 训练/选择数据 | BRSET train/validation |
| 测试集 | 未访问 internal test 或 mBRSET |

运行目录：`outputs/experiments/20261012_b2_convnext_tiny_v1_pilot1epoch_seed20260929`

## 2. 运行状态

- 状态：完成，无 OOM、NaN 或数据读取错误；
- 引擎记录耗时：155.99 秒；该时间包含一个训练 epoch、epoch 后 validation、检查点写入及最佳检查点的最终 validation；
- train：4,326 名患者、8,670 张图像；
- validation：618 名患者、1,236 张图像；
- 运行中 `nvidia-smi` 观察到 GPU 总显存约 4.9 GB，说明 batch size 4 有较大显存余量；该值是采样观察值而非 PyTorch 精确峰值；
- 最佳检查点可读取，大小约 334 MB；
- validation 预测文件含 618 名唯一患者和 40 名阳性患者。

## 3. 单 epoch validation 结果

| AUROC | AUPRC | Brier | Sensitivity@0.5 | Specificity@0.5 | PPV@0.5 | NPV@0.5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9497 | 0.7575 | 0.0330 | 0.6750 | 0.9775 | 0.6750 | 0.9775 |

这些数值来自用于模型选择的 validation，且模型只训练一个 epoch。它们只说明预训练图像特征和训练链路有效，不能用于论文主结果、模型比较或泛化结论。

## 4. 试跑发现的问题

本次进程启动时没有预先设置 `CUBLAS_WORKSPACE_CONFIG`，PyTorch 因此提示部分 cuBLAS 操作未达到严格确定性。本次 pilot 结果保留，但不作为正式可复现实验。训练入口已修正：当 `runtime.deterministic=true` 时，在 CUDA 初始化前自动设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并在运行元数据中记录该值。

后续引擎还将精确记录 PyTorch 的 peak allocated/reserved GPU memory，避免只依赖 `nvidia-smi` 抽样值。

## 5. 正式训练建议

1. 采用已验证的 batch size 8、梯度累积 2 配置完成种子 20260929；
2. 按约 2.3 分钟/epoch 的当前观察估算，30 epoch 单种子约 65–75 分钟，实际可能因早停缩短；
3. 第一种子训练曲线稳定后，再运行其余两个预注册种子；
4. 在模型与阈值冻结前继续禁止访问 internal test 和 mBRSET。

## 6. batch size 8 严格确定性对照

运行目录：`outputs/experiments/20261012_b2_convnext_tiny_v1_pilot_bs8_seed20260929`

| 项目 | batch 4 pilot | batch 8 pilot |
|---|---:|---:|
| 梯度累积 | 4 | 2 |
| 等效 batch size | 16 | 16 |
| 严格 cuBLAS 确定性 | 否 | 是 |
| 耗时 | 155.99 s | 137.90 s |
| 峰值 allocated 显存 | 未记录 | 6.24 GiB |
| 峰值 reserved 显存 | 未记录 | 8.64 GiB |
| validation AUROC | 0.9497 | 0.9699 |
| validation AUPRC | 0.7575 | 0.8333 |
| validation Brier | 0.0330 | 0.0268 |

batch size 8 运行完成且无 OOM、NaN 或数据读取错误，耗时较 batch size 4 减少约 11.6%，显存仍有充足余量。两次运行的确定性环境不同，而且都只训练一个 epoch，因此验证指标差异不应解读为 batch size 8 具有性能优势。本对照仅支持在后续正式训练中选用 batch size 8 以提高吞吐。

本次配置哈希为 `134678fdfeecedff1dff7a681d21d88d5b2974e38ad6ed3b45e86502ca26e549`；预测文件含 618 名唯一验证患者，其中 40 名阳性患者；检查点已成功回读。internal test 和 mBRSET 仍未被访问。
