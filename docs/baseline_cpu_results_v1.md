# B0、B1 与 B1-mask CPU 基线结果 v1

日期：2026-10-12  
随机种子：20260929  
数据：BRSET train（4,326 人，rDR 283 人）与 validation（618 人，rDR 40 人）  
用途：模型开发阶段诊断，不包含 internal test 或 mBRSET external test

## 1. 完整 validation 结果

| 模型 | 最佳 epoch | AUROC | AUPRC | Brier | Sensitivity@0.5 | Specificity@0.5 | PPV@0.5 | NPV@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 患病率常数 | 0 | 0.5000 | 0.0647 | 0.0605 | 0.0000 | 1.0000 | NA | 0.9353 |
| B1 四项临床数值 | 26 | 0.8343 | 0.4539 | 0.1530 | 0.6000 | 0.9135 | 0.3243 | 0.9706 |
| B1-mask 仅缺失掩码 | 5 | 0.8051 | 0.2182 | 0.1714 | 0.6250 | 0.8858 | 0.2747 | 0.9715 |

所有预测文件均包含 618 名唯一患者和 40 名 rDR 阳性患者。B1 在第 34 epoch 触发早停，B1-mask 在第 13 epoch 触发早停；表中使用各自最佳 AUPRC 检查点。

## 2. 初步解释

1. B0 的 AUPRC 等于 validation 患病率（40/618=0.0647），符合常数预测的理论预期，说明基础指标实现正常。
2. B1-mask 仅使用年龄、性别、糖尿病病程和胰岛素使用的四位“是否观测”掩码，仍得到 AUROC 0.8051、AUPRC 0.2182。这表明 BRSET 的记录缺失模式携带明显的 rDR 相关信号，可能反映糖尿病诊断和临床记录流程，而非纯粹生理信息。
3. B1 的 AUROC/AUPRC 高于 B1-mask，但缺失值经训练集均值填补后恰好为 z-score 0，因此 B1 仍可能从精确零值间接识别缺失。当前不能把 B1 相对 B0 的全部增益解释为临床数值贡献。
4. B1 与 B1-mask 的原始 Brier score 均差于 B0。训练采用类别加权 BCE，原始输出概率尚未校准；判别提高不等于风险概率可靠。
5. 0.5 阈值仅用于开发监控。论文阈值仍应在 validation 上按预注册的至少 90% 敏感度规则冻结，并只在内部/外部测试上评价。

## 3. 结论边界

- 当前结果来自 BRSET validation，validation 同时承担早停和模型选择，不能作为最终无偏性能估计。
- 当前仅完成一个随机种子，不报告置信区间，也不进行模型显著性检验。
- internal test 和 mBRSET external test 尚未读取或评价。
- B1-mask 的高判别力支持“缺失捷径审计”的研究动机，但不能证明该捷径在外部域仍然有效。

## 4. 可追溯运行

| 模型 | 运行目录 | 配置 SHA256 |
|---|---|---|
| B0 | `outputs/experiments/20261012_b0_prevalence_v1_seed20260929` | `29c22c78c73edc50ba09bd7dd713ad4d2d3fa2b6a9d443b150654e80ee0edd4f` |
| B1 | `outputs/experiments/20261012_b1_clinical_v1_seed20260929` | `d0753551d93079d77854a91ba5d7866f379a2dc490a488a4d16d5065e649a31e` |
| B1-mask | `outputs/experiments/20261012_b1_mask_v1_seed20260929` | `aa4ce8570cc01ab1911639e2b37fd5f70e998593925c3ed545b43db886cccc5d` |

三个运行使用相同的冻结患者清单：
`1e8b6e135c3cdb7e7d1840d5ac7ccb9dc4039656eb410a26921c53535f8341b3`。

## 5. 下一步

1. 使用 GPU 1 对 B2 ConvNeXt-Tiny 进行单 epoch 吞吐、显存和损失稳定性试跑；
2. 实现 validation 上的 90% 敏感度阈值选择与校准模块；
3. B2 稳定后再决定 B1/B1-mask 是否补跑其余随机种子；
4. 不在当前阶段查看 internal test 或 mBRSET。
