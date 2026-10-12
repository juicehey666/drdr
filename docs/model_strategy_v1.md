# BRSET–mBRSET 模型选择与训练策略 v1

日期：2026-09-29

## 1. 对 CNN + Transformer 构想的评价

该构想在机制上合理：CNN 的局部归纳偏置和多尺度特征图有利于保留点状出血、微动脉瘤和小渗出；Transformer 有利于聚合跨区域信息、较大病灶分布和全眼上下文。但它不能单独作为论文创新，因为直接相关工作已经较多：

- DR-CTFN 已融合 ConvNeXt 与 Swin Transformer；
- EffTNet 已使用 EfficientNet-B3a + ViT-B/16 建模局部和全局特征；
- LRMVDR 已研究 CNN 注入 Transformer、病灶重建和多视野 DR 分级；
- 另有多篇眼底疾病 CNN–Transformer 混合分类研究。

参考：[DR-CTFN, 2025](https://pubmed.ncbi.nlm.nih.gov/41151189/)；[EffTNet, 2026](https://pubmed.ncbi.nlm.nih.gov/42386883/)；[LRMVDR, 2026](https://pubmed.ncbi.nlm.nih.gov/42419158/)。

因此，应将混合编码器定位为“适配微小与全局病灶的图像表征模块”，论文主创新仍是：真实非随机缺失、双眼不对称和复合外部偏移下的可靠患者级融合。

## 2. 推荐的分阶段模型路线

### 阶段 A：三种图像编码器筛选，只跑 B2

| 类型 | 推荐模型 | 作用 |
|---|---|---|
| CNN | ConvNeXt-Tiny，ImageNet 预训练 | 稳定、成本适中的局部/层级基线 |
| Transformer | DINOv3 ViT-S/16 | 强预训练全局表征基线 |
| 现成混合模型 | MaxViT-Tiny 或同级局部—全局混合骨干 | 低风险验证 CNN+Transformer 假设 |

三者都使用 `fundus_512_v1`、同一患者划分、同一增强和同一评价代码。只在 BRSET validation 上选择后续主骨干，不能依据 mBRSET 外测结果选择。

若 MaxViT 没有稳定优于 ConvNeXt/DINOv3，就不要为了“混合”而坚持复杂结构。

### 阶段 B：固定一个骨干，完成患者级基线

1. B2：左右眼独立预测，患者概率取较高值；
2. B3：双眼特征融合；
3. B4：双眼特征 + 临床表格朴素拼接；
4. B1-mask：只使用临床变量缺失掩码，诊断捷径。

这一阶段不修改图像编码器，否则无法判断改进来自骨干还是融合方法。

### 阶段 C：实现论文主模型

建议名称暂定为 `Image-Anchored Asymmetry-aware Multimodal Network`（IAAM-Net）。模型应保持轻量，不采用两个完整的大型骨干并行训练。

## 3. 推荐主模型结构

### 3.1 图像编码器

优先方案：使用阶段 A 选出的预训练编码器。若确实需要自定义 CNN–Transformer，则采用单干式结构：

```text
512×512 CFP
    ↓
CNN stem + hierarchical stages
    ├── 中层高分辨率特征 → local attention pooling → z_local
    └── 深层低分辨率特征 → 2个Transformer blocks → z_global
                                   ↓
                         gated fusion(z_local, z_global)
                                   ↓
                               z_image
```

要点：

- Transformer 只处理低分辨率深层 token，控制显存和过拟合；
- CNN 中层特征负责“对小病灶更敏感”，不能在没有定位真值时直接声称完成病灶定位；
- 不推荐 ConvNeXt 与完整 ViT 双大骨干并行，因为 BRSET 主队列只有 404 名 rDR 阳性；
- 使用共享图像编码器处理所有眼和视野，不能为左右眼建立两套独立参数。

### 3.2 同眼多视野聚合

```text
{z_view_1, ..., z_view_n}
        ↓ masked mean/attention pooling
      z_left 或 z_right
```

第一版使用 masked mean，方法版再加入一个很小的 attention pooling。输入必须支持 BRSET 每眼约一张和 mBRSET 每眼约两张的变长集合。

### 3.3 双眼融合

使用对称表征，避免简单拼接过度依赖固定位置：

```text
z_bilateral = concat(
    mean(z_left, z_right),
    max(z_left, z_right),
    abs(z_left - z_right)
)
```

同时为两眼建立共享的辅助眼级 rDR 预测头：

```text
p_left = head(z_left)
p_right = head(z_right)
asymmetry = abs(p_left - p_right)
```

由 `asymmetry` 控制跨眼信息门控：两眼预测高度不一致时，减少强制融合，避免重眼证据错误传播到健眼。

### 3.4 临床表格分支

只有约四个跨库统一变量，不建议使用大型 FT-Transformer。使用两层 MLP 即可：

```text
[标准化数值, 缺失掩码]
        ↓
MLP(32 → 64) + LayerNorm + Dropout
        ↓
z_tabular
```

数值填补、均值和标准差仅由 BRSET train 计算。另保留 mask-only 基线，验证模型是否利用记录行为而非生理信息。

### 3.5 图像锚定的最终融合

不要直接把三类特征拼接后完全重新预测。推荐残差式融合：

```text
logit_final = logit_image
            + gate_eye * delta_bilateral
            + gate_tab * delta_tabular
```

其中 gate 受图像置信度、双眼不对称和模态可用性控制。这样图像是锚点，临床信息或另一只眼缺失时仍能自然退化为图像模型。

## 4. 损失函数

第一版只保留必要项：

```text
L = L_patient_rDR
  + 0.3 * (L_left_eye + L_right_eye)
  + λ_cf * L_counterfactual_mask_consistency
```

- `L_patient_rDR`：类别加权 BCE；
- 眼级辅助损失：防止双眼融合后丢失眼级临床语义；
- 反事实掩码一致性：同一患者在不同表格缺失模式下预测不应剧烈变化；
- `λ_cf` 从 `{0.05, 0.1, 0.2}` 在 BRSET validation 选择。

不要第一轮同时加入 focal、contrastive、ordinal、重建和蒸馏五种损失。每增加一项损失都必须能被单独消融解释。

## 5. 训练策略

### 第一阶段：头部训练

- 冻结预训练图像编码器；
- 训练分类头、视野池化、双眼融合和表格分支 3–5 epoch；
- AdamW；融合头学习率约 `1e-3`。

### 第二阶段：部分解冻

- 解冻图像编码器最后 1–2 个 stage/block；
- 编码器学习率约 `1e-5`，新模块约 `1e-4`；
- cosine decay + warmup；
- mixed precision；
- early stopping 依据 BRSET validation AUPRC；
- 每个最终模型运行 3 个随机种子。

只有在部分解冻稳定后才尝试全量微调。由于阳性患者少，不建议一开始端到端高学习率训练。

## 6. 必需消融

| 消融 | 回答的问题 |
|---|---|
| CNN vs Transformer vs hybrid | 混合编码器是否真的需要 |
| 去掉 local 分支 | 小尺度分支是否贡献增益 |
| 去掉 global Transformer | 全局关系模块是否贡献增益 |
| B2 vs B3 | 双眼信息是否有价值 |
| B3 vs B4 | 临床表格是否有增益 |
| 去掉 asymmetry gate | 增益是否来自不对称感知 |
| 去掉 counterfactual consistency | 是否提高缺失模式稳健性 |
| mask-only | 缺失记录是否形成标签捷径 |

模型参数量、FLOPs、显存和推理时间也应报告。若复杂模型只提高很小且置信区间重叠，应优先选择简单模型。

## 7. 不建议当前采用的方案

- 两个完整大骨干（如 ConvNeXt-L + ViT-L）并行；
- 在 404 个阳性患者上从头训练 Transformer；
- 同时引入病灶分割、生成式补全、对比学习和多模态大模型；
- 用 Grad-CAM 的漂亮热图证明模型识别了病灶；
- 根据 mBRSET 外测表现反复修改模型；
- 把 CNN+Transformer 本身写成主要新颖点。

## 8. 最终推荐

短期先训练 `ConvNeXt-Tiny / DINOv3-S/16 / MaxViT-Tiny` 三个 B2 图像模型。若现成混合骨干有稳定收益，将其作为共享图像编码器；若没有，就使用性能和校准更好的单一骨干，把精力集中在双眼不对称门控、临床缺失稳健性和外部验证上。

论文方法的推荐最小版本为：

```text
共享预训练图像编码器
+ 同眼变长视野池化
+ 对称双眼特征与不对称门控
+ 小型表格 MLP
+ 图像锚定残差融合
+ 反事实缺失一致性
```

该组合比“再做一个 CNN+Transformer 分类器”更贴合当前数据，也更有机会形成可解释、可验证的研究贡献。
