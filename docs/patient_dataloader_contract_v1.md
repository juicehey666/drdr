# patient_dataloader_v1 数据接口约定

日期：2026-10-10  
输入队列：`patient_split_v1`  
图像缓存：`fundus_512_v1`

## 1. 分析单位

一个 `Dataset` item 对应一名患者，而不是一张图像。左右眼始终分开返回；同眼全部合格视野均被保留，并按 `image_id` 稳定排序。

## 2. Batch 张量

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `left_images` | `[B, V_left, 3, 512, 512]` float32 | 左眼图像，按 batch 内最大视野数 padding |
| `right_images` | `[B, V_right, 3, 512, 512]` float32 | 右眼图像，按 batch 内最大视野数 padding |
| `left_view_mask` | `[B, V_left]` bool | `True` 表示真实视野 |
| `right_view_mask` | `[B, V_right]` bool | `True` 表示真实视野 |
| `clinical_features` | `[B, 4]` float32 | 年龄、性别、糖尿病病程、胰岛素使用 |
| `clinical_mask` | `[B, 4]` bool | 对应临床变量是否真实观测 |
| `patient_grade` | `[B]` int64 | 患者级 ICDR 0–4 |
| `patient_rdr` | `[B]` float32 | 患者级可转诊 DR 标签 |
| `left_eye_grade` | `[B]` int64 | 左眼眼级 ICDR |
| `right_eye_grade` | `[B]` int64 | 右眼眼级 ICDR |

`V_left` 与 `V_right` 只由当前 batch 决定，不截断视野。padding 图像张量值为 0，但模型做同眼池化时必须应用对应的 `view_mask`，禁止把 padding 当作真实图像参与均值。

## 3. 图像变换

- train：应用 `aug_moderate_v1` 后进行 ImageNet ConvNeXt mean/std 归一化；
- validation、internal test、external test：只做张量转换与确定性归一化；
- 每张图像独立采样增强；
- 不使用垂直翻转、RandomResizedCrop、MixUp、CutMix 或 RandomErasing；
- mBRSET 全部视野均输入，不按质量或模型置信度事后挑图。

## 4. 临床变量

顺序固定为：

```text
[age, sex, diabetes_duration_years, insulin_use]
```

- `sex` 编码为 female=0、male=1；
- 均值和标准差只使用 BRSET train 中真实观测值拟合；
- 缺失值先用训练集观测均值填补，因此 z-score 后为 0；
- `clinical_mask` 必须与特征同时输入或用于缺失压力测试；
- validation、internal test 和 mBRSET 不参与统计量估计。

冻结统计位于：

```text
data/processed/patient_dataloader_v1/clinical_normalization.json
```

## 5. 使用示例

```python
from pathlib import Path

from dr_grading.data.patient_dataset import PatientDataModule

root = Path("/home/stu1/research/dr-grading-workspace")
data = PatientDataModule(
    project_root=root,
    config_path=root / "configs/data/patient_dataloader_v1.yaml",
    clinical_normalizer_path=(
        root / "data/processed/patient_dataloader_v1/clinical_normalization.json"
    ),
)

train_loader = data.dataloader(
    "train",
    batch_size=8,
    num_workers=4,
    training=True,
)
```

从命令行运行时应使用 Conda 环境 `dr-grading`，并确保项目 `src/` 位于 `PYTHONPATH`；仓库内正式脚本已自动完成路径设置。

## 6. 防泄漏约束

1. 不重新随机划分患者；
2. 不用 mBRSET 计算临床统计、阈值或类别权重；
3. 评估集合不使用随机增强；
4. 任何模型必须同时记录患者清单、图像清单、DataLoader 配置和临床标准化文件的 SHA256；
5. B2、B3、B4 与 Proposed 必须复用同一数据接口。
