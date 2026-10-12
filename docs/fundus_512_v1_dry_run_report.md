# fundus_512_v1 小规模 dry-run 报告

运行时间：2026-09-29  
运行环境：`dr-grading`（Python 3.10.19、Pillow 12.0.0、OpenCV 4.8.1）

## 命令

```bash
/home/stu1/anaconda3/envs/dr-grading/bin/python scripts/preprocess_fundus.py \
  --output-root data/processed/fundus_512_v1_dry_run \
  --limit-per-dataset 6
```

## 结果

- 总记录：12；
- BRSET：6；mBRSET：6；
- 成功：12；错误：0；安全回退：0；
- 相机覆盖：Canon CR 4、NIKON NF5050 2、Phelcom Eyer 6；
- BRSET 眼别编码：1/2 各3；mBRSET：left/right 各3；
- 前景比例范围：0.75606679–0.78983307；
- 全部输出重新解码为 RGB、512×512；
- 全部输出 SHA256 与 manifest 记录一致；
- 抽查原图与处理图，未观察到由算法新增的视盘、黄斑或视网膜边缘截断；
- 原始文件仅以只读方式打开，未覆盖或修改。

配置 SHA256：

```text
3fe348cddf620d8153149ba0f73d3206350abcc103740777be94ec651a4d0822
```

manifest SHA256：

```text
61bca326faf3e57c48eec6d3493da7422e5079763f8f3a25f34bcbacda7a54aa
```

## 输出

- `data/processed/fundus_512_v1_dry_run/preprocessing_manifest.csv`
- `data/processed/fundus_512_v1_dry_run/resolved_config.yaml`
- `data/processed/fundus_512_v1_dry_run/run_metadata.json`
- `data/processed/fundus_512_v1_dry_run/brset/`
- `data/processed/fundus_512_v1_dry_run/mbrset/`

## 解释范围

本次 dry-run 只验证代码路径、格式、哈希、拒绝覆盖和少量视觉结果，不替代协议规定的正式分层QC。开始全量预处理前，仍需先对 BRSET-Canon、BRSET-Nikon、mBRSET 各至少100张图生成覆盖层并人工检查；全量完成后再随机检查至少200张。
