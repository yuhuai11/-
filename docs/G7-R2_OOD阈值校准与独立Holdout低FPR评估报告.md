# G7-R2：OOD 阈值校准与独立 Holdout 低 FPR 评估报告

## 1. 实验目的

本报告记录 G7-R2 四个已训练配置在 OOD 开发数据上的阈值校准、独立 Holdout 测试和低 FPR 评估。阈值只使用 Tune 集选择，Holdout 集只用于一次性评估，不能反向调参。

本轮仍属于开发实验，不是全新外部最终确认；所有结果为 `seed42` 首轮结果。

## 2. 数据集及用途

| 数据 | 文件 | 用途 |
|---|---|---|
| DADS 训练集 | `artifacts/g7_leakage_fixed_v2/data/manifest.csv` | 训练四个 G7-R2 模型；使用修正后的原生 0.5 秒内容组件去重协议，共 417221 行 |
| OOD Tune | `artifacts/val_ood/manifests/val_ood_tune_manifest.csv` | 拟合温度、选择阈值；共 1527 个样本 |
| OOD Holdout | `artifacts/val_ood/manifests/val_ood_holdout_manifest.csv` | 固定阈值后的独立测试；共 1473 个样本 |

Tune 与 Holdout 的样本哈希不重叠，预测链和低 FPR脚本均通过数据防火墙检查。Holdout 没有参与阈值选择。

## 3. 评估流程

1. 使用冻结的 G7-R2 检查点分别对 Tune 和 Holdout 推理。
2. 仅在 Tune 上拟合 temperature scaling。
3. 低 FPR 阈值使用 Tune 集负类分数的保守经验分位数，分别对应目标 FPR=1% 和 5%。
4. 将阈值原封不动迁移到 Holdout，报告实际 FPR、TPR、F1 和排序指标。
5. 由于阈值只由 Tune 决定，Holdout 指标用于评估阈值迁移稳定性。

## 4. Holdout 结果

### 4.1 低 FPR 工作点

| 配置 | Tune 温度 | 目标 FPR | Tune 阈值 | Holdout 实际 FPR | Holdout TPR | Holdout F1 |
|---|---:|---:|---:|---:|---:|---:|
| `pt_control` | 15.8050 | 1% | 0.5453 | 1.8843% | 16.5753% | 0.2798 |
| `pt_mic` | 12.0703 | 1% | 0.5280 | 0.9421% | 18.2192% | 0.3057 |
| `pt_mic_bg` | 4.3603 | 1% | 0.5908 | 1.6151% | 21.3699% | 0.3474 |
| `pt_mic_bg_freq` | 4.3757 | 1% | 0.5675 | 1.6151% | **21.5068%** | **0.3493** |
| `pt_control` | 15.8050 | 5% | 0.4560 | 7.6716% | 33.9726% | 0.4792 |
| `pt_mic` | 12.0703 | 5% | 0.4677 | **3.7685%** | 30.4110% | 0.4531 |
| `pt_mic_bg` | 4.3603 | 5% | 0.4643 | 5.7873% | **32.4658%** | 0.4693 |
| `pt_mic_bg_freq` | 4.3757 | 5% | 0.4718 | 4.5760% | 31.6438% | 0.4643 |

### 4.2 Holdout 排序指标

| 配置 | ROC-AUC | PR-AUC | 标准化 pAUC（FPR≤0.05） |
|---|---:|---:|---:|
| `pt_control` | 0.69996 | 0.72534 | 0.58024 |
| `pt_mic` | 0.70750 | 0.74603 | 0.61013 |
| `pt_mic_bg` | **0.73372** | **0.75350** | 0.60096 |
| `pt_mic_bg_freq` | 0.72430 | 0.75276 | **0.60791** |

## 5. 结果解释

- `pt_mic_bg_freq` 在 1% 目标 FPR 下取得最高 Holdout TPR（21.51%）和 F1（0.3493），但实际 FPR 为 1.615%，高于名义 1%。
- `pt_mic` 的 1% 工作点实际 FPR 最接近并低于目标（0.942%），但召回低于两个背景增强配置。
- `pt_mic_bg` 的 Holdout ROC-AUC 和 PR-AUC 最高，说明整体排序能力最好；其 1% 工作点 TPR 略低于 `pt_mic_bg_freq`。
- 所有配置的 Tune 约束 `recall≥0.80、specificity≥0.90` 均未同时满足，因此不能把常规选定阈值解释为满足这两个约束；本报告的主要结论以负类保守 1%/5% FPR 工作点为准。

## 6. 结论边界与下一步

本轮证明了阈值可以从 OOD Tune 集迁移到独立 Holdout 集，并量化了跨域 FPR 膨胀。结果不能直接声称外部泛化已经解决，因为：

- 只完成 `seed42`，尚无 `seed43/44` 的均值和标准差；
- 未包含 IDMT 原生 0.5 秒/录音级误报门禁；
- `val_ood` 是开发数据，不是 `data/future_final_holdout/` 的一次性最终数据。

建议的下一个步骤是先按同一协议比较四个配置的 IDMT 严格 FPR，再对通过门禁的候选（优先 `pt_mic_bg_freq` 与 `pt_mic_bg`）运行 `seed43/44`，最后才考虑全新最终 Holdout。

## 7. 结果文件

- 校准结果：`artifacts/g7_r2_generalization/val_ood/calibration/<candidate>/seed_42/calibration.json`
- Tune 预测：`artifacts/g7_r2_generalization/val_ood/predictions/tune/<candidate>/seed_42/predictions.csv`
- Holdout 预测：`artifacts/g7_r2_generalization/val_ood/predictions/holdout/<candidate>/seed_42/predictions.csv`
- 低 FPR 结果：`artifacts/g7_r2_generalization/low_fpr/<candidate>/seed_42/metrics.json`
- 低 FPR 工作点表：`artifacts/g7_r2_generalization/low_fpr/<candidate>/seed_42/operating_points.csv`
