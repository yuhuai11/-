# 历史模型归档

本目录集中保存已经被替代的基线、未晋级候选模型及其历史日志。归档的目的不是删除
实验，而是把冷数据从工程根目录移出，同时保留检查点、指标、预测和复现配置。

当前正式模型不在本目录。正式G7仍位于：

```text
artifacts_g7_panns_pt/runs/seed_42/best.pt
```

## 目录

```text
historical_models/
├─ baselines/                 # 原始CRNN和ResNet10-CBAM对照模型
├─ superseded_champions/      # 曾经晋级、后来被G7替代的阶段冠军
├─ failed_candidates/         # 未通过门控、未完成或未获晋级的候选
├─ logs/
│  ├─ baseline_evaluations/
│  ├─ superseded_champion/
│  └─ failed_candidates/
├─ MODEL_REGISTRY.csv
└─ CHECKSUMS.sha256
```

## 分类原则

### baselines

这些模型不是“无效模型”，而是论文复现和后续提升比较所需的历史基线：

- CRNN 15000/类；
- CRNN全量；
- ResNet10-CBAM 15000/类；
- ResNet10-CBAM全量。

### superseded_champions

G2全量增强CRNN曾经是阶段冠军，后来被正式G7替代，因此单独保存，不能标记为失败。

### failed_candidates

- G3来源均衡：内部F1显著下降；
- G4低SNR轻量覆盖：门控输出`keep_g2`；
- G5 MFCC-64：Validation/Test F1非劣门失败；
- G6时序注意力：val_ood门控失败；
- G7 Scratch：训练曾受GPU/图形会话故障影响，且没有替代预训练G7；
- G9 PANNs-HN：后续G10/G11门控未允许整体替代G7。

“failed”表示没有通过预注册晋级门槛，不表示相关方法或产物没有研究价值。

## 使用规则

1. 默认不从本目录恢复正式服务；
2. 历史配置已经更新为新的归档路径；
3. 生成产物中的旧绝对或相对路径作为历史记录保留，不修改其内容；
4. 复算结果时优先使用`MODEL_REGISTRY.csv`中的当前路径；
5. 不要删除检查点、指标JSON和预测NPY，除非完成独立离线备份及SHA256复核；
6. 当前开发应优先阅读`docs/G17双采样率互补表征与安全后融合方案.md`。

## 路径迁移

| 旧路径 | 新路径 |
|---|---|
| `artifacts_15000/` | `archive/historical_models/baselines/artifacts_15000/` |
| `artifacts_crnn_full/` | `archive/historical_models/baselines/artifacts_crnn_full/` |
| `artifacts_resnet10_cbam_15000/` | `archive/historical_models/baselines/artifacts_resnet10_cbam_15000/` |
| `artifacts_resnet10_cbam_full/` | `archive/historical_models/baselines/artifacts_resnet10_cbam_full/` |
| `artifacts_crnn_full_augmented_g2/` | `archive/historical_models/superseded_champions/artifacts_crnn_full_augmented_g2/` |
| `artifacts_crnn_full_augmented_g3_source_balanced/` | `archive/historical_models/failed_candidates/artifacts_crnn_full_augmented_g3_source_balanced/` |
| `artifacts_crnn_full_augmented_g4_low_snr/` | `archive/historical_models/failed_candidates/artifacts_crnn_full_augmented_g4_low_snr/` |
| `artifacts_crnn_full_augmented_g5_mfcc64/` | `archive/historical_models/failed_candidates/artifacts_crnn_full_augmented_g5_mfcc64/` |
| `artifacts_crnn_full_augmented_g6_temporal_attention/` | `archive/historical_models/failed_candidates/artifacts_crnn_full_augmented_g6_temporal_attention/` |
| `artifacts_g7_panns_scratch/` | `archive/historical_models/failed_candidates/artifacts_g7_panns_scratch/` |
| `artifacts_g9_panns_hn_bce/` | `archive/historical_models/failed_candidates/artifacts_g9_panns_hn_bce/` |

