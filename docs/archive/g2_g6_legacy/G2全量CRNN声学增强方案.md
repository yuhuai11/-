# G2：Full CRNN训练专用声学增强方案

## 1. 实验目的

OOD校准阶段证明temperature scaling能改善ECE/NLL，却无法同时达到80% Recall和90% Specificity；即使UAV-only的holdout Recall也只有约63%。G2因此验证一个单一问题：保持Full CRNN和当时的验证/回归口径不变，只扩大训练音频的声学条件，是否能改善val_ood开发域表现。

## 2. 冻结项

以下内容与`configs/crnn_dads_full.yaml`一致：

- Full DADS manifest及train/val/test划分；
- 64-bin Log-Mel参数；
- CRNN结构和参数量；
- Adam、学习率、batch size、early stopping；
- `pos_weight: auto`；
- DADS val选择最佳checkpoint；
- DADS原`test`行不参与梯度更新或checkpoint早停；两个最终外部测试集不参与训练及模型选择。

需要按最终审计修正历史口径：`prepare_manifest`实际是在每个标签内随机把Parquet
行（`AudioRef`）分配到Train、Validation和Test，再把每一行展开成1秒片段，因此
同一Parquet行的片段不会跨split；它没有把`source_path`作为分组键。三个split的
`source_path`交集为0只是事后身份审计结果，不能反推为按路径分组。原始WAV字节级
SHA256审计发现15个重复哈希组，共涉及30条Parquet记录；每组保留1条并移除15条，
其中6组的成员原本跨至少两个split。移除这15条冗余记录会连带移除244条历史1秒
manifest片段。

DADS原`test`还在本实验及后续多轮候选门禁中用于决定停止或继续，因此下文将其称为
“历史DADS同数据源已消费内部回归集”，而不是独立测试。旧`dads_dedup_v2`只修复
上述原始WAV字节级精确重复，仍复用把不足1秒音频进行`loop/tail_loop`扩展的历史
1秒预处理与缓存，未消除标签相关的时长/重复捷径。新的主实验复现应改用
`dads_native_half_second_content_component_v2`：从原始WAV构造16 kHz、8000采样点
的原生0.5秒窗口，不循环、不补齐、不拉伸，并让原始与模型输入内容组件整体划分。
因此本方案的历史数值只用于回归对照，不代表跨会话、跨设备、跨场景或跨无人机型号
的独立泛化能力。

首轮只训练seed 42。只有G2在DADS validation和val_ood holdout达到预设门槛，才扩展seed 43、44。

## 3. 唯一变化：训练split在线增强

### 3.1 正类背景混合

70%的训练正类随机混入同一DADS train split的负类片段。目标SNR及条件概率：

| SNR | 概率 |
|---:|---:|
| −5 dB | 10% |
| 0 dB | 25% |
| +5 dB | 35% |
| +10 dB | 30% |

不使用DADS val、历史原test（已消费内部回归）、val_ood、Unseen或Real-world作为
背景池。第一版不加入`-15/-10 dB`，避免极弱无人机正标签主导训练。

### 3.2 两类共同声学扰动

- 50%麦克风频响扰动：8个频率锚点，标准差3 dB，限制在±6 dB；
- 30%轻混响：1～3个12～80 ms延迟，反射增益0.05～0.25；
- 30%粉红型彩色噪声：10～30 dB SNR；
- 50%非循环时间平移：最大±100 ms，空缺补零。

所有增强后统一峰值归一化。单纯整体增益未加入，因为现有波形峰值归一化和逐样本频谱标准化会基本抵消它。

## 4. 可复现性和I/O

Full训练集包含212,502个segment，现有缓存约19 GB。G2在Dataset中在线增强，不复制缓存。配置固定`num_workers: 0`，使随机序列由训练seed确定；若未来启用多worker，必须先实现独立worker seed。

增强只在`training=True`的train Dataset启用，val/test调用路径不变。每个checkpoint保存完整增强配置。

## 5. 运行前审计

```bash
PYTHONPATH=src python -m dads_crnn.audit_augmentation \
  --config configs/crnn_dads_full_augmented_g2.yaml \
  --samples-per-class 1000
```

审计检查：操作频率、SNR档位、目标/实际SNR误差、有限值、峰值范围、背景池及val/test未增强声明。

## 6. 训练

服务器后台运行（在工程根目录执行）：

```bash
mkdir -p logs
nohup env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.train \
  --config configs/crnn_dads_full_augmented_g2.yaml \
  --manifest artifacts_full/manifests/dads_all_seed42.csv \
  --seeds 42 \
  > logs/g2_crnn_full_seed42.log 2>&1 &
echo $!
```

输出：

```text
archive/historical_models/superseded_champions/artifacts_crnn_full_augmented_g2/runs/seed_42/
```

监控：

```bash
tail -f logs/g2_crnn_full_seed42.log
nvidia-smi
```

## 7. 评估顺序

1. 检查DADS val和历史 DADS 同数据源已消费内部回归集，要求F1相对Full CRNN下降不超过1个百分点；
2. 只在val_ood tune拟合temperature和阈值；
3. 在val_ood holdout冻结确认；
4. 达标后再运行seed 43、44；
5. Unseen和Real-world不用于G2筛选。

训练完成后，只用既有val_ood tune/holdout进行校准和冻结评估：

```bash
nohup env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.calibrate_ood \
  --checkpoint archive/historical_models/superseded_champions/artifacts_crnn_full_augmented_g2/runs/seed_42/best.pt \
  --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
  --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
  --output-dir artifacts/val_ood \
  --experiment crnn_full_augmented_g2 \
  --batch-size 256 --num-workers 0 --device auto \
  --target-recall 0.80 --target-specificity 0.90 \
  > logs/g2_crnn_full_seed42_val_ood.log 2>&1 &
echo $!
```

G2通过门槛：

```text
历史 DADS 内部回归F1下降 ≤ 1个百分点
val_ood holdout Recall ≥ 80%
val_ood holdout Specificity ≥ 90%
```

若G2仍无法达标，下一项不应继续堆叠增强，而应单独增加独立无人机来源覆盖或进行Log-Mel/MFCC/PCEN特征消融。

## 8. Seed 42实测结果（2026-07-16）

训练在epoch 43早停，最佳checkpoint为epoch 33，总耗时109.71分钟。历史 DADS 同数据源
已消费内部回归指标满足保护门槛：

| 模型 | 历史 DADS 内部回归F1 | Recall | Specificity | AUC |
|---|---:|---:|---:|---:|
| 原Full CRNN seed 42 | 0.99835 | 0.99824 | 0.99775 | 0.99989 |
| G2 seed 42 | 0.99522 | 0.99467 | 0.99384 | 0.99970 |

F1下降0.31个百分点，小于1个百分点。

val_ood阈值仅由tune确定，冻结后应用于holdout：

| 模型 | 阈值 | AUC | F1 | Recall | Specificity | tune双约束可行 |
|---|---:|---:|---:|---:|---:|---:|
| 原Full CRNN seed 42 | 0.45080 | 0.67536 | 0.62084 | 0.60000 | 0.67295 | 否 |
| G2 seed 42 | 0.41933 | 0.76072 | 0.70955 | 0.71781 | 0.69987 | 否 |

G2使AUC提高0.08536、F1提高0.08871、Recall提高0.11781，证明训练期声学增强有效；但Recall 80%和Specificity 90%仍无法同时满足，因此不扩展seed 43/44。

按条件看，G2将UAV-only Recall从0.6417提高到0.7500，`+5 dB`从0.6505提高到0.8738；但`-15 dB`从0.5306下降到0.4388。下一实验应保持G2增强不变，单独处理来源均衡与困难背景/极低SNR覆盖，仍只运行seed 42筛选。
