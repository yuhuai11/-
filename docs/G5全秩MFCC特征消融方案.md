# G5：全秩MFCC-64特征消融与安全晋级方案

## 1. 实验目的

G3证明替换式来源均衡会破坏正类覆盖；G4证明增加低SNR样本虽然能改善目标子组，却会损害UAV-only和高SNR条件。后续不再叠加数据分布调整，回到当前最佳G2，只检验前端表示是否限制跨域分离能力。

G5采用champion–challenger：G2永久保留，G5独立训练；任何审计、内部指标、整体OOD指标或条件级指标失败，都输出`keep_g2`。

## 2. 为什么使用MFCC-64而不是现有MFCC-40

现有ResNet配置使用40维MFCC、512点窗、256点hop和0.97预加重。直接复制会同时改变：

- 频谱轴由64降至40；
- CRNN参数量由1,012,193降至717,281（下降29.1%）；
- 时间帧数、窗函数和高频权重。

这样无法区分收益来自MFCC还是模型容量/时间分辨率。G5因此使用全秩MFCC-64：

```yaml
features:
  type: mfcc
  n_mfcc: 64
  n_mels: 64
  n_fft: 512
  win_length: 400
  hop_length: 160
  f_min: 20
  f_max: 8000
  preemphasis: 0.0
  window_type: hann
```

与G2相比，STFT、Hann窗、帧数、Mel滤波器、全局标准化和CRNN容量均不变，唯一有效变化是对64维Log-Mel执行完整正交DCT，从频谱基底变换为倒谱基底。

MFCC默认仍保持Hamming窗，确保旧ResNet checkpoint重建行为不变；只有G5显式选择Hann窗。

## 3. 历史证据与预期

未增强Full模型的seed 42结果仅提供弱先验：ResNet+MFCC的val_ood AUC为0.71191，高于CRNN+Log-Mel的0.67536，但Recall较低，且模型结构混杂，不能证明MFCC本身更优。

G5的目标不是预先承诺提升，而是进行严格的同容量特征消融：

- 若AUC/Recall整体提升且所有条件不退化，MFCC可晋级；
- 若仅提高Specificity或少数条件，保留G2；
- 若MFCC-64失败，不直接推导MFCC-40也失败；40维压缩必须作为后续“特征+容量”联合实验单独标注。

## 4. 冻结项

以下内容逐字段等于G2：

- Full DADS manifest、自然shuffle和212,502个样本/epoch；
- G2全部声学增强及SNR分布；
- CRNN结构、参数量、dropout；
- batch size、Adam、学习率、early stopping；
- `pos_weight: auto`、`num_workers: 0`；
- DADS val checkpoint选择和0.40/0.50/0.65评估阈值；
- seed 42首轮筛选。

独立输出：

```text
archive/historical_models/failed_candidates/artifacts_crnn_full_augmented_g5_mfcc64/runs/
```

## 5. 训练前审计

```bash
PYTHONPATH=src /usr/local/anaconda3/envs/dads-crnn/bin/python -u \
  -m dads_crnn.audit_feature_ablation \
  --baseline configs/crnn_dads_full_augmented_g2.yaml \
  --candidate configs/crnn_dads_full_augmented_g5_mfcc64.yaml \
  --guardrails configs/g5_mfcc64_guardrails.yaml \
  --manifest artifacts_full/manifests/dads_all_seed42.csv \
  --mode preflight
```

当前审计已通过：

- G2/G5特征形状均为`[2,1,64,101]`；
- 两个CRNN均为1,012,193参数；
- DCT正交最大误差`2.265e-6`；
- 特征、logit、loss及梯度均为有限值；
- 唯一配置差异为特征类型、MFCC/Hann显式参数及独立输出目录；
- G2、DADS/val_ood manifests、基线逐样本预测和所有关键运行代码已记录SHA256。

保护库存：

```text
artifacts/g5_mfcc64/baseline_inventory.json
artifacts/g5_mfcc64/delta_audit.json
```

## 6. Seed 42服务器训练

```bash
cd /home/user1/JJZ/ABDDV-CRNN
mkdir -p logs

nohup env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.train \
  --config configs/crnn_dads_full_augmented_g5_mfcc64.yaml \
  --manifest artifacts_full/manifests/dads_all_seed42.csv \
  --seeds 42 \
  > logs/g5_crnn_mfcc64_seed42.log 2>&1 &
echo $!
```

监控：

```bash
tail -f logs/g5_crnn_mfcc64_seed42.log
```

## 7. 第一关：DADS内部保护

训练完成后必须先把审计切换为verify；它会核对保护库存、候选checkpoint、metrics和checkpoint内嵌配置：

```bash
PYTHONPATH=src /usr/local/anaconda3/envs/dads-crnn/bin/python \
  -m dads_crnn.audit_feature_ablation --mode verify

PYTHONPATH=src /usr/local/anaconda3/envs/dads-crnn/bin/python \
  -m dads_crnn.gate_candidate \
  --config configs/g5_mfcc64_guardrails.yaml \
  --stage internal
```

最低要求：Val/Test F1相对G2最多下降0.002，Val AUC最多下降0.001，Test Recall和Specificity均不低于0.99，Test AUC不低于0.995。失败时立即`keep_g2`，不运行val_ood。

## 8. 第二关：val_ood严格晋级

只有第一关通过才运行：

```bash
nohup env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.calibrate_ood \
  --checkpoint archive/historical_models/failed_candidates/artifacts_crnn_full_augmented_g5_mfcc64/runs/seed_42/best.pt \
  --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
  --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
  --output-dir artifacts/val_ood \
  --experiment crnn_full_augmented_g5_mfcc64 \
  --batch-size 256 --num-workers 0 --device auto \
  --target-recall 0.80 --target-specificity 0.90 \
  > logs/g5_crnn_mfcc64_seed42_val_ood.log 2>&1 &
echo $!
```

完成后：

```bash
PYTHONPATH=src /usr/local/anaconda3/envs/dads-crnn/bin/python \
  -m dads_crnn.gate_candidate \
  --config configs/g5_mfcc64_guardrails.yaml \
  --stage final
```

最终门控会确认G2/G5的tune与holdout逐样本身份、顺序和标签完全一致。晋级要求：

1. holdout F1、Balanced Accuracy、Recall、Specificity、AUC全部不低于G2；
2. F1或AUC至少提高0.02，或Recall至少提高0.03；
3. UAV-only及六个SNR条件Recall、背景Specificity，任一不得下降超过0.03；
4. DADS内部保护和所有SHA256仍通过。

只有全部满足才输出`promote_g5_mfcc64`。否则输出`keep_g2`，不训练seed 43/44，也不查看Unseen/Real-world。

## 9. 方法学边界

val_ood holdout已参与多轮候选判断，应视为开发验证集。即使G5晋级，也只能说明相对G2在当前开发域更优；论文最终泛化结论仍需新的、从未查看过的外部测试集。

## 10. Seed 42实际结果与终止判定

训练于2026-07-17完成，共运行72个epoch，最佳checkpoint出现在epoch 62；训练耗时约181.29分钟。训练后`verify`审计全部通过：G2与G5的输入形状、模型参数量及除特征基底外的配置保持一致，保护库存中的基线模型、数据清单、逐样本预测和关键代码SHA256均未发生变化。

在固定阈值0.50下，内部结果如下：

| Split | 指标 | G2 Log-Mel | G5 MFCC-64 | G5-G2 | 门控结果 |
|---|---|---:|---:|---:|---|
| Val | F1 | 0.995003 | 0.991483 | -0.003520 | 失败 |
| Val | AUC | 0.999764 | 0.999248 | -0.000517 | 通过 |
| Test | F1 | 0.995222 | 0.992127 | -0.003095 | 失败 |
| Test | Recall | 0.994673 | 0.990669 | -0.004004 | 通过绝对下限 |
| Test | Specificity | 0.993840 | 0.990680 | -0.003160 | 通过绝对下限 |
| Test | AUC | 0.999696 | 0.999352 | -0.000344 | 通过绝对下限 |

Val F1与Test F1的下降均超过预注册的最大允许值0.002，因此内部硬门控判定为`keep_g2`。按既定协议立即停止G5，不运行val_ood、不扩展seed 43/44，也不查看Unseen或Real-world结果。

该结果说明：全秩DCT虽然不丢失64维输入信息，但会改变局部卷积看到的邻域关系；对当前CRNN而言，MFCC-64的倒谱局部性不如Log-Mel的频率局部性匹配。它不能证明MFCC对所有结构都无效，但足以否决“用MFCC-64直接替换G2前端”这一候选。后续不得通过放宽门槛或在val_ood上补看结果来挽救G5。

正式产物：

```text
artifacts/g5_mfcc64/delta_audit.json
artifacts/g5_mfcc64/gate.json
artifacts/g5_mfcc64/gate.md
archive/historical_models/failed_candidates/artifacts_crnn_full_augmented_g5_mfcc64/runs/seed_42/
```
