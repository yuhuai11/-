# G22旋翼谐波特征与PANNs后融合方案

## 1. 研究问题

G18已经利用PANNs 2048维Embedding获得较好的九型号分类结果，但三个随机种子的
Known Tune平均Accuracy为0.9670、Macro-F1为0.9669，最低类别Recall为0.8571。
G19—G21没有稳定超过它。G22研究一个更具体的问题：

> 旋翼基频、谐波间距和谱峰稳定性等物理声学特征，能否为G18的通用预训练Embedding
> 提供互补证据？

G22不更换数据集、不研究Unknown拒识，也不修改正式G7检测器。

## 2. 公开研究依据

无人机旋翼声音通常表现为基频及其多阶谐波。G22参考的主要公开工作包括：

- Fan等，*Research on Sound Recognition of Long-Distance UAV Based on
  Harmonic Features*，Drones 2026：使用高阶谐波特征，在120米条件下对Phantom 4
  Pro V2.0获得78.03%识别率，相比MFCC提高16.03个百分点；
- He等，*SVM-based drone sound recognition using the combination of HLA
  and WPT techniques in practical noisy environment*：结合谐波线关联和子带功率；
- Uluskan，*Precise acoustic drone localization and tracking via drone noise:
  SRP-PHAT around harmonics*，2024：只聚焦谐波邻域进行声学处理；
- Kielce原始论文使用MFCC判别分析完成多型号分类。

这些方法和特征均不是本工程原创。G22的候选贡献是把可解释的旋翼谐波先验与PANNs
下游型号Logit在严格录音隔离协议下进行低权重融合。

## 3. 特征与融合

每个1秒、16 kHz片段提取61维特征：

- 32个对数频带能量；
- 频谱质心、带宽、平坦度、熵和三个roll-off；
- 40—500 Hz范围内的谐波基频搜索；
- 12阶谐波突出度和有效谐波比例；
- 四个0.25秒子帧的基频与谐波强度稳定性。

一条原始录音最多包含20个片段。片段特征的均值和标准差拼接为122维录音特征，使用
`StandardScaler + class-balanced LogisticRegression(C=0.1)`训练九分类谐波分支。

G18与谐波分支的每条录音类别分数分别中心化并除以标准差，固定融合为：

```text
score_G22 = normalized_score_G18 + 0.25 × normalized_score_harmonic
```

`0.25`在运行前写入配置并由代码锁定。其他权重只输出诊断曲线，不能事后替换主结果。

## 4. Seed 42探针结果

| 方法 | Accuracy | Macro-F1 | 最低Recall |
|---|---:|---:|---:|
| G18 seed 42 | 0.978022 | 0.979675 | 0.857143 |
| 谐波分支单独使用 | 0.912088 | 0.898718 | 0.714286 |
| G18＋谐波，固定0.25 | 0.978022 | 0.980531 | 0.928571 |

谐波分支明显弱于PANNs，不能独立替代G18。固定融合纠正了一条
`MAVICAIR2S→PHANTOM4`错误，但新增一条`MAVIC2PRO→MAVICAIR2`错误，因此总正确
录音数不变；提升主要体现在类别均衡和最低Recall。

诊断权重0.5和0.75达到0.9890 Accuracy，但它们是在查看Tune曲线后观察到的，不能
改写成G22正式主结果。

## 5. 三随机种子固定权重消融

固定0.25权重应用于G18 seed 42、43、44：

| Seed | G18 Accuracy | G22 Accuracy | G18 Macro-F1 | G22 Macro-F1 |
|---:|---:|---:|---:|---:|
| 42 | 0.978022 | 0.978022 | 0.979675 | 0.980531 |
| 43 | 0.956044 | 0.978022 | 0.957272 | 0.980541 |
| 44 | 0.967033 | 0.978022 | 0.963802 | 0.980541 |

聚合结果：

| 指标 | G18三种子均值 | G22三种子均值 | 变化 |
|---|---:|---:|---:|
| Accuracy | 0.967033 | 0.978022 | +0.010989 |
| Macro-F1 | 0.966916 | 0.980537 | +0.013621 |
| 最低Recall | 0.857143 | 0.928571 | +0.071429 |

三个门槛全部通过：

- 平均Accuracy不下降；
- 平均Macro-F1严格提升；
- 每个seed的最低Recall均不下降。

G22因此是当前Known Tune上的开发冠军候选，但还不是新的正式最终冠军。原因是91条
Known Tune已经参与方法评价，G18原有Known Holdout也已在P6消费。必须使用未来新增、
从未参与开发的录音级Holdout才能确认真实泛化提升。

## 6. 数据边界

G22只使用：

- `known_train`：拟合谐波分类器；
- `known_tune`：开发评价；
- 冻结的G18三个随机种子检查点。

没有读取Unknown Tune、Known/Unknown Holdout、X6D或Y6，也没有重新利用G18 P6结果
进行权重选择。

## 7. 复现命令

```bash
cd /home/user1/JJZ/ABDDV-CRNN
bash scripts/run_g22_harmonic_fusion_probe.sh
bash scripts/run_g22_harmonic_multiseed.sh
```
