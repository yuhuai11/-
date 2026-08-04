# G7-R2：参考Reuter等（2026）的泛化增强消融实验方案

## 1. 实验目标

本实验只改进G7已知二分类模块，即判断音频中是否存在无人机；暂不优化未知型号拒识。
目标从内部Accuracy转向低误报工作点上的跨域召回、阈值迁移和背景误报稳定性。

所有实验使用修正后的`dads_native_half_second_content_component_v2`数据协议。旧1秒DADS
和已经删除的未晋级模型产物不会重新启用。Val/OOD与IDMT已经被使用，因此本阶段只能
称为开发实验，不能称为新的独立外部最终测试。

## 2. 论文中可以借鉴的证据

Reuter等使用AudioSet预训练的紧凑SE-ResNet18和逐项增强消融，在校准背景上确定1%及
5%目标FPR，再保持阈值不变测量不同无人机域的TPR。其10次随机运行结果表明：

| 配置 | IDMT-Test TPR@1%FPR | Berne TPR@1%FPR | AuDroK OOD TPR@1%FPR |
|---|---:|---:|---:|
| No-PT | 0.860 | 0.816 | 0.701 |
| PT | 0.920 | 0.892 | 0.783 |
| PT+mic | 0.923 | 0.891 | 0.816 |
| PT+mic+bg | 0.926 | 0.889 | 0.818 |
| PT+mic+bg+spec | 0.925 | 0.890 | 0.818 |
| PT+mic+bg+spec+pitch | 0.921 | 0.883 | 0.825 |

最主要的提升来自AudioSet预训练；增强链主要改善声学条件不匹配的OOD子集。完整增强
并非在所有域上都更好，ESC-50的1%目标工作点FPR也从PT的0.019升至完整链的0.025，
说明必须同时约束外部背景误报。

## 3. 不能直接照搬的部分

| 论文 | 当前工程 | 本实验处理 |
|---|---|---|
| SE-ResNet18、80 Mel | PANNs Cnn14_16k、64 Mel | 保持G7架构，避免同时改变模型与增强 |
| 1秒训练片段 | 修正后的原生0.5秒片段 | 保持0.5秒内容去重协议 |
| 内部IDMT阵列数据训练 | DADS训练 | 不假设训练域等价 |
| 10个随机种子 | 当前常用42/43/44 | 首轮seed42筛选，冠军再做三种子 |
| SoX保时长pitch shift | 环境中没有SoX | 首轮暂缓，之后独立实现和审计 |
| 完整增强一次叠加 | R1麦克风增强曾使IDMT FPR恶化 | 必须逐项消融和误报保护 |

尤其需要注意，旧G7-R1把结构化麦克风IIR加入更激进的原增强链后，IDMT严格FPR从
`0.066789`恶化到`0.179854`。这不否定论文方法，但说明数据域和增强组合不同，不能
直接认定麦克风模拟一定提高本工程泛化。

## 4. 预注册消融顺序

| 名称 | AudioSet预训练 | 麦克风IIR | 10–20dB背景混合 | 仅频率遮挡 |
|---|---:|---:|---:|---:|
| `pt_control` | 是 | 否 | 否 | 否 |
| `pt_mic` | 是 | 0.33概率 | 否 | 否 |
| `pt_mic_bg` | 是 | 0.33概率 | 0.33概率 | 否 |
| `pt_mic_bg_freq` | 是 | 0.33概率 | 0.33概率 | 0.50概率 |

麦克风IIR范围采用论文公开范围：60–120Hz一阶高通、6–8kHz一阶低通、2–3个峰值EQ、
增益±4dB。背景只混入正类，SNR从[10,20]dB连续均匀采样。频率遮挡最多3块，每块宽度
不超过64个Mel频带的15%，使用该录音Log-Mel均值填充，不进行时间遮挡。

原G7中的混响、彩色噪声和时间平移在这一组实验中全部关闭，以保证每一步的增量可解释。

## 5. 评价协议

第一阶段每个配置只运行seed42：

1. DADS Train用于训练，Validation选checkpoint；
2. 已消费内部Test只作为安全回归，F1相对同轮`pt_control`最多下降0.005；
3. Val/OOD Tune负类按保守经验分位数校准1%和5% FPR阈值；
4. 固定阈值后在Val/OOD Holdout报告TPR、FPR、来源宏平均TPR、ROC-AUC、PR-AUC和pAUC；
5. 在IDMT 0.5秒原生窗口及录音级聚合上比较FPR，并进行会话配对Bootstrap；
6. 只有所有保护门通过的最佳候选才运行seeds 43/44。

候选排序以`TPR@校准FPR=1%`的来源宏平均为主，但有两个否决条件：内部F1明显下降，
或IDMT误报率相对同协议控制组恶化。不能只凭内部Accuracy选模型。

IDMT现有清单以1秒窗口为单位，而新G7使用0.5秒输入。正式IDMT比较前必须生成不丢弃
后半段的0.5秒视图，并同时报告片段级与录音聚合结果；禁止直接截取每个1秒片段的前半段。

## 6. 当前实现

- 统一协议：`configs/g7_r2_generalization_protocol.yaml`
- 四个训练配置：`configs/g7_r2_pt_*.yaml`
- 运行入口：`scripts/run_g7_r2_generalization.sh`
- 连续均匀SNR：`WaveformAugmenter.mix_snr_uniform_db`
- 仅频率遮挡：`RecordingMeanFrequencyMasking`
- 低FPR评估：复用`calibrate_ood`生成冻结预测，再由`evaluate_low_fpr`按负类选择阈值。

## 7. 运行命令

```bash
# 数据与论文完整性
bash scripts/run_g7_r2_generalization.sh validate-data
bash scripts/run_g7_r2_generalization.sh paper-checksum

# 四配置GPU预检
CUDA_VISIBLE_DEVICES=0 bash scripts/run_g7_r2_generalization.sh preflight-all

# 首轮seed42训练示例
CUDA_VISIBLE_DEVICES=0 bash scripts/run_g7_r2_generalization.sh train pt_control 42
CUDA_VISIBLE_DEVICES=0 bash scripts/run_g7_r2_generalization.sh train pt_mic 42
CUDA_VISIBLE_DEVICES=0 bash scripts/run_g7_r2_generalization.sh train pt_mic_bg 42
CUDA_VISIBLE_DEVICES=0 bash scripts/run_g7_r2_generalization.sh train pt_mic_bg_freq 42

# Val/OOD冻结预测与低FPR报告
CUDA_VISIBLE_DEVICES=0 bash scripts/run_g7_r2_generalization.sh calibrate pt_control 42
bash scripts/run_g7_r2_generalization.sh low-fpr pt_control 42
```

## 8. 当前结论边界

代码与协议完成、预检通过只表示实验可运行，不表示泛化已经提升。只有完成同协议控制组、
逐项消融、IDMT误报门禁和三随机种子复现后，才能把冠军称为开发候选；最终泛化结论仍
需要`data/future_final_holdout/`中的全新数据一次性确认。

## 9. 执行状态（2026-08-01）

- 四个配置均已在RTX 3080上通过单批次GPU预检；
- 每个配置参数量为79,955,457，预检峰值GPU分配显存约1.62GB；
- 四项预检的训练/验证Logit和梯度均为有限值，AMP未发生非有限梯度跳步；
- `pt_control seed42`已经启动正式训练；
- 日志：`logs/g7_r2_generalization/pt_control_seed42.log`；
- 当前阶段尚无正式Accuracy、F1或跨域TPR结果，不得提前声称泛化提升。
