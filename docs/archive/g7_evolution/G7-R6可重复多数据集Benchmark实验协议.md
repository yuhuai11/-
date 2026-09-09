# G7-R6可重复多数据集Benchmark实验协议

## 1. 实验定位

本协议允许多个模型版本、消融方案和随机种子反复运行同一组内部与外部测试数据，用于
展示模型效果、稳定性和相对改进。测试集名称统一为`Reusable Benchmark Test`，不再称为
一次性Final Holdout或全新独立盲测。

重复测试指不同检查点、不同seed或不同算法在相同样本上的配对比较。对同一个确定性
检查点重复执行完全相同的推理不会产生新的统计证据，因此没有必要。

## 2. 数据安排

| 角色 | 数据 | 用途 |
|---|---|---|
| Train | DADS train、Kielce train、TAU train、G9 train | 更新模型参数 |
| Model Validation | DADS validation、Kielce tune、TAU Lisbon tune | 早停、选模型和增强 |
| Threshold Calibration | TAU Prague tune | 单独确定1%和5% FPR阈值 |
| Reusable Internal Benchmark | DADS test | 多版本域内回归比较 |
| Reusable External Benchmark | Kielce/TAU holdout、G13、IDMT-TRAFFIC、ESC-50 | 多版本跨域效果比较 |

Train、Validation和Calibration继续保持音频哈希、片段哈希、录音组和来源组零重叠。
Benchmark可以重复推理，但不得用其标签重新拟合阈值。同一系列实验中，如果某个Benchmark
数据被并入训练，必须建立新版本系列并从Benchmark列表移除该数据。

## 3. 多次测试方式

每个算法至少训练seed 42、43、44，然后在完全相同的Benchmark清单上测试。报告：

- 三随机种子均值与标准差；
- G7-R2、G7-R4-C和新候选模型的配对差值；
- 录音级bootstrap 95%置信区间；
- 固定0.5阈值及Calibration确定的1%/5% FPR工作点；
- Accuracy、Precision、Recall、F1、FPR、ROC-AUC、PR-AUC、标准化pAUC；
- 每型号、每会话Recall和每小时误告警数；
- 片段级作为辅助结果，录音级作为主结果。

这种做法能够充分展示模型效果，但论文中必须写明测试集被多个模型重复使用，不能描述为
“从未查看的一次性独立测试”。

## 4. 增加数据集

新增数据接入后可以建立固定版本的公开Benchmark：优先考虑IDMT Berne 2022和新采集的
地面无人机会话、部署环境背景。AuDroK必须先完成与DADS的内容重叠审计；UaVirBASE和
DroneAudioSet只作为任务域不一致的压力测试；NASA数据在许可问题解除前继续禁用。

所有新增数据必须先登记许可、原始录音会话、设备、地点、型号、距离和SHA256，然后固定
为`benchmark_v1`。后续模型都在同一个版本上测试；新增样本则发布`benchmark_v2`，不能
静默改变旧版本组成。

## 5. 当前状态

当前已生成Train、Model Validation和Threshold Calibration清单，可重复Benchmark策略
已启用。G7-R5的一次性测试结果保留为历史结果，但从G7-R6开始允许在登记的Benchmark上
重复评估不同模型。

2026-08-04已从Figshare官方下载DroneNoise Database v3到
`data/g7_r6_new_sources/drone_noise_v3`。共175个文件、742123543字节，其中174个WAV；
全部官方MD5通过。排除9个校准录音后有165个无人机录音通道，归属于20个飞行事件，
采样率50 kHz、单通道float32，总通道时长约1.03小时。与DADS正样本原始哈希及对齐
0.5秒PCM指纹均为零匹配。当前状态为“可进行事件级元数据映射”，尚未加入训练集。

2026-08-04已使用seed 42按飞行事件完成60/20/20划分。同一次飞行的全部麦克风
录音保持在同一集，Validation和Test在每种无人机中各保留1个完整九麦克风事件。
最终Train为12个事件/92个录音，Validation为4个事件/36个录音，Test为4个事件/
36个录音；各集的音频哈希、录音组和事件组交集均为0。划分排除9个校准文件和1个
官方重复清单项。

DroneNoise只有无人机正类，因此其Validation/Test可单独报告按型号和事件的Recall；
若要计算FPR、Specificity和完整二分类Accuracy，必须与事先固定且不重叠的负类语料
联合评估。此外Test只有4个独立飞行事件，它适合可重复对比，不应单独宣称为大规模
外部泛化证据。

2026-08-04已完成50 kHz到16 kHz重采样和原生连续0.5秒缓存，生成7,100个片段：
Train 4,040、Model Validation 1,620、Reusable Positive Test 1,440。无静音片段、
无非有限值、无完整尾部丢弃，集合间原始音频、0.5秒片段、录音组和事件组交集均为0。

G7-R6主协议已升级为`g7_r6_reuter_reusable_multicorpus_v3`。合并后Train为406,157个
0.5秒样本，Model Validation为91,698个样本，Threshold Calibration仍为4,480个
TAU Prague负类，DroneNoise Test作为独立正类测试清单保存1,440个片段。

2026-08-04已建立`g7_r6_dronenoise_data_expansion_control_v1`训练控制组。模型、特征、
训练超参、增强和评估方式与已晋级的G7-R2 `pt_mic_bg_freq`完全一致，唯一变量是
Train和Model Validation加入DroneNoise。训练拟合清单共560,439行：Train 406,157、
Model Validation 91,698、已消费Reusable Internal Test 62,584。DroneNoise Test未进入该清单。
GPU预检已通过，seed 42正式训练已启动。
