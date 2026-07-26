# 2024—2026 年声学无人机识别研究综述

**整理日期：** 2026-07-13  
**范围：** 2024-01—2026-07 公开的期刊论文、会议论文、预印本和数据集论文。  
**主题：** 使用麦克风或麦克风阵列对无人机进行检测、机型/类别识别、定位和声学身份认证。

> 说明：不同论文的任务定义、无人机数量、录音距离、背景噪声、训练/测试划分差异很大。下文的准确率只用于复述原论文结果，不能直接作为统一排行榜。

## 一、主要结论

1. **单麦克风频谱分类仍是最容易复现的基线。** 常见输入是 Log-Mel、MFCC 或 STFT 频谱图，模型以 2D-CNN、CRNN、BLSTM 为主。对随机切分的封闭数据集，准确率常达到 95% 以上，但跨录音会话、跨地点和跨机型的性能通常明显下降。
2. **研究重点正在从“高准确率”转向“可泛化检测”。** 2026 年 Reuter 等将 AudioSet 预训练、无人机专用增强和会话独立评测结合，在 AuDroK 跨域测试集上比无预训练模型更稳定；这比单一数据集的随机切分准确率更有工程意义。
3. **阵列信号处理是远距离部署的关键。** 波束形成、MVDR、DOA 和三维阵列定位可以抑制背景噪声并估计方向/距离，但它们解决的主要是“哪里有无人机”，不等同于“是哪一种无人机”。
4. **融合声学特征成为主流趋势。** AUDRON 将 MFCC、STFT、Mel-BiLSTM 和原始波形自编码器融合；另有工作融合 RF 与声学特征，或融合声音与视觉。融合通常有收益，但也提高了数据同步、模型复杂度和消融验证要求。
5. **数据集建设开始成为独立研究方向。** DroneAudioSet、UaVirBASE、NASA 小型 UAS 飞越数据、32 类多类别数据集以及 ERAU 的 YAMNet 嵌入数据，为低信噪比、定位和跨域实验提供了基础。
6. **适合做论文的空白点是“跨域、低误报、开放集”。** 目前不少工作仍采用同一录音随机切片划分，容易让相邻片段同时出现在训练集和测试集；真正面向未知地点、未知背景和未知机型的评测仍然不足。

## 二、代表性论文与工作

| 年份 | 文献 | 任务与方法 | 数据/结果（按原文） | 可复现性与备注 |
|---|---|---|---|---|
| 2024 | Kümmritz, *The Sound of Surveillance: Enhancing Machine Learning-Driven Drone Detection with Advanced Acoustic Augmentation*, **Drones** | VGGish 嵌入；两级分类：无人机/非无人机，再分 C0–C3 重量级别；谐波失真、噪声、变调、延迟等增强 | 40 种无人机、约 23.42 h、1 s 片段；检测无人机 99.1%、非无人机 97.2%；C0–C3 平均约 93.8–94.7% | [论文](https://www.mdpi.com/2504-446X/8/3/105)；[MATLAB 代码](https://github.com/H2ThinkResearchInstitute/DroneClassifier)。训练/验证片段相关，外部机型泛化较弱，适合做增强和复现基线 |
| 2024 | Mięsikowska 等, *Classification of Unmanned Aerial Vehicles Based on Acoustic Signals Obtained in External Environmental Conditions*, **Sensors** | 12 个 MFCC + 判别函数分析 | 17 架无人机、13 个型号组，4 个室外地点，8 m 悬停；报告分类率 98.8% | [论文](https://www.mdpi.com/1424-8220/24/17/5663)。封闭集、固定距离/高度，适合传统特征基线，不代表开放环境性能 |
| 2024 | Sun 等, *Deep Learning-based drone acoustic event detection system for microphone arrays*, **Multimedia Tools and Applications** | 麦克风阵列、波束形成/方向性前端与 Log-Mel/MFCC、CNN | 研究阵列相对单麦克风的检测收益，并报告可探测距离可达约 135 m | [DOI](https://doi.org/10.1007/s11042-023-17477-1)。重点是事件检测和阵列前端，不是细粒度机型识别 |
| 2024 | Sun 等, *Improved method for drone sound event detection system aiming at the impact of background noise and angle deviation*, **Sensors and Actuators A: Physical** | 面向背景噪声和角度偏差的阵列/深度学习改进 | 重点考察噪声与入射角变化对检测的影响 | [DOI](https://doi.org/10.1016/j.sna.2024.115676)。适合作为鲁棒阵列检测方向的补充 |
| 2024 | *Drones Detection Using a Fusion of RF and Acoustic Features and Deep Neural Networks*, **Sensors** | MFCC、GTCC 与 RF 特征融合；比较 CNN、RNN、SVM 等 | 采用 16 频带 Mel 特征，验证声学与射频互补性 | [论文](https://www.mdpi.com/1424-8220/24/8/2427)。属于多模态/多传感器，不是纯声学方案 |
| 2024 | Andryushchenko 等, *Detection of Unmanned Aerial Vehicles Using Microphone Arrays* | 麦克风阵列、声源方向/距离估计 | 讨论阵列检测范围和工程布置 | [论文页面](https://journals.rcsi.science/2075-3608/article/view/320037)。可作为阵列硬件和定位背景阅读 |
| 2025 | Lei 等, *UAV Audio Detection and Identification Using Short-Time Fourier Transform Spectrograms with Deep Learning Models*, **ICUAS** | STFT 频谱图；比较 CNN、RNN、CRNN | 同时研究检测与识别；公开信息未给出统一跨域指标 | [DOI](https://doi.org/10.1109/ICUAS65942.2025.11007784)。适合作为 STFT-CRNN 基线 |
| 2025 | Azizi 等, *Convolutional Neural Network Classifier for Unmanned Aerial Vehicles Detection and Identification Using Mel-Frequency Spectrograms*, **The Journal of Engineering** | 64 Mel 滤波器、FFT 512/hop 256、2D-CNN | Bebop/Mambo 两类；检测准确率 98.23%，识别准确率 95.39%；70/30 划分 | [论文](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/tje2.70135)。类别和环境较少，需避免过度外推 |
| 2025 | Wu Canbo、Han Gangtao, 《基于时频注意力和软阈值化 CNN 的无人机声学检测与识别》 | Mel 频谱；时频注意力 + 软阈值 CNN | 采集 8 类无人机及背景噪声；报告准确率、精确率、召回率和 F1 优于对比方法 | [论文信息](https://openurl.ebsco.com/contentitem/gcd%3A186939226)。中文方法论文；公开元数据未显示全部数值 |
| 2025 | Tegler 等, *Detection and Localization of Drones and UAVs Using Sound and Vision*, **CVPRW Anti-UAV** | 麦克风阵列声学定位 + 固定/PTZ 相机 | 真实飞行场景，声学 DOA 可在数百米范围内辅助定位 | [CVF 全文](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Tegler_Detection_and_Localization_of_Drones_and_UAVs_Using_Sound_and_Vision_CVPRW_2025_paper.html)。核心是定位和声视融合，不是音色分类 |
| 2025 | Li 等, *DroneAudioID: A Lightweight Acoustic Fingerprint-Based Drone Authentication System for Secure Drone Delivery*, **IEEE TIFS** | 基频/谐波声纹，轻量化声学指纹认证 | 目标是同型号无人机的个体身份认证和安全配送 | [作者页面](https://lynnlilu.github.io/publication/2025-01-journal-droneaudioid-tifs)。任务从“类别识别”扩展到“个体认证”，需关注攻击和重放鲁棒性 |
| 2025 | Chatterjee 等, *AUDRON: A Deep Learning Framework with Fused Acoustic Signatures for Drone Type Recognition* | MFCC-Conv1D、STFT-CNN、Mel-BiLSTM 注意力、原始波形自编码器四分支融合 | 合成数据 99.92%；二分类 98.51%；Bebop/Mambo/Noise 多类 97.11%；去掉 MFCC 后下降 4.70 个百分点 | [arXiv](https://arxiv.org/abs/2512.20407)。预印本/会议论文；随机切片、类别不平衡、无独立测试集，暂未发现官方代码和权重 |
| 2025 | Jekateryńczuk 等, *UaVirBASE: A Public-Access Unmanned Aerial Vehicle Sound Source Localization Dataset*, **Applied Sciences** | 多麦克风同步录音，用于距离、方位角、侧向姿态估计 | 公开约 13.8 GB 数据；基线 MAE：距离/高度约 0.5 m，方位角约 1°，侧向姿态小于 10° | [论文](https://www.mdpi.com/2076-3417/15/10/5378)；[数据与代码](https://zenodo.org/records/15391924)；[GitLab](https://gitlab.com/g.jekaterynczuk/uavirbase_ssl)。定位数据集，不是分类排行榜 |
| 2025 | Gupta 等, *DroneAudioset: An Audio Dataset for Drone-based Search and Rescue*, **NeurIPS 2025 Datasets / arXiv** | 低信噪比无人机听觉、检测和搜索救援 | 约 23.5 h 标注音频，SNR 约 −57.2 至 −2.5 dB，包含不同无人机、油门、麦克风和环境 | [arXiv](https://arxiv.org/abs/2510.15383)；[Hugging Face 数据集](https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet)。适合低 SNR、噪声鲁棒性和事件级评测 |
| 2025 | Wang 等, *A Multiclass Acoustic Dataset and Interactive Tool for Analyzing Drone Signatures in Real-World Environments* | 多类别音频、频谱图和 MFCC 可视化工具 | 32 个品牌/型号类别，提供原始音频与特征交互分析 | [arXiv](https://arxiv.org/abs/2509.04715)；[交互工具](https://mackenzie-jane.github.io/drone-visualization/)。更偏数据资源和探索性分析 |
| 2025 | ERAU, *Uncrewed Aircraft Detection from YAMNET Embedding Dataset* | 直接使用 YAMNet 嵌入训练浅层全连接模型 | 9108 个 1 s 嵌入，含 M100、Mavic 3、Mini 2 与非无人机；两层全连接测试准确率约 96% | [数据集](https://datacommons.erau.edu/datasets/5dmcszvym4/3)。适合快速验证分类器，但嵌入损失了部分原始声学信息 |
| 2025 | NASA, *Small UAS Flyover Acoustics Data* | 小型 UAS 飞越声学数据资源 | 真实飞越录音，适合飞越事件检测、距离和环境变化研究 | [NASA 数据集](https://data.nasa.gov/dataset/small-uas-flyover-acoustics-data)。需自行检查许可、标注和切分方式 |
| 2026 | Reuter、Ohlenbusch、Rollwage, *Improving Acoustic Drone Detection Generalization Through Pretraining and Data Augmentation* | 18 层 2D SE-ResNet；AudioSet 预训练；变调、噪声混合、麦克风 IIR、SpecAugment | 16 kHz、1 s 片段；会话独立和跨数据集评测。FPR=1% 时，AuDroK OOD TPR：无预训练 0.701，预训练 0.783，预训练+增强 0.825；最难 Audacity 子集从 0.166 提升到 0.662 | [arXiv](https://arxiv.org/abs/2605.31329)。2026 预印本/Quiet Drones 2026；公开页面未发现官方代码或权重。论文指出 1% 片段级 FPR 仍约等于每小时 36 个误报片段，需事件聚合 |
| 2026 | Ghenescu 等, *Acoustic Source Drone Detection System Using Tetrahedral Microphone Array and Deep Neural Networks*, **Sensors** | 四面体阵列、方向性特征与传感器几何元数据融合，三维声源定位 | RTX 4070M 上约 26.63 ms/帧；在 7–15 m 场景报告较高定位精度 | [论文](https://www.mdpi.com/1424-8220/26/6/1778)。重点是实时检测/定位，未来方向包括 RNN、Transformer 和多传感器融合 |
| 2026 | *Ensemble learning models for micro-drone detection using integrated acoustic signatures*, **Discover AI** | 13 维 MFCC + 32 维自相关系数；RF、AdaBoost、XGBoost 与 stacking ensemble | 2664 个 1 s 片段，70/30 分层切分；stacking 集成准确率约 97.12%，XGBoost AUC 报告为 1.00 | [论文](https://link.springer.com/article/10.1007/s44163-026-00869-1)。轻量非深度学习对照；随机切分、封闭集，跨域证据有限 |

## 三、方法路线的演进

### 1. 频谱图 CNN/CRNN：最稳妥的起点

典型流程是：16 kHz 重采样 → 1–2 s 分帧 → STFT 或 Log-Mel/MFCC → 2D-CNN/CRNN → 类别概率。STFT 保留时间—频率纹理，MFCC 更强调谱包络，Log-Mel 在计算量和可视化之间较平衡。Lei 等的 STFT-CNN/RNN/CRNN 对比、Azizi 等的 Mel-CNN，以及 Kümmritz 的 VGGish 嵌入，都适合做第一组基线。

### 2. 多分支特征融合：提高互补性，但要防止“复杂度换分数”

AUDRON 将四种声学表征拼接后再融合。其消融结果表明 MFCC 分支贡献最大，但该结论是在其数据划分下得到的，不能直接证明 MFCC 在所有场景都优于原始波形。复现时应报告单分支、两分支和全融合模型的参数量、推理时延及跨域性能，而不能只报告融合模型准确率。

### 3. 预训练与增强：2026 年最值得跟进的方向

Kümmritz 主要考察物理启发增强：噪声、变调、延迟和谐波失真；其总体类均值改善有限，但个别室外机型在增强后明显改善。Reuter 等进一步把 AudioSet 预训练、无人机专用增强和会话独立测试结合起来，并使用 TPR@固定 FPR 衡量泛化。该路线适合研究：

- 声音域预训练（AudioSet/YAMNet/PANNs/自监督音频编码器）；
- 设备响应模拟（麦克风频响、带通、混响、压缩失真）；
- 背景混音和低 SNR 训练；
- 跨地点、跨距离、跨机型的域泛化；
- 片段级分类到事件级告警的后处理。

### 4. 麦克风阵列：从“识别”走向“检测+定位”

阵列方案先利用波束形成、MVDR 或 GCC-PHAT 获得方向性信号，再用 CNN/CRNN 做事件判断。UaVirBASE 和 Ghenescu 的工作说明，阵列数据可以把距离、方位角、姿态作为监督目标。若最终系统要在室外固定点部署，建议把“是否有无人机”和“方位/距离”分成多任务头；不要只把阵列通道简单拼接后做分类。

### 5. 开放集、个体认证和多模态

DroneAudioID 关注同型号不同个体的声学指纹认证，属于身份安全问题；RF+声学和声视融合则针对恶劣环境下的互补传感。未来可以把分类器改成“已知型号分类 + 未知类拒识 + 个体嵌入检索”三级结构，以应对数据库没有登记的新机型。

## 四、数据集与复现实验资源

| 资源 | 规模/内容 | 最适合的实验 |
|---|---|---|
| [Kümmritz 数据/代码](https://github.com/H2ThinkResearchInstitute/DroneClassifier) | 约 23.42 h、40 种无人机；MATLAB 增强与 VGGish 分类流程 | VGGish、传统增强、两级分类复现 |
| [DroneAudioSet](https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet) | 约 23.5 h，极低 SNR、多设备、多环境 | 低 SNR 检测、事件级召回、噪声泛化 |
| [UaVirBASE](https://zenodo.org/records/15391924) | 公开多麦克风同步数据，含距离/高度/方位/姿态 | DOA、三维定位、多任务学习 |
| [Wang 多类别数据](https://arxiv.org/abs/2509.04715) | 32 个品牌/型号类别，原始音频和可视化工具 | 多类别探索、特征可解释性、开放集初测 |
| [ERAU YAMNet 嵌入](https://datacommons.erau.edu/datasets/5dmcszvym4/3) | 9108 个 1 s 嵌入，3 种无人机 + 非无人机 | 快速验证 MLP、校准和类别不平衡 |
| [NASA Small UAS Flyover](https://data.nasa.gov/dataset/small-uas-flyover-acoustics-data) | 真实小型 UAS 飞越录音 | 飞越检测、距离变化、事件级评测 |
| Reuter 使用的 AuDroK、DroneNoise、SoundSnap、IDMT-TRAFFIC/ESC50 等 | 多数据集、跨域和背景 OOD | 会话独立、跨域泛化、低 FPR |

使用公开数据时，应保存原始文件哈希、采样率、通道数、机型/地点/日期元数据和许可信息。切分时优先按“录音会话或飞行批次”分组，而不是把同一段录音随机切成训练和测试。

## 五、评测时最容易被忽略的问题

### 1. 随机切片造成数据泄漏

同一段长录音切出的相邻 1 s 片段高度相似。若随机分层切分，测试片段可能只是在“记忆录音背景”，而不是识别旋翼声。最低要求是按录音会话、地点、日期和设备做 group split；更严格的设置是留一地点或留一机型测试。

### 2. 准确率不能替代低误报指标

固定监测点通常非无人机音频远多于无人机音频。应至少报告 ROC-AUC、PR-AUC、TPR@1% FPR、TPR@5% FPR、每小时误报数以及事件级召回率。Reuter 的实验提醒：片段级 FPR=1% 在连续监测中仍可能产生大量告警，因此必须对相邻阳性片段做事件聚合、最短持续时间约束和迟滞阈值。

### 3. “型号分类”“重量级分类”“个体认证”不是同一任务

Kümmritz 的 C0–C3 是重量类别，AUDRON 主要是 Bebop/Mambo 类型，DroneAudioID 进一步区分同型号个体。论文比较时需要明确标签层级，否则会出现“任务更简单但指标更高”的误判。

### 4. 类别不平衡与未知类

Noise/非无人机往往远多于每个无人机型号。除 class weight 或 focal loss 外，应单独设置未知机型和未知背景；报告 macro-F1、每类召回率、混淆矩阵和拒识率。

## 六、可复现性与开源情况

| 工作 | 代码/权重 | 复现判断 |
|---|---|---|
| Kümmritz 2024 | 有 GitHub MATLAB 代码；数据获取和本地 SQL 组织仍需处理 | **最适合直接复现**；先复现原始两级分类，再重做 session-disjoint 测试 |
| AUDRON 2025 | 论文给出分支结构和超参数，但未找到官方代码/权重 | **可按论文重写**；需自行构造数据、核对标签和处理类别不平衡 |
| Reuter 2026 | 训练流程、增强概率、跨域划分和指标描述较完整；公开页面未见官方代码/权重 | **可复现实验思想，难以逐点复现数值**；重点复现 SE-ResNet、AudioSet 预训练和 FPR 评测 |
| UaVirBASE 2025 | 数据和 GitLab 代码公开 | **适合阵列定位复现**，不直接提供细粒度机型分类基线 |
| DroneAudioSet 2025 | Hugging Face 数据公开（MIT） | **适合低 SNR 检测**；需自行定义分类标签和跨域划分 |
| ERAU YAMNet 2025 | 嵌入数据公开 | **适合快速做分类器原型**，不宜替代原始波形端到端实验 |

## 七、面向深度学习无人机识别的建议课题

### 推荐主线

建议把课题定义为：

> **面向跨场景和低误报约束的声学无人机检测与机型分类**

它兼顾了可复现性和研究创新，难度比直接做阵列三维定位可控，也比只在小数据集上追求准确率更有论文价值。

### 建议系统结构

1. **两级任务：** 一级二分类“无人机/非无人机”，二级在检测到无人机后进行型号分类；同时增加未知机型拒识分支。
2. **输入：** 16 kHz 单通道音频，1 s 和 2 s 两种窗口；Log-Mel 作为主输入，MFCC/STFT 作为消融输入。
3. **基线：** 轻量 CRNN（CNN 提取频谱纹理 + 双向 GRU/LSTM 建模时间变化）。
4. **增强版：** 在 CRNN 前加入 SE/残差块，或直接复现 18 层 SE-ResNet；加入变调、背景混音、麦克风频响、混响和 SpecAugment。
5. **预训练版：** 使用 AudioSet/YAMNet/PANNs 等音频嵌入初始化，再在无人机数据上微调；必须与从头训练模型在相同 group split 下比较。
6. **可选扩展：** 如果有阵列硬件，再增加 GCC-PHAT/波束形成通道和方位角多任务头；如果没有阵列，不建议为了“看起来复杂”强行加入定位模块。

### 最小实验矩阵

| 实验 | 目的 | 必报指标 |
|---|---|---|
| Log-Mel + CRNN | 可复现基线 | macro-F1、ROC-AUC、TPR@1%/5% FPR |
| CRNN + 物理/噪声增强 | 验证增强收益 | 同上 + 各 SNR/距离分层 |
| AudioSet 预训练 + 微调 | 验证迁移学习 | 同上 + 收敛速度、参数量 |
| 跨地点/跨设备/跨机型 | 验证泛化 | 留一域测试、每小时误报 |
| 已知型号 + 未知型号 | 验证开放集 | AUROC、FPR95、未知类拒识率 |
| 1 s 与 2 s 事件聚合 | 验证部署可用性 | 事件级召回、告警延迟、每小时误报 |

## 八、最终判断

- **想快速做出可运行系统：** 复现 Kümmritz 的 VGGish/两级分类，再换成 Log-Mel-CRNN。
- **想做有明显研究价值的深度学习论文：** 以 Reuter 的“预训练 + 数据增强 + 跨域低 FPR”作为主线，把 AUDRON 的多分支融合做成受控消融，而不是直接堆叠四个分支。
- **想做工程部署：** 选择阵列 + 波束形成/DOA + 轻量 CRNN，并把声学定位与类别识别分成多任务。
- **想做安全方向：** 在型号分类后增加 DroneAudioID 式声学指纹嵌入，用于个体认证和未知设备拒识。

## 参考文献与链接

1. [Kümmritz et al., Drones 2024, 8, 105](https://www.mdpi.com/2504-446X/8/3/105)
2. [Mięsikowska et al., Sensors 2024, 24, 5663](https://www.mdpi.com/1424-8220/24/17/5663)
3. [Sun et al., Multimedia Tools and Applications](https://doi.org/10.1007/s11042-023-17477-1)
4. [Sun et al., Sensors and Actuators A: Physical](https://doi.org/10.1016/j.sna.2024.115676)
5. [Drones Detection Using a Fusion of RF and Acoustic Features](https://www.mdpi.com/1424-8220/24/8/2427)
6. [Lei et al., ICUAS 2025](https://doi.org/10.1109/ICUAS65942.2025.11007784)
7. [Azizi et al., The Journal of Engineering](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/tje2.70135)
8. [Tegler et al., CVPRW Anti-UAV 2025](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Tegler_Detection_and_Localization_of_Drones_and_UAVs_Using_Sound_and_Vision_CVPRW_2025_paper.html)
9. [Li et al., DroneAudioID, IEEE TIFS](https://lynnlilu.github.io/publication/2025-01-journal-droneaudioid-tifs)
10. [Chatterjee et al., AUDRON](https://arxiv.org/abs/2512.20407)
11. [Jekateryńczuk et al., UaVirBASE](https://www.mdpi.com/2076-3417/15/10/5378)
12. [Gupta et al., DroneAudioSet](https://arxiv.org/abs/2510.15383)
13. [Wang et al., Multiclass Acoustic Dataset](https://arxiv.org/abs/2509.04715)
14. [ERAU YAMNet Embedding Dataset](https://datacommons.erau.edu/datasets/5dmcszvym4/3)
15. [Reuter et al., arXiv:2605.31329](https://arxiv.org/abs/2605.31329)
16. [Ghenescu et al., Sensors 2026, 26, 1778](https://www.mdpi.com/1424-8220/26/6/1778)
17. [Ensemble learning models for micro-drone detection, Discover AI 2026](https://link.springer.com/article/10.1007/s44163-026-00869-1)
18. [NASA Small UAS Flyover Acoustics Data](https://data.nasa.gov/dataset/small-uas-flyover-acoustics-data)

### 本次重点阅读的本地附件

- [Kümmritz 2024 PDF](</home/user1/.codex/attachments/a1ce04a8-d905-479e-96ee-adf45f9132e7/Kümmritz - 2024 - The Sound of Surveillance Enhancing Machine Learning-Driven Drone Detection with Advanced Acoustic.pdf>)
- [AUDRON 2025 PDF](</home/user1/.codex/attachments/dcbb9dae-238c-47d6-9fe1-a734d5fa5ecf/Chatterjee 等 - 2025 - AUDRON A Deep Learning Framework with Fused Acoustic Signatures for Drone Type Recognition.pdf>)
- [Reuter 2026 PDF](</home/user1/.codex/attachments/77f0fb39-aa08-408e-b346-d851d1e7822f/Reuter 等 - 2026 - Improving acoustic drone detection generalization through pretraining and data augmentation.pdf>)
