# 2024—2026 年较新声学无人机识别模型整理

**整理日期：** 2026-07-13  
**研究范围：** 2024—2026 年公开的无人机声学识别、检测、定位和通用音频表征模型。  
**说明：** “较新”按论文公开时间判断；部分模型是无人机专用模型，部分是可迁移到无人机音频的通用音频模型。

## 一、快速结论

当前最值得关注的方向不是继续加深普通 CNN，而是：

1. **Mamba/状态空间模型：** Audio Mamba、TAME、LTFM；
2. **预训练音频编码器：** BEATs、OpenBEATs、AudioMAE；
3. **音频 Transformer：** AST、HTS-AT、PaSST；
4. **多模态自监督：** AV-DTEC；
5. **麦克风阵列图模型：** MicGraphNet、GNN；
6. **开放集和小样本识别：** CLAP、度量学习、原型网络和能量模型；
7. **自监督音频预训练：** AudioMAE、OpenBEATs、STAR-MAE。

如果数据量只有几千个 1 秒片段，推荐优先使用：

> **BEATs/OpenBEATs 特征提取 + 轻量分类头**

如果希望增加论文创新性，可以研究：

> **Log-Mel + Audio Mamba + 跨场景泛化/未知机型拒识**

## 二、2024—2026 年较新模型时间线

| 年份 | 模型/工作 | 核心思想 | 与无人机任务的关系 | 开源情况 |
|---|---|---|---|---|
| 2024 | Audio Mamba | 用选择性状态空间模型替代 Transformer 自注意力，建模长时序音频 | 通用音频分类，可用于无人机声学分类 | [论文](https://arxiv.org/abs/2406.03344) |
| 2024 | TAME | Temporal Audio-based Mamba，同时学习音频时间/频率特征和三维轨迹 | 无人机分类、检测、三维轨迹估计 | [论文](https://arxiv.org/abs/2412.13037)；[代码](https://github.com/AmazingDay1/TAME) |
| 2024 | AV-DTEC | Audio Mamba + Vision Mamba + 自监督音视融合 | 无人机识别、定位和多模态威胁检测 | [论文](https://arxiv.org/abs/2412.16928)；[代码](https://github.com/AmazingDay1/AV-DETC) |
| 2025 | OpenBEATs | 进一步开放 BEATs 的训练代码、模型和多域预训练资源 | 小样本无人机分类、迁移学习 | [论文](https://arxiv.org/abs/2507.14129)；[项目页](https://shikhar-s.github.io/OpenBEATs) |
| 2025 | LTFM | Light Temporal-Frequency Mamba、多尺度时频融合、知识蒸馏 | 轻量级无人机分类和三维轨迹预测 | [论文](https://doi.org/10.1016/j.phycom.2025.102897) |
| 2025 | PEFT + 预训练模型 | 参数高效微调、定向数据增强和预训练模型组合 | 面向小数据 UAV 音频分类 | [论文](https://arxiv.org/abs/2506.11049) |
| 2026 | MicGraphNet | 将麦克风阵列几何和 TDOA/互相关特征建模为图 | 多麦克风声源定位、阵列检测 | [论文](https://www.sciencedirect.com/science/article/pii/S0263224125029112) |
| 2026 | STAR-MAE | 时空音频掩码自编码器和分布感知重建 | 通用音频自监督预训练，可迁移到无人机识别 | [论文](https://doi.org/10.1016/j.patcog.2026.113133) |

## 三、无人机专用的新模型

### 1. TAME：Temporal Audio-based Mamba

TAME 是目前与无人机声学任务结合较直接的 Mamba 模型之一。它使用并行选择性状态空间模块同时提取：

- 音频时间变化；
- 频谱变化；
- 声音传播信息；
- 无人机类别信息；
- 三维运动轨迹信息。

它还使用残差交叉注意力，将频谱信息注入时间特征，形成分类和轨迹估计的多任务模型。

**适合：**

- 单麦克风或阵列音频；
- 分类与定位联合建模；
- 需要论文创新性的研究。

**注意：**

- 任务不只是机型分类，还需要轨迹标签；
- 如果只有少量分类数据，直接复现完整 TAME 可能过重；
- 更适合拆出 Audio Mamba 主干，先做分类，再增加定位分支。

### 2. LTFM：Light Temporal-Frequency Mamba

LTFM 是面向边缘部署的轻量模型，主要模块包括：

- 时间域和频率域多尺度特征融合；
- Mamba 时序建模；
- 分类与三维轨迹预测多任务输出；
- 教师模型到学生模型的知识蒸馏。

论文使用四通道 Mel 频谱，并在包含 Mavic 2、Mavic 3、Avata、Phantom 4 和 M300 的多模态数据上进行实验。论文摘要报告分类准确率超过 95%，模型规模和计算量比 TFMamba 降低超过 60%。

**适合：**

- 麦克风阵列；
- 边缘设备和实时部署；
- 分类、定位和模型压缩联合研究。

**注意：**

- 需要多通道音频和轨迹标注；
- 不能简单用于单麦克风数据而不改动输入层；
- 论文任务更偏检测、定位和轨迹预测，不是传统封闭集机型分类。

### 3. AV-DTEC：音视自监督 Mamba

AV-DTEC 使用：

- Audio Mamba；
- Vision Mamba；
- 残差交叉注意力；
- 教师—学生自监督学习；
- LiDAR 或视觉生成的伪标签。

其思路是训练时利用声学、视觉和 LiDAR 的同步信息，推理阶段尽量降低对视觉传感器的依赖。

**适合：**

- 声音 + 摄像头 + LiDAR 多模态系统；
- 昼夜变化和视觉遮挡场景；
- 轨迹估计、目标存在性判断和分类联合任务。

**不适合：**

- 只有单麦克风、没有视觉数据的项目；
- 只需要简单无人机型号分类的课题。

## 四、通用音频预训练模型

### 1. BEATs 和 OpenBEATs

BEATs 通过声学 Tokenizer 和掩码预测学习通用音频表示，OpenBEATs 则进一步开放了训练代码、预训练模型和多域数据资源。

推荐使用方式：

~~~text
无人机音频 → BEATs/OpenBEATs → 音频嵌入 → 线性层或 MLP 分类器
~~~

**优势：**

- 对少量标注数据更友好；
- 可以先冻结编码器，只训练分类头；
- 比从头训练 Transformer 更稳定；
- 可用于跨机型和跨环境迁移。

**推荐实验：**

1. 冻结全部编码器，只训练 MLP；
2. 解冻最后 1—2 个 Transformer block；
3. 使用 LoRA 或 Adapter 做参数高效微调；
4. 比较随机初始化和预训练初始化；
5. 使用跨录音会话测试检验泛化。

### 2. AudioMAE

AudioMAE 使用掩码频谱图重建进行自监督学习。它适合先利用大量无标签环境音频学习声学结构，再用少量无人机标签进行微调。

**适合：**

- 无标签无人机录音较多；
- 背景噪声复杂；
- 需要跨设备、跨地点泛化。

**推荐结构：**

~~~text
无标签音频 → AudioMAE 预训练
                    ↓
无人机标注音频 → 编码器微调 → 分类/检测头
~~~

### 3. STAR-MAE

STAR-MAE 是 2026 年较新的时空音频掩码自编码器，针对直接把音频频谱当作图像处理的问题，引入时空结构化掩码和分布感知重建损失。

目前它更适合作为通用音频预训练研究方向。无人机领域还需要自行验证：

- 预训练域与旋翼声域的差异；
- 对低 SNR 背景的迁移效果；
- 1 秒短音频是否足以发挥其优势；
- 是否需要无人机专用继续预训练。

## 五、音频 Transformer：成熟但仍然有效

这些模型不算最新，但仍然是非常重要的对比基线。

| 模型 | 核心机制 | 评价 |
|---|---|---|
| AST | 将频谱图划分为 Patch，使用纯自注意力 | 结构清晰，适合作为 Transformer 基线 |
| HTS-AT | 分层 Token 和 Token-Semantic 模块 | 同时支持分类和声音事件定位 |
| PaSST | Patchout 随机丢弃频谱 Patch | 降低 Transformer 训练和推理成本 |
| MAST | 多尺度、层次化的音频 Transformer | 适合较长音频和复杂时频结构 |
| Conformer | Transformer 全局注意力 + 局部卷积 | 适合同时建模旋翼局部纹理和长时序变化 |

参考论文：[AST](https://arxiv.org/abs/2104.01778)、[HTS-AT](https://arxiv.org/abs/2202.00874)、[PaSST](https://www.isca-archive.org/interspeech_2022/koutini22_interspeech.pdf)、[Conformer](https://arxiv.org/abs/2005.08100)。

### 在 Lei 论文上的改造方式

Lei 论文采用 1 秒、16 kHz、STFT 频谱图。若改用 AST/HTS-AT，需要注意：

1. 许多预训练音频 Transformer 使用较长的音频窗口；
2. Lei 的 128×55/56 输入尺寸与预训练模型的 Mel 频谱尺寸不同；
3. 需要调整 Mel 前端、输入长度和位置编码；
4. 直接把短 STFT 图送入预训练 Transformer，可能导致预训练权重利用不足。

因此，建议先将 Lei 的 STFT 输入改为 Log-Mel，再比较 AST、HTS-AT 和 BEATs。

## 六、麦克风阵列模型

### 1. GNN/Graph Transformer

当输入来自多个麦克风时，可以把每个麦克风作为图节点，把麦克风之间的距离、方向和互相关特征作为边。

输入可以表示为：

~~~text
节点：每个麦克风的频谱或时域特征
边：麦克风距离、TDOA、GCC-PHAT、阵列几何关系
输出：无人机类别、方位角、距离
~~~

Relation Network GNN 已被用于分布式麦克风声源定位，并验证了不同麦克风数量下的适应能力。[GNN 声源定位论文](https://arxiv.org/abs/2306.16081)

2026 年的 MicGraphNet 进一步将麦克风阵列几何和互相关特征统一建模，并直接回归连续 TDOA，目标是提高噪声、混响和阵列安装误差下的 DOA 精度。[MicGraphNet](https://www.sciencedirect.com/science/article/pii/S0263224125029112)

### 2. 适用边界

- 单麦克风：不建议使用 GNN；
- 规则阵列：可以使用 GNN 或 Graph Transformer；
- 分布式麦克风：GNN 的优势更明显；
- 需要型号分类和定位：使用图网络多任务头；
- 只需要机型分类：先比较多通道融合和单通道模型。

## 七、开放集和小样本模型

无人机实际部署时，测试集可能出现训练阶段没有见过的新机型。因此，可以在上述编码器后增加度量学习或开放集模块。

### 可选模型

- Siamese Network；
- Triplet Network；
- Prototypical Network；
- ArcFace/AM-Softmax；
- Deep SVDD；
- Energy-based Open-set Recognition；
- CLAP 音频—文本嵌入。

推荐结构：

~~~text
音频编码器 → 归一化声学嵌入
                    ├── 已知型号分类
                    ├── 原型距离匹配
                    └── 未知型号拒识
~~~

CLAP 能把音频和文字描述映射到共同向量空间，并支持零样本或少样本推理，但无人机声学属于专业非语音领域，使用时应进行无人机数据微调或将其作为辅助嵌入，而不建议完全依赖零样本结果。[CLAP](https://arxiv.org/abs/2206.04769)

## 八、模型选择建议

| 研究条件 | 首选模型 | 次选模型 | 不建议优先使用 |
|---|---|---|---|
| 单麦克风、数据较少 | BEATs/OpenBEATs + MLP | AudioMAE + MLP | 从头训练大型 AST |
| 单麦克风、希望有新意 | Audio Mamba | Conformer | 盲目堆叠多分支 |
| 麦克风阵列 | TAME/LTFM | GNN/Graph Transformer | 只拼接所有通道 |
| 需要实时边缘部署 | LTFM、轻量 Audio Mamba | PaSST-S、Tiny Conformer | 大型 AST/ViT |
| 声音 + 视频 | AV-DTEC | Cross-Attention Transformer | 只做简单特征拼接 |
| 未知机型/小样本 | BEATs + 度量学习 | CLAP + 原型分类 | 普通 Softmax 封闭集 |
| 无标签音频较多 | AudioMAE、OpenBEATs、STAR-MAE | BYOL-A/对比学习 | 只使用少量有标签数据从头训练 |

## 九、推荐实验路线

### 路线 A：小数据集、单麦克风

~~~text
Log-Mel → BEATs/OpenBEATs → MLP
                  ↓
        已知型号分类 + 未知类拒识
~~~

这是最稳妥、最容易得到可靠结果的路线。

### 路线 B：论文创新型

~~~text
Log-Mel → 时频双分支 Audio Mamba
                  ↓
           注意力/门控融合
                  ↓
       型号分类 + 开放集拒识
~~~

重点研究：

- 预训练与从头训练的差异；
- Mamba 与 Transformer 的计算量；
- 跨地点和跨设备泛化；
- 未知机型拒识；
- 1 秒和 2 秒窗口的影响。

### 路线 C：阵列定位型

~~~text
多通道音频 → GCC-PHAT/TDOA + Log-Mel
                    ↓
          MicGraphNet/GNN
                    ↓
       无人机分类 + DOA/距离估计
~~~

### 路线 D：边缘部署型

~~~text
TFMamba/Audio Mamba 教师模型
                  ↓ 知识蒸馏
              LTFM 学生模型
                  ↓
            Jetson/嵌入式设备
~~~

## 十、建议的实验对比表

| 实验组 | 输入 | 模型 | 目的 |
|---|---|---|---|
| Baseline 1 | Log-Mel | CNN | 与传统方法比较 |
| Baseline 2 | STFT | Lei 风格 CRNN | 复现已有论文 |
| Experiment 1 | Log-Mel | AST/HTS-AT | 验证 Transformer |
| Experiment 2 | 原始音频或 Log-Mel | BEATs/OpenBEATs + MLP | 验证预训练 |
| Experiment 3 | Log-Mel | Audio Mamba | 验证状态空间模型 |
| Experiment 4 | Log-Mel | Conformer | 验证局部—全局联合建模 |
| Experiment 5 | 多通道特征 | GNN/MicGraphNet | 验证阵列空间信息 |
| Experiment 6 | 任意最佳编码器 | ArcFace/ProtoNet | 验证未知机型拒识 |

必须统一报告：

- Macro-F1；
- Balanced Accuracy；
- ROC-AUC；
- PR-AUC；
- TPR@1% FPR；
- 跨录音会话准确率；
- 跨设备准确率；
- 未知机型拒识率；
- 参数量、FLOPs 和推理延迟。

## 十一、重要提醒

### 1. 新模型不一定优于简单模型

2025 年一项 UAV 音频研究在约 3100 个片段、31 种无人机上比较了预训练模型和参数高效微调策略，作者报告 EfficientNet-B0 在三种增强下达到 95.95% 验证准确率，并超过 AST。这说明数据划分、增强和微调方式有时比模型名称更重要。[相关研究](https://arxiv.org/abs/2506.11049)

### 2. 不要只做随机切片划分

应优先采用：

- 按录音会话划分；
- 按地点划分；
- 按录音设备划分；
- 留一机型测试；
- 未知背景噪声测试。

### 3. 1 秒片段的片段级准确率不等于部署性能

实际系统需要进行：

- 连续音频滑窗；
- 相邻阳性片段合并；
- 最短事件持续时间限制；
- 阈值迟滞；
- 每小时误报统计。

### 4. Mamba 和 Transformer 都需要足够数据或预训练

如果只有几千个短片段，不建议直接从头训练大规模 AST、ViT 或 Mamba。更合理的方式是：

1. 使用预训练编码器；
2. 冻结大部分参数；
3. 只训练分类头；
4. 再逐步解冻最后几层；
5. 最后进行跨域测试。

## 十二、最终推荐

### 最实用

**BEATs/OpenBEATs + MLP 分类头**

### 最有论文创新性

**Audio Mamba + 跨场景泛化 + 未知机型拒识**

### 最适合阵列系统

**TAME/LTFM + MicGraphNet/GNN**

### 最适合多模态系统

**AV-DTEC 或音频—视觉 Cross-Attention Transformer**

### 最适合当前 CRNN 基础

建议保留 Lei 论文中的 CNN-STFT 和 CRNN-STFT 作为基线，然后增加：

~~~text
CNN-STFT → AST → BEATs → Audio Mamba → Conformer
~~~

最终重点比较的不是单一随机测试集准确率，而是：

> **跨场景性能、低误报率、未知机型识别能力和边缘推理速度。**

## 参考文献

1. [Audio Mamba: Bidirectional State Space Model for Audio Representation Learning](https://arxiv.org/abs/2406.03344)
2. [TAME: Temporal Audio-based Mamba for Enhanced Drone Trajectory Estimation and Classification](https://arxiv.org/abs/2412.13037)
3. [AV-DTEC: Self-Supervised Audio-Visual Fusion for Drone Trajectory Estimation and Classification](https://arxiv.org/abs/2412.16928)
4. [OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder](https://arxiv.org/abs/2507.14129)
5. [LTFM: A lightweight model for UAV classification and trajectory prediction](https://doi.org/10.1016/j.phycom.2025.102897)
6. [15,500 Seconds: Lean UAV Classification Using EfficientNet and Lightweight Fine-Tuning](https://arxiv.org/abs/2506.11049)
7. [MicGraphNet: Microphone graph network for sound source localization](https://www.sciencedirect.com/science/article/pii/S0263224125029112)
8. [Masked autoencoders for spatio-temporal audio representations: STAR-MAE](https://doi.org/10.1016/j.patcog.2026.113133)
9. [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)
10. [HTS-AT: A Hierarchical Token-Semantic Audio Transformer](https://arxiv.org/abs/2202.00874)
11. [PaSST: Efficient Training of Audio Transformers with Patchout](https://www.isca-archive.org/interspeech_2022/koutini22_interspeech.pdf)
12. [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058)
13. [AudioMAE: Masked Autoencoders that Listen](https://github.com/facebookresearch/AudioMAE)
14. [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
15. [CLAP: Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769)
16. [Graph neural networks for sound source localization on distributed microphone networks](https://arxiv.org/abs/2306.16081)
