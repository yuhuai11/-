# 声学无人机检测与开放集型号识别系统

本工程是一个基于深度学习的声学无人机检测与开放集型号识别实验系统。当前正式检测器
为G7 PANNs Cnn14_16k，G18在冻结G7表征上进一步完成九个已知型号的录音级分类，并
实验性地拒识未知型号。

## 项目名称与兼容标识

当前正式展示名称为：

```text
声学无人机检测与开放集型号识别系统
Acoustic UAV Detection and Open-Set Model Identification
```

`ABDDV-CRNN`是工程早期复现论文
`Audio-Based Drone Detection Via CRNN: An Investigation into Threshold Sensitivity and Stability`
时形成的历史仓库名称。它继续作为以下兼容标识保留：

- 现有仓库目录 `/home/user1/JJZ/ABDDV-CRNN`；
- Python包导入名 `dads_crnn`；
- 历史脚本路径、配置、检查点及审计产物中的冻结字符串。

保留这些技术标识不表示当前正式模型仍是CRNN。早期CRNN现在只用于论文复现、历史
对照和消融实验，不代表当前正式检测器。新文档、论文和界面应优先使用当前正式展示
名称；只有在说明仓库路径、Python入口或历史实验时才使用`ABDDV-CRNN`。

Python发行包的当前名称为`acoustic-uav-open-set-identification`；导入路径仍为
`dads_crnn`，因此现有的`python -m dads_crnn.<module>`命令不受影响。

## DADS历史测试口径与精确内容泄漏修复

历史DADS清单生成程序以Parquet行（`AudioRef`）为分配单元，同一行展开的片段不会
跨split。在已审计的`dads_all_seed42`及相关子清单中，`source_path`跨split交集为0，
但这是后验结果，不是程序的显式分组规则。原始WAV审计共发现15个精确重复SHA组、涉及
30条记录，去重时移除15条重复记录；其中6组的成员分布在至少两个split。历史Test又已
被多轮候选门禁使用，因此旧表中的DADS Test统一解释为“历史DADS同数据源已消费内部
回归集”，不能再称为独立测试，也不能用于证明跨会话、设备、场景或无人机型号泛化。

需要区分两个名称相近但作用不同的版本：`dads_dedup_v2`只对原始WAV做精确哈希去重，
保留历史1秒循环补齐缓存和原split；新的
`dads_native_half_second_content_component_v2`则从原始WAV重新构造原生0.5秒
输入，并按原始音频及最终模型输入哈希形成不可拆分组件。后者修复的是已审计到的精确
内容交叉和类别相关循环补齐，不表示已经排除近重复、会话或设备层面的全部泄漏。过程、
数据统计、拒绝实验和新结果统一记录在
[DADS历史测试口径更正与G7精确内容泄漏修复重训报告](docs/current/binary_g7/DADS历史测试口径更正与G7精确内容泄漏修复重训报告.md)。
原生0.5秒协议的seed 42重训已经完成：固定阈值0.50下，Val F1为`0.990233`，
已消费内部开发Test F1为`0.988123`，最佳epoch为7。新检查点位于
`artifacts/g7_leakage_fixed_v2/runs/seed_42/best.pt`。它是当前DADS协议修复后的
内部基线。随后seed 42/43/44严格重训、外部Benchmark复核和1秒录音级聚合均已完成，
该严格版本现已成为二分类正式主线；外部集合因已被多轮查看，仍只能称为开发Benchmark。

## 当前工程状态

当前正式二分类模型已经从原始CRNN基线发展为：

```text
G7严格版PANNs Cnn14_16k AudioSet预训练模型（seed 42/43/44）
```

正式检查点：

```text
artifacts/g7_strict_retrain_v1/runs/seed_{42,43,44}/best.pt
```

部署首选seed42，并使用原生0.5秒输入、连续两个片段概率的非重叠1秒均值和固定阈值0.5。
完整独立交付目录为`releases/g7_binary_best_strict_v1/`。旧路径
`artifacts_g7_panns_pt/`保留为历史G7预训练基线，不再代表当前正式版本。

当前文档入口见[文档索引](docs/README.md)。后续开发参考：

- [G7二分类完整分析](docs/current/binary_g7/G7二分类模型原理创新实验结果与改进分析.md)
- [G7内外部结果与论文对比](docs/current/binary_g7/G7二分类内外部测试结果与三篇论文对比分析.md)
- [G7严格重训练实验记录](docs/current/binary_g7/G7严格重训练实验记录.md)
- [G7严格外部基线与录音级聚合](docs/current/binary_g7/G7严格重训外部基线与1秒聚合实验报告.md)
- [G7-R9后续优化方案与执行记录](docs/current/binary_g7/G7-R9来源类别配额跨域优化实验方案与执行记录.md)
- [历史工程整理说明](docs/archive/general/工程目录整理与历史模型归档说明.md)

G17-P0高频可行性审计已通过；正式结果位于
`artifacts/g17_dual_rate/p0_audit/decision.json`。当前只允许进入P1结构与训练路径
预检，G7仍是正式基线模型。

G18在G7之后增加开放集机型识别，已经完成P0注册、P1预检、P2/P4型号头训练、
P3系列Unknown方法比较、P5多种子复现和P6一次性最终Holdout。最终结果表明，九个
已知型号具有较好的录音级分类可行性，但X6D/Y6未知型号拒识泛化失败，因此当前不能
视为已经具备可靠的通用开放集或跨设备型号识别能力。

完整方法与最终结果见：

- [G18方法原理与完整流程说明](docs/archive/model_identification_g18_g22/G18方法原理与完整流程说明.md)
- [G18实验结果综合汇总报告](docs/archive/model_identification_g18_g22/G18实验结果综合汇总报告.md)

G19开发分支已经完成监督对比Embedding、门控注意力录音聚合和类别条件
PCA-OAS Mahalanobis边界实验；Known Tune录音级Accuracy为0.9560、Macro-F1为
0.9478、最低类别Recall为0.8571。G20多头注意力统计池化也已经完成，但门槛未通过：
Accuracy为0.9451、Macro-F1为0.9236、最低类别Recall为0.5714。G20注意力头平均
相似度0.9939且标准化熵0.9950，说明增加注意力复杂度没有产生有效的片段分工。

G21因此保留G18冠军分类流程，不再增加注意力头；它只在G7的私有型号识别副本中以
低学习率解冻PANNs `fc1`，并用L2-SP约束参数偏移。G21 seed 42已经完成，Known
Tune的Accuracy为0.9780、Macro-F1为0.9797、最低类别Recall为0.8571，与G18 seed
42完全相同。因此G21属于“非劣但未提升”的消融结果，不能替换G18冠军。训练后期指标
波动明显，当前不扩大解冻范围。G18 P6中的X6D/Y6已经消费，G19—G21均不会将其重新
用作最终测试。

G22增加61维旋翼谐波片段特征，并以预注册固定权重0.25与G18录音级类别分数后融合。
在seed 42/43/44上，Known Tune平均Accuracy由0.9670提高到0.9780，平均Macro-F1由
0.9669提高到0.9805，最低类别Recall由0.8571提高到0.9286，三个预注册门槛均通过。
G22是当前开发集冠军候选，但Known Tune已经参与评价，不能把该结果替代已消费的G18
最终Holdout结论；正式确认需要未来新增的录音级Holdout。

历史CRNN、ResNet10-CBAM和G2产物已经集中到：

```text
archive/historical_models/
```

历史模型状态和当前路径见
[历史模型归档索引](archive/historical_models/README.md)。归档模型只用于复现和对照，
不得误认为当前正式模型。

未晋级的G3、G4、G5、G6、G7-Scratch和G9模型产物已于2026-08-01按用户要求永久
删除，释放空间约4.8GB；其配置、源码、文档、历史日志及删除前摘要哈希继续保留，详见
[未晋级模型删除记录](archive/historical_models/failed_candidates/DELETION_RECORD.md)。

## 数据版本封存与使用边界

工程数据已经按实验用途集中登记到
[数据版本封存中心](archive/data_versions/README.md)。为保持配置、脚本、审计哈希和历史
结果可复现，数据没有物理搬迁；各原目录增加`DATA_STATUS.md`，中央登记表记录实际路径、
状态、允许用途、禁止用途和消费历史。

当前分为四类：历史内部数据、当前修正内部基线、已使用外部数据和未来独立最终测试集。
其中G7修正后的0.5秒DADS数据可用于内部训练/验证/测试；Val/OOD、G11、G13、G14、
IDMT Traffic以及G18-G22 Kielce数据均已参与开发或评估，不再视为全新的独立最终测试
集。`data/future_final_holdout/`当前为空，只有完成来源登记、跨库去重、划分审计以及模型/
阈值冻结后才能接收未来最终测试数据。

封存校验命令：

```bash
bash scripts/verify_data_archive.sh
```

## 历史CRNN论文复现设置（非当前正式G7）

以下设置和命令用于复现工程起点的CRNN论文实验，不用于训练当前正式G7或G18。

- 数据集：DADS，16 kHz、16-bit、mono WAV。
- 抽样：Drone 5000 个原始音频文件，No-Drone 5000 个原始音频文件，保持原始文件级类别均衡。
- 划分：历史复现流程先把Parquet行（`AudioRef`）按70% train、15% validation、
  15% test分配，再在各split内分段；同一行的片段不跨split。`source_path`交集为0
  只是该清单的后验检查结果，程序没有把它作为显式分组键，也没有阻止内容相同但标识
  不同的录音跨split，因此该流程只能作为历史内部回归协议。
- 预处理：损坏文件和完全静音文件删除；超过 1 秒的音频切成非重叠 1 秒片段；短于 1 秒的音频循环补齐；峰值归一化；转 Log-Mel 频谱。
- 模型：CRNN，CNN 提取时频特征，GRU 建模时间依赖，最后输出 drone 概率。
- 训练：Adam，binary cross entropy，batch size 32，learning rate 0.001，最多 150 epoch，early stopping patience 10。
- 稳定性：seed 42、43、44。
- 阈值：0.40、0.50、0.65，对比 Accuracy、Precision、Recall、F1、AUC。

## 环境

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

当前代码不依赖 `torchaudio` 或 `librosa`，Mel 频谱在 PyTorch 中直接计算，减少版本匹配问题。

## 生成历史CRNN复现manifest

数据集已位于 `date/DADS/data`。下面命令会先跳过损坏/完全静音文件，再固定随机抽样每类 5000 个原始音频文件，并按原始文件划分 train/val/test。之后只在各自 split 内把长音频切成 1 秒片段，短音频和长音频最后不足 1 秒的尾段会循环补齐，并把生成的 1 秒片段缓存为 `.npy`。

```powershell
python -m dads_crnn.prepare_manifest --config configs/crnn_dads.yaml --extract-audio
```

调试时可先小样本运行：

```powershell
python -m dads_crnn.prepare_manifest --config configs/crnn_dads.yaml --per-class 20 --extract-audio
```

## 训练历史CRNN

```powershell
python -m dads_crnn.train --config configs/crnn_dads.yaml
```

只跑一个随机种子：

```powershell
python -m dads_crnn.train --config configs/crnn_dads.yaml --seeds 42
```

训练产物会写入 `artifacts/runs/seed_<seed>`：

- `best.pt`：验证集 F1 最优模型。
- `history.csv`：每个 epoch 的训练/验证指标。
- `metrics.json`：验证集和测试集在三个阈值下的 segment-level 与 file-level 指标，以及训练耗时。
- `val_probabilities.npy` / `val_labels.npy`：验证集阈值复算用输出。
- `test_probabilities.npy` / `test_labels.npy`：测试集阈值复算用输出。

## 复算历史CRNN阈值指标

```powershell
python -m dads_crnn.evaluate_thresholds --config configs/crnn_dads.yaml --run-dir artifacts/runs/seed_42
```

## 计算历史CRNN原始文件级指标

训练默认按 1 秒 segment 计算指标。若要把同一个原始音频文件的所有 segment 聚合成一个预测，可运行：

```powershell
python -m dads_crnn.evaluate_file_level `
  --config configs/crnn_dads.yaml `
  --manifest artifacts\manifests\dads_balanced_5000_seed42.csv `
  --run-dir artifacts\runs\seed_43 `
  --splits val test
```

脚本会同时输出 `mean` 和 `max` 两种聚合方式。

## 汇总全部实验

将所有已有运行的 segment-level、file-level、阈值和跨随机种子统计统一输出到
`artifacts/comparison/实验结果汇总.md`：

```powershell
python -m dads_crnn.summarize_experiments
```

## 外部泛化测试

使用服务器上的 Unseen 与 Real-world 数据冻结评估现有 CRNN 和 ResNet10-CBAM，详见
[外部泛化测试方案](docs/archive/general/外部泛化测试方案.md)。正式测试包括数据清单与重复审计、逐样本预测、
分组失效分析、bootstrap 置信区间和三随机种子汇总。

外部测试完成后的改进路线及独立OOD验证规范见
[泛化能力改进方案](docs/archive/general/泛化能力改进方案.md)。

如果尚未执行 `pip install -e .`，可在项目根目录运行：

```powershell
$env:PYTHONPATH="src"
python -m dads_crnn.summarize_experiments
```

## 与论文的可复现差异

论文没有给出完整 CNN 层数、卷积通道数、Mel bins、FFT window/hop 等细节；本工程选择了常见的 64-bin Log-Mel、25 ms window、10 ms hop，以及 3 层 CNN + 双向 GRU。训练协议、数据规模、划分比例、随机种子、阈值实验和公开描述的音频预处理流程按论文复现。

原始论文PDF与文献综述统一位于：

```text
docs/references/papers/
docs/references/reviews/
```
