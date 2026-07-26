# Audio-Based Drone Detection Via CRNN 复现

本工程复现论文 `Audio-Based Drone Detection Via CRNN: An Investigation into Threshold Sensitivity and Stability` 在 DADS 数据集上的二分类实验。

## 当前工程状态

当前正式模型已经从原始CRNN基线发展为：

```text
G7 PANNs Cnn14_16k AudioSet预训练模型（seed 42）
```

正式检查点：

```text
artifacts_g7_panns_pt/runs/seed_42/best.pt
```

后续开发参考：

- [G7模型结果与性能分析报告](docs/G7模型结果与性能分析报告.md)
- [G17双采样率互补表征与安全后融合方案](docs/G17双采样率互补表征与安全后融合方案.md)
- [G18基于G7的分层开放集无人机型号识别方案](docs/G18基于G7的分层开放集无人机型号识别方案.md)
- [工程目录整理与历史模型归档说明](docs/工程目录整理与历史模型归档说明.md)

G17-P0高频可行性审计已通过；正式结果位于
`artifacts/g17_dual_rate/p0_audit/decision.json`。当前只允许进入P1结构与训练路径
预检，G7仍是正式基线模型。

G18在G7之后增加开放集机型识别，P0录音级隔离注册表已生成。该功能仍处于开发预检
阶段，不能视为已经具备真实跨设备型号识别能力。

G18-P1冻结G7训练路径已经在RTX 3080通过，P2 seed 42机型头可行性训练入口为
`scripts/run_g18_p2_seed42.sh`。

历史CRNN、ResNet10-CBAM、G2–G6、G7 Scratch和G9产物已经集中到：

```text
archive/historical_models/
```

历史模型状态和当前路径见
[历史模型归档索引](archive/historical_models/README.md)。归档模型只用于复现和对照，
不得误认为当前正式模型。

## 论文设置

- 数据集：DADS，16 kHz、16-bit、mono WAV。
- 抽样：Drone 5000 个原始音频文件，No-Drone 5000 个原始音频文件，保持原始文件级类别均衡。
- 划分：先按原始文件做 70% train，15% validation，15% test；再在各 split 内分段，避免同一原始文件的片段跨 split。
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

## 生成复现 manifest

数据集已位于 `date/DADS/data`。下面命令会先跳过损坏/完全静音文件，再固定随机抽样每类 5000 个原始音频文件，并按原始文件划分 train/val/test。之后只在各自 split 内把长音频切成 1 秒片段，短音频和长音频最后不足 1 秒的尾段会循环补齐，并把生成的 1 秒片段缓存为 `.npy`。

```powershell
python -m dads_crnn.prepare_manifest --config configs/crnn_dads.yaml --extract-audio
```

调试时可先小样本运行：

```powershell
python -m dads_crnn.prepare_manifest --config configs/crnn_dads.yaml --per-class 20 --extract-audio
```

## 训练

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

## 复算阈值指标

```powershell
python -m dads_crnn.evaluate_thresholds --config configs/crnn_dads.yaml --run-dir artifacts/runs/seed_42
```

## 计算原始文件级指标

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
[外部泛化测试方案](docs/外部泛化测试方案.md)。正式测试包括数据清单与重复审计、逐样本预测、
分组失效分析、bootstrap 置信区间和三随机种子汇总。

外部测试完成后的改进路线及独立OOD验证规范见
[泛化能力改进方案](docs/泛化能力改进方案.md)。

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
