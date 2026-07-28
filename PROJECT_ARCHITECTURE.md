# 声学无人机检测与开放集型号识别系统：工程文件架构说明

本文档说明“声学无人机检测与开放集型号识别系统”的目录结构、代码模块职责、数据流
和常用运行命令。工程起点是复现论文
**Audio-Based Drone Detection Via CRNN: An Investigation into Threshold Sensitivity and Stability**
中的DADS音频无人机检测实验，当前正式检测器已经演进为G7 PANNs Cnn14_16k，并在
G18中增加冻结G7表征上的开放集型号识别实验。

`ABDDV-CRNN`仍是历史仓库目录和兼容标识，不再作为当前模型结构的简称。现有目录、
Python包`dads_crnn`、脚本路径以及冻结产物中的名称保持不变，避免破坏历史实验和
可复现身份。

## 1. 工程总览

```text
ABDDV-CRNN/                    历史仓库目录名，兼容保留
├─ README.md
├─ PROJECT_ARCHITECTURE.md
├─ requirements.txt
├─ environment.yml
├─ pyproject.toml
├─ .gitignore
├─ archive/
│  └─ historical_models/        历史基线、未晋级候选及日志
├─ configs/
├─ src/
├─ date/
├─ artifacts/
├─ artifacts_full/
├─ artifacts_g7_panns_pt/       当前正式G7
├─ docs/
│  └─ references/               论文PDF与文献综述
├─ scripts/
└─ tests/
```

工程整体可以分为五部分：

| 部分 | 作用 |
|---|---|
| `date/` | 原始 DADS 数据集，包含 parquet 格式音频数据 |
| `configs/` | 实验配置，包括数据、特征、模型、训练和评估参数 |
| `src/dads_crnn/` | 复现代码主体，包括数据处理、模型、训练和评估 |
| `artifacts/` | 运行生成物，包括 manifest、音频缓存、模型和指标 |
| `artifacts_g7_panns_pt/` | 当前正式G7检查点、指标和预测 |
| `archive/historical_models/` | 历史基线、已替代冠军、未晋级候选和旧日志 |
| `docs/` | 当前方案、结果、归档说明和文献资料 |
| 根目录文件 | 环境配置和项目入口说明 |

当前目录状态及历史路径映射以
`docs/工程目录整理与历史模型归档说明.md`为准。本文件后续章节仍保留原始CRNN复现
架构说明，便于理解工程起点。

当前开发方向为G17双采样率互补分支。P0只读高频审计的入口与结果为：

```text
configs/g17_p0_high_rate.yaml
scripts/run_g17_p0_audit.sh
src/dads_crnn/audit_g17_high_rate.py
artifacts/g17_dual_rate/p0_audit/
```

P0不训练模型，也不替代当前正式G7检查点。

分层开放集型号识别入口：

```text
configs/g18_model_id_registry.yaml
configs/g18_model_id_preflight.yaml
scripts/run_g18_p0_registry.sh
scripts/run_g18_p1_preflight.sh
src/dads_crnn/model_identification.py
artifacts/g18_model_identification/
```

G18位于G7检测门之后；G7仍保持冻结并负责背景/无人机检测。G18已经完成P0至P6
实验链路和一次性最终Holdout。当前结论是九个Known型号的录音级分类具有可行性，
但X6D/Y6的Unknown拒识未能泛化，不能视为已经具备可靠的通用开放集识别能力。

## 2. 原始数据目录

```text
date/
└─ DADS/
   ├─ README.md
   └─ data/
      ├─ train-00000-of-00039.parquet
      ├─ train-00001-of-00039.parquet
      ├─ ...
      └─ train-00038-of-00039.parquet
- no-drone：5000 个原始音频文件
- drone：5000 个原始音频文件
- 总计：10000 条
- 划分比例：70% train，15% validation，15% test

原始数据目录只读使用，训练代码不会修改这里的 parquet 文件。

## 3. 配置文件目录

```text
configs/
└─ crnn_dads.yaml
```

`configs/crnn_dads.yaml` 是复现实验的主配置文件，控制完整实验流程。

主要配置包括：

```yaml
data:
  parquet_dir: date/DADS/data
  manifest_dir: artifacts/manifests
  audio_cache_dir: artifacts/audio_cache
  sample_rate: 16000
  clip_seconds: 1.0
  per_class: 5000
  manifest_seed: 42
  splits:
    train: 0.70
    val: 0.15
    test: 0.15
```

这部分定义数据来源、音频长度、采样率、类别抽样数量和数据集划分比例。

```yaml
features:
  n_mels: 64
  n_fft: 512
  win_length: 400
  hop_length: 160
```

这部分定义 Log-Mel Spectrogram 特征参数。

```yaml
train:
  seeds: [42, 43, 44]
  batch_size: 32
  epochs: 150
  learning_rate: 0.001
  patience: 10
```

这部分对应论文训练设置：三组随机种子、batch size 32、学习率 0.001、最多 150 epoch、early stopping patience 10。

```yaml
eval:
  thresholds: [0.40, 0.50, 0.65]
```

这部分用于复现论文中的阈值敏感性实验。

## 4. 源码目录

```text
src/
└─ dads_crnn/
   ├─ __init__.py
   ├─ audio.py
   ├─ config.py
   ├─ dataset.py
   ├─ evaluate_thresholds.py
   ├─ features.py
   ├─ metrics.py
   ├─ model.py
   ├─ prepare_manifest.py
   └─ train.py
```

### 4.1 `prepare_manifest.py`

作用：从原始 DADS parquet 文件中抽样并生成训练清单。

主要功能：

- 扫描 `date/DADS/data/*.parquet`
- 按类别收集可用原始音频索引
- 固定随机种子抽样每类 5000 个原始音频文件
- 先按原始文件做 70/15/15 划分 train/val/test
- 再在每个 split 内展开为 1 秒 segment，避免同一原始文件的片段跨 split
- 生成 manifest CSV
- 可选地把音频提取为 1 秒 `.npy` 缓存

输出示例：

```text
artifacts/manifests/dads_balanced_5000_seed42.csv
artifacts/audio_cache/train/0/*.npy
artifacts/audio_cache/train/1/*.npy
artifacts/audio_cache/val/0/*.npy
artifacts/audio_cache/test/1/*.npy
```

### 4.2 `audio.py`

作用：音频底层预处理。

主要功能：

- 从 WAV bytes 解码音频
- 转为 mono float32
- 重采样到 16 kHz
- 长音频切分为非重叠 1 秒片段，最后不足 1 秒的尾段保留并循环补齐
- 短音频循环补齐到 1 秒
- 峰值归一化

对应论文中的预处理步骤：

```text
WAV mono 16 kHz
→ 1 秒分段
→ 长音频分段，尾段循环补齐
→ 短音频循环补齐
→ 归一化
```

### 4.3 `dataset.py`

作用：定义 PyTorch Dataset。

训练时会根据 manifest 加载样本。优先读取已经缓存好的 `.npy` 文件：

```text
artifacts/audio_cache/
```

如果缓存不存在，则回退到 parquet 文件中读取原始 WAV bytes。

### 4.4 `features.py`

作用：把 waveform 转换为 Log-Mel Spectrogram。

处理流程：

```text
1 秒 waveform
→ STFT
→ Mel filterbank
→ log 压缩
→ 标准化
→ shape: [batch, 1, n_mels, frames]
```

当前实现不依赖 `torchaudio` 或 `librosa`，而是直接使用 PyTorch 的 `torch.stft` 计算频谱。

### 4.5 `model.py`

作用：定义 CRNN 模型。

模型结构：

```text
Log-Mel Spectrogram
→ CNN blocks
→ GRU
→ Fully Connected layers
→ Sigmoid probability
```

其中：

- CNN 用于提取时频局部特征
- GRU 用于建模时间序列依赖
- 最终输出一个 logit，表示 drone 类概率

### 4.6 `train.py`

作用：训练入口。

主要流程：

```text
读取配置
→ 读取 manifest
→ 构建 train/val/test Dataset
→ 构建 Log-Mel 特征模块
→ 构建 CRNN 模型
→ 训练模型
→ 验证集 early stopping
→ 保存 best.pt
→ 在测试集上评估多个阈值
→ 输出 metrics.json 和 summary.json
```

训练结果保存在：

```text
artifacts/runs/seed_42/
artifacts/runs/seed_43/
artifacts/runs/seed_44/
```

### 4.7 `metrics.py`

作用：计算二分类评估指标。

包括：

- Accuracy
- Precision
- Recall
- F1-score
- AUC

### 4.8 `evaluate_thresholds.py`

作用：在不重新训练的情况下，基于保存的预测概率重新计算不同阈值下的指标。

输入：

```text
test_probabilities.npy
test_labels.npy
```

输出：不同 threshold 下的分类指标。

## 5. 运行产物目录

```text
artifacts/
├─ manifests/
├─ audio_cache/
└─ runs/
```

### 5.1 `artifacts/manifests/`

保存数据划分清单。

当前重要文件：

```text
dads_balanced_5000_seed42.csv
```

这是正式复现实验使用的 manifest。

如果存在：

```text
dads_balanced_20_seed42.csv
```

这是之前用于 smoke test 的小样本 manifest，不应用于正式结果。

### 5.2 `artifacts/audio_cache/`

保存从 parquet 中提取并预处理后的 1 秒音频缓存。

结构：

```text
artifacts/audio_cache/
├─ train/
│  ├─ 0/
│  └─ 1/
├─ val/
│  ├─ 0/
│  └─ 1/
└─ test/
   ├─ 0/
   └─ 1/
```

其中：

- `0`：no-drone
- `1`：drone

缓存文件格式为 `.npy`，每个文件是一段 1 秒、16 kHz 的 waveform。

### 5.3 `artifacts/runs/`

保存训练结果。

每个随机种子对应一个目录：

```text
artifacts/runs/seed_42/
├─ best.pt
├─ history.csv
├─ metrics.json
├─ test_labels.npy
└─ test_probabilities.npy
```

文件说明：

| 文件 | 作用 |
|---|---|
| `best.pt` | 验证集 F1 最优模型权重 |
| `history.csv` | 每个 epoch 的训练 loss、验证 loss、验证指标 |
| `metrics.json` | 测试集在不同阈值下的指标 |
| `test_labels.npy` | 测试集真实标签 |
| `test_probabilities.npy` | 测试集预测概率 |

`artifacts/runs/summary.json` 汇总多个随机种子的最终结果。

## 6. 环境和项目文件

### 6.1 `requirements.txt`

pip 依赖文件，用于安装 Python 包：

```powershell
python -m pip install -r requirements.txt
```

### 6.2 `environment.yml`

Conda 环境定义文件，用于创建 `dads-crnn` 环境：

```powershell
conda env create -f environment.yml
```

### 6.3 `pyproject.toml`

Python 项目配置文件，使项目可以用 editable 模式安装：

```powershell
python -m pip install -e .
```

安装后可以直接运行：

```powershell
python -m dads_crnn.train
```

### 6.4 `.gitignore`

忽略运行产物和缓存，例如：

```text
artifacts/
__pycache__/
*.pyc
```

`artifacts/` 中包含大量缓存和模型文件，不适合纳入版本管理。

## 7. 完整数据流

整个复现流程如下：

```text
DADS parquet 原始数据
→ prepare_manifest.py
→ balanced manifest CSV
→ audio_cache 1 秒 waveform 缓存
→ DADSDataset
→ Log-Mel Spectrogram
→ CRNN 模型
→ 训练与验证
→ best.pt
→ 测试集预测概率
→ 阈值评估
→ metrics.json / summary.json
```

对应命令流程：

```powershell
python -m dads_crnn.prepare_manifest --config configs/crnn_dads.yaml --extract-audio

python -m dads_crnn.train `
  --config configs/crnn_dads.yaml `
  --manifest artifacts\manifests\dads_balanced_5000_seed42.csv `
  --seeds 42
```

如果要完整复现论文的三个随机种子：

```powershell
python -m dads_crnn.train `
  --config configs/crnn_dads.yaml `
  --manifest artifacts\manifests\dads_balanced_5000_seed42.csv
```

## 8. 正式训练注意事项

1. 正式训练必须指定完整 manifest：

```powershell
--manifest artifacts\manifests\dads_balanced_5000_seed42.csv
```

否则可能误用小样本文件 `dads_balanced_20_seed42.csv`。

2. 当前训练如果显示：

```text
device: cpu
```

说明使用 CPU 训练。全量训练会比较慢。如果有 NVIDIA GPU，应安装 CUDA 版 PyTorch。

3. `best_epoch` 不是总 epoch 数，而是验证集 F1 最优模型所在的 epoch。实际训练了多少 epoch 应查看：

```powershell
Get-Content artifacts\runs\seed_42\history.csv
```

4. `artifacts/audio_cache/` 可以删除后重新生成，但删除后下一次训练前需要重新运行 manifest 提取缓存命令。

## 9. 与论文设置的对应关系

| 论文设置 | 本工程实现 |
|---|---|
| DADS 数据集 | `date/DADS/data/*.parquet` |
| Drone/No-drone 各 5000 个原始音频文件 | `prepare_manifest.py` 均衡抽样并按原始文件划分 |
| 1 秒音频片段 | `prepare_manifest.py` 分段，`audio.py` 循环补齐 |
| Mel-spectrogram | `features.py` |
| CRNN | `model.py` |
| Adam optimizer | `train.py` |
| BCE loss | `train.py` |
| batch size 32 | `configs/crnn_dads.yaml` |
| learning rate 0.001 | `configs/crnn_dads.yaml` |
| 150 epoch | `configs/crnn_dads.yaml` |
| early stopping patience 10 | `configs/crnn_dads.yaml` |
| seeds 42/43/44 | `configs/crnn_dads.yaml` |
| thresholds 0.40/0.50/0.65 | `metrics.py` / `evaluate_thresholds.py` |

## 10. 推荐阅读顺序

如果要理解代码，建议按下面顺序阅读：

1. `configs/crnn_dads.yaml`
2. `src/dads_crnn/prepare_manifest.py`
3. `src/dads_crnn/audio.py`
4. `src/dads_crnn/dataset.py`
5. `src/dads_crnn/features.py`
6. `src/dads_crnn/model.py`
7. `src/dads_crnn/train.py`
8. `src/dads_crnn/metrics.py`

这样可以从数据准备一路看到模型训练和评估。
