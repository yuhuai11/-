# G7严格二分类运行手册

## 环境

整理时验证环境：

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.5.1+cu121 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| PyArrow | 24.0.0 |
| PyYAML | 6.0.3 |

默认Python：`/usr/local/anaconda3/envs/dads-crnn/bin/python`。

## 完整性验证

快速检查文件存在、大小和小文件哈希：

```bash
bash scripts/run.sh verify-fast
```

完整检查包括13GB缓存和模型哈希：

```bash
bash scripts/run.sh verify
```

## 对单条WAV推理

```bash
bash scripts/run.sh infer /absolute/path/example.wav
```

可选参数：

```bash
bash scripts/run.sh infer /absolute/path/example.wav \
  --seed 42 \
  --threshold 0.5 \
  --device cuda \
  --output prediction.json
```

输出同时包含每个0.5秒窗口概率和非重叠1秒均值决策。最后不足0.5秒的尾段使用零填充，并在结果中记录真实时长。

## GPU预检

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh preflight
```

预检读取完整DADS缓存并执行一次真实前向、反向和优化器步骤。

## 重新训练

警告：重新训练会在发布包内部的`artifacts/g7_strict_retrain_v1/runs/`写入结果。若要保留发布检查点不变，应先复制发布包或修改配置中的`output_dir`。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh train
```

中断后恢复：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh resume
```

## 重新执行外部评价

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh external
bash scripts/run.sh aggregate-one-second
```

现有预测和指标已经保存在发布包内。重新评价仍属于对已消费Benchmark的复验，不会变成新的独立测试。

## 回归测试

```bash
bash scripts/run.sh test
```

