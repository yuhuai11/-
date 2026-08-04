# G7-R6按Reuter严格多数据集实验协议（已被可重复Benchmark协议替代）

> 状态：`SUPERSEDED_BEFORE_FINAL_TEST`。用户选择使用可重复外部测试展示多版本模型效果，
> 当前有效协议见`docs/G7-R6可重复多数据集Benchmark实验协议.md`。本文件仅保留旧方案记录。

## 1. 目的

G7-R6参考Reuter等人的多数据集泛化实验，但采用比论文更严格的角色隔离：模型验证、
阈值校准、可重复历史Benchmark和最终盲测四者分开。任何已经查看过结果的数据都不能
重新包装为新的Final Holdout。

## 2. 数据角色

| 层级 | 数据 | 用途 | 是否允许反向影响模型 |
|---|---|---|---|
| Train | DADS train、Kielce train、TAU train、G9 train | 更新参数和数据增强 | 是 |
| Model Validation | DADS validation、Kielce tune、TAU Lisbon tune | 早停、结构和超参数选择 | 是，仅选模 |
| Threshold Calibration | TAU Prague tune | 独立确定1%和5% FPR阈值 | 仅阈值 |
| Historical Internal Benchmark | 已消费DADS test | 域内历史回归对照 | 否 |
| Historical OOD Benchmark | 已消费Kielce/TAU holdout、G13、IDMT-TRAFFIC、ESC-50 | 可重复论文基准与误差分析 | 否，不产生新独立结论 |
| Final Locked Test | 尚未接入的新正负会话 | 冻结模型后一次性最终结论 | 否 |

TAU tune按城市划分：Lisbon只进入Model Validation，Prague只进入Threshold
Calibration。两者在录音组、来源组和音频内容上必须保持零重叠。阈值校准集只包含
背景负样本，不能用于选择网络结构、增强方法或随机种子。

## 3. 建议增加的数据

第一优先级是IDMT Berne 2022，可作为新的公开OOD正样本Benchmark，进行距离分层和
跨会话检测。但在下载前必须找到官方归档、确认许可并与DADS逐音频哈希审计。它一旦被
用于比较多个方案，就是公开Benchmark，不是最终盲测。

真正的Final Holdout必须同时增加：

1. 至少5个全新的地面阵列无人机录音会话；
2. 至少5个目标部署环境的无无人机背景会话；
3. 新地点、新日期，并优先包含新设备或新型号；
4. 保存原始连续录音、设备、距离、天气、时间和型号元数据；
5. 接入时只做许可、格式和哈希审计，不运行模型预览。

AuDroK需要先排除与DADS的内容重叠。UaVirBASE主要是阵列定位数据，DroneAudioSet主要
是机载麦克风搜救听觉数据，两者只能作为补充压力测试。NASA Small UAS在现有许可审计
解除前禁止用于训练、校准和模型推理。

## 4. 固定实验顺序

1. 仅使用Train训练seed 42、43、44；
2. 仅使用Model Validation选结构、训练轮次和增强；
3. 模型完全确定后，在Threshold Calibration上确定1%和5% FPR阈值；
4. 冻结检查点SHA256、三种子集成、阈值、0.5秒录音均值聚合与指标；
5. 可在Historical Benchmark上报告与旧版本的可重复比较；
6. Final Holdout准入完成后只运行一次，不允许根据结果重训、重设阈值或选seed。

主指标包括ROC-AUC、PR-AUC、FPR不超过5%的标准化pAUC、TPR@1%/5%校准FPR、
各背景数据集实际FPR、各无人机型号/会话Recall、录音级Recall和每小时误告警数。

## 5. 当前状态

现有数据已经完成开发层重新划分，但Final Holdout仍为空。这是有意的安全状态，而不是
缺少测试步骤。在新数据通过准入审计之前，G7-R6只能进行Train、Validation、Calibration
和历史Benchmark实验，不能产生新的独立最终测试结论。

运行：

```bash
bash scripts/run_g7_r6_protocol.sh prepare
bash scripts/run_g7_r6_protocol.sh audit
```
