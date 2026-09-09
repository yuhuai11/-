# G7严格二分类数据卡

## DADS内部数据

| 划分 | 0.5秒窗口 | 角色 |
|---|---:|---|
| Train | 292,053 | 参数训练 |
| Validation | 62,584 | 早停与检查点选择 |
| Internal Test | 62,584 | 已消费内部回归测试 |
| 合计 | 417,221 | 内容去重后的最终窗口 |

标签总数：背景222,415，无人机194,806。

主要资产：

- `artifacts/g7_leakage_fixed_v2/data/manifest.csv`；
- `artifacts/g7_leakage_fixed_v2/data/cache/native_half_second_audio.npy`；
- `artifacts/g7_leakage_fixed_v2/data/audit.json`；
- `artifacts/g7_leakage_fixed_v2/data/source_registry.csv`；
- `artifacts/g7_leakage_fixed_v2/data/duplicate_windows.csv`。

训练缓存约13GB，完整包含在发布目录中。

## 外部评价数据

| 数据 | 评价视图 | 作用 |
|---|---:|---|
| TAU Prague | 4,480 | 背景阈值校准 |
| Kielce/TAU | 38,126 | 外部无人机Recall与背景FPR |
| G13 DDL+AeroSonic | 110,000 | 外部无人机与配套背景评价 |
| IDMT Traffic | 58,692 | 交通背景FPR压力测试 |
| ESC-50 | 480 | 通用背景保护测试 |

外部音频及缓存位于：

- `artifacts/g14_domain_generalization/segment_cache/`；
- `data/external_confirmation_v2/`；
- `data/g7_cross_domain/idmt_traffic_active/development_test/`；
- `artifacts/g9_hard_negatives/cache/guard/`。

G13和IDMT清单中的原工程绝对路径已在发布副本中改为相对`data/...`路径，使其能从发布包根目录读取。原工程清单没有被修改。

## 数据可信度边界

- DADS已排除跨split的精确原始音频、float32输入、PCM16等价输入和内容组件交集；
- 近重复审计不能替代缺失的采集会话元数据；
- Kielce正类和TAU负类存在“来源与标签绑定”的混杂；
- G13部分来源组为标签纯组；
- IDMT和ESC-50只有负类；
- 所有外部集合均为已消费开发Benchmark。

## 数据许可提醒

本目录是本地科研整理包，不自动授予公开再分发权。对外复制或发表数据前，必须分别核对DADS、Kielce、TAU、DDL/AeroSonic、IDMT和ESC-50的原始许可证、署名要求及使用范围。模型结果可以引用，但原始音频是否可随论文公开需要单独确认。

