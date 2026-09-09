# 工程文档索引

本目录从2026-09-09起按“当前正式内容、实验状态、历史归档、论文资料”组织。历史文档被移动到归档区，不表示其数据或结论被删除；它们继续用于实验追溯、消融说明和论文写作。

## 当前二分类模型

- [G7二分类模型原理、创新、实验结果与改进分析](current/binary_g7/G7二分类模型原理创新实验结果与改进分析.md)
- [G7二分类内部与外部测试结果及论文对比](current/binary_g7/G7二分类内外部测试结果与三篇论文对比分析.md)
- [G7严格重训练实验记录](current/binary_g7/G7严格重训练实验记录.md)
- [G7严格重训外部基线与1秒聚合实验报告](current/binary_g7/G7严格重训外部基线与1秒聚合实验报告.md)
- [DADS历史口径更正与精确内容泄漏修复](current/binary_g7/DADS历史测试口径更正与G7精确内容泄漏修复重训报告.md)
- [G7数据泄露与结果虚高修整报告](current/binary_g7/G7数据泄露与结果虚高修整报告.md)
- [G7-R9来源类别配额跨域优化方案与记录](current/binary_g7/G7-R9来源类别配额跨域优化实验方案与执行记录.md)

当前可独立交付的最佳二分类版本位于`releases/g7_binary_best_strict_v1/`。它与本目录的当前文档共同构成正式入口。

## 未晋级实验

- [G7-R7频率MixStyle单变量实验](experiments/failed/G7-R7频率MixStyle单变量实验报告.md)
- [G7-R8城市负样本实验](experiments/failed/G7-R8城市负样本评估方案.md)

这里保留未晋级实验的设计、指标和失败原因，避免重复尝试。恢复训练检查点不作为长期科研记录。

## 历史归档

- `archive/g2_g6_legacy/`：早期CRNN及轻量改进方案。
- `archive/g7_evolution/`：G7从R0到R6的演进、旧方案和已被严格协议替代的报告。
- `archive/g8_g17_binary/`：G8至G17的二分类泛化、校准和受约束适配研究。
- `archive/model_identification_g18_g22/`：G18至G22型号识别与开放集研究；当前二分类主线不依赖它们。
- `archive/general/`：旧阶段总结、通用规划和工程整理记录。

归档内容默认只读。若重新启用某个方案，应先在实验登记表中将状态改为`ACTIVE`，再恢复到当前开发区。

## 论文与外部资料

- `references/papers/`：工程内保存的论文。
- `references/reviews/`：论文综述、对比说明和配图。

## 登记与清理记录

- `registries/experiment_registry.csv`：实验状态、模型保留策略和文档位置。
- `registries/cleanup_inventory_before_2026-09-09.csv`：清理前文件大小、SHA256及inode清单。
- `registries/cleanup_inventory_after_2026-09-09.csv`：清理后复核清单；当前只有表头，表示目标临时文件和`last.pt`均已清除。
- `registries/cleanup_record_2026-09-09.md`：本次清理范围、空间变化和恢复说明。
