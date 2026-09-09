# G14 阶段二：DADS 去重审计结果

## 1. 结论

`dads_dedup_v2` 已生成并通过审计：

> **通过：15个精确重复SHA组均只保留一个代表；处理后train、val、test之间的原始
> 音频SHA、录音组和来源路径交叉均为0。**

本阶段只读取 DADS parquet 和历史 manifest，复用既有音频缓存，没有覆盖历史
manifest，没有改写缓存，也没有启动训练。

这里的“通过”只针对原始WAV精确哈希去重和清单完整性。该版本保留原来的split及历史
1秒loop/tail-loop缓存，不是重新划分后的独立测试协议，也没有消除后来发现的类别相关
时长捷径。

## 2. 输入

历史全量 manifest：

```text
artifacts_full/manifests/dads_all_seed42.csv
```

输入 SHA256：

```text
be1a6293b90a15208c75d4f2aa8cfe4efefc68c8cc382ca57f9c637e08f09c45
```

原始规模：

| 单位 | 总数 | label 0 | label 1 |
|---|---:|---:|---:|
| 原始录音记录 | 180,320 | 16,729 | 163,591 |
| 1秒片段 | 305,006 | 125,314 | 179,692 |

## 3. 去重方法

1. 以 `(parquet_file, row_group, row_in_group)` 唯一定位原始记录；
2. 读取 parquet 中的原始 WAV 字节并计算 SHA256；
3. 将相同 SHA256 视为相同原始音频；
4. 对重复组只保留一个确定性代表；
5. 优先保留能维持原类别split比例的代表；
6. 保留代表原来的split和现有cache路径；
7. 为每个保留来源增加：
   - `raw_audio_sha256`；
   - `recording_group=dads_raw_sha256:<sha256>`；
   - `original_split`。

其余非重复来源也全部保留原split。因此该方法不会对180,305个唯一来源做全局重划，
也不会重新导出或复制19GB音频缓存。

## 4. 重复情况

审计结果：

- 唯一原始音频：180,305；
- 重复SHA组：15，共涉及30条原始记录；
- 重复记录移除：15；
- 原来跨split的重复SHA组：6，即组内成员分布在至少两个split；
- 移除1秒片段：244；
- 重复组中label 0组数：14；
- 重复组中label 1组数：1。

移除片段分布：

| 标签 | split | 移除片段 |
|---:|---|---:|
| 0 | train | 144 |
| 0 | val | 48 |
| 0 | test | 32 |
| 1 | train | 20 |

## 5. 去重后规模

| 单位 | 总数 | label 0 | label 1 |
|---|---:|---:|---:|
| 唯一原始录音 | 180,305 | 16,715 | 163,590 |
| 1秒片段 | 304,762 | 125,090 | 179,672 |

分split片段数：

| split | 总数 | label 0 | label 1 |
|---|---:|---:|---:|
| train | 212,338 | 87,830 | 124,508 |
| val | 46,565 | 18,622 | 27,943 |
| test | 45,859 | 18,638 | 27,221 |

表中的`test`是为兼容历史清单保留的split字段，角色是已消费内部回归，不是独立最终
测试集。

相对于原数据仅减少244/305,006个片段，约为0.08%，类别与split分布没有发生
实质性改变。这只能说明raw-WAV去重对样本数量影响很小，不能据此推断旧G7的高指标
不受1秒循环补齐时长捷径影响。

## 6. 隔离与完整性

去重后：

- train/val 原始音频SHA交叉：0；
- train/test 原始音频SHA交叉：0；
- val/test 原始音频SHA交叉：0；
- 三个split的 `recording_group` 交叉：0；
- 三个split的 `source_path` 交叉：0；
- 缓存缺失：0；
- 标签冲突重复组：0。

其中`source_path`交集为0是对产出清单的后验检查结果，不是历史清单按
`source_path`显式分组的证明。“缓存缺失为0”也只表示历史缓存文件存在；本阶段没有
逐条验证最终float32/PCM16模型输入内容是否等价。

## 7. 产物

```text
artifacts/dads_dedup_v2/
├── source_registry.csv
├── duplicate_groups.csv
├── dads_dedup_v2_all.csv
├── train_manifest.csv
├── val_manifest.csv
├── test_manifest.csv
└── audit.json
```

关键产物：

- combined manifest：304,762行；
- combined manifest SHA256：
  `45c858f63bae8f87923aab83d3adedcac4dd71be9a8c057f9b1a3b7acc565113`；
- train manifest SHA256：
  `8e093d56446e37a375f5455e1b2197b50a6e845528fc18319f4f0dfb95670a23`；
- val manifest SHA256：
  `b0ed1dd1bcb85a41f5f241ce28c047f4d321efc101ae7880239358a7331aa8c8`；
- test manifest SHA256：
  `0d30e7dfaaa912f596a293a229702dc06437b2047e632223812d766f274d4054`。

执行入口：

```text
scripts/run_g14_dads_dedup_v2.sh
```

## 8. 验证

- `dads_dedup_v2` 专项测试：3项通过；
- 工程完整单元测试：93项通过；
- Python源码编译检查：通过；
- PANNs训练入口身份审计通过：212,338行，未创建DataLoader；
- 历史DADS manifest SHA256保持不变；
- `training_started=false`。

## 9. 使用限制

`dads_dedup_v2` 是新的开发数据版本，不追溯修改 G7、G9、G12 或 G13 的历史
实验。它后来作为G14-A历史开发流程的DADS训练输入：

```text
artifacts/dads_dedup_v2/train_manifest.csv
```

其`test_manifest.csv`沿用已被历史开发消费的数据池，只能称为已消费内部回归划分。
该版本不能被解释为独立最终测试，也不能替代后来的
`dads_native_half_second_content_component_v2`。

## 10. 当时记录的下一步

本节保留阶段二完成时登记的后续步骤，用于历史追溯，不表示当前工程仍停留在这里：

1. 列出与 AeroSonicDB、DDL Real Data 独立的候选来源；
2. 核验标签、设备/会话元数据、许可证和下载完整性；
3. 建立统一source registry；
4. 与 DADS、G9、G13和旧最终集进行哈希排重；
5. 按来源组生成train、tune、holdout，不运行模型预测。
