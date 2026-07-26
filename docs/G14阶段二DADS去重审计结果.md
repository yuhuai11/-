# G14 阶段二：DADS 去重审计结果

## 1. 结论

`dads_dedup_v2` 已生成并通过审计：

> **通过：精确重复原始音频已删除，train、val、test之间的原始音频SHA、录音组和来源路径交叉均为0。**

本阶段只读取 DADS parquet 和历史 manifest，复用既有音频缓存，没有覆盖历史
manifest，没有改写缓存，也没有启动训练。

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

该方法不会重新导出或复制19GB音频缓存。

## 4. 重复情况

审计结果：

- 唯一原始音频：180,305；
- 重复SHA组：15；
- 重复记录移除：15；
- 原来跨split的重复SHA组：6；
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

相对于原数据仅减少244/305,006个片段，约为0.08%，类别与split分布没有发生
实质性改变。

## 6. 隔离与完整性

去重后：

- train/val 原始音频SHA交叉：0；
- train/test 原始音频SHA交叉：0；
- val/test 原始音频SHA交叉：0；
- 三个split的 `recording_group` 交叉：0；
- 三个split的 `source_path` 交叉：0；
- 缓存缺失：0；
- 标签冲突重复组：0。

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
实验。后续 G14-A 可以使用：

```text
artifacts/dads_dedup_v2/train_manifest.csv
```

但在接入新的G14多来源开发数据并完成冻结G7诊断之前，不启动G14-A训练。

## 10. 下一步

进入G14新多来源开发数据接入阶段：

1. 列出与 AeroSonicDB、DDL Real Data 独立的候选来源；
2. 核验标签、设备/会话元数据、许可证和下载完整性；
3. 建立统一source registry；
4. 与 DADS、G9、G13和旧最终集进行哈希排重；
5. 按来源组生成train、tune、holdout，不运行模型预测。
